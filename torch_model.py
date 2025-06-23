import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from typing import Tuple, List
import random

class YOLOv8Detector:
    def __init__(self, model_path: str = 'yolo11.yaml', weights_path: str = '/root/work/runs/yolo11s.pt'):
        """Initialize YOLOv8 detection model with configuration and weights"""
        # 直接从 .pt 权重文件加载完整 DetectionModel，避免维度不匹配
        self.ckpt = torch.load(weights_path, map_location='cpu')
        self.model = self.ckpt['model'] if isinstance(self.ckpt, dict) and 'model' in self.ckpt else self.ckpt
        assert isinstance(self.model, DetectionModel), 'checkpoint does not contain DetectionModel'
        # self.model.fuse() # 提升推理速度，可选但推荐。
        self.model.float() # 保证 dtype 统一与算子支持度，通常必要；除非你已经确定要全链路 FP16。
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.confidence_thres = 0.25
        self.iou_thres = 0.5
        self.input_width = 640
        self.input_height = 640

        # 类别与调色板
        self.classes = self.model.names  # 类别名
        random.seed(42)
        self.color_palette = [tuple(random.randint(0,255) for _ in range(3)) 
                              for _ in range(len(self.classes))]

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """Draw bounding boxes (left, top, width, height) and labels on the image."""
        x1, y1, w, h = box  # postprocess 给出的格式
        x2, y2 = x1 + w, y1 + h

        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        label = f"{self.classes[class_id]}: {score:.2f}"

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = int(x1)
        label_y = int(y1) - 10 if y1 - 10 > label_height else int(y1) + 10

        cv2.rectangle(
            img,
            (label_x, label_y - label_height),
            (label_x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            pad (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, output: List[np.ndarray], pad: Tuple[int, int]) -> np.ndarray:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def predict(self, image_path: str):
        self.input_image = image_path
        img_data, pad = self.preprocess()
        print(img_data.shape)
        print(img_data)
        # """Run inference on a single image"""
        # # Load image
        # img = cv2.imread(image_path)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # # Letterbox and get padding
        # input_image, pad = self.letterbox(img_rgb)
        
        # # Convert to tensor
        input_tensor = torch.from_numpy(img_data).to(torch.float32).to(self.device)
        # input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # # Run inference
        with torch.no_grad():
            preds = self.model(input_tensor)[0]  # shape: (1, 84, 8400)

        # 将 tensor 转成 numpy 并包装成 list，符合 postprocess 需求
        preds_np = preds.detach().cpu().numpy()
        result_img = self.postprocess(self.img, [preds_np], pad)
        
        # # Save and show result
        cv2.imwrite('result.jpg', result_img)
        return result_img

# Run detection
if __name__ == '__main__':
    detector = YOLOv8Detector()
    detector.predict("/root/work/datasets/coco128/images/train2017/000000000308.jpg")
