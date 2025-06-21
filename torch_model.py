import cv2
import torch
from torchvision import transforms
from ultralytics.nn.tasks import DetectionModel  # YOLOv8 检测模型

# 从配置文件重建模型（需与训练时一致）
model = DetectionModel(cfg='yolov8s.yaml')  # cfg 可以是字典或文件路径

# 加载参数字典（通常保存在 'model' 键中）
ckpt = torch.load('./runs/detect/train2/weights/best.pt')
# 加载参数到模型
model.load_state_dict(ckpt['model'].state_dict())

# Load your image
image_path = "./datasets/coco128/images/train2017/000000000308.jpg"
image = cv2.imread(image_path)

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image
image = cv2.resize(image, (640, 640))

# Convert to tensor
image_tensor = transforms.ToTensor()(image)

# Add batch dimension and convert to float
image_tensor = image_tensor.unsqueeze(0).float()

# Perform inference
with torch.no_grad():
    preds = model(image_tensor)
    print(preds)