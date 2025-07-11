from ultralytics import YOLO
from PIL import Image


# Define path to the image file
source = "./datasets/coco128/images/train2017/000000000308.jpg"

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train2/weights/best.pt")

# Run inference on the source
results1 = model(source)  # list of Results objects

# Load the exported ONNX model
onnx_model = YOLO("./runs/detect/train2/weights/best.onnx")

# Run inference
results2 = onnx_model(source)

# Load the exported TensorRT model
tensorrt_model = YOLO("./runs/detect/train2/weights/best.engine")

# # Run inference
results3 = tensorrt_model(source)


# print(results1)
# print(results2)
print(results3)

# # Or equivalent to the following line
# # Run inference
# # model.predict(source, save=True, imgsz=320, conf=0.5)

# Visualize the results
for i, r in enumerate(results3):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()