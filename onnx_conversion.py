from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
# Load the YOLO11 model
model = YOLO("./runs/detect/train2/weights/best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # model will be saved in the same directory as "best.onnx"

# Export the model to TensorRT format
model.export(format="engine", device=0)  # creates 'yolo11n.engine'
