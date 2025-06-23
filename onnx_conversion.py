from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/root/work/runs/yolo11s.pt")

# Export the model to ONNX format
model.export(format="onnx")  # model will be saved in the same directory as "best.onnx"