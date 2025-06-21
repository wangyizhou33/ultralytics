from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8.yaml")  # build a new model from YAML
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco128.yaml", epochs=5, imgsz=640)