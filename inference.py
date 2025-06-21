from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./runs/detect/train/weights/best.pt")

# Define path to the image file
source = "./datasets/coco128/images/train2017/000000000308.jpg"

# Run inference on the source
results = model(source)  # list of Results objects

print(results)

# Run inference on 'bus.jpg' with arguments
model.predict(source, save=True, imgsz=320, conf=0.5)