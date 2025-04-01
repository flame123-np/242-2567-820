from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")

# Perform object detection on an image
results = model("test/testbutter.jpg")
results[0].show()