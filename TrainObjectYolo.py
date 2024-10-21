from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('Models/yolov8n.pt')  # You can choose yolov8s.pt for better accuracy
device = "cpu"
model.to(device)
# Train the model
model.train(data='Models/custom_yolo_dataset.yaml', epochs=50, imgsz=640)
