from ultralytics import YOLO

# Load YOLO model once
yolo_model = YOLO("models/detection_model.pt")

def detect_objects(image_path, conf=0.20):
    """Returns list of detected class names for given image path."""
    results = yolo_model.predict(image_path, conf=conf)[0]
    class_ids = results.boxes.cls.tolist()
    class_names = [results.names[int(cid)] for cid in class_ids]
    return class_names
