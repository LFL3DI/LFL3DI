from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path="/home/g05/LFL3DI/yolov8s_custom/weights/best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)

    def detect_objects(self, image):
        results = self.model(image, conf=0.5)
        return results
