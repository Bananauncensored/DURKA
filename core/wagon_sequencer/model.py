from ultralytics import YOLO

class WagonDetectionModel:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def count_boxes(self, image_path: str) -> int:
        result = self.model.predict(image_path, verbose=False)[0]
        return len(result.boxes) if result.boxes else 0
