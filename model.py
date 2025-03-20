import cv2
import numpy as np
import torch
from ultralytics import YOLO

class FaceDetector:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('face_yolov8n.pt')

    def detect_faces(self, frame):
        """
        Detect faces in the frame using YOLO
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            list: Detected faces with bounding boxes and confidence
        """
        results = self.model(frame)
        faces = []
        for detection in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence = map(int, detection[:5])
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
            })
        return faces