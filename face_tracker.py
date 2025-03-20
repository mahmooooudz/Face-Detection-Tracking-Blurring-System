import numpy as np
import cv2
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from datetime import datetime
import torch
import random

class AdvancedFaceTracker:
    def __init__(self, similarity_threshold=0.5, max_lost_frames=30):
        """
        Initialize the advanced face tracker
        
        Args:
            similarity_threshold (float): Embedding similarity threshold for face matching
            max_lost_frames (int): Maximum frames a tracked face can be lost before being removed
        """
        # Initialize face analysis model
        self.face_detector = YOLO('face_yolov8n.pt')


        
        # Tracking parameters
        self.tracked_faces = {}
        self.next_id = 1
        self.similarity_threshold = similarity_threshold
        self.max_lost_frames = max_lost_frames
        self.appearances = {}
        
        # Recurrence logging
        self.face_recurrence_log = {}
        
        # Custom names for faces
        self.face_custom_names = {}

    def detect_and_extract_faces(self, frame):
        """
        Detect faces in the frame using YOLO
        """
        results = self.face_detector(frame)
        faces = []
        for detection in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence = map(int, detection[:5])
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
            })
        return faces

    def compute_embedding_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two face embeddings
        
        Args:
            emb1 (numpy.ndarray): First face embedding
            emb2 (numpy.ndarray): Second face embedding
        
        Returns:
            float: Similarity score between 0 and 1
        """
        return 1 - cosine(emb1, emb2)
    

    def get_face_color(self, face_id):
        """
        Get a unique color for each face ID (consistent with tracking).
        Ensures the same RGB color is used in both OpenCV and Tkinter.
        """
        np.random.seed(face_id)
        base_color = np.random.randint(50, 200, 3).tolist()  # Generate base RGB color

        # Convert to HEX format for Tkinter UI
        return f"#{base_color[2]:02x}{base_color[1]:02x}{base_color[0]:02x}"  # Reverse order (BGR to RGB)



    def track_faces(self, frame, current_frame_index):
        detected_faces = self.detect_and_extract_faces(frame)
        current_tracked_faces = []

        for face in detected_faces:
            bbox = list(map(int, face['bbox']))
            confidence = face.get('confidence', 0.0)
            best_match_id = None
            best_iou = 0.0

            for tracked_id, tracked_info in self.tracked_faces.items():
                tracked_bbox = tracked_info['bbox']
                iou = self.compute_iou(bbox, tracked_bbox)

                if iou > best_iou and iou > 0.4:
                    best_match_id = tracked_id
                    best_iou = iou

            if best_match_id:
                face_id = best_match_id
                self.tracked_faces[face_id]['bbox'] = bbox

                # Detect recurrence only if the face was absent for `max_lost_frames`
                if self.tracked_faces[face_id]['lost_counter'] > self.max_lost_frames:
                    self.appearances[face_id] = self.appearances.get(face_id, 0) + 1

                # Reset last seen & lost counter
                self.tracked_faces[face_id]['last_seen'] = current_frame_index
                self.tracked_faces[face_id]['lost_counter'] = 0

                self._update_face_recurrence_log(face_id, bbox)
            else:
                face_id = self.next_id
                self.tracked_faces[face_id] = {
                    'bbox': bbox,
                    'last_seen': current_frame_index,
                    'lost_counter': 0  # Initialize lost counter
                }
                self.appearances[face_id] = 1  # First appearance of new face
                self.next_id += 1

            tracked_face_info = {
                'id': face_id,
                'bbox': bbox,
                'confidence': confidence,
                'name': self.face_custom_names.get(face_id, f"Face {face_id}")
            }
            current_tracked_faces.append(tracked_face_info)

        # Update lost counter for each tracked face
        for tracked_id in self.tracked_faces.keys():
            if 'lost_counter' in self.tracked_faces[tracked_id]:
                self.tracked_faces[tracked_id]['lost_counter'] += 1
            else:
                self.tracked_faces[tracked_id]['lost_counter'] = 1  # Initialize if missing



        # Remove faces that have not been seen for more than max_lost_frames
        self.tracked_faces = {id: info for id, info in self.tracked_faces.items() if current_frame_index - info['last_seen'] <= self.max_lost_frames}
        return current_tracked_faces

    def _update_face_recurrence_log(self, face_id, bbox):
        """
        Update recurrence log for an existing face
        
        Args:
            face_id (int): Unique face identifier
            bbox (list): Bounding box coordinates
        """
        if face_id not in self.face_recurrence_log:
            self._create_face_recurrence_log(face_id, bbox)
        else:
            log = self.face_recurrence_log[face_id]
            log['appearances'] += 1
            log['last_seen'] = datetime.now()
            log['locations'].append(bbox)

    def _create_face_recurrence_log(self, face_id, bbox):
        """
        Create initial recurrence log for a new face
        
        Args:
            face_id (int): Unique face identifier
            bbox (list): Bounding box coordinates
        """
        self.face_recurrence_log[face_id] = {
            'first_seen': datetime.now(),
            'last_seen': datetime.now(),
            'appearances': 1,
            'locations': [bbox]
        }

    def rename_face(self, face_id, new_name):
        """
        Rename a tracked face
            
        Args:
            face_id (int): Unique face identifier
            new_name (str): New name for the face
        """
        if face_id in self.tracked_faces:
            self.face_custom_names[face_id] = new_name

    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list): [x1, y1, x2, y2] for the first box
            box2 (list): [x1, y1, x2, y2] for the second box

        Returns:
            float: IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def get_face_recurrence_summary(self):
        """
        Generate a comprehensive summary of face recurrences
        
        Returns:
            dict: Detailed face recurrence information
        """
        summary = {}
        for face_id, log in self.face_recurrence_log.items():
            summary[face_id] = {
                'total_appearances': self.appearances.get(face_id, 1),  # Start at 1 since the first appearance counts
                'first_seen': log.get('first_seen', "N/A"),
                'last_seen': log.get('last_seen', "N/A"),
                'name': self.face_custom_names.get(face_id, f'Face {face_id}'),
                'unique_locations': len(log.get('locations', []))
            }
        return summary

    def draw_tracking_info(self, frame, tracked_faces, draw_bbox=True):
        """
        Draw modern bounding boxes with rounded corners, dynamic text size, and highly saturated colors.
        """
        for face in tracked_faces:
            if 'id' not in face:
                continue

            x1, y1, x2, y2 = face['bbox']
            face_id = face['id']
            label = self.face_custom_names.get(face_id, f"Face {face_id}")

            # Assign a unique color per ID with increased saturation
            np.random.seed(face_id)
            base_color = np.random.randint(50, 200, 3).tolist()  # RGB color
            hsv_color = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_RGB2HSV)
            hsv_color[0][0][1] = 255  # Max saturation
            color = tuple(cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0].tolist())

            # ✅ ارسم الـ ID حتى لو الـ bounding box متوقف
            # Adjust font size dynamically based on face size
            face_width = x2 - x1
            font_scale = max(0.8, face_width / 200)  # Minimum size 0.8, increases with face width
            font_thickness = max(2, int(font_scale * 2))

            # Text background for better readability
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x_end = x1 + text_size[0] + 10
            text_y_start = y1 - text_size[1] - 5

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, text_y_start), (text_x_end, y1), color, -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # ✅ النص يفضل ظاهر مهما كان وضع الـ bounding box
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            # ✅ لو المستخدم مفعل الـ bounding boxes، نرسم الإطار
            if draw_bbox:
                thickness = 2
                radius = 15  # Corner radius

                # Draw rounded corners using polylines
                corner_points = np.array([
                    [x1 + radius, y1], [x2 - radius, y1], [x2, y1 + radius],
                    [x2, y2 - radius], [x2 - radius, y2], [x1 + radius, y2],
                    [x1, y2 - radius], [x1, y1 + radius]
                ], np.int32)

                cv2.polylines(frame, [corner_points], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        return frame
