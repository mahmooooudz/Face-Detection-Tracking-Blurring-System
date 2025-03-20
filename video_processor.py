import cv2
from face_tracker import AdvancedFaceTracker

class VideoProcessor:
    def __init__(self, video_path):
        """
        Initialize video processor with advanced face tracking
        
        Args:
            video_path (str): Path to input video file
        """
        self.video_path = video_path
        
        # Initialize advanced face tracker
        self.tracker = AdvancedFaceTracker(
            similarity_threshold=0.5,  # Adjust as needed
            max_lost_frames=30
        )
        
        # Open video capture
        self.cap = cv2.VideoCapture(video_path)
        
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def process_video(self, output_path=None, show_bbox=True):
        all_tracked_faces = {}
        frame_count = 0

        # تهيئة VideoWriter إذا تم تحديد مسار للفيديو الخارج
        if output_path:
            # احصل على ارتفاع وعرض الفريم من الكاميرا
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            tracked_faces = self.tracker.track_faces(frame, frame_count)
            for face_id, face_info in tracked_faces.items():
                if face_id in all_tracked_faces:
                    all_tracked_faces[face_id].append(face_info)
                else:
                    all_tracked_faces[face_id] = [face_info]

            frame_count += 1
            
            # رسم معلومات التتبع على الفريم إذا مطلوب
            if show_bbox:
                for face_info in tracked_faces:
                    cv2.rectangle(frame, tuple(face_info['bbox'][0:2]), tuple(face_info['bbox'][2:4]), (0, 255, 0), 2)

            # كتابة الفريم إذا تم توفير مسار خارج
            if output_path:
                out.write(frame)
            
        self.cap.release()
        # تأكد من إغلاق الكائن out بعد الانتهاء من التسجيل
        if output_path:
            out.release()

        return all_tracked_faces
    
    def get_face_recurrence_summary(self):
        """
        Retrieve face recurrence summary
        
        Returns:
            dict: Detailed face recurrence information
        """
        return self.tracker.get_face_recurrence_summary()