import torch
import cv2
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np
import os
import sys
from face_tracker import AdvancedFaceTracker
from video_processor import VideoProcessor
import tkinter.simpledialog as simpledialog
SETTINGS_FILE = "settings.txt"


try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not found. Please install it using: pip install ultralytics")
    sys.exit(1)


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




class FaceBlurApp:
    def __init__(self, root):
        # ✅ Load saved theme mode
        self.current_mode = self.load_theme_mode()

        # ✅ Apply theme mode
        ctk.set_appearance_mode(self.current_mode)
        ctk.set_default_color_theme("blue")

        # GPU Device Setup - FIRST THING
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Root window setup
        self.root = root
        self.root.title("Falcon-9")
        self.root.geometry("1200x800")

        # Initialize Advanced Face Tracker
        self.face_tracker = AdvancedFaceTracker(
            similarity_threshold=0.5,
            max_lost_frames=30
        )
        
        # Load model once
        self.model = self.load_model()

        # Main frame with padding
        self.main_frame = ctk.CTkFrame(root, corner_radius=10)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Create a grid layout within the main frame
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Header
        self.header = ctk.CTkLabel(self.main_frame, 
                                   text="Falon-9", 
                                   font=("Arial", 24, "bold"),
                                   text_color=("black" if self.current_mode == "Light" else "white"))
        self.header.grid(row=0, column=0, columnspan=3, pady=20, padx=20, sticky="ew")

        # Sidebar for controls
        # ✅ Load theme icons
        self.theme_icon_light = ctk.CTkImage(light_image=Image.open("sun.png"), size=(30, 30))
        self.theme_icon_dark = ctk.CTkImage(light_image=Image.open("moon.png"), size=(30, 30))

        # ✅ Add theme toggle button
        self.theme_button = ctk.CTkButton(
            self.root, text="",
            width=40, height=40,
            fg_color="transparent",  # 🔥 Remove background
            hover_color="gray",  # 🔥 Hover effect
            image=self.theme_icon_light if self.current_mode == "Dark" else self.theme_icon_dark,
            command=self.toggle_theme
        )
        self.theme_button.place(relx=0.95, rely=0.05, anchor="ne")  # 🔥 Adjust position
        self.sidebar_frame = ctk.CTkFrame(self.main_frame, width=200, corner_radius=10)
        self.sidebar_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        # Create sidebar buttons and UI elements
        self.create_sidebar_buttons()

        # Video and model state initialization
        self.initialize_video_state()

        # Add a new method for renaming faces
        self.rename_face_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Rename Face", 
            command=self.rename_face,
            corner_radius=10
        )
        self.rename_face_button.grid(row=11, column=0, padx=20, pady=10, sticky="ew")

        self.current_frame_index = 0






    def rename_face(self):
        """
        Allow renaming of a selected face
        """
        selected_indices = self.face_id_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select a face to rename")
            return
        
        selected_face = self.detected_faces[selected_indices[0]]
        face_id = selected_face.get('id', 'Unknown')
        
        new_name = simpledialog.askstring(
            "Rename Face", 
            f"Enter new name for Face {face_id}:"
        )
        
        if new_name:
            # Update the name in face tracker
            self.face_tracker.rename_face(face_id, new_name)
            
            # Refresh the listbox to reflect the changes
            self.update_face_list(self.detected_faces)


    def save_theme_mode(self, mode):
        """Save the selected theme mode to a file."""
        with open(SETTINGS_FILE, "w") as f:
            f.write(mode)

    def load_theme_mode(self):
        """Load the saved theme mode from a file."""
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return f.read().strip()
        return "Dark"  # Default mode



    def toggle_theme(self):
        """Toggle between Dark and Light mode."""
        self.current_mode = "Light" if self.current_mode == "Dark" else "Dark"

        # ✅ Apply theme mode
        ctk.set_appearance_mode(self.current_mode)

        # ✅ Change icon
        self.theme_button.configure(
            image=self.theme_icon_light if self.current_mode == "Dark" else self.theme_icon_dark,
            bg_color="transparent"  # 🔥 تأكيد عدم وجود خلفية شفافة غير مرغوبة
        )

        # ✅ تحديث لون الهيدر بناءً على المود
        self.header.configure(text_color=("black" if self.current_mode == "Light" else "white"))

        # ✅ تحديث ألوان الـ Listbox الخاصة بـ Detected Faces
        self.update_face_list(self.detected_faces)

        # ✅ Save mode
        self.save_theme_mode(self.current_mode)





    def create_sidebar_buttons(self):
        # Load Video Button
        self.load_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Load Video", 
            command=self.load_video,
            corner_radius=10
        )
        self.load_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        # Add a checkbox for bounding box visibility during export
        self.show_bbox_var = ctk.BooleanVar(value=True)
        self.show_bbox_checkbox = ctk.CTkCheckBox(
            self.sidebar_frame, 
            text="Show Bounding Boxes", 
            variable=self.show_bbox_var
        )
        self.show_bbox_checkbox.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        # Face Recurrence Button
        self.recurrence_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Face Recurrence", 
            command=self.show_face_recurrence_info,
            corner_radius=10,
            fg_color="orange"
        )
        self.recurrence_button.grid(row=10, column=0, padx=20, pady=10, sticky="ew")

        # Blur Face Button
        self.blur_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Blur Face", 
            command=self.blur_face,
            corner_radius=10,
            fg_color="red"
        )
        self.blur_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Export Video Button
        self.export_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Export Video", 
            command=self.export_video,
            corner_radius=10,
            fg_color="purple"
        )
        self.export_button.grid(row=8, column=0, padx=20, pady=10, sticky="ew")

        # Unblur Face Button
        self.unblur_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Unblur Face", 
            command=self.unblur_face,
            corner_radius=10,
            fg_color="green"
        )
        self.unblur_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Play/Pause Button
        self.play_pause_button = ctk.CTkButton(
            self.sidebar_frame, 
            text="Play", 
            command=self.toggle_play_pause,
            corner_radius=10
        )
        self.play_pause_button.grid(row=6, column=0, padx=20, pady=10, sticky="ew")

        # Video Progress Slider
        self.progress_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0, 
            to=100, 
            command=self.seek_video
        )
        self.progress_slider.grid(row=7, column=0, padx=20, pady=10, sticky="ew")

        # Face Detection List Label
        self.face_list_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Detected Faces",
            font=("Arial", 16)
        )
        self.face_list_label.grid(row=4, column=0, padx=20, pady=(10,5), sticky="w")

        # Face ID Listbox
        self.face_id_listbox = tk.Listbox(
            self.sidebar_frame, 
            height=10, 
            font=("Arial", 12),
            bg="#333333", 
            fg="white",
            selectbackground="#4CAF50"
        )
        self.face_id_listbox.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")


    def export_video(self):
        if not self.video_source:
            messagebox.showwarning("Export Error", "No video loaded!")
            return

        # Ask user for export location
        export_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("Video Files", "*.mp4")]
        )

        if not export_path:
            return

        try:
            # Reopen the original video
            cap = cv2.VideoCapture(self.video_source)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(export_path, fourcc, fps, (width, height))

            # Progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Exporting Video")
            progress_window.geometry("300x100")
            progress_window.grab_set()

            progress_label = tk.Label(progress_window, text="Exporting video...")
            progress_label.pack(pady=10)

            progress_bar = tk.ttk.Progressbar(
                progress_window, 
                orient="horizontal", 
                length=250, 
                mode="determinate"
            )
            progress_bar.pack(pady=10)

            # Reset video to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process and export each frame
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame with or without bounding boxes
                processed_frame, detected_faces = self.process_frame(
                    frame, draw_bbox=self.show_bbox_var.get()
                )
                
                # Write the processed frame
                out.write(processed_frame)

                # Update progress
                progress = int((frame_num / total_frames) * 100)
                progress_bar["value"] = progress
                progress_window.update()

            # Clean up
            cap.release()
            out.release()
            progress_window.destroy()

            messagebox.showinfo("Export Successful", f"Video exported to {export_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export video: {e}")




    def initialize_video_state(self):
        """Initialize video and model-related state variables"""
        # Video Display Area
        self.video_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.video_frame.grid(row=1, column=1, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.canvas = tk.Canvas(
            self.video_frame, 
            bg="black", 
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Model and Video Initialization
        self.video_source = None
        self.cap = None
        self.frame = None
        self.original_frame = None
        self.detected_faces = []

        # Video playback state variables
        self.is_playing = False
        self.current_frame = 0
        self.fps = 30  # Default fps if not set
        self.total_frames = 0

        # Dictionary to store blurred faces
        self.blurred_faces = {}



    def update_face_list(self, detected_faces):
        """
        Update the face ID listbox with detected faces, applying the correct colors.
        """
        # Clear existing items
        self.face_id_listbox.delete(0, tk.END)

        for face in detected_faces:
            face_id = face.get('id', 'Unknown')
            name = self.face_tracker.face_custom_names.get(face_id, f"Face {face_id}")
            color = self.face_tracker.get_face_color(face_id)  # Get consistent color
            
            # Insert face name into listbox
            self.face_id_listbox.insert(tk.END, name)
            self.face_id_listbox.itemconfig(tk.END, fg=color)  # Apply color




    def load_model(self):
        """
        Robust model loading with multiple fallback strategies
        """
        try:
            # Potential model paths with your specific model names
            model_paths = [
                'face_yolov8n.pt',   # Prioritize face-specific model
                'face_yolov8s.pt',
                'yolov8n.pt',         # Fallback to general model
                os.path.join(os.path.dirname(__file__), 'face_yolov8n.pt'),
                os.path.join(os.path.dirname(__file__), 'face_yolov8s.pt'),
                os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
            ]

            # Try loading from existing paths
            for path in model_paths:
                print(f"Attempting to load model from: {path}")
                if os.path.exists(path):
                    try:
                        model = YOLO(path).to(self.device)
                        print(f"Model loaded successfully from {path}")
                        model.to(self.device)
                        return model
                    except Exception as load_error:
                        print(f"Error loading model from {path}: {load_error}")
                        continue

            # If no model is found
            error_message = (
                "Face Detection Model Not Found\n\n"
                "Available Models:\n"
                "- face_yolov8n.pt\n"
                "- face_yolov8s.pt\n"
                "- yolov8n.pt\n\n"
                "Ensure at least one model is present in the script directory."
            )
            
            messagebox.showerror("Model Error", error_message)
            return None

        except Exception as e:
            messagebox.showerror("Initialization Error", 
                f"Unexpected error during model initialization: {e}")
            return None
        

    def verify_model(model_path):
        """
        Additional model verification step
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
                
            # Test model with a sample image
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(test_image)
                
            print(f"Model {model_path} loaded successfully")
            return True
        except Exception as e:
            print(f"Model verification failed for {model_path}: {e}")
            return False



    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_source = file_path
            self.cap = cv2.VideoCapture(file_path)
        
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
        
            # Reset UI elements
            self.progress_slider.set(0)
            self.play_pause_button.configure(text="Play")
            self.is_playing = False

    def update_frame(self):
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.is_playing = False
                self.play_pause_button.configure(text="Play")
                return

            self.original_frame = frame.copy()
            self.frame, self.detected_faces = self.process_frame(frame)
            self.display_frame(self.frame)

            # تحديث السلايدر فقط لو المستخدم مش بيغيره يدويًا
            if not self.video_slider._command_active:
                progress_percentage = (self.current_frame / self.total_frames) * 100
                self.video_slider.set(progress_percentage)

            delay = max(1, int(1000 / self.fps))
            self.root.after(delay, self.update_frame)



    def process_frame(self, frame, draw_bbox=True):
        if frame is None:
            return frame, []

        # Track faces using the face tracker, now passing current_frame_index
        tracked_faces = self.face_tracker.track_faces(frame, self.current_frame_index)
        self.current_frame_index += 1  # Update the frame index after processing each frame

        # Update the UI list of detected faces
        self.update_face_list(tracked_faces)

        # Apply blurring only to selected faces and only if necessary
        for face in tracked_faces:
            if 'id' in face and face['id'] in self.blurred_faces:
                x1, y1, x2, y2 = face['bbox']
                face_region = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred_face

        # ✅ استدعاء draw_tracking_info مع تمرير draw_bbox للتحكم في إظهار الفريمات
        frame = self.face_tracker.draw_tracking_info(frame, tracked_faces, draw_bbox)

        return frame, tracked_faces





    def show_face_recurrence_info(self):
        if not hasattr(self, 'face_tracker'):
            messagebox.showwarning("No Data", "No faces have been tracked yet.")
            return

        # Get recurrence summary
        recurrence_summary = self.face_tracker.get_face_recurrence_summary()
        
        # Create a new window for recurrence information
        recurrence_window = tk.Toplevel(self.root)
        recurrence_window.title("Face Recurrence Detailed Information")
        recurrence_window.geometry("600x400")

        # Create a frame with scrollbar
        frame = tk.Frame(recurrence_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview for detailed information
        tree = ttk.Treeview(frame, 
            columns=('Total', 'First Time', 'Unique Locations'), 
            show='headings',
            yscrollcommand=scrollbar.set
        )

        # Define column headings
        tree.heading('Total', text='Face ID')
        tree.heading('First Time', text='Appearances')
        tree.heading('Unique Locations', text='Seen In Locations')


        # Configure column widths
        tree.column('Total', width=100, anchor='center')
        tree.column('First Time', width=150, anchor='center')
        tree.column('Unique Locations', width=150, anchor='center')

        # Populate the treeview
        for face_id, info in recurrence_summary.items():
            tree.insert('', 'end', values=(
                info['name'],
                info['total_appearances'],
                info['unique_locations']
            ))


        # Connect scrollbar
        scrollbar.config(command=tree.yview)

        # Pack the treeview
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Optional: Add a details button
        details_button = tk.Button(recurrence_window, text="Show Details", command=lambda: show_face_details(tree))
        details_button.pack(pady=10)

        # Optional: Add a details button
        def show_face_details(tree):
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("No Selection", "Please select a face")
                return

            face_id = int(tree.item(selected_item)['values'][0].split()[-1])
            details = recurrence_summary.get(face_id, {})

            # Create a detailed popup
            details_window = tk.Toplevel(recurrence_window)
            details_window.title(f"Face {face_id} Details")
            details_window.geometry("400x300")

            # Create text widget to display details
            details_text = tk.Text(details_window, wrap=tk.WORD)
            details_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            # Insert detailed information
            details_text.insert(tk.END, f"Face ID: {face_id}\n")
            details_text.insert(tk.END, f"Total Appearances: {details['total_appearances']}\n")
            details_text.insert(tk.END, f"First Time Seen: {'Yes' if details['is_first_time'] else 'No'}\n")
            details_text.insert(tk.END, f"Unique Locations: {details['unique_locations']}\n")
            details_text.insert(tk.END, f"First Seen Bbox: {details['first_seen_bbox']}\n")
            details_text.config(state=tk.DISABLED)




    def find_best_match(self, previous_bbox, current_faces, iou_threshold=0.2):
        """
        Find the best matching face for a previously blurred face
        """
        px, py, pw, ph = previous_bbox
        previous_center = (px + pw/2, py + ph/2)

        best_match = None
        best_score = 0

        for face in current_faces:
            cx, cy, cw, ch = face['bbox']
            current_center = (cx + cw/2, cy + ch/2)

            # Calculate Intersection over Union (IoU)
            intersection_x = max(0, min(px+pw, cx+cw) - max(px, cx))
            intersection_y = max(0, min(py+ph, cy+ch) - max(py, cy))
            intersection_area = intersection_x * intersection_y

            union_area = pw*ph + cw*ch - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0

            # Calculate center distance
            center_distance = ((previous_center[0] - current_center[0])**2 + 
                            (previous_center[1] - current_center[1])**2)**0.5
            
            # Normalize distance (smaller is better)
            normalized_distance = 1 / (1 + center_distance)

            # Combine IoU and distance for matching
            score = iou * normalized_distance

            if score > best_score and iou > iou_threshold:
                best_match = face
                best_score = score

        return best_match
        


    def find_matching_face(self, previous_bbox, current_faces, iou_threshold=0.3):
        """
        Find the best matching face for a previously blurred face
        """
        px, py, pw, ph = previous_bbox
        previous_center = (px + pw/2, py + ph/2)

        best_match = None
        best_iou = 0

        for face in current_faces:
            cx, cy, cw, ch = face['bbox']
            current_center = (cx + cw/2, cy + ch/2)

            # Calculate Intersection over Union (IoU)
            intersection_x = max(0, min(px+pw, cx+cw) - max(px, cx))
            intersection_y = max(0, min(py+ph, cy+ch) - max(py, cy))
            intersection_area = intersection_x * intersection_y

            union_area = pw*ph + cw*ch - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0

            # Calculate center distance
            center_distance = ((previous_center[0] - current_center[0])**2 + 
                            (previous_center[1] - current_center[1])**2)**0.5

            # Combine IoU and distance for matching
            if iou > iou_threshold:
                if best_match is None or iou > best_iou:
                    best_match = face
                    best_iou = iou

        return best_match

    


    def is_same_face(self, current_face, blur_info):
        # Implement a method to determine if the current face is the same as a previously blurred face
        # This could be based on position, size, or other characteristics
        current_x, current_y, current_w, current_h = current_face['bbox']
        blur_x, blur_y, blur_w, blur_h = blur_info['bbox']

        # Check if the faces overlap significantly
        overlap_x = max(0, min(current_x + current_w, blur_x + blur_w) - max(current_x, blur_x))
        overlap_y = max(0, min(current_y + current_h, blur_y + blur_h) - max(current_y, blur_y))
    
        # Calculate overlap area
        overlap_area = overlap_x * overlap_y
        current_area = current_w * current_h

        # If overlap is more than 50% of the current face, consider it the same face
        return overlap_area > 0.5 * current_area


    def play_video(self):
        if not self.cap or not self.is_playing:
            return

        # Read the next frame
        ret, frame = self.cap.read()

        if not ret:
            # Reset video to the beginning if it reaches the end
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.is_playing = False
            self.play_pause_button.configure(text="Play")
            return

        # Update current frame position and slider
        self.current_frame += 1
        progress_percentage = (self.current_frame / self.total_frames) * 100
        self.progress_slider.set(progress_percentage)

        # Save original frame for unblur functionality
        self.original_frame = frame.copy()

        # Process and display the frame
        self.frame, self.detected_faces = self.process_frame(frame)
        self.display_frame(self.frame)

        # Schedule next frame with a delay based on the video's native frame rate
        delay = max(1, int(1000 / self.fps))
        self.root.after(delay, self.play_video)



    def toggle_play_pause(self):
        if not self.cap:
            return

        if self.is_playing:
            # Pause the video
            self.is_playing = False
            self.play_pause_button.configure(text="Play")
        else:
            # Play the video
            self.is_playing = True
            self.play_pause_button.configure(text="Pause")
            self.play_video()






    def seek_video(self, value):
        if not self.cap or not self.total_frames:
            return

        target_frame = int((value / 100) * self.total_frames)

        # استخدم set() مع تحديث الفريم يدويًا لتحسين الدقة
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)  # ضبط الموضع للفريم الصحيح
        self.current_frame = target_frame

        ret, frame = self.cap.read()
        if ret:
            self.original_frame = frame.copy()
            self.frame, self.detected_faces = self.process_frame(frame)
            self.display_frame(self.frame)

        # تأكد إن السلايدر ما يتحركش لو المستخدم هو اللي غيره
        self.video_slider.set((self.current_frame / self.total_frames) * 100)




    def display_frame(self, frame):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
        self.canvas.image = image_tk





    def blur_face(self):
        selected_face_indices = self.face_id_listbox.curselection()
        if selected_face_indices:
            selected_face = self.detected_faces[selected_face_indices[0]]
            x, y, w, h = selected_face['bbox']
        
            # Store the blurred face information
            self.blurred_faces[selected_face['id']] = {
                'bbox': [x, y, w, h]
            }
        
            # Reprocess the frame to apply all blurs
            self.frame, _ = self.process_frame(self.original_frame)
            self.display_frame(self.frame)

            

    def unblur_face(self):
        selected_face_indices = self.face_id_listbox.curselection()
        if selected_face_indices:
            selected_face = self.detected_faces[selected_face_indices[0]]
        
            # Remove the face from blurred faces
            if selected_face['id'] in self.blurred_faces:
                del self.blurred_faces[selected_face['id']]
        
            # Reprocess the frame to apply remaining blurs
            self.frame, _ = self.process_frame(self.original_frame)
            self.display_frame(self.frame)





# Main execution
if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceBlurApp(root)
    root.mainloop()