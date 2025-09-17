"""
Extract pose keypoints using YOLO + MediaPipe.
- Supports single and multi-person.
- Filters low-confidence detections.
- Tracks people across frames using SORT (optional).
- Optimized for CPU or GPU.
- Skips frames for faster processing.
- Resizes frames smaller for speed.

Folder structure:
- Place videos in ./input/
- CSV outputs in ./output/
"""

import os
import sys
import subprocess
import glob
import numpy as np
import torch

# Install required packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try imports
for pkg in ["opencv-python", "pandas", "mediapipe", "ultralytics"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

import cv2
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO

# Optional: Install SORT for tracking
try:
    from sort import Sort
    sort_available = True
except ImportError:
    sort_available = False
    print("🔔 SORT tracking not installed. Run `pip install sort` to enable tracking.")

# Load YOLO model
def load_yolo_model(device):
    model = YOLO("yolov8n.pt").to(device)
    return model

class PoseExtractor:
    def __init__(self, input_folder="input", output_folder="output",
                 confidence_threshold=0.5, use_sort_tracker=False,
                 frame_skip=2, resized_width=640, resized_height=360):
        
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.confidence_threshold = confidence_threshold
        self.use_sort_tracker = use_sort_tracker and sort_available
        self.frame_skip = frame_skip
        self.resized_width = resized_width
        self.resized_height = resized_height

        # Optimize Torch
        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Using device: {self.device}")

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # use lighter pose model
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.model = load_yolo_model(self.device)

        if self.use_sort_tracker:
            self.tracker = Sort()

        os.makedirs(self.output_folder, exist_ok=True)

    def extract(self):
        video_files = glob.glob(os.path.join(self.input_folder, "*.mp4"))
        print(f"Found {len(video_files)} video(s) in '{self.input_folder}'")

        for video_path in video_files:
            print(f"Processing {video_path}")
            self.process_video(video_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        keypoints_by_person = dict()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            frame = cv2.resize(frame, (self.resized_width, self.resized_height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model.predict([rgb], classes=[0], verbose=False, imgsz=320)
            boxes_raw = results[0].boxes
            scores = boxes_raw.conf.cpu().numpy()
            boxes = boxes_raw.xyxy.cpu().numpy().astype(int)

            # Filter low-confidence detections
            good_boxes = []
            for box, score in zip(boxes, scores):
                if score > self.confidence_threshold:
                    good_boxes.append(box)
            good_boxes = np.array(good_boxes)

            if good_boxes.shape[0] == 0:
                frame_idx += 1
                continue

            # Apply SORT tracking if enabled
            if self.use_sort_tracker:
                if good_boxes.shape[1] == 4:
                    good_boxes = np.hstack((good_boxes, np.ones((good_boxes.shape[0], 1))))
                tracked_objects = self.tracker.update(good_boxes)
                detections = [(int(track[-1]), *map(int, track[:4])) for track in tracked_objects]
            else:
                detections = [(person_id, *box) for person_id, box in enumerate(good_boxes)]

            # Pose Extraction
            for person_id, x1, y1, x2, y2 in detections:
                crop = rgb[y1:y2, x1:x2]
                result = self.pose.process(crop)

                if result.pose_landmarks:
                    row = [frame_idx, person_id]
                    for lmk in result.pose_landmarks.landmark:
                        row.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])

                    if person_id not in keypoints_by_person:
                        keypoints_by_person[person_id] = []
                    keypoints_by_person[person_id].append(row)

            frame_idx += 1

        cap.release()
        self.save_csvs(video_path, keypoints_by_person)

    def save_csvs(self, video_path, keypoints_by_person):
        columns = ['frame', 'person_id']
        landmark_names = [
            'Nose', 'Left_eye_inner', 'Left_eye', 'Left_eye_outer', 'Right_eye_inner', 'Right_eye', 'Right_eye_outer',
            'Left_ear', 'Right_ear', 'Mouth_left', 'Mouth_right', 'Left_shoulder', 'Right_shoulder', 'Left_elbow',
            'Right_elbow', 'Left_wrist', 'Right_wrist', 'Left_pinky', 'Right_pinky', 'Left_index', 'Right_index',
            'Left_thumb', 'Right_thumb', 'Left_hip', 'Right_hip', 'Left_knee', 'Right_knee', 'Left_ankle', 'Right_ankle',
            'Left_heel', 'Right_heel', 'Left_foot_index', 'Right_foot_index'
        ]
        for name in landmark_names:
            columns += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        for person_id, keypoints in keypoints_by_person.items():
            df = pd.DataFrame(keypoints, columns=columns)
            output_file = os.path.join(self.output_folder, f"{base_name}_person_{person_id}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {output_file}")

if __name__ == "__main__":
    # ===== USER SETTINGS =====
    CONFIDENCE_THRESHOLD = 0.5    # Filter weak detections
    USE_SORT_TRACKER = True       # Track people across frames
    FRAME_SKIP = 2                # Skip every N-th frame
    RESIZED_WIDTH = 640           # Resize width
    RESIZED_HEIGHT = 360          # Resize height
    # =========================

    extractor = PoseExtractor(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_sort_tracker=USE_SORT_TRACKER,
        frame_skip=FRAME_SKIP,
        resized_width=RESIZED_WIDTH,
        resized_height=RESIZED_HEIGHT
    )
    extractor.extract()
