'''
This is the script for extracting pose and embedding pose information from a video (single or multiperson)
using MediaPipe and YOLO

'''

#import all packages
import os
import cv2
import mediapipe as mp
import pandas as pd
import torch

import subprocess
import sys

# Helper function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Try to import YOLOv5; install if missing
try:
    from yolov5 import YOLOv5
except ImportError:
    install("yolov5")
    from yolov5 import YOLOv5

# Download yolov5s weights if not already present
def download_yolov5_weights():
    weights_path = "yolov5s.pt"
    if not os.path.exists(weights_path):
        print("Downloading yolov5s.pt weights...")
        import requests
        url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
        response = requests.get(url)
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print("Downloaded yolov5s.pt")

download_yolov5_weights()


# Core pose processor class
class PoseProcessor:
    def __init__(self, output_csv_folder, output_video_folder=None, status_callback=None):
        self.output_csv_folder = output_csv_folder
        self.output_video_folder = output_video_folder
        self.status_callback = status_callback  
        self.enable_multi_person_pose = False  # Default to single person mode
        
        
        # Initialize Mediapipe Pose and YOLOv5
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.drawing_utils = mp.solutions.drawing_utils
        self.yolo = YOLOv5("yolov5s.pt")  # Load YOLOv5 once

    def set_multi_person_mode(self, enabled: bool):
        """Enable or disable multi-person pose mode."""
        self.enable_multi_person_pose = enabled

    def extract_pose_features(self, video_path):
        """Extract pose features from video, saving one CSV per person."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        keypoints_by_person = {} # Dictionary to store keypoints per person

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
              
            if self.status_callback:
                self.status_callback(f"📸 Extracting pose from: {os.path.basename(video_path)}")

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.enable_multi_person_pose:
              # Detect multiple people using YOLO
                results = self.yolo.predict(image_rgb, size=640)
                boxes = results.xyxy[0]
                person_boxes = [b[:4].int().tolist() for b in boxes if int(b[5]) == 0]
                
                # Process each detected person
                for person_id, (x1, y1, x2, y2) in enumerate(person_boxes):
                    cropped = image_rgb[y1:y2, x1:x2]
                    result = self.pose.process(cropped)

                    if result.pose_landmarks:
                        row = [frame_idx, person_id]
                        for lmk in result.pose_landmarks.landmark:
                            row.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
                        if person_id not in keypoints_by_person:
                            keypoints_by_person[person_id] = []
                        keypoints_by_person[person_id].append(row)

            else:
                # Single-person mode
                result = self.pose.process(image_rgb)
                if result.pose_landmarks:
                    row = [frame_idx, 0]
                    for lmk in result.pose_landmarks.landmark:
                        row.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
                    keypoints_by_person[0] = keypoints_by_person.get(0, []) + [row]

            frame_idx += 1

        cap.release()
        
        
        # Define DataFrame column names
        columns = ['frame', 'person_id']
        names = [
            'Nose', 'Left_eye_inner', 'Left_eye', 'Left_eye_outer', 'Right_eye_inner', 
            'Right_eye', 'Right_eye_outer', 'Left_ear', 'Right_ear', 'Mouth_left', 
            'Mouth_right', 'Left_shoulder', 'Right_shoulder', 'Left_elbow', 'Right_elbow',
            'Left_wrist', 'Right_wrist', 'Left_pinky', 'Right_pinky', 'Left_index', 
            'Right_index', 'Left_thumb', 'Right_thumb', 'Left_hip', 'Right_hip',
            'Left_knee', 'Right_knee', 'Left_ankle', 'Right_ankle', 'Left_heel',
            'Right_heel', 'Left_foot_index', 'Right_foot_index'
        ]
        for n in names:
            columns.extend([f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_confidence"])

        suffix = "_multi" if self.enable_multi_person_pose else ""
        base_filename = os.path.splitext(os.path.basename(video_path))[0] + suffix
        
        
        # Save separate CSV for each person
        for person_id, keypoints in keypoints_by_person.items():
            df = pd.DataFrame(keypoints, columns=columns)
            filename = f"{base_filename}_ID_{int(person_id)}.csv"
            csv_path = os.path.join(self.output_csv_folder, filename)
            df.to_csv(csv_path, index=False)

        return

    def embed_pose_video(self, video_path):
        """Overlay pose landmarks on the video and save output."""
        if not self.output_video_folder:
            return None

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        suffix = "_multi" if self.enable_multi_person_pose else ""
        filename = os.path.splitext(os.path.basename(video_path))[0] + f"{suffix}_pose.mp4"
        out_path = os.path.join(self.output_video_folder, filename)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.enable_multi_person_pose:
                # Use YOLO to detect person boxes
                results = self.yolo.predict(image_rgb, size=640)
                boxes = results.xyxy[0]
                person_boxes = [b[:4].int().tolist() for b in boxes if int(b[5]) == 0]
                
                # Draw pose for each person (approximate)
                for (x1, y1, x2, y2) in person_boxes:
                    cropped = image_rgb[y1:y2, x1:x2]
                    result = self.pose.process(cropped)

                    if result.pose_landmarks:
                        try:
                            # Draw landmarks within the cropped region (can be improved with coordinate shifting)
                            self.drawing_utils.draw_landmarks(frame[y1:y2, x1:x2], result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                        except Exception:
                            pass

            else:
                # Single-person mode
                result = self.pose.process(image_rgb)
                if result.pose_landmarks:
                    self.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            out.write(frame)

        cap.release()
        out.release()
        return out_path
