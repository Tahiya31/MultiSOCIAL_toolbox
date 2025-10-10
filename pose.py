'''
This is the script for extracting pose and embedding pose information from a video (single or multiperson)
using MediaPipe and YOLO

'''

import os
import cv2
import mediapipe as mp
import pandas as pd
from yolov5 import YOLOv5

def ensure_yolov5_weights():
    """Ensure yolov5s weights exist without triggering network calls at import in other modules."""
    weights_path = "yolov5s.pt"
    if not os.path.exists(weights_path):
        try:
            import requests
            url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            with open(weights_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded YOLOv5 weights to {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download YOLOv5 weights: {e}")


# Core pose processor class
class PoseProcessor:
    def __init__(self, output_csv_folder, output_video_folder=None, status_callback=None):
        self.output_csv_folder = output_csv_folder
        self.output_video_folder = output_video_folder
        self.status_callback = status_callback  
        self.enable_multi_person_pose = False  # Default to single person mode
        
        # Initialize Mediapipe Pose
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)
        self.drawing_utils = mp.solutions.drawing_utils
        
        # Initialize YOLO lazily (only when needed for multi-person mode)
        self.yolo = None

    def _expand_and_clip_bbox(self, x1, y1, x2, y2, image_width, image_height, margin_ratio=0.12):
        """Expand bbox by margin_ratio on all sides and clip to image bounds.

        Returns clipped integer coordinates (nx1, ny1, nx2, ny2) with nx1 < nx2 and ny1 < ny2.
        """
        # Convert to float for precise expansion
        bx1, by1, bx2, by2 = float(x1), float(y1), float(x2), float(y2)
        bw = max(0.0, bx2 - bx1)
        bh = max(0.0, by2 - by1)
        if bw <= 0 or bh <= 0:
            return int(max(0, min(image_width - 1, x1))), int(max(0, min(image_height - 1, y1))), int(max(0, min(image_width, x2))), int(max(0, min(image_height, y2)))

        margin_w = bw * margin_ratio
        margin_h = bh * margin_ratio

        ex1 = bx1 - margin_w
        ey1 = by1 - margin_h
        ex2 = bx2 + margin_w
        ey2 = by2 + margin_h

        # Clip to image bounds
        ex1 = max(0.0, min(ex1, image_width - 1.0))
        ey1 = max(0.0, min(ey1, image_height - 1.0))
        ex2 = max(0.0, min(ex2, image_width - 0.0))
        ey2 = max(0.0, min(ey2, image_height - 0.0))

        # Ensure correct ordering after clipping
        if ex2 <= ex1:
            ex2 = min(image_width * 1.0, ex1 + 1.0)
        if ey2 <= ey1:
            ey2 = min(image_height * 1.0, ey1 + 1.0)

        return int(ex1), int(ey1), int(ex2), int(ey2)

    def set_multi_person_mode(self, enabled: bool):
        """Enable or disable multi-person pose mode."""
        self.enable_multi_person_pose = enabled

    def _ensure_yolo(self):
        """Lazily initialize YOLO only when needed for multi-person detection."""
        if self.yolo is None:
            try:
                ensure_yolov5_weights()
                self.yolo = YOLOv5("yolov5s.pt")
                if self.status_callback:
                    self.status_callback("🤖 YOLOv5 model loaded for multi-person detection")
            except Exception as e:
                if self.status_callback:
                    self.status_callback(f"❌ Failed to load YOLOv5: {e}")
                raise RuntimeError(f"Failed to initialize YOLOv5: {e}")

    def extract_pose_features(self, video_path, progress_callback=None):
        """Extract pose features from video, saving one CSV per person."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        keypoints_by_person = {} # Dictionary to store keypoints per person
        
        # Get video dimensions for coordinate normalization
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback: estimate frames from FPS and duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                total_frames = int(fps * duration)

        # Maintain locked ROIs across frames for multi-person mode
        # Each ROI holds its own MediaPipe Pose instance; index in list is stable person_id
        locked_rois = []  # Each item: {"x1":int,"y1":int,"x2":int,"y2":int,"lost":int, "pose": Pose}

        def _roi_center(x1, y1, x2, y2):
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            return cx, cy

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
              
            if self.status_callback:
                self.status_callback(f"📸 Extracting pose from: {os.path.basename(video_path)} (Frame {frame_idx + 1}/{total_frames})")

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.enable_multi_person_pose:
                # Ensure YOLO is loaded
                self._ensure_yolo()

                # Seed locked ROIs once (or when none exist)
                if not locked_rois:
                    results = self.yolo.predict(image_rgb, size=640)
                    boxes = results.xyxy[0]
                    person_boxes = [b[:4].int().tolist() for b in boxes if int(b[5]) == 0]
                    if not person_boxes:
                        if self.status_callback:
                            self.status_callback(f"⚠️ No people detected in frame {frame_idx}")
                        frame_idx += 1
                        if progress_callback and total_frames > 0:
                            progress_percent = int((frame_idx / total_frames) * 100)
                            progress_callback(progress_percent)
                        continue
                    for (x1, y1, x2, y2) in person_boxes:
                        x1e, y1e, x2e, y2e = self._expand_and_clip_bbox(x1, y1, x2, y2, w, h, margin_ratio=0.25)
                        roi_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2)
                        locked_rois.append({"x1": x1e, "y1": y1e, "x2": x2e, "y2": y2e, "lost": 0, "pose": roi_pose})

                # If still none, skip frame
                if not locked_rois:
                    frame_idx += 1
                    if progress_callback and total_frames > 0:
                        progress_percent = int((frame_idx / total_frames) * 100)
                        progress_callback(progress_percent)
                    continue

                # Process each locked ROI
                for person_id, roi in enumerate(locked_rois):
                    x1e, y1e, x2e, y2e = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                    cropped = image_rgb[y1e:y2e, x1e:x2e]
                    result = roi["pose"].process(cropped)

                    if result.pose_landmarks:
                        roi["lost"] = 0
                        row = [frame_idx, person_id]
                        for lmk in result.pose_landmarks.landmark:
                            # Transform coordinates back to original frame and normalize to [0,1]
                            orig_x = ((lmk.x * (x2e - x1e)) + x1e) / w
                            orig_y = ((lmk.y * (y2e - y1e)) + y1e) / h
                            row.extend([orig_x, orig_y, lmk.z, lmk.visibility])
                        if person_id not in keypoints_by_person:
                            keypoints_by_person[person_id] = []
                        keypoints_by_person[person_id].append(row)
                    else:
                        roi["lost"] += 1
                        if roi["lost"] >= 10:
                            # Re-seed this ROI using nearest current YOLO detection
                            results = self.yolo.predict(image_rgb, size=640)
                            boxes = results.xyxy[0]
                            person_boxes = [b[:4].int().tolist() for b in boxes if int(b[5]) == 0]
                            if person_boxes:
                                cx, cy = _roi_center(x1e, y1e, x2e, y2e)
                                best = None
                                best_d = None
                                for (rx1, ry1, rx2, ry2) in person_boxes:
                                    rcx, rcy = _roi_center(rx1, ry1, rx2, ry2)
                                    d = (rcx - cx) * (rcx - cx) + (rcy - cy) * (rcy - cy)
                                    if best is None or d < best_d:
                                        best = (rx1, ry1, rx2, ry2)
                                        best_d = d
                                if best is not None:
                                    nx1, ny1, nx2, ny2 = self._expand_and_clip_bbox(best[0], best[1], best[2], best[3], w, h, margin_ratio=0.25)
                                    # Close old Pose and create a new one tied to the new ROI
                                    try:
                                        roi_pose = roi.get("pose")
                                        if roi_pose is not None:
                                            roi_pose.close()
                                    except Exception:
                                        pass
                                    roi["x1"], roi["y1"], roi["x2"], roi["y2"] = nx1, ny1, nx2, ny2
                                    roi["pose"] = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2)
                                    roi["lost"] = 0

            else:
                # Single-person mode
                result = self.pose.process(image_rgb)
                if result.pose_landmarks:
                    row = [frame_idx, 0]
                    for lmk in result.pose_landmarks.landmark:
                        row.extend([lmk.x, lmk.y, lmk.z, lmk.visibility])
                    keypoints_by_person[0] = keypoints_by_person.get(0, []) + [row]

            frame_idx += 1
            
            # Update progress if callback provided
            if progress_callback and total_frames > 0:
                progress_percent = int((frame_idx / total_frames) * 100)
                progress_callback(progress_percent)

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

    def embed_pose_video(self, video_path, progress_callback=None):
        """Overlay pose landmarks on the video and save output."""
        if not self.output_video_folder:
            return None

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback: estimate frames from FPS and duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                total_frames = int(fps * duration)

        suffix = "_multi" if self.enable_multi_person_pose else ""
        filename = os.path.splitext(os.path.basename(video_path))[0] + f"{suffix}_pose.mp4"
        out_path = os.path.join(self.output_video_folder, filename)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_idx = 0
        # Maintain locked ROIs across frames for multi-person mode
        # Each ROI holds its own MediaPipe Pose instance
        locked_rois = []  # Each item: {"x1":int,"y1":int,"x2":int,"y2":int,"lost":int, "pose": Pose}

        def _roi_center(x1, y1, x2, y2):
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            return cx, cy
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.enable_multi_person_pose:
                # Ensure YOLO is loaded
                self._ensure_yolo()

                # Seed locked ROIs once, with larger margin for stability
                if not locked_rois:
                    results = self.yolo.predict(image_rgb, size=640)
                    boxes = results.xyxy[0]
                    person_boxes = [b[:4].int().tolist() for b in boxes if int(b[5]) == 0]
                    for (x1, y1, x2, y2) in person_boxes:
                        x1e, y1e, x2e, y2e = self._expand_and_clip_bbox(x1, y1, x2, y2, w, h, margin_ratio=0.25)
                        roi_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2)
                        locked_rois.append({"x1": x1e, "y1": y1e, "x2": x2e, "y2": y2e, "lost": 0, "pose": roi_pose})

                # If no ROIs found, write frame and continue
                if not locked_rois:
                    out.write(frame)
                    frame_idx += 1
                    if progress_callback and total_frames > 0:
                        progress_percent = int((frame_idx / total_frames) * 100)
                        progress_callback(progress_percent)
                    continue

                # Process each locked ROI without re-running YOLO every frame
                for roi in locked_rois:
                    x1e, y1e, x2e, y2e = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                    # Draw ROI rectangle for inspection
                    try:
                        cv2.rectangle(frame, (x1e, y1e), (x2e, y2e), (255, 255, 0), 2)
                    except Exception:
                        pass

                    cropped = image_rgb[y1e:y2e, x1e:x2e]
                    result = roi["pose"].process(cropped)

                    if result.pose_landmarks:
                        roi["lost"] = 0
                        try:
                            original_landmarks = result.pose_landmarks
                            # Map back to full-frame
                            for lmk in original_landmarks.landmark:
                                lmk.x = (lmk.x * (x2e - x1e) + x1e) / w
                                lmk.y = (lmk.y * (y2e - y1e) + y1e) / h
                            self.drawing_utils.draw_landmarks(
                                frame, 
                                original_landmarks, 
                                mp.solutions.pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )
                        except Exception as e:
                            if self.status_callback:
                                self.status_callback(f"❌ Error drawing landmarks: {e}")
                            print(f"Error drawing landmarks: {e}")
                    else:
                        roi["lost"] += 1
                        if roi["lost"] >= 10:
                            # Re-seed this ROI using nearest current YOLO detection
                            results = self.yolo.predict(image_rgb, size=640)
                            boxes = results.xyxy[0]
                            person_boxes = [b[:4].int().tolist() for b in boxes if int(b[5]) == 0]
                            if person_boxes:
                                cx, cy = _roi_center(x1e, y1e, x2e, y2e)
                                best = None
                                best_d = None
                                for (rx1, ry1, rx2, ry2) in person_boxes:
                                    rcx, rcy = _roi_center(rx1, ry1, rx2, ry2)
                                    d = (rcx - cx) * (rcx - cx) + (rcy - cy) * (rcy - cy)
                                    if best is None or d < best_d:
                                        best = (rx1, ry1, rx2, ry2)
                                        best_d = d
                                if best is not None:
                                    nx1, ny1, nx2, ny2 = self._expand_and_clip_bbox(best[0], best[1], best[2], best[3], w, h, margin_ratio=0.25)
                                    # Close old Pose and create a new one tied to the new ROI
                                    try:
                                        roi_pose = roi.get("pose")
                                        if roi_pose is not None:
                                            roi_pose.close()
                                    except Exception:
                                        pass
                                    roi["x1"], roi["y1"], roi["x2"], roi["y2"] = nx1, ny1, nx2, ny2
                                    roi["pose"] = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2)
                                    roi["lost"] = 0

            else:
                # Single-person mode
                result = self.pose.process(image_rgb)
                if result.pose_landmarks:
                    # Debug output to verify landmarks are being detected
                    if self.status_callback:
                        self.status_callback(f"🎯 Drawing {len(result.pose_landmarks.landmark)} landmarks (single-person mode)")
                    
                    # Draw landmarks with visible style
                    self.drawing_utils.draw_landmarks(
                        frame, 
                        result.pose_landmarks, 
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                else:
                    # Debug output when no landmarks detected in single-person mode
                    if self.status_callback:
                        self.status_callback("⚠️ No landmarks detected (single-person mode)")

            out.write(frame)
            
            frame_idx += 1
            
            # Update progress if callback provided
            if progress_callback and total_frames > 0:
                progress_percent = int((frame_idx / total_frames) * 100)
                progress_callback(progress_percent)

        cap.release()
        out.release()
        # Cleanup ROI Pose instances
        try:
            for roi in locked_rois:
                roi_pose = roi.get("pose")
                if roi_pose is not None:
                    roi_pose.close()
        except Exception:
            pass
        return out_path
