import os
import json
import cv2
import numpy as np
import pandas as pd
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


def _landmark_columns():
    names = [
        'Nose', 'Left_eye_inner', 'Left_eye', 'Left_eye_outer', 'Right_eye_inner',
        'Right_eye', 'Right_eye_outer', 'Left_ear', 'Right_ear', 'Mouth_left',
        'Mouth_right', 'Left_shoulder', 'Right_shoulder', 'Left_elbow', 'Right_elbow',
        'Left_wrist', 'Right_wrist', 'Left_pinky', 'Right_pinky', 'Left_index',
        'Right_index', 'Left_thumb', 'Right_thumb', 'Left_hip', 'Right_hip',
        'Left_knee', 'Right_knee', 'Left_ankle', 'Right_ankle', 'Left_heel',
        'Right_heel', 'Left_foot_index', 'Right_foot_index'
    ]
    cols = []
    for n in names:
        cols.extend([f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_confidence"])
    return cols


def sanity_check_csv(csv_path):
    problems = []
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return [f"Failed to read CSV: {e}"]

    expected_cols = 2 + 33 * 4
    if df.shape[1] != expected_cols:
        problems.append(f"Unexpected column count: {df.shape[1]} != {expected_cols}")

    for col in df.columns:
        if col.endswith('_x') or col.endswith('_y'):
            s = df[col]
            if s.isna().any():
                problems.append(f"{col} has NaNs")
            if (s < 0).any() or (s > 1).any():
                problems.append(f"{col} out of [0,1] range")

    return problems


def _load_csv_group(csv_path):
    df = pd.read_csv(csv_path)
    lmk_cols = _landmark_columns()
    subset_cols = ['frame', 'person_id'] + lmk_cols
    df = df[subset_cols]
    return df.groupby('frame'), lmk_cols


def _draw_from_row(img, row, lmk_cols, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for i in range(0, len(lmk_cols), 4):
        x = row[lmk_cols[i]]
        y = row[lmk_cols[i + 1]]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        px = int(np.clip(x, 0.0, 1.0) * w)
        py = int(np.clip(y, 0.0, 1.0) * h)
        cv2.circle(img, (px, py), 3, color, -1)
    return img


def _is_green_pixel(bgr):
    # Heuristic: green landmarks drawn as (0,255,0) with antialias; allow tolerance
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return (g >= 150) and (r <= 80) and (b <= 80) and (g >= r + 40) and (g >= b + 40)


def _landmark_hit(frame_bgr, px, py, window=5):
    h, w = frame_bgr.shape[:2]
    x1 = max(0, px - window)
    y1 = max(0, py - window)
    x2 = min(w - 1, px + window)
    y2 = min(h - 1, py + window)
    roi = frame_bgr[y1:y2 + 1, x1:x2 + 1]
    # any pixel sufficiently green?
    for yy in range(roi.shape[0]):
        for xx in range(roi.shape[1]):
            if _is_green_pixel(roi[yy, xx]):
                return True
    return False


def _green_overlay_mask(bgr_img):
    """Extract a tight mask of green overlay pixels from a BGR image.
    - HSV gate for vivid green (avoid whites/greys)
    - AND with channel-dominance (G much higher than R/B and absolute G high)
    - Morphology + area filtering to keep dot-like blobs only
    Returns uint8 mask in {0,255}.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # Tighter green; tune if needed for your theme
    lower = np.array([50, 120, 90], dtype=np.uint8)   # H,S,V
    upper = np.array([85, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower, upper)

    b, g, r = cv2.split(bgr_img)
    g_i = g.astype(np.int32)
    r_i = r.astype(np.int32)
    b_i = b.astype(np.int32)
    dom = (g_i - np.maximum(r_i, b_i)) >= 50
    high_g = g_i >= 170
    mask_dom = (dom & high_g).astype(np.uint8) * 255

    # Conservative intersection to avoid background
    mask = cv2.bitwise_and(mask_hsv, mask_dom)

    # Morphology to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Area filter: keep small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    min_area = 2
    max_area = 250
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == lbl] = 255

    return filtered


def verify(video_path, csv_paths, output_size=None, stride=1, hit_threshold=0.8, max_worst=10, worst_dir=None, processed_only=True, window=5, conf_threshold=0.0, metric='hit_rate', ssim_threshold=0.98):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_size is None:
        output_size = (fw, fh)

    per_person = []
    csv_problems = {}
    processed_indices = set()
    for csv in csv_paths:
        probs = sanity_check_csv(csv)
        if probs:
            csv_problems[os.path.basename(csv)] = probs
        grp, lmk_cols = _load_csv_group(csv)
        per_person.append((grp, lmk_cols))
        try:
            processed_indices.update(grp.groups.keys())
        except Exception:
            pass

    frame_idx = 0
    hit_rates = []  # per-frame hit rate
    ssim_scores = []  # per-frame ssim when computed
    worst_hit = []  # list of (score, frame_idx, marks_bgr)
    worst_ssim = []  # list of (score, frame_idx, masks_img)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != output_size:
            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)

        gate = True
        if processed_only:
            gate = frame_idx in processed_indices
        else:
            gate = (stride == 1) or (frame_idx % stride == 0)

        if gate:
            if metric in ('hit_rate', 'both'):
                total = 0
                hits = 0
                marks = frame.copy()
                for grp, lmk_cols in per_person:
                    if frame_idx in grp.groups:
                        rows = grp.get_group(frame_idx)
                        for _, row in rows.iterrows():
                            for i in range(0, len(lmk_cols), 4):
                                x = row[lmk_cols[i]]
                                y = row[lmk_cols[i + 1]]
                                conf = row[lmk_cols[i + 3]]
                                if not np.isfinite(x) or not np.isfinite(y):
                                    continue
                                if conf_threshold > 0.0 and (not np.isfinite(conf) or conf < conf_threshold):
                                    continue
                                px = int(np.clip(x, 0.0, 1.0) * output_size[0])
                                py = int(np.clip(y, 0.0, 1.0) * output_size[1])
                                total += 1
                                if _landmark_hit(frame, px, py, window=window):
                                    hits += 1
                                    cv2.circle(marks, (px, py), 3, (0, 255, 255), -1)
                                else:
                                    cv2.circle(marks, (px, py), 3, (255, 0, 255), -1)

                rate = float(hits) / float(max(1, total))
                hit_rates.append(rate)
                score = rate
                if len(worst_hit) < max_worst:
                    worst_hit.append((score, frame_idx, marks))
                    worst_hit.sort(key=lambda x: x[0])
                else:
                    if score < worst_hit[-1][0]:
                        worst_hit[-1] = (score, frame_idx, marks)
                        worst_hit.sort(key=lambda x: x[0])
            if metric in ('ssim_green', 'both'):
                if ssim is None:
                    raise RuntimeError("scikit-image is required for SSIM (pip install scikit-image)")
                # Build synthetic overlay canvas from CSV landmarks
                canvas = np.zeros_like(frame)
                for grp, lmk_cols in per_person:
                    if frame_idx in grp.groups:
                        rows = grp.get_group(frame_idx)
                        for _, row in rows.iterrows():
                            canvas = _draw_from_row(canvas, row, lmk_cols, color=(0, 255, 0))
                actual_mask = _green_overlay_mask(frame)
                synth_mask = _green_overlay_mask(canvas)
                # Harmonize stroke thickness/antialias
                k = np.ones((3, 3), np.uint8)
                actual_mask = cv2.dilate(actual_mask, k, iterations=1)
                synth_mask = cv2.dilate(synth_mask, k, iterations=1)
                score = float(ssim(actual_mask, synth_mask, data_range=255))
                ssim_scores.append(score)
                # For worst images, store both masks stacked horizontally
                marks = np.concatenate([actual_mask, synth_mask], axis=1)
                if len(worst_ssim) < max_worst:
                    worst_ssim.append((score, frame_idx, marks))
                    worst_ssim.sort(key=lambda x: x[0])
                else:
                    if score < worst_ssim[-1][0]:
                        worst_ssim[-1] = (score, frame_idx, marks)
                        worst_ssim.sort(key=lambda x: x[0])

        frame_idx += 1

    cap.release()

    if metric == 'hit_rate':
        report = {
            "frames_compared": len(hit_rates),
            "mean_hit_rate": float(np.mean(hit_rates)) if hit_rates else None,
            "min_hit_rate": float(np.min(hit_rates)) if hit_rates else None,
            "below_threshold": int(sum(1 for r in hit_rates if r < hit_threshold)),
            "stride": int(stride),
            "output_size": [int(output_size[0]), int(output_size[1])],
            "hit_threshold": float(hit_threshold),
            "window": int(window),
            "conf_threshold": float(conf_threshold),
            "csv_problems": csv_problems,
            "metric": "hit_rate",
        }
    elif metric == 'ssim_green':
        report = {
            "frames_compared": len(ssim_scores),
            "mean_ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
            "min_ssim": float(np.min(ssim_scores)) if ssim_scores else None,
            "below_threshold": int(sum(1 for s in ssim_scores if s < ssim_threshold)),
            "stride": int(stride),
            "output_size": [int(output_size[0]), int(output_size[1])],
            "ssim_threshold": float(ssim_threshold),
            "csv_problems": csv_problems,
            "metric": "ssim_green",
        }
    else:
        report = {
            "frames_compared": int(max(len(hit_rates), len(ssim_scores))),
            "mean_hit_rate": float(np.mean(hit_rates)) if hit_rates else None,
            "min_hit_rate": float(np.min(hit_rates)) if hit_rates else None,
            "below_threshold_hit": int(sum(1 for r in hit_rates if r < hit_threshold)) if hit_rates else None,
            "mean_ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
            "min_ssim": float(np.min(ssim_scores)) if ssim_scores else None,
            "below_threshold_ssim": int(sum(1 for s in ssim_scores if s < ssim_threshold)) if ssim_scores else None,
            "stride": int(stride),
            "output_size": [int(output_size[0]), int(output_size[1])],
            "hit_threshold": float(hit_threshold),
            "ssim_threshold": float(ssim_threshold),
            "window": int(window),
            "conf_threshold": float(conf_threshold),
            "csv_problems": csv_problems,
            "metric": "both",
        }

    worst_frames_hit = []
    worst_frames_ssim = []
    if worst_dir is not None:
        os.makedirs(worst_dir, exist_ok=True)
        # Hit-rate worst frames
        target_dir = worst_dir if metric != 'both' else os.path.join(worst_dir, "hit_rate")
        if (metric in ('hit_rate', 'both')) and worst_hit:
            os.makedirs(target_dir, exist_ok=True)
            for rank, (score, fidx, marks_img) in enumerate(worst_hit):
                stem = f"rank_{rank:02d}_frame_{fidx}_hit_{score:.4f}"
                marks_path = os.path.join(target_dir, stem + "_marks.png")
                try:
                    cv2.imwrite(marks_path, marks_img)
                    worst_frames_hit.append({"frame": int(fidx), "marks": marks_path, "hit_rate": float(score)})
                except Exception:
                    pass

        # SSIM worst frames
        target_dir = worst_dir if metric != 'both' else os.path.join(worst_dir, "ssim_green")
        if (metric in ('ssim_green', 'both')) and worst_ssim:
            os.makedirs(target_dir, exist_ok=True)
            for rank, (score, fidx, masks_img) in enumerate(worst_ssim):
                stem = f"rank_{rank:02d}_frame_{fidx}_ssim_{score:.4f}"
                marks_path = os.path.join(target_dir, stem + "_masks.png")
                try:
                    cv2.imwrite(marks_path, masks_img)
                    worst_frames_ssim.append({"frame": int(fidx), "marks": marks_path, "ssim": float(score)})
                except Exception:
                    pass

    if metric == 'hit_rate':
        report["worst_frames"] = worst_frames_hit
    elif metric == 'ssim_green':
        report["worst_frames"] = worst_frames_ssim
    else:
        report["worst_frames_hit"] = worst_frames_hit
        report["worst_frames_ssim"] = worst_frames_ssim
    return report


def save_report(report, out_json_path, out_csv_path=None):
    out_dir = os.path.dirname(out_json_path)
    if out_dir:  # Only create directory if path has a directory component
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(report, f, indent=2)
    if out_csv_path is not None:
        metric = report.get("metric")
        if metric == 'both':
            hit_csv = out_csv_path.replace('.csv', '_hit.csv')
            ssim_csv = out_csv_path.replace('.csv', '_ssim.csv')
            # Hit
            rows = []
            for wf in report.get("worst_frames_hit", []):
                rows.append({"frame": wf.get("frame"), "hit_rate": wf.get("hit_rate"), "marks": wf.get("marks")})
            pd.DataFrame(rows).to_csv(hit_csv, index=False)
            # SSIM
            rows = []
            for wf in report.get("worst_frames_ssim", []):
                rows.append({"frame": wf.get("frame"), "ssim": wf.get("ssim"), "marks": wf.get("marks")})
            pd.DataFrame(rows).to_csv(ssim_csv, index=False)
        else:
            rows = []
            for wf in report.get("worst_frames", []):
                row = {"frame": wf.get("frame"), "marks": wf.get("marks")}
                if "hit_rate" in wf:
                    row["hit_rate"] = wf.get("hit_rate")
                if "ssim" in wf:
                    row["ssim"] = wf.get("ssim")
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv_path, index=False)


