"""MediaPipe adapters and deterministic mask helpers; no disk I/O."""
import cv2
import mediapipe as mp
import numpy as np
from .errors import CaptureError

SYMMETRIC_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]
SWAP_MAP = {i: i for i in range(33)}
for a, b in SYMMETRIC_PAIRS:
    SWAP_MAP[a] = b
    SWAP_MAP[b] = a


def detect_pose(image, name, warnings):
    """Stage 1: 33 landmarks; model_complexity=2 + horizontal flip smoothing + profile recovery."""
    h, w = image.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                min_detection_confidence=0.5) as detector:
        res1 = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        flipped = cv2.flip(image, 1)
        res2 = detector.process(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

    if res1.pose_landmarks is None and res2.pose_landmarks is None:
        raise CaptureError(f"No person detected in the {name} image. Use a clear full-body photo.")

    if (res1.pose_landmarks is not None and res2.pose_landmarks is not None
            and res1.pose_landmarks is not res2.pose_landmarks):
        lms1 = res1.pose_landmarks.landmark
        pts1 = np.array([(p.x * w, p.y * h) for p in lms1])
        vis1 = np.array([p.visibility for p in lms1])

        lms2 = res2.pose_landmarks.landmark
        pts2 = np.zeros_like(pts1)
        vis2 = np.zeros_like(vis1)
        for i, p in enumerate(lms2):
            orig_idx = SWAP_MAP[i]
            pts2[orig_idx] = ((1.0 - p.x) * w, p.y * h)
            vis2[orig_idx] = p.visibility

        pts1[:, 0] = np.clip(pts1[:, 0], 0, w - 1)
        pts1[:, 1] = np.clip(pts1[:, 1], 0, h - 1)
        pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
        pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)

        w1 = np.maximum(0.01, vis1)[:, None]
        w2 = np.maximum(0.01, vis2)[:, None]
        points = (pts1 * w1 + pts2 * w2) / (w1 + w2)
        visibility = np.maximum(vis1, vis2)
    elif res1.pose_landmarks is not None:
        lms1 = res1.pose_landmarks.landmark
        points = np.array([(p.x * w, p.y * h) for p in lms1])
        visibility = np.array([p.visibility for p in lms1])
    else:
        lms2 = res2.pose_landmarks.landmark
        points = np.zeros((33, 2))
        visibility = np.zeros(33)
        for i, p in enumerate(lms2):
            orig_idx = SWAP_MAP[i]
            points[orig_idx] = ((1.0 - p.x) * w, p.y * h)
            visibility[orig_idx] = p.visibility

    outside = ((points[:, 0] < 0) | (points[:, 0] > w-1) |
               (points[:, 1] < 0) | (points[:, 1] > h-1))
    if np.any(outside & (visibility >= 0.5)):
        warnings.append(f"{name.title()} landmarks extend outside the frame; coordinates were clamped and measurements may be inaccurate.")
    points[:, 0] = np.clip(points[:, 0], 0, w-1)
    points[:, 1] = np.clip(points[:, 1], 0, h-1)

    groups = {"shoulders": (11, 12), "hips": (23, 24), "knees": (25, 26), "ankles": (27, 28)}
    for label, pair in groups.items():
        scores = visibility[list(pair)]
        if name == "front":
            score = float(min(scores))
            if score < 0.5:
                raise CaptureError(f"{name.title()} {label} have low visibility. Retake the photo with joints unobstructed.")
            if score < 0.75:
                warnings.append(f"{name.title()} {label} have reduced visibility; related estimates may be unreliable.")
        else:
            score = float(max(scores))
            if score < 0.5:
                recovered = False
                if label == "knees" and visibility[23] >= 0.5 and visibility[27] >= 0.5:
                    hip_pt = points[23] if visibility[23] >= visibility[24] else points[24]
                    ank_pt = points[27] if visibility[27] >= visibility[28] else points[28]
                    knee_pt = hip_pt + 0.52 * (ank_pt - hip_pt)
                    points[pair[0]] = knee_pt
                    points[pair[1]] = knee_pt
                    visibility[pair[0]] = 0.6
                    visibility[pair[1]] = 0.6
                    warnings.append("Side knees were occluded and recovered from leg proportions.")
                    recovered = True
                elif label == "ankles" and visibility[25] >= 0.5 and visibility[23] >= 0.5:
                    hip_pt = points[23] if visibility[23] >= visibility[24] else points[24]
                    knee_pt = points[25] if visibility[25] >= visibility[26] else points[26]
                    ank_pt = knee_pt + 0.92 * (knee_pt - hip_pt)
                    points[pair[0]] = ank_pt
                    points[pair[1]] = ank_pt
                    visibility[pair[0]] = 0.6
                    visibility[pair[1]] = 0.6
                    warnings.append("Side ankles were occluded and recovered from leg proportions.")
                    recovered = True
                if not recovered:
                    raise CaptureError(f"Side {label} have low visibility. Retake the photo with joints unobstructed.")
            else:
                better = pair[0] if visibility[pair[0]] >= visibility[pair[1]] else pair[1]
                weaker = pair[1] if better == pair[0] else pair[0]
                if visibility[weaker] < 0.5:
                    points[weaker] = points[better].copy()
                    visibility[weaker] = visibility[better]
                    warnings.append(f"Side {label} had occluded landmark; recovered from opposite side.")

    if not np.isfinite(points).all():
        raise CaptureError(f"Invalid pose in the {name} image; retake the photo.")
    return points, visibility


def segment_body(image, name, warnings=None):
    """Stage 2: Selfie Segmentation, morphology, largest connected component."""
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        result = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return clean_mask(result.segmentation_mask, name, warnings)


def clean_mask(probabilities, name, warnings=None):
    """Pure mask cleanup; component count is a warning, not a people detector."""
    mask = (probabilities > 0.5).astype(np.uint8)
    k = max(3, int(min(mask.shape) / 300) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        raise CaptureError(f"No usable body silhouette in the {name} image. Try a plain contrasting background.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    areas = stats[1:, cv2.CC_STAT_AREA]
    if np.count_nonzero(areas >= max(mask.size*.03, areas.max()*.25)) > 1 and warnings is not None:
        warnings.append(f"{name.title()} has multiple large silhouette components; possible multiple people or segmentation fragments. Include only one person.")
    mask = (labels == largest).astype(np.uint8)
    if int(mask.sum()) < mask.size * 0.015:
        raise CaptureError(f"Body silhouette is too small in the {name} image. Move closer with the full body visible.")
    return mask


def detect_chin(image, points, name, warnings):
    """Stage 3: Face Mesh chin 152 in a pose-guided head crop, in image pixels."""
    h, w = image.shape[:2]
    shoulder_y = float(np.mean(points[[11, 12], 1]))
    nose = points[0]
    extent = max(32.0, abs(shoulder_y - nose[1]) * 1.5)
    x0, x1 = np.clip([int(nose[0]-extent), int(nose[0]+extent)], 0, w).astype(int)
    y0, y1 = np.clip([int(nose[1]-extent), int(shoulder_y)], 0, h).astype(int)
    chin = None
    if x1 > x0 and y1 > y0:
        crop = cv2.resize(image[y0:y1, x0:x1], None, fx=2, fy=2)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             min_detection_confidence=0.5) as face:
            result = face.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            landmark = result.multi_face_landmarks[0].landmark[152]
            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                chin = y0 + landmark.y * (y1 - y0)
    return chin


def detect_face_mesh_details(image, points, name):
    """Extract chin, forehead, eyes and head pitch from Face Mesh if available."""
    h, w = image.shape[:2]
    shoulder_y = float(np.mean(points[[11, 12], 1]))
    nose = points[0]
    extent = max(32.0, abs(shoulder_y - nose[1]) * 1.5)
    x0, x1 = np.clip([int(nose[0]-extent), int(nose[0]+extent)], 0, w).astype(int)
    y0, y1 = np.clip([int(nose[1]-extent), int(shoulder_y)], 0, h).astype(int)
    details = {"chin": None, "forehead": None, "pitch_cosine": 1.0, "eyes_y": None}
    if x1 > x0 and y1 > y0:
        crop = cv2.resize(image[y0:y1, x0:x1], None, fx=2, fy=2)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             min_detection_confidence=0.5) as face:
            result = face.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            chin_lm = lms[152]
            forehead_lm = lms[10]
            nose_lm = lms[1]
            eye_l = lms[33]
            eye_r = lms[263]
            if 0 <= chin_lm.x <= 1 and 0 <= chin_lm.y <= 1:
                details["chin"] = float(y0 + chin_lm.y * (y1 - y0))
            if 0 <= forehead_lm.x <= 1 and 0 <= forehead_lm.y <= 1:
                details["forehead"] = float(y0 + forehead_lm.y * (y1 - y0))
            if 0 <= eye_l.y <= 1 and 0 <= eye_r.y <= 1:
                details["eyes_y"] = float(y0 + 0.5 * (eye_l.y + eye_r.y) * (y1 - y0))
            upper = abs(nose_lm.y - forehead_lm.y)
            lower = abs(chin_lm.y - nose_lm.y)
            if upper > 0.01 and lower > 0.01:
                ratio = upper / lower
                tilt_rad = min(0.6, abs(ratio - 1.0) * 0.7)
                details["pitch_cosine"] = float(max(0.75, np.cos(tilt_rad)))
    return details


def edges_touched(mask, tol=2):
    """Identify silhouette pixels within tol pixels of each image boundary."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return []
    h, w = mask.shape[:2]
    touched = []
    if ys.min() <= tol:
        touched.append("top")
    if ys.max() >= h-1-tol:
        touched.append("bottom")
    if xs.min() <= tol:
        touched.append("left")
    if xs.max() >= w-1-tol:
        touched.append("right")
    return touched
