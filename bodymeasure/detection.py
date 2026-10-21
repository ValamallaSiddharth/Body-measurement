"""MediaPipe adapters and deterministic mask helpers; no disk I/O."""
import cv2
import mediapipe as mp
import numpy as np
from .errors import CaptureError

def detect_pose(image, name, warnings):
    """Stage 1: 33 landmarks; a profile permits one visible side per joint pair."""
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                min_detection_confidence=0.5) as detector:
        result = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        raise CaptureError(f"No person detected in the {name} image. Use a clear full-body photo.")
    landmarks = result.pose_landmarks.landmark
    points = np.array([(p.x * image.shape[1], p.y * image.shape[0]) for p in landmarks])
    visibility = np.array([p.visibility for p in landmarks])
    groups = {"shoulders": (11, 12), "hips": (23, 24), "knees": (25, 26), "ankles": (27, 28)}
    for label, pair in groups.items():
        scores = visibility[list(pair)]
        score = float(min(scores) if name == "front" else max(scores))
        if score < 0.5:
            raise CaptureError(f"{name.title()} {label} have low visibility. Retake the photo with joints unobstructed.")
        if min(scores) < 0.75:
            warnings.append(f"{name.title()} {label} have reduced visibility; related estimates may be unreliable.")
    if not np.isfinite(points).all():
        raise CaptureError(f"Invalid pose in the {name} image; retake the photo.")
    h, w = image.shape[:2]
    outside = ((points[:, 0] < 0) | (points[:, 0] > w-1) |
               (points[:, 1] < 0) | (points[:, 1] > h-1))
    if np.any(outside & (visibility >= 0.5)):
        warnings.append(f"{name.title()} landmarks extend outside the frame; coordinates were clamped and measurements may be inaccurate.")
    points[:, 0] = np.clip(points[:, 0], 0, w-1)
    points[:, 1] = np.clip(points[:, 1], 0, h-1)
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
    # Exclusive slice ends may equal the dimension; starts cannot exceed it.
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


