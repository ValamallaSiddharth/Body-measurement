"""Uncalibrated adult estimates; all image processing stays in memory.

Coordinates remain pixels; aligned slices use normalized body-height units.
"""
from dataclasses import dataclass
import math
import cv2
import mediapipe as mp
import numpy as np


class CaptureError(Exception):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class View:
    name: str
    points: np.ndarray
    visibility: np.ndarray
    mask: np.ndarray
    top: float
    ground: float
    chin: float | None
    cm_per_pixel: float = 0.0
    touched_edges: tuple[str, ...] = ()

    @property
    def height_px(self):
        return self.ground - self.top

    def y(self, indices):
        return float(np.mean(self.points[indices, 1]))


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


def segment_body(image, name):
    """Stage 2: Selfie Segmentation, morphology, largest connected component."""
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        result = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = (result.segmentation_mask > 0.5).astype(np.uint8)
    k = max(3, int(min(mask.shape) / 300) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        raise CaptureError(f"No usable body silhouette in the {name} image. Try a plain contrasting background.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    if int(mask.sum()) < image.shape[0] * image.shape[1] * 0.015:
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


def boundary_confidence(confidence, views):
    """Reduce one level, once per request, when head or feet meet a boundary."""
    if any({"top", "bottom"}.intersection(view.touched_edges) for view in views):
        return {"high": "medium", "medium": "low", "low": "low"}[confidence]
    return confidence


def prepare_view(image, name, warnings):
    points, visibility = detect_pose(image, name, warnings)
    mask = segment_body(image, name)
    ys, _ = np.where(mask)
    top, ground = float(ys.min()), float(ys.max())
    touched = edges_touched(mask, tol=2)
    if touched:
        warnings.append(f"{name.title()} silhouette is close to the image edge")
        warnings.append(f"{name.title()} touched edges: {', '.join(touched)}")
        if "bottom" in touched:
            warnings.append("Feet may be cut off; height may be underestimated")
        if "top" in touched:
            warnings.append("Head may be cut off; height and scale may be inaccurate")
    h, w = mask.shape[:2]
    visible_ankles = [i for i in (27, 28) if visibility[i] >= 0.5]
    ankles = float(np.clip(np.max(points[visible_ankles, 1]) if visible_ankles else ground, 0, h-1))
    if "bottom" in touched:
        ground = ankles
    height = ground - top
    if height < 100 or abs(ground-ankles) > 0.12*height:
        raise CaptureError(f"{name.title()} silhouette does not cover the feet reliably. Retake with a plain background and full body visible.")
    for i in (11, 12, 23, 24, 25, 26, 27, 28):
        if visibility[i] < 0.5:
            continue
        x, y = np.rint(np.clip(points[i], [0, 0], [w-1, h-1])).astype(int)
        r = max(3, int(height*0.025))
        x0, x1 = max(0, x-r), min(w, x+r+1)
        y0, y1 = max(0, y-r), min(h, y+r+1)
        if not mask[y0:y1, x0:x1].any():
            if touched and (x <= r or x >= w-1-r or y <= r or y >= h-1-r):
                warnings.append(f"{name.title()} boundary joints are outside the visible silhouette; related measurements may be inaccurate.")
                continue
            raise CaptureError(f"{name.title()} segmentation misses body joints. Improve lighting and background contrast.")
    return View(name, points, visibility, mask, top, ground,
                detect_chin(image, points, name, warnings), touched_edges=tuple(touched))


def estimate_scale(view, gender, warnings):
    """Stage 4: independent metric anchor for each photograph.

    Adult vertex-to-chin priors: male 23.2 cm, female 21.8 cm (task constants).
    No face: assume 7.5 head lengths per adult. This is a population prior.
    Expected engineering error budget: roughly 10-20% or worse for lengths,
    15-30% or worse for girths; NOT validated bounds/confidence intervals.
    Hair, head pitch, age and population differences can exceed these ranges.
    """
    head_cm = 23.2 if gender == "male" else 21.8
    head_px = view.chin - view.top if view.chin is not None else None
    anchored = head_px is not None and 0.085 <= head_px/view.height_px <= 0.20
    if not anchored:
        if head_px is None:
            warnings.append(f"{view.name.title()} face/chin not detected; scale uses the adult 7.5-head proportion prior.")
        else:
            warnings.append(f"{view.name.title()} head geometry is unreliable; using the adult 7.5-head proportion prior.")
        head_px = view.height_px / 7.5
    view.cm_per_pixel = head_cm / head_px
    return view.height_px * view.cm_per_pixel, anchored


def normalize_views(front, side, heights, anchored, warnings):
    """Stage 5: top/ground become 0/1; both photos share one estimated stature.

    Estimate per-view scales first to accommodate different camera distances.
    Weight a face-derived height 1 and a proportion fallback 0.25.
    """
    height = float(np.average(heights, weights=[1 if a else 0.25 for a in anchored]))
    if abs(heights[0]-heights[1])/height > 0.1:
        warnings.append("Front and side height anchors disagree by more than 10%; scale confidence is low.")
    for view in (front, side):
        view.cm_per_pixel = height / view.height_px
    for label, indices in (("shoulder", [11, 12]), ("hip", [23, 24]), ("knee", [25, 26])):
        levels = [(v.y(indices)-v.top)/v.height_px for v in (front, side)]
        if abs(levels[0]-levels[1]) > 0.05:
            warnings.append(f"Front/side {label} levels differ; posture or perspective may misalign the slices.")
    ratio = np.linalg.norm(front.points[11]-front.points[12]) / front.height_px
    if not 0.17 <= ratio <= 0.29:
        warnings.append("Shoulder-to-height proportion is unusual; check that the front photo faces the camera squarely.")
    return height


def ellipse_circumference(width, depth):
    a, b = width/2, depth/2
    return math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))


def slice_width(view, normalized_y, target_x, label, warnings, limb=False):
    """Median local horizontal run at the body/limb, excluding separated arms.

    Hidden boundaries of merged limbs cannot be recovered: return null with a
    warning rather than reporting the entire torso or both legs as one limb.
    """
    h, w = view.mask.shape[:2]
    raw_y = view.top + normalized_y*view.height_px
    y = int(np.clip(round(raw_y), 0, h-1))
    clipped_x = float(np.clip(target_x, 0, w-1))
    if not (0 <= raw_y <= h-1 and 0 <= target_x <= w-1):
        warnings.append(f"{view.name.title()} {label} slice falls outside the image; this measurement is unavailable.")
        return None
    target_x = clipped_x
    radius = max(1, int(view.height_px*0.003))
    widths = []
    for row_y in range(max(0, y-radius), min(h, y+radius+1)):
        changes = np.diff(np.pad(view.mask[row_y].astype(np.int16), (1, 1)))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        for start, end in zip(starts, ends):
            if start-2 <= target_x < end+2:
                widths.append(float(end-start))
                break
    if not widths:
        warnings.append(f"{view.name.title()} {label} slice is unavailable; retake with limbs separated and fitted clothing.")
        return None
    width = float(np.median(widths))
    if limb and width/view.height_px > (0.06 if label == "wrist" else 0.17):
        warnings.append(f"{view.name.title()} {label} merges with another body part; this girth is unavailable.")
        return None
    return width * view.cm_per_pixel


def metric(value):
    if value is None or not math.isfinite(value):
        return {"cm": None, "in": None}
    return {"cm": round(float(value), 1), "in": round(float(value)/2.54, 1)}


def extract_measurements(front, side, height, warnings):
    """Stage 6: matched normalized slices and scaled Euclidean joint lengths."""
    views = (front, side)
    levels = {}
    for view in views:
        shoulder, hip = view.y([11, 12]), view.y([23, 24])
        levels[view.name] = {"neck": (view.y([7, 8])+shoulder)/2,
                            "chest": shoulder+0.23*(hip-shoulder),
                            "waist": shoulder+0.72*(hip-shoulder), "hips": hip}
    girths, dimensions = {}, {}
    for label in ("neck", "chest", "waist", "hips"):
        normalized_y = float(np.mean([(levels[v.name][label]-v.top)/v.height_px for v in views]))
        diameters = []
        for view in views:
            indices = [7, 8, 11, 12] if label == "neck" else [11, 12, 23, 24]
            x = float(np.mean(view.points[indices, 0]))
            diameters.append(slice_width(view, normalized_y, x, label, warnings))
        dimensions[label] = diameters
        girths[label] = None if None in diameters else ellipse_circumference(*diameters)

    # Same anatomical limb in both views, selected by combined visibility.
    for label, pairs in (("wrist", ((15,), (16,))), ("thigh", ((23, 25), (24, 26)))):
        scores = [min(float(v.visibility[list(ids)].min()) for v in views) for ids in pairs]
        ids = pairs[int(np.argmax(scores))]
        if max(scores) < 0.5:
            warnings.append(f"{label.title()} landmarks have low visibility; girth is unavailable.")
            girths[label] = None
            continue
        anchors = [v.points[ids[0]] if label == "wrist" else
                   0.65*v.points[ids[0]]+0.35*v.points[ids[1]] for v in views]
        anchor_levels = [(p[1]-v.top)/v.height_px for v, p in zip(views, anchors)]
        if abs(anchor_levels[0]-anchor_levels[1]) > 0.035:
            warnings.append(f"{label.title()} heights differ between views; matching slices would measure different anatomy, so girth is unavailable.")
            girths[label] = None
            continue
        normalized_y = float(np.mean(anchor_levels))
        diameters = []
        for view, point in zip(views, anchors):
            slice_y = view.top + normalized_y*view.height_px
            proximal = (13 if ids[0] == 15 else 14) if label == "wrist" else ids[0]
            distal = ids[0] if label == "wrist" else ids[1]
            p, q = view.points[proximal], view.points[distal]
            x = float(point[0])
            if abs(q[1]-p[1]) > 1:
                x = float(p[0] + (slice_y-p[1])/(q[1]-p[1])*(q[0]-p[0]))
            diameters.append(slice_width(view, normalized_y, x, label, warnings, limb=True))
        girths[label] = None if None in diameters else ellipse_circumference(*diameters)

    def length_chain(indices):
        return sum(float(np.linalg.norm(front.points[a]-front.points[b]))
                   for a, b in zip(indices, indices[1:])) * front.cm_per_pixel

    arm_lengths = [length_chain(ids) for ids in ((11, 13, 15), (12, 14, 16))
                   if min(front.visibility[list(ids)]) >= 0.5]
    arm = float(np.mean(arm_lengths)) if arm_lengths else None
    if arm is None:
        warnings.append("Arm joints have low visibility; arm length is unavailable.")
    hip_midpoint = front.points[[23, 24]].mean(axis=0)
    inseam = float(np.linalg.norm(hip_midpoint-front.points[[27, 28]].mean(axis=0))) * front.cm_per_pixel
    torso = float(np.linalg.norm(front.points[[11, 12]].mean(axis=0)-hip_midpoint)) * front.cm_per_pixel
    warnings.append("Inseam is a hip-midpoint-to-ankle proxy; MediaPipe has no crotch landmark. It may exceed a tape-measured inseam.")
    results = {
        "head_neck": {"neck": metric(girths["neck"])},
        "torso": {"shoulder_width": metric(length_chain((11, 12))), "torso_length": metric(torso),
                  **{k: metric(girths[k]) for k in ("chest", "waist", "hips")}},
        "arms": {"arm_length": metric(arm), "wrist": metric(girths["wrist"])},
        "legs": {"inseam": metric(inseam), "thigh": metric(girths["thigh"])}
    }
    return results, dimensions


def sanity_checks(height, measurements, dimensions, warnings):
    """Stage 7: anomalous estimates are warnings, never crashes or clamped values."""
    if not 120 <= height <= 220:
        warnings.append("Estimated height is outside 120–220 cm; the adult scale prior may not apply.")
    torso = measurements["torso"]
    waist, hips = torso["waist"]["cm"], torso["hips"]["cm"]
    if waist is not None and hips is not None and waist > hips:
        warnings.append("Estimated waist exceeds hips. Check clothing, posture and slice alignment.")
    for label, (width, depth) in dimensions.items():
        if width and depth and depth > width*1.25:
            warnings.append(f"{label.title()} side depth is much larger than front width; arms may be merged with the torso.")
    bounds = {"neck": (20, 65), "shoulder_width": (20, 65), "torso_length": (25, 80),
              "chest": (50, 180), "waist": (40, 180), "hips": (50, 190),
              "arm_length": (30, 100), "wrist": (10, 30), "inseam": (40, 120), "thigh": (25, 100)}
    for group in measurements.values():
        for name, value in group.items():
            cm = value["cm"]
            if cm is not None and not bounds[name][0] <= cm <= bounds[name][1]:
                warnings.append(f"{name.replace('_', ' ').title()} is outside a broad adult plausibility range.")


def run_pipeline(front_image, side_image, gender):
    if gender not in ("male", "female"):
        raise CaptureError("Select Male or Female.", 400)
    warnings = []
    front = prepare_view(front_image, "front", warnings)
    side = prepare_view(side_image, "side", warnings)
    estimates = [estimate_scale(view, gender, warnings) for view in (front, side)]
    heights, anchored = [item[0] for item in estimates], [item[1] for item in estimates]
    height = normalize_views(front, side, heights, anchored, warnings)
    measurements, dimensions = extract_measurements(front, side, height, warnings)
    sanity_checks(height, measurements, dimensions, warnings)
    # Only the always-present inseam note permits medium; never claim high.
    confidence = "medium" if all(anchored) and len(warnings) == 1 else "low"
    confidence = boundary_confidence(confidence, (front, side))
    warnings.insert(0, "Absolute size cannot be determined uniquely from two uncalibrated photos. These adult-prior estimates may be substantially wrong.")
    warnings.append("Girths assume elliptical cross-sections. Hair, clothing, perspective and limb overlap affect results; null means unavailable.")
    return {"status": "ok", "height": metric(height), "measurements": measurements,
            "confidence": confidence, "warnings": list(dict.fromkeys(warnings))}
