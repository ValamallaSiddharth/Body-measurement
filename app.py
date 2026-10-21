"""Two-photo measurements. Run with Python 3.11: python app.py."""
import io
import base64
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from threading import BoundedSemaphore

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, Request, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from bodymeasure.pipeline import CaptureError, detect_pose, segment_body, detect_chin, edges_touched

# Adult engineering priors, NOT population-validated anthropometric standards.
# All anthropometric tuning belongs here. Length error may be 10-20% or worse;
# circumference error may be 15-30% or worse. These are not validated bounds.
HEAD_REF_CM = {"male": 23.2, "female": 21.8}
BASELINE_HEIGHT_CM = {"male": 175.0, "female": 162.0}
HEAD_BODY_RATIO_RANGE = (0.11, 0.16)  # Reject unreliable front chin/vertex anchors.
HEAD_REGION_HALF_WIDTH_TO_SHOULDERS = 0.55  # Excludes raised hands from vertex search.
HEAD_REGION_MIN_HALF_WIDTH_TO_IMAGE = 0.06
SHOULDER_BAND_HEIGHT_RATIO = 0.03  # Maximum outer span in shoulder row +/- 3%.
DELTOID_TO_HEIGHT_RATIO = {"male": 0.25, "female": 0.235}
DELTOID_HEIGHT_RANGE = (0.16, 0.35)
CHEST_LEVEL_IN_TORSO = 0.23
WAIST_LEVEL_IN_TORSO = 0.72
CHEST_TO_DELTOID_MAX = 0.95
TORSO_WIDTH_TO_DELTOID = {"neck": 0.30, "chest": 0.80, "waist": 0.72, "hips": 0.90}
TORSO_WIDTH_MIN = {"neck": 0.15, "chest": 0.40, "waist": 0.30, "hips": 0.40}
TORSO_WIDTH_MAX = {"neck": 0.55, "chest": CHEST_TO_DELTOID_MAX, "waist": 1.20, "hips": 1.30}
DEPTH_WIDTH_MIN, DEPTH_WIDTH_MAX = 0.40, 1.00
TORSO_DEPTH_TO_WIDTH = {"neck": 0.85, "chest": 0.68, "waist": 0.72, "hips": 0.72}
ARM_HINT_ROW_HEIGHT_RATIO = 0.025
ARM_HINT_OUTER_RUN_FRACTION = 0.25
ARM_HINT_PRIOR_EXCESS = 1.15  # Hints trigger fallback only with excessive spans.
WAIST_HIP_DEPTH_TRIGGER = 1.05
WAIST_HIP_DEPTH_FALLBACK = 0.90  # Applied ONLY to an arm-contaminated waist.
WRIST_TO_HEIGHT_RATIO = {"male": 0.100, "female": 0.095}  # Circumference prior.
WRIST_HEIGHT_RANGE = (0.065, 0.145)
WRIST_DIAMETER_HEIGHT_MAX = 0.045
WRIST_SCAN_HALF_HEIGHT_RATIO = 0.10
WRIST_HIP_PROXIMITY_HEIGHT_RATIO = 0.12
THIGH_HIP_TO_KNEE_FRACTION = 0.30
THIGH_TO_HEIGHT_RATIO = {"male": 0.32, "female": 0.31}  # Circumference prior.
THIGH_HEIGHT_RANGE = (0.20, 0.50)
THIGH_DEPTH_WIDTH_RANGE = (0.40, 2.0)
LENGTH_TO_HEIGHT = {"arm_length": 0.34, "torso_length": 0.29, "inseam": 0.45}
LENGTH_HEIGHT_RANGE = {"arm_length": (0.20, 0.50), "torso_length": (0.18, 0.40), "inseam": (0.25, 0.60)}
ADULT_HEIGHT_RANGE_CM = (120, 220)
LANDMARK_VISIBILITY_MIN = 0.5
EDGE_TOLERANCE_PX = 2

ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 20_000_000
PROCESSING_SLOT = BoundedSemaphore(1)


@dataclass
class View:
    name: str
    image: np.ndarray
    mask: np.ndarray
    points: np.ndarray
    visibility: np.ndarray
    vertex: float
    ground: float
    touched_edges: tuple = ()
    chin: float | None = None
    cm_per_px: float = 0.0
    traces: list = field(default_factory=list)

    @property
    def body_px_height(self):
        return max(1.0, self.ground - self.vertex)


def point_in_image(point, shape):
    return np.clip(np.asarray(point, dtype=float), [0, 0], [shape[1]-1, shape[0]-1])


def prepare_view(image, name, notes):
    points, visibility = detect_pose(image, name, notes)
    mask = segment_body(image, name)
    h, w = mask.shape
    points = np.clip(points, [0, 0], [w-1, h-1])
    ys, _ = np.where(mask > 0)
    # Search only the head region above shoulders, centered on the pose nose.
    half_width = max(abs(points[11, 0]-points[12, 0])*HEAD_REGION_HALF_WIDTH_TO_SHOULDERS,
                     w*HEAD_REGION_MIN_HALF_WIDTH_TO_IMAGE)
    x0, x1 = np.clip([math.floor(points[0, 0]-half_width), math.ceil(points[0, 0]+half_width)+1], 0, w).astype(int)
    head_bottom = int(np.clip(math.ceil(points[[11, 12], 1].mean()), 1, h))
    head_ys, _ = np.where(mask[:head_bottom, x0:x1] > 0)
    vertex = float(head_ys.min()) if head_ys.size else float(ys.min())
    if not head_ys.size:
        notes.append(f"{name.title()} head region is uncertain; silhouette top was used.")
    touched = tuple(edges_touched(mask, EDGE_TOLERANCE_PX))
    if touched:
        notes += [f"{name.title()} silhouette is close to the image edge",
                  f"{name.title()} touched edges: {', '.join(touched)}"]
        if "bottom" in touched:
            notes.append("Feet may be cut off; height may be underestimated")
        if "top" in touched:
            notes.append("Head may be cut off; height and scale may be inaccurate")
    feet = [i for i in (29, 30, 31, 32) if visibility[i] > LANDMARK_VISIBILITY_MIN]
    if not feet:
        feet = [i for i in (27, 28) if visibility[i] > LANDMARK_VISIBILITY_MIN]
    foot_y = float(points[feet, 1].max()) if feet else float(ys.max())
    ground = foot_y if "bottom" in touched else max(foot_y, float(ys.max()))
    ground = float(np.clip(ground, 0, h-1))
    if ground <= vertex:
        vertex, ground = float(ys.min()), float(ys.max())
        notes.append(f"{name.title()} foot geometry failed; using visible silhouette span.")
    chin = detect_chin(image, points, name, notes) if name == "front" else None
    return View(name, image, mask, points, visibility, vertex, ground, touched, chin)


def metric(value, source):
    return {"cm": round(float(value), 1), "in": round(float(value)/2.54, 1), "source": source}


def estimate_scale(front, side, gender, notes):
    """Derive absolute scale ONCE from the front head; normalize side separately."""
    head_px = front.chin-front.vertex if front.chin is not None else float("nan")
    ratio = head_px/front.body_px_height
    if math.isfinite(ratio) and HEAD_BODY_RATIO_RANGE[0] <= ratio <= HEAD_BODY_RATIO_RANGE[1]:
        front.cm_per_px = HEAD_REF_CM[gender]/head_px
        height_cm = front.body_px_height*front.cm_per_px
        source = "measured"
    else:
        height_cm = BASELINE_HEIGHT_CM[gender]
        front.cm_per_px = height_cm/front.body_px_height
        source = "assumed"
        reason = "front face/chin not detected" if front.chin is None else "front head/body ratio outside 0.11-0.16"
        notes.append(f"Height is not measured: {reason}; assuming {height_cm:.1f} cm adult baseline.")
        notes.append("All dimensions using the assumed height scale are labeled estimated.")
    side.cm_per_px = height_cm/side.body_px_height
    return height_cm, source


def row_runs(mask, y):
    row = int(np.clip(round(y), 0, mask.shape[0]-1))
    changes = np.diff(np.pad((mask[row] > 0).astype(np.int16), (1, 1)))
    return row, list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1))


def mask_run(view, y, x):
    """Only the continuous foreground run containing the target axis."""
    if not np.isfinite([x, y]).all():
        return None
    point = point_in_image([x, y], view.mask.shape)
    row, runs = row_runs(view.mask, point[1])
    if x < 0 or x > view.mask.shape[1]-1 or y < 0 or y > view.mask.shape[0]-1:
        return None  # Clamped safely, but do not invent geometry outside the photo.
    for left, right in runs:
        if left <= point[0] <= right:
            return float(right-left+1), np.array([left, row]), np.array([right, row])
    return None


def axis_x(view, y, upper=(11, 12), lower=(23, 24)):
    p, q = view.points[list(upper)].mean(axis=0), view.points[list(lower)].mean(axis=0)
    t = (y-p[1])/(q[1]-p[1]) if abs(q[1]-p[1]) > 1 else 0.0
    return float(np.clip(p[0]+t*(q[0]-p[0]), 0, view.mask.shape[1]-1))


def record_span(view, name, span, source, used_cm=None):
    if span is None:
        return
    width, p, q = span
    view.traces.append({"label": name, "start": point_in_image(p, view.mask.shape).tolist(),
                        "end": point_in_image(q, view.mask.shape).tolist(),
                        "observed_cm": float(width*view.cm_per_px),
                        "used_cm": float(used_cm if used_cm is not None else width*view.cm_per_px),
                        "source": source})


def outer_shoulder_width(front, height, gender, scale_source, notes):
    center = float(front.points[[11, 12], 1].mean())
    radius = SHOULDER_BAND_HEIGHT_RATIO*front.body_px_height
    lo, hi = np.clip([math.floor(center-radius), math.ceil(center+radius)], 0, front.mask.shape[0]-1).astype(int)
    candidates = []
    for y in range(lo, hi+1):
        row, runs = row_runs(front.mask, y)
        if runs:
            left, right = runs[0][0], runs[-1][1]
            candidates.append((float(right-left+1), np.array([left, row]), np.array([right, row])))
    span = max(candidates, key=lambda item: item[0]) if candidates else None
    cm = span[0]*front.cm_per_px if span else float("nan")
    source = "measured" if scale_source == "measured" else "estimated"
    if (not math.isfinite(cm) or not DELTOID_HEIGHT_RANGE[0] <= cm/height <= DELTOID_HEIGHT_RANGE[1]
            or min(front.visibility[[11, 12]]) <= LANDMARK_VISIBILITY_MIN):
        cm, source = DELTOID_TO_HEIGHT_RATIO[gender]*height, "estimated"
        notes.append("Shoulder width estimated from height: outer shoulder band is missing, occluded or implausible.")
    record_span(front, "deltoids", span, source, cm)
    return cm, source


def arm_hint(view, y, span):
    if span is None:
        return False
    width, p, q = span
    axis = axis_x(view, y)
    hints = []
    for elbow, wrist in ((13, 15), (14, 16)):
        for index in (elbow, wrist):
            if view.visibility[index] > LANDMARK_VISIBILITY_MIN and abs(view.points[index, 1]-y) <= ARM_HINT_ROW_HEIGHT_RATIO*view.body_px_height:
                hints.append(view.points[index, 0])
        # Interpolate only when BOTH forearm landmarks are visible.
        if min(view.visibility[[elbow, wrist]]) > LANDMARK_VISIBILITY_MIN:
            a, b = view.points[elbow], view.points[wrist]
            if min(a[1], b[1]) <= y <= max(a[1], b[1]) and abs(b[1]-a[1]) > 1:
                hints.append(a[0]+(y-a[1])/(b[1]-a[1])*(b[0]-a[0]))
    return any(p[0] <= x <= q[0] and abs(x-axis) > ARM_HINT_OUTER_RUN_FRACTION*width for x in hints)


def ellipse_circumference(width_cm, depth_cm):
    a, b = width_cm/2, depth_cm/2
    return math.pi*(3*(a+b)-math.sqrt((3*a+b)*(a+3*b)))


def constrain_waist_depth(dimensions, notes):
    waist, hips = dimensions["waist"], dimensions["hips"]
    if waist["contaminated"] and waist["depth"] > hips["depth"]*WAIST_HIP_DEPTH_TRIGGER:
        waist["depth"] = hips["depth"]*WAIST_HIP_DEPTH_FALLBACK
        waist["source"] = "estimated"
        notes.append("Arm-contaminated waist depth exceeded hip depth; estimated as 90% of hip depth.")
    elif waist["depth"] > hips["depth"]*WAIST_HIP_DEPTH_TRIGGER:
        notes.append("Waist depth exceeds hip depth; retained because no waist arm contamination was detected.")


def torso_measurements(front, side, height, gender, scale_source, deltoid_cm, deltoid_source, notes):
    shoulder_y, hip_y = front.points[[11, 12], 1].mean(), front.points[[23, 24], 1].mean()
    chin_y = front.chin if front.chin is not None else front.vertex + HEAD_REF_CM[gender]/front.cm_per_px
    levels = {"neck": (chin_y+shoulder_y)/2,
              "chest": shoulder_y+CHEST_LEVEL_IN_TORSO*(hip_y-shoulder_y),
              "waist": shoulder_y+WAIST_LEVEL_IN_TORSO*(hip_y-shoulder_y), "hips": hip_y}
    dimensions = {}
    for name, front_y in levels.items():
        normalized_y = (front_y-front.vertex)/front.body_px_height
        side_y = side.vertex+normalized_y*side.body_px_height
        front_span = mask_run(front, front_y, axis_x(front, front_y))
        side_span = mask_run(side, side_y, axis_x(side, side_y))
        width = front_span[0]*front.cm_per_px if front_span else float("nan")
        depth = side_span[0]*side.cm_per_px if side_span else float("nan")
        estimated_width = TORSO_WIDTH_TO_DELTOID[name]*deltoid_cm
        reasons = []
        contaminated = False
        if not math.isfinite(width) or not math.isfinite(depth) or width <= 0 or depth <= 0:
            reasons.append("a torso-axis mask run is missing")
        else:
            if width > TORSO_WIDTH_MAX[name]*deltoid_cm:
                contaminated = True
                reasons.append("front run is excessively wide and may include arms")
            if width < TORSO_WIDTH_MIN[name]*deltoid_cm:
                reasons.append("front mask run is too narrow")
            if not DEPTH_WIDTH_MIN <= depth/width <= DEPTH_WIDTH_MAX:
                contaminated = True
                reasons.append("side depth/front width ratio is implausible")
            if (arm_hint(front, front_y, front_span) and width > estimated_width*ARM_HINT_PRIOR_EXCESS) or (
                    arm_hint(side, side_y, side_span) and depth > width*TORSO_DEPTH_TO_WIDTH[name]*ARM_HINT_PRIOR_EXCESS):
                contaminated = True
                reasons.append("visible forearm hints indicate a merged arm")
        if min(front.visibility[[11, 12, 23, 24]]) <= LANDMARK_VISIBILITY_MIN or max(side.visibility[[11, 12]]) <= LANDMARK_VISIBILITY_MIN or max(side.visibility[[23, 24]]) <= LANDMARK_VISIBILITY_MIN:
            reasons.append("torso landmarks have low visibility")
        source = "measured" if scale_source == "measured" else "estimated"
        if reasons:
            width, depth = estimated_width, estimated_width*TORSO_DEPTH_TO_WIDTH[name]
            source = "estimated"
            notes.append(f"{name.title()} estimated from deltoid proportions: {'; '.join(reasons)}.")
            if deltoid_source == "estimated":
                notes.append(f"{name.title()} fallback also depends on estimated shoulder width.")
        if name == "neck" and front.chin is None:
            source = "estimated"
            notes.append("Neck row estimated because the front chin was not detected.")
        dimensions[name] = {"width": width, "depth": depth, "source": source, "contaminated": contaminated,
                            "front_span": front_span, "side_span": side_span}
    constrain_waist_depth(dimensions, notes)
    values = {}
    for name, dimension in dimensions.items():
        values[name] = metric(ellipse_circumference(dimension["width"], dimension["depth"]), dimension["source"])
        record_span(front, name, dimension["front_span"], dimension["source"], dimension["width"])
        record_span(side, name, dimension["side_span"], dimension["source"], dimension["depth"])
    if values["waist"]["cm"] > values["hips"]["cm"]:
        notes.append("Waist circumference exceeds hips; this can be legitimate. Values were not silently clamped.")
    return values


def perpendicular_wrist_span(view, elbow, wrist):
    center = view.points[wrist]
    axis = center-view.points[elbow]
    norm = float(np.linalg.norm(axis))
    if norm < 1:
        return None
    direction = np.array([-axis[1], axis[0]])/norm
    radius = max(1, int(math.ceil(WRIST_SCAN_HALF_HEIGHT_RATIO*view.body_px_height)))
    offsets = np.arange(-radius, radius+1)
    positions = center+offsets[:, None]*direction
    valid = ((positions >= [0, 0]) & (positions <= [view.mask.shape[1]-1, view.mask.shape[0]-1])).all(axis=1)
    indices = np.rint(np.clip(positions, [0, 0], [view.mask.shape[1]-1, view.mask.shape[0]-1])).astype(int)
    foreground = (view.mask[indices[:, 1], indices[:, 0]] > 0) & valid
    if not foreground[radius]:
        return None
    left = right = radius
    while left > 0 and foreground[left-1]:
        left -= 1
    while right+1 < len(foreground) and foreground[right+1]:
        right += 1
    if left == 0 or right == len(foreground)-1:
        return None  # Diameter did not terminate inside the search window.
    return float(right-left+1), positions[left], positions[right]


def wrist_measurement(front, side, height, gender, scale_source, notes):
    candidates = []
    for elbow, wrist in ((13, 15), (14, 16)):
        if any(min(v.visibility[[elbow, wrist]]) <= LANDMARK_VISIBILITY_MIN for v in (front, side)):
            continue
        fspan = perpendicular_wrist_span(front, elbow, wrist)
        point = side.points[wrist]
        sspan = mask_run(side, point[1], point[0])
        if fspan is None or sspan is None:
            continue
        diameters = [fspan[0]*front.cm_per_px, sspan[0]*side.cm_per_px]
        merged = max(diameters)/height > WRIST_DIAMETER_HEIGHT_MAX
        for view, span in ((front, fspan), (side, sspan)):
            wrist_y = view.points[wrist, 1]
            hip_y = view.points[[23, 24], 1].mean()
            torso_x = axis_x(view, wrist_y)
            if (abs(wrist_y-hip_y) < WRIST_HIP_PROXIMITY_HEIGHT_RATIO*view.body_px_height and
                    min(span[1][0], span[2][0]) <= torso_x <= max(span[1][0], span[2][0])):
                merged = True
        circumference = ellipse_circumference(*diameters)
        if not merged and WRIST_HEIGHT_RANGE[0] <= circumference/height <= WRIST_HEIGHT_RANGE[1]:
            candidates.append((circumference, fspan, sspan))
    if not candidates:
        notes.append("Wrist estimated from height: low visibility, missing cross-section, implausible diameter or merged hip/thigh silhouette.")
        return metric(WRIST_TO_HEIGHT_RATIO[gender]*height, "estimated")
    source = "measured" if scale_source == "measured" else "estimated"
    for _, fspan, sspan in candidates:
        record_span(front, "wrist", fspan, source)
        record_span(side, "wrist", sspan, source)
    return metric(float(np.mean([item[0] for item in candidates])), source)


def thigh_measurement(front, side, height, gender, scale_source, notes):
    estimates = []
    source = "measured" if scale_source == "measured" else "estimated"
    midline = float(front.points[[23, 24], 0].mean())
    for hip, knee in ((23, 25), (24, 26)):
        p = front.points[hip] + THIGH_HIP_TO_KNEE_FRACTION*(front.points[knee]-front.points[hip])
        span = mask_run(front, p[1], p[0])
        leg_source = "measured" if scale_source == "measured" else "estimated"
        if span is not None:
            other_hip, other_knee = (24, 26) if hip == 23 else (23, 25)
            other = front.points[other_hip] + THIGH_HIP_TO_KNEE_FRACTION*(front.points[other_knee]-front.points[other_hip])
            _, left, right = span
            if left[0] <= other[0] <= right[0]:
                split = float(np.clip(midline, left[0], right[0]))
                if front.points[hip, 0] < front.points[other_hip, 0]:
                    right = np.array([split, left[1]])
                else:
                    left = np.array([split, right[1]])
                span = float(right[0]-left[0]+1), left, right
                leg_source = "estimated"
                notes.append("Thigh width estimated by splitting merged front legs at the hip midpoint.")
        normalized_y = (p[1]-front.vertex)/front.body_px_height
        side_y = side.vertex+normalized_y*side.body_px_height
        sspan = mask_run(side, side_y, axis_x(side, side_y, (23, 24), (25, 26)))
        reliable = min(front.visibility[[hip, knee]]) > LANDMARK_VISIBILITY_MIN and max(side.visibility[[23, 24]]) > LANDMARK_VISIBILITY_MIN and max(side.visibility[[25, 26]]) > LANDMARK_VISIBILITY_MIN
        value = float("nan")
        if reliable and span is not None and sspan is not None:
            width, depth = span[0]*front.cm_per_px, sspan[0]*side.cm_per_px
            if width > 0 and THIGH_DEPTH_WIDTH_RANGE[0] <= depth/width <= THIGH_DEPTH_WIDTH_RANGE[1]:
                value = ellipse_circumference(width, depth)
        if not math.isfinite(value) or not THIGH_HEIGHT_RANGE[0] <= value/height <= THIGH_HEIGHT_RANGE[1]:
            value, leg_source = THIGH_TO_HEIGHT_RATIO[gender]*height, "estimated"
            notes.append("Thigh circumference estimated from height: one leg slice is missing, low visibility or implausible.")
        estimates.append(value)
        if leg_source == "estimated":
            source = "estimated"
        record_span(front, "thigh", span, leg_source)
        record_span(side, "thigh", sspan, leg_source)
    return metric(float(np.mean(estimates)), source)


def length_measurements(front, height, scale_source, notes):
    points = front.points
    values = {}
    arms = [sum(float(np.linalg.norm(points[a]-points[b])) for a, b in zip(ids, ids[1:]))*front.cm_per_px
            for ids in ((11, 13, 15), (12, 14, 16)) if min(front.visibility[list(ids)]) > LANDMARK_VISIBILITY_MIN]
    hip_mid = points[[23, 24]].mean(axis=0)
    raw = {"arm_length": float(np.mean(arms)) if arms else float("nan"),
           "torso_length": float(np.linalg.norm(points[[11, 12]].mean(axis=0)-hip_mid))*front.cm_per_px,
           "inseam": float(np.linalg.norm(points[[27, 28]].mean(axis=0)-hip_mid))*front.cm_per_px}
    for name, value in raw.items():
        source = "measured" if scale_source == "measured" else "estimated"
        bounds = LENGTH_HEIGHT_RANGE[name]
        ids = [11, 12, 23, 24] if name == "torso_length" else [23, 24, 27, 28]
        reliable = bool(arms) if name == "arm_length" else min(front.visibility[ids]) > LANDMARK_VISIBILITY_MIN
        if not math.isfinite(value) or not bounds[0] <= value/height <= bounds[1] or not reliable:
            value, source = LENGTH_TO_HEIGHT[name]*height, "estimated"
            notes.append(f"{name.replace('_', ' ').title()} estimated from height: joint length is missing or implausible.")
        if name == "inseam":
            source = "estimated"
            notes.append("Inseam is an estimated hip-midpoint-to-ankle proxy; Pose has no crotch landmark.")
        values[name] = metric(value, source)
    return values


def measurement_confidence(height_source, torso, views):
    # Quality tiers, NOT statistical accuracy. Each key estimate lowers one tier.
    penalties = int(height_source != "measured")
    penalties += sum(torso[name]["source"] != "measured" for name in ("chest", "waist", "hips"))
    penalties += int(any({"top", "bottom"}.intersection(v.touched_edges) for v in views))
    return ("high", "medium", "low")[min(2, penalties)]


def debug_overlay(view):
    """Return in-memory PNG data URI plus exact observed/used span metadata."""
    canvas = view.image.copy()
    foreground = view.mask > 0
    canvas[foreground] = (0.55*canvas[foreground] + 0.45*np.array([70, 180, 70])).astype(np.uint8)
    h, w = view.mask.shape
    for label, y in (("vertex", view.vertex), ("ground", view.ground)):
        y = int(np.clip(round(y), 0, h-1))
        cv2.line(canvas, (0, y), (w-1, y), (255, 160, 0), 1)
        cv2.putText(canvas, label, (5, int(np.clip(y-5, 12, h-1))), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 160, 0), 1)
    if view.chin is not None:
        y = int(np.clip(round(view.chin), 0, h-1))
        cv2.line(canvas, (0, y), (w-1, y), (255, 160, 0), 1)
        cv2.putText(canvas, "front chin 152", (5, int(np.clip(y-5, 12, h-1))), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 160, 0), 1)
    for number, trace in enumerate(view.traces):
        p, q = [tuple(np.rint(point_in_image(trace[key], view.mask.shape)).astype(int)) for key in ("start", "end")]
        color = (0, 200, 255) if trace["source"] != "measured" else (255, 80, 30)
        cv2.line(canvas, (0, p[1]), (w-1, p[1]), (150, 150, 150), 1)
        cv2.line(canvas, p, q, color, 2)
        cv2.circle(canvas, p, 3, color, -1)
        cv2.circle(canvas, q, 3, color, -1)
        # Numbered legend prevents nearby neck/chest/thigh labels overlapping.
        cv2.putText(canvas, str(number+1), (p[0], int(np.clip(p[1]-5, 12, h-1))), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    legend = np.full((max(h, 40+24*len(view.traces)), 390, 3), 245, np.uint8)
    cv2.putText(legend, f"{view.name}: observed -> used cm", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1)
    for number, trace in enumerate(view.traces):
        line = f"{number+1} {trace['label']}: {trace['observed_cm']:.1f} -> {trace['used_cm']:.1f} ({trace['source']})"
        cv2.putText(legend, line, (10, 46+24*number), cv2.FONT_HERSHEY_SIMPLEX, .40, (30, 30, 30), 1)
    canvas = cv2.copyMakeBorder(canvas, 0, legend.shape[0]-h, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    ok, encoded = cv2.imencode(".png", np.concatenate([canvas, legend], axis=1))
    if not ok:
        raise RuntimeError("Could not encode debug overlay")
    return {"overlay": "data:image/png;base64,"+base64.b64encode(encoded).decode("ascii"),
            "cm_per_px": view.cm_per_px, "vertex_y": view.vertex, "ground_y": view.ground,
            "touched_edges": list(view.touched_edges), "slices": view.traces}


def run_pipeline(front_image, side_image, gender, debug=False):
    if gender not in HEAD_REF_CM:
        raise CaptureError("Select Male or Female.", 400)
    notes = []
    front, side = prepare_view(front_image, "front", notes), prepare_view(side_image, "side", notes)
    height, height_source = estimate_scale(front, side, gender, notes)
    deltoid, deltoid_source = outer_shoulder_width(front, height, gender, height_source, notes)
    torso = torso_measurements(front, side, height, gender, height_source, deltoid, deltoid_source, notes)
    lengths = length_measurements(front, height, height_source, notes)
    measurements = {"head_neck": {"neck": torso["neck"]},
                    "torso": {"shoulder_width": metric(deltoid, deltoid_source),
                              **{name: torso[name] for name in ("chest", "waist", "hips")}, "torso_length": lengths["torso_length"]},
                    "arms": {"arm_length": lengths["arm_length"], "wrist": wrist_measurement(front, side, height, gender, height_source, notes)},
                    "legs": {"inseam": lengths["inseam"], "thigh": thigh_measurement(front, side, height, gender, height_source, notes)}}
    if not ADULT_HEIGHT_RANGE_CM[0] <= height <= ADULT_HEIGHT_RANGE_CM[1]:
        notes.append("Height is outside the broad adult plausibility range.")
    notes.append("Measured means image-derived using an adult head-size prior, not a validated tape measurement. Clothing, hair and perspective can cause substantial error.")
    result = {"status": "ok", "height": metric(height, height_source), "measurements": measurements,
              "confidence": measurement_confidence(height_source, torso, (front, side)),
              "warnings": list(dict.fromkeys(notes))}
    if debug:
        result["debug"] = {view.name: debug_overlay(view) for view in (front, side)}
    return result


class MemoryRequest(Request):
    # Werkzeug otherwise spools multipart files above ~500 KB to a temp file.
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def validate_upload(upload, view):
    """Check encoded type, byte/pixel limits and decoding, entirely in RAM."""
    if upload is None or not upload.filename:
        raise CaptureError(f"Upload the {view} image.", 400)
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in formats:
        raise CaptureError(f"{view.title()} image must be JPG, PNG or WebP.", 400)
    data = upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise CaptureError(f"{view.title()} image exceeds the 12 MB limit.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format != formats[suffix]:
                    raise CaptureError(f"{view.title()} image contents do not match its extension.", 400)
                if source.width * source.height > MAX_PIXELS:
                    raise CaptureError(f"{view.title()} image exceeds the 20 megapixel limit.", 413)
                if min(source.size) < 128:
                    raise CaptureError(f"{view.title()} image must be at least 128 pixels on each side.", 400)
                if getattr(source, "is_animated", False):
                    raise CaptureError(f"{view.title()} image must be a still photo.", 400)
                source.load()
                oriented = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", oriented.size, "white")
                background.alpha_composite(oriented)
                background.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                rgb = np.asarray(background.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise CaptureError(f"{view.title()} image dimensions are too large.", 413) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CaptureError(f"{view.title()} image cannot be decoded. Upload a valid JPG, PNG or WebP.", 400) from exc


def create_app():
    app = Flask(__name__)
    app.request_class = MemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

    @app.get("/")
    def index():
        return send_file(ROOT / "index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", service="two-photo-body-measurement",
                       boundary_policy="warn")

    @app.post("/measure")
    def measure():
        gender = request.form.get("gender", "").strip().lower()
        if gender not in ("male", "female"):
            raise CaptureError("Select Male or Female.", 400)
        if not PROCESSING_SLOT.acquire(blocking=False):
            raise CaptureError("Another measurement is processing. Please retry shortly.", 503)
        try:
            images = [validate_upload(request.files.get(f"{view}_image"), view)
                      for view in ("front", "side")]
            return jsonify(run_pipeline(*images, gender))
        finally:
            PROCESSING_SLOT.release()

    @app.errorhandler(CaptureError)
    def capture_error(exc):
        return jsonify(status="error", error=str(exc)), exc.status_code

    @app.errorhandler(HTTPException)
    def http_error(exc):
        message = ("Request exceeds 25 MB; each image must be at most 12 MB."
                   if exc.code == 413 else exc.description)
        response = exc.get_response()
        response.data = app.json.dumps({"status": "error", "error": message})
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def internal_error(exc):
        app.logger.exception("Measurement request failed")
        return jsonify(status="error", error="Internal processing error. Please retry; see the server log for details."), 500

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")), debug=False)
