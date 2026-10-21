"""Shared body measurement geometry and orchestration; images stay in RAM."""
import base64
import math
from dataclasses import dataclass, field
import cv2
import numpy as np
from .errors import CaptureError
from .detection import detect_pose, segment_body, detect_chin, detect_face_mesh_details, edges_touched

# Adult engineering priors, NOT population-validated anthropometric standards.
HEAD_REF_CM = {"male": 23.2, "female": 21.8}
BASELINE_HEIGHT_CM = {"male": 175.0, "female": 162.0}
POPULATION_PRIOR = {
    "male": {"mean": 175.0, "sd": 7.0},
    "female": {"mean": 162.0, "sd": 6.5}
}
HEAD_BODY_RATIO_RANGE = (0.11, 0.16)
HEAD_REGION_HALF_WIDTH_TO_SHOULDERS = 0.55
HEAD_REGION_MIN_HALF_WIDTH_TO_IMAGE = 0.06
SHOULDER_BAND_HEIGHT_RATIO = 0.03
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
ARM_HINT_PRIOR_EXCESS = 1.15
WAIST_HIP_DEPTH_TRIGGER = 1.05
WAIST_HIP_DEPTH_FALLBACK = 0.90
WRIST_TO_HEIGHT_RATIO = {"male": 0.100, "female": 0.095}
WRIST_HEIGHT_RANGE = (0.065, 0.145)
WRIST_DIAMETER_HEIGHT_MAX = 0.045
WRIST_SCAN_HALF_HEIGHT_RATIO = 0.10
WRIST_HIP_PROXIMITY_HEIGHT_RATIO = 0.12
THIGH_HIP_TO_KNEE_FRACTION = 0.30
THIGH_TO_HEIGHT_RATIO = {"male": 0.32, "female": 0.31}
THIGH_HEIGHT_RANGE = (0.20, 0.50)
THIGH_DEPTH_WIDTH_RANGE = (0.40, 2.0)
LENGTH_TO_HEIGHT = {"arm_length": 0.34, "torso_length": 0.29, "inseam": 0.45}
LENGTH_HEIGHT_RANGE = {"arm_length": (0.20, 0.50), "torso_length": (0.18, 0.40), "inseam": (0.25, 0.60)}
ADULT_HEIGHT_RANGE_CM = (120, 220)
LANDMARK_VISIBILITY_MIN = 0.5
EDGE_TOLERANCE_PX = 2


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
    torso_mask: np.ndarray | None = None
    arm_mask: np.ndarray | None = None
    face_meta: dict = field(default_factory=dict)
    scale_meta: dict = field(default_factory=dict)

    @property
    def body_px_height(self):
        return max(1.0, self.ground - self.vertex)


def point_in_image(point, shape):
    return np.clip(np.asarray(point, dtype=float), [0, 0], [shape[1]-1, shape[0]-1])


def camera_tilt_cosine(view):
    """Estimate camera vertical tilt from body midline vector."""
    if min(view.visibility[[11, 12, 27, 28]]) >= LANDMARK_VISIBILITY_MIN:
        mid_s = view.points[[11, 12]].mean(axis=0)
        mid_a = view.points[[27, 28]].mean(axis=0)
        dx = abs(mid_s[0] - mid_a[0])
        dy = abs(mid_s[1] - mid_a[1])
        if dy > 50:
            angle = math.atan2(dx, dy)
            if angle > math.radians(2.5):
                return float(max(0.85, math.cos(angle)))
    return 1.0


def prepare_view(image, name, notes):
    points, visibility = detect_pose(image, name, notes)
    mask = segment_body(image, name, notes)
    h, w = mask.shape
    points = np.clip(points, [0, 0], [w-1, h-1])
    ys, _ = np.where(mask > 0)
    
    face_meta = detect_face_mesh_details(image, points, name) if name == "front" else {}
    chin = face_meta.get("chin")
    if chin is None and name == "front":
        chin = detect_chin(image, points, name, notes)

    half_width = max(abs(points[11, 0]-points[12, 0])*HEAD_REGION_HALF_WIDTH_TO_SHOULDERS,
                     w*HEAD_REGION_MIN_HALF_WIDTH_TO_IMAGE)
    x0, x1 = np.clip([math.floor(points[0, 0]-half_width), math.ceil(points[0, 0]+half_width)+1], 0, w).astype(int)
    head_bottom = int(np.clip(math.ceil(points[[11, 12], 1].mean()), 1, h))
    head_ys, _ = np.where(mask[:head_bottom, x0:x1] > 0)
    vertex = float(head_ys.min()) if head_ys.size else float(ys.min())
    if not head_ys.size:
        notes.append(f"{name.title()} head region is uncertain; silhouette top was used.")

    # Skull curve vs hair volume
    if face_meta.get("eyes_y") is not None and chin is not None:
        d_eye_chin = chin - face_meta["eyes_y"]
        if d_eye_chin > 5:
            skull_limit = face_meta["eyes_y"] - 1.25 * d_eye_chin
            if vertex < skull_limit:
                vertex = float(skull_limit)

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
        if "bottom" in touched and feet:
            ank_y = float(points[feet, 1].max())
            foot_offset = max(8.0, 0.045 * (ank_y - vertex))
            ground = float(min(h - 1, ank_y + foot_offset))
            notes.append("Feet appear cut off; ground level estimated from ankles with foot-height offset.")
        else:
            foot_y = float(points[feet, 1].max()) if feet else float(ys.max())
            ground = foot_y if "bottom" in touched else max(foot_y, float(ys.max()))
    else:
        foot_y = float(points[feet, 1].max())
        ground = foot_y if "bottom" in touched else max(foot_y, float(ys.max()))

    ground = float(np.clip(ground, 0, h-1))
    if ground <= vertex:
        vertex, ground = float(ys.min()), float(ys.max())
        notes.append(f"{name.title()} foot geometry failed; using visible silhouette span.")
    if ground-vertex < 100:
        raise CaptureError(f"{name.title()} body is too small or incomplete; include the full body with a clear background.")

    for i in (11, 12, 23, 24, 25, 26, 27, 28):
        if visibility[i] < LANDMARK_VISIBILITY_MIN:
            continue
        x, y = np.rint(points[i]).astype(int)
        radius = max(3, int((ground-vertex)*.025))
        if not mask[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)].any():
            if touched and (x <= radius or x >= w-1-radius or y <= radius or y >= h-1-radius):
                notes.append(f"{name.title()} boundary joint is outside the visible silhouette.")
            else:
                raise CaptureError(f"{name.title()} segmentation misses body joints; improve lighting and background contrast.")

    view_obj = View(name, image, mask, points, visibility, vertex, ground, touched, chin, face_meta=face_meta)
    build_torso_mask(view_obj)
    return view_obj


def metric(value, source):
    return {"cm": round(float(value), 1), "in": round(float(value)/2.54, 1), "source": source}


def validate_height(value):
    if value is None or value == "":
        return None
    try:
        height = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureError("Known height must be a number in cm (120-220).", 400) from exc
    if isinstance(value, bool) or not math.isfinite(height) or not 120 <= height <= 220:
        raise CaptureError("Known adult height must be between 120 and 220 cm.", 400)
    return height


def estimate_scale(front, side, gender, notes, height_cm=None):
    """Derive multi-cue fused scale without user height; optional known stature overrides."""
    known_height = validate_height(height_cm)
    head_px = (front.chin - front.vertex) if front.chin is not None else float("nan")
    ratio = head_px / front.body_px_height

    if known_height is not None:
        if math.isfinite(ratio) and HEAD_BODY_RATIO_RANGE[0] <= ratio <= HEAD_BODY_RATIO_RANGE[1]:
            est_h = front.body_px_height * (HEAD_REF_CM[gender] / head_px)
            if abs(est_h - known_height) / known_height > 0.10:
                notes.append("Head-based height and user-supplied height disagree by more than 10%; check capture and entered height.")
        height_cm = known_height
        source = "measured"
        front.cm_per_px = height_cm / front.body_px_height
        side.cm_per_px = height_cm / side.body_px_height
        front.scale_meta = {
            "uncertainty_cm": 0.0,
            "uncertainty_band": f"{round(height_cm, 1)} cm +/- 0.0",
            "cues_used": ["user_height"],
        }
        return height_cm, source

    cos_tilt = camera_tilt_cosine(front)
    eff_front_body_h = front.body_px_height / cos_tilt

    cues = []
    pitch_cos = front.face_meta.get("pitch_cosine", 1.0) if hasattr(front, "face_meta") else 1.0
    if math.isfinite(head_px) and head_px > 10:
        head_px_unproj = head_px / pitch_cos
        r = head_px_unproj / eff_front_body_h
        if HEAD_BODY_RATIO_RANGE[0] <= r <= HEAD_BODY_RATIO_RANGE[1]:
            h_head = eff_front_body_h * (HEAD_REF_CM[gender] / head_px_unproj)
            cues.append(("head_prior", float(h_head), 7.0))

    if np.all(front.visibility[[7, 8, 11, 12]] >= LANDMARK_VISIBILITY_MIN):
        ear_y = float(front.points[[7, 8], 1].mean())
        sh_y = float(front.points[[11, 12], 1].mean())
        neck_px = abs(sh_y - ear_y)
        if neck_px > 5:
            ref_neck = 22.0 if gender == "male" else 20.0
            h_neck = eff_front_body_h * (ref_neck / neck_px)
            if 140 <= h_neck <= 210:
                cues.append(("neck_shoulder_cue", float(h_neck), 9.0))

    if min(front.visibility[[11, 12, 23, 24]]) >= LANDMARK_VISIBILITY_MIN:
        mid_s = front.points[[11, 12]].mean(axis=0)
        mid_h = front.points[[23, 24]].mean(axis=0)
        torso_px = float(np.linalg.norm(mid_s - mid_h))
        ref_torso = 49.8 if gender == "male" else 45.8
        h_torso = eff_front_body_h * (ref_torso / torso_px)
        if 140 <= h_torso <= 210:
            cues.append(("torso_proportion", float(h_torso), 8.5))

    if min(front.visibility[[23, 24, 27, 28]]) >= LANDMARK_VISIBILITY_MIN:
        mid_h = front.points[[23, 24]].mean(axis=0)
        mid_a = front.points[[27, 28]].mean(axis=0)
        leg_px = float(np.linalg.norm(mid_h - mid_a))
        ref_leg = 82.0 if gender == "male" else 75.0
        h_leg = eff_front_body_h * (ref_leg / leg_px)
        if 140 <= h_leg <= 210:
            cues.append(("leg_proportion", float(h_leg), 9.0))

    if side is not None and side.chin is not None:
        side_head_px = side.chin - side.vertex
        if side_head_px > 10:
            h_side = side.body_px_height * (HEAD_REF_CM[gender] / side_head_px)
            if 140 <= h_side <= 210:
                cues.append(("side_head_cue", float(h_side), 9.0))

    prior_mean = POPULATION_PRIOR[gender]["mean"]
    prior_sd = POPULATION_PRIOR[gender]["sd"]

    if not cues:
        height_cm = prior_mean
        source = "assumed"
        uncertainty = 10.0
        cues_used = ["adult_baseline"]
        notes.append(f"Height is not measured: landmarks uncertain; assuming {height_cm:.1f} cm adult baseline.")
    else:
        vals = np.array([c[1] for c in cues])
        med = float(np.median(vals))
        valid_cues = []
        if len(cues) > 2:
            diffs = np.abs(vals - med)
            mad = float(np.median(diffs)) or 5.0
            for c_name, h_val, sd_val in cues:
                if abs(h_val - med) <= max(14.0, 2.5 * 1.4826 * mad):
                    valid_cues.append((c_name, h_val, sd_val))
                else:
                    notes.append(f"Dropped outlier scale cue {c_name} ({h_val:.1f} cm).")
        else:
            valid_cues = cues

        if not valid_cues:
            valid_cues = cues

        if len(valid_cues) == 1:
            w_prior = (1.0 / (prior_sd ** 2)) * 1.5
            notes.append("Only a single scale cue was available; height shrunk toward population prior.")
        elif len(valid_cues) >= 3:
            w_prior = (1.0 / (prior_sd ** 2)) * 0.35
        else:
            w_prior = (1.0 / (prior_sd ** 2)) * 0.8
        weights = [1.0 / (sd_val ** 2) for _, _, sd_val in valid_cues]
        total_w = sum(weights) + w_prior
        height_cm = float((sum(w * h for (_, h, _), w in zip(valid_cues, weights)) + w_prior * prior_mean) / total_w)
        uncertainty = float(min(8.0, max(3.5, 1.96 * np.sqrt(1.0 / total_w))))
        cues_used = [c[0] for c in valid_cues] + ["population_regularizer"]
        source = "estimated"
        notes.append("Height estimated via multi-cue fusion with population regularization; absolute size is uncalibrated.")

    if side is not None and cues:
        side_cues = [c[1] for c in cues if "side" in c[0]]
        front_cues = [c[1] for c in cues if "side" not in c[0]]
        if side_cues and front_cues:
            if abs(np.mean(side_cues) - np.mean(front_cues)) / height_cm > 0.06:
                notes.append("Front and side view height cues disagree by more than 6%; perspective or posture differences may affect accuracy.")

    front.cm_per_px = height_cm / front.body_px_height
    side.cm_per_px = height_cm / side.body_px_height
    front.scale_meta = {
        "uncertainty_cm": round(uncertainty, 1),
        "uncertainty_band": f"{round(height_cm, 1)} cm +/- {round(uncertainty, 1)}",
        "cues_used": cues_used,
    }
    return height_cm, source


def build_torso_mask(view):
    """Fix 1: Subtract arms (shoulder-elbow-wrist lines + wrist hand disc) from the silhouette."""
    arm_mask = np.zeros_like(view.mask)
    body_h = view.body_px_height
    t_arm = max(4, int(0.045 * body_h))
    t_forearm = max(3, int(t_arm * 0.85))
    r_hand = max(4, int(t_forearm * 1.3))

    if view.name == "front":
        arm_chains = [(11, 13, 15), (12, 14, 16)]
    else:
        vis_l = min(view.visibility[[11, 13, 15]])
        vis_r = min(view.visibility[[12, 14, 16]])
        if vis_l >= 0.4 and vis_r >= 0.4:
            arm_chains = [(11, 13, 15), (12, 14, 16)]
        elif vis_l >= vis_r:
            arm_chains = [(11, 13, 15)]
        else:
            arm_chains = [(12, 14, 16)]

    for s_idx, e_idx, w_idx in arm_chains:
        s = tuple(np.rint(view.points[s_idx]).astype(int))
        e = tuple(np.rint(view.points[e_idx]).astype(int))
        wr = tuple(np.rint(view.points[w_idx]).astype(int))
        cv2.line(arm_mask, s, e, 255, thickness=t_arm)
        cv2.line(arm_mask, e, wr, 255, thickness=t_forearm)
        cv2.circle(arm_mask, wr, r_hand, 255, thickness=-1)

    torso_mask = view.mask.copy()
    torso_mask[arm_mask > 0] = 0
    view.arm_mask = arm_mask
    view.torso_mask = torso_mask
    return torso_mask


def get_torso_corridor(view, y):
    """Left and right corridor bounds between shoulders and hips, expanded by margin."""
    p11, p12 = view.points[11], view.points[12]
    p23, p24 = view.points[23], view.points[24]
    s_y = 0.5 * (p11[1] + p12[1])
    h_y = 0.5 * (p23[1] + p24[1])
    dy = max(1.0, h_y - s_y)
    t = float(np.clip((y - s_y) / dy, -0.2, 1.2))

    x_l = p11[0] + t * (p23[0] - p11[0])
    x_r = p12[0] + t * (p24[0] - p12[0])
    x_min, x_max = min(x_l, x_r), max(x_l, x_r)
    margin = max(6.0, 0.05 * view.body_px_height, 0.15 * abs(p11[0] - p12[0]))
    return float(x_min - margin), float(x_max + margin)


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
        return None
    for left, right in runs:
        if left <= point[0] <= right:
            return float(right-left+1), np.array([left, row]), np.array([right, row])
    return None


def mask_run_custom(mask, y, x):
    """Continuous foreground run in given mask containing the target x."""
    if mask is None or not np.isfinite([x, y]).all():
        return None
    h, w = mask.shape
    row = int(np.clip(round(y), 0, h - 1))
    target_x = int(np.clip(round(x), 0, w - 1))
    changes = np.diff(np.pad((mask[row] > 0).astype(np.int16), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for left, right in zip(starts, ends):
        if left - 2 <= target_x <= right + 2:
            return float(right - left + 1), np.array([left, row]), np.array([right, row])
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
    
    if front.torso_mask is None:
        build_torso_mask(front)
    if side.torso_mask is None:
        build_torso_mask(side)

    dimensions = {}
    fallback_levels = []
    for name, front_y in levels.items():
        normalized_y = (front_y-front.vertex)/front.body_px_height
        side_y = side.vertex+normalized_y*side.body_px_height

        axis_front = axis_x(front, front_y)
        axis_side = axis_x(side, side_y)

        raw_front_span = mask_run(front, front_y, axis_front)
        raw_side_span = mask_run(side, side_y, axis_side)

        c_left, c_right = get_torso_corridor(front, front_y)
        t_front_span = mask_run_custom(front.torso_mask, front_y, axis_front)
        t_side_span = mask_run_custom(side.torso_mask, side_y, axis_side)

        contaminated = False
        reasons = []

        if raw_front_span is not None:
            raw_w, p_left, p_right = raw_front_span
            if p_left[0] < c_left or p_right[0] > c_right:
                contaminated = True
                reasons.append("front run is excessively wide and may include arms")
            row_idx = int(np.clip(round(front_y), 0, front.mask.shape[0]-1))
            if front.arm_mask is not None and np.any(front.arm_mask[row_idx, int(p_left[0]):int(p_right[0])+1] > 0):
                contaminated = True
                reasons.append("visible forearm hints indicate a merged arm")

        chosen_front_span = None
        if t_front_span is not None:
            w_t, pl_t, pr_t = t_front_span
            cl = max(pl_t[0], c_left)
            cr = min(pr_t[0], c_right)
            if cr > cl:
                chosen_front_span = (float(cr - cl + 1), np.array([cl, front_y]), np.array([cr, front_y]))
        elif raw_front_span is not None:
            w_r, pl_r, pr_r = raw_front_span
            cl = max(pl_r[0], c_left)
            cr = min(pr_r[0], c_right)
            if cr > cl:
                chosen_front_span = (float(cr - cl + 1), np.array([cl, front_y]), np.array([cr, front_y]))

        chosen_side_span = t_side_span or raw_side_span

        width = chosen_front_span[0]*front.cm_per_px if chosen_front_span else float("nan")
        depth = chosen_side_span[0]*side.cm_per_px if chosen_side_span else float("nan")
        estimated_width = TORSO_WIDTH_TO_DELTOID[name]*deltoid_cm

        if not math.isfinite(width) or not math.isfinite(depth) or width <= 0 or depth <= 0:
            reasons.append("a torso-axis mask run is missing")
        else:
            if width > TORSO_WIDTH_MAX[name]*deltoid_cm:
                contaminated = True
                if "front run is excessively wide and may include arms" not in reasons:
                    reasons.append("front run is excessively wide and may include arms")
            if width < TORSO_WIDTH_MIN[name]*deltoid_cm:
                reasons.append("front mask run is too narrow")
            if not DEPTH_WIDTH_MIN <= depth/width <= DEPTH_WIDTH_MAX:
                contaminated = True
                reasons.append("side depth/front width ratio is implausible")
            if (arm_hint(front, front_y, raw_front_span) and width > estimated_width*ARM_HINT_PRIOR_EXCESS) or (
                    arm_hint(side, side_y, raw_side_span) and depth > width*TORSO_DEPTH_TO_WIDTH[name]*ARM_HINT_PRIOR_EXCESS):
                contaminated = True
                if "visible forearm hints indicate a merged arm" not in reasons:
                    reasons.append("visible forearm hints indicate a merged arm")

        if min(front.visibility[[11, 12, 23, 24]]) <= LANDMARK_VISIBILITY_MIN or max(side.visibility[[11, 12]]) <= LANDMARK_VISIBILITY_MIN or max(side.visibility[[23, 24]]) <= LANDMARK_VISIBILITY_MIN:
            reasons.append("torso landmarks have low visibility")

        source = "measured" if scale_source == "measured" else "estimated"
        if reasons or contaminated:
            width, depth = estimated_width, estimated_width*TORSO_DEPTH_TO_WIDTH[name]
            source = "estimated"
            fallback_levels.append(name)
            notes.append(f"{name.title()} estimated from deltoid proportions: {'; '.join(reasons)}.")

        if name == "neck" and front.chin is None:
            source = "estimated"
            notes.append("Neck row estimated because the front chin was not detected.")

        dimensions[name] = {"width": width, "depth": depth, "source": source, "contaminated": contaminated,
                            "front_span": chosen_front_span, "side_span": chosen_side_span}

    if fallback_levels:
        notes.append("Tip: Keep arms slightly away from the body in the front and side photos for accurate torso measurements.")

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
        return None
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


def find_crotch(view):
    """Fix 2: Find crotch point where the central run splits into two legs."""
    h, w = view.mask.shape
    hip_y = float(view.points[[23, 24], 1].mean())
    knee_y = float(view.points[[25, 26], 1].mean())
    mid_x = float(view.points[[23, 24], 0].mean())
    scan_start = int(np.clip(round(hip_y + 0.05 * (knee_y - hip_y)), 0, h - 1))
    scan_end = int(np.clip(round(knee_y), 0, h))

    for y in range(scan_start, scan_end):
        changes = np.diff(np.pad((view.mask[y] > 0).astype(np.int16), (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        runs = list(zip(starts, ends))
        mid_idx = int(np.clip(round(mid_x), 0, w - 1))
        if view.mask[y, mid_idx] == 0:
            left_runs = [r for r in runs if r[1] < mid_x + 2]
            right_runs = [r for r in runs if r[0] > mid_x - 2]
            if left_runs and right_runs:
                return float(y)
    return None


def thigh_measurement(front, side, height, gender, scale_source, notes):
    estimates = []
    source = "measured" if scale_source == "measured" else "estimated"
    midline = float(front.points[[23, 24], 0].mean())
    crotch_y = find_crotch(front)
    hip_y = float(front.points[[23, 24], 1].mean())
    knee_y = float(front.points[[25, 26], 1].mean())

    if crotch_y is not None:
        thigh_y = crotch_y + 0.30 * (knee_y - crotch_y)
    else:
        thigh_y = hip_y + THIGH_HIP_TO_KNEE_FRACTION * (knee_y - hip_y)

    legs_merged = False
    for hip, knee in ((23, 25), (24, 26)):
        p_x = float(front.points[hip, 0] + THIGH_HIP_TO_KNEE_FRACTION * (front.points[knee, 0] - front.points[hip, 0]))
        span = mask_run(front, thigh_y, p_x)
        leg_source = "measured" if scale_source == "measured" else "estimated"
        if span is not None:
            other_hip = 24 if hip == 23 else 23
            other_x = float(front.points[other_hip, 0])
            _, left, right = span
            if (left[0] < midline < right[0]) and (left[0] <= other_x <= right[0]):
                legs_merged = True
                split = float(np.clip(midline, left[0], right[0]))
                if front.points[hip, 0] < other_x:
                    right = np.array([split, left[1]])
                else:
                    left = np.array([split, right[1]])
                span = float(right[0] - left[0] + 1), left, right
                leg_source = "estimated"

        normalized_y = (thigh_y - front.vertex) / front.body_px_height
        side_y = side.vertex + normalized_y * side.body_px_height
        sspan = mask_run(side, side_y, axis_x(side, side_y, (23, 24), (25, 26)))
        reliable = (min(front.visibility[[hip, knee]]) > LANDMARK_VISIBILITY_MIN and
                    max(side.visibility[[23, 24]]) > LANDMARK_VISIBILITY_MIN and
                    max(side.visibility[[25, 26]]) > LANDMARK_VISIBILITY_MIN)

        value = float("nan")
        if reliable and span is not None and sspan is not None:
            width, depth = span[0] * front.cm_per_px, sspan[0] * side.cm_per_px
            if width > 0 and THIGH_DEPTH_WIDTH_RANGE[0] <= depth / width <= THIGH_DEPTH_WIDTH_RANGE[1]:
                value = ellipse_circumference(width, depth)
        if not math.isfinite(value) or not THIGH_HEIGHT_RANGE[0] <= value / height <= THIGH_HEIGHT_RANGE[1]:
            value, leg_source = THIGH_TO_HEIGHT_RATIO[gender] * height, "estimated"

        estimates.append(value)
        if leg_source == "estimated":
            source = "estimated"
        record_span(front, "thigh", span, leg_source)
        record_span(side, "thigh", sspan, leg_source)

    if legs_merged:
        notes.append("Thigh width estimated by splitting merged front legs at the pelvis midline: legs merged.")

    return metric(float(np.mean(estimates)), source)


def length_measurements(front, height, scale_source, notes):
    points = front.points
    values = {}
    arms = [sum(float(np.linalg.norm(points[a]-points[b])) for a, b in zip(ids, ids[1:]))*front.cm_per_px
            for ids in ((11, 13, 15), (12, 14, 16)) if min(front.visibility[list(ids)]) > LANDMARK_VISIBILITY_MIN]
    hip_mid = points[[23, 24]].mean(axis=0)

    crotch_y = find_crotch(front)
    ankles_y = float(points[[27, 28], 1].mean())
    if crotch_y is not None and ankles_y > crotch_y + 15:
        inseam_val = (ankles_y - crotch_y) * front.cm_per_px
        inseam_src = "measured" if scale_source == "measured" else "estimated"
        notes.append("Inseam measured from segmented crotch level to ankle.")
    else:
        inseam_val = float(np.linalg.norm(points[[27, 28]].mean(axis=0)-hip_mid))*front.cm_per_px
        inseam_src = "estimated"
        notes.append("Inseam is an estimated hip-midpoint-to-ankle proxy; Pose has no crotch landmark.")

    raw = {"arm_length": float(np.mean(arms)) if arms else float("nan"),
           "torso_length": float(np.linalg.norm(points[[11, 12]].mean(axis=0)-hip_mid))*front.cm_per_px,
           "inseam": inseam_val}
    for name, value in raw.items():
        source = inseam_src if name == "inseam" else ("measured" if scale_source == "measured" else "estimated")
        bounds = LENGTH_HEIGHT_RANGE[name]
        ids = [11, 12, 23, 24] if name == "torso_length" else [23, 24, 27, 28]
        reliable = bool(arms) if name == "arm_length" else min(front.visibility[ids]) > LANDMARK_VISIBILITY_MIN
        if not math.isfinite(value) or not bounds[0] <= value/height <= bounds[1] or not reliable:
            value, source = LENGTH_TO_HEIGHT[name]*height, "estimated"
            notes.append(f"{name.replace('_', ' ').title()} estimated from height: joint length is missing or implausible.")
        values[name] = metric(value, source)
    return values


def measurement_confidence(height_source, torso, views, flagged=(), capture_warnings=()):
    """Conservative quality tier, capped at medium or low."""
    degraded = (height_source != "measured" or bool(flagged) or bool(capture_warnings)
                or any(torso[name]["source"] != "measured" for name in ("chest", "waist", "hips"))
                or any({"top", "bottom"}.intersection(v.touched_edges) for v in views))
    return "low" if degraded else "medium"


PLAUSIBILITY = {
    "neck": ((25, 50), (23, 48), (.13, .30)),
    "chest": ((65, 140), (60, 135), (.35, .90)),
    "waist": ((55, 130), (50, 125), (.28, .85)),
    "hips": ((65, 140), (60, 140), (.35, .95)),
    "thigh": ((35, 85), (32, 85), (.18, .50)),
    "wrist": ((12, 25), (10, 24), (.06, .15)),
    "shoulder_width": ((35, 65), (32, 60), (.18, .34)),
    "torso_length": ((25, 80), (23, 75), (.16, .42)),
    "arm_length": ((40, 95), (38, 90), (.22, .52)),
    "inseam": ((50, 110), (45, 105), (.28, .62)),
}


def sanity_checks(height, measurements, gender, notes):
    """Flag specific suspicious values; retain numbers rather than silently clamp."""
    flagged = []
    for group in measurements.values():
        for name, value in group.items():
            male, female, relative = PLAUSIBILITY[name]
            lower, upper = male if gender == "male" else female
            cm = value["cm"]
            if cm is None or not math.isfinite(cm) or not (lower <= cm <= upper and relative[0] <= cm/height <= relative[1]):
                value["source"] = "estimated"
                flagged.append(name)
                notes.append(f"Implausible {name.replace('_', ' ')}: {cm} cm is outside the {gender} screening range ({lower}-{upper} cm) or height-relative range ({relative[0]:.2f}-{relative[1]:.2f}). Retake or verify with a tape measure.")
    return flagged


def enforce_plausibility_guards(measurements, height, gender, deltoid_cm, notes):
    """Fix 5: Re-estimate any measurement that violates physical screening bounds with fallback."""
    flagged = []
    for group_name, group in measurements.items():
        for name, val in group.items():
            cm = val.get("cm")
            if cm is None or not math.isfinite(cm):
                continue
            male, female, relative = PLAUSIBILITY[name]
            lower, upper = male if gender == "male" else female
            rel_lo, rel_hi = relative

            tripped = not (lower <= cm <= upper and rel_lo <= cm / height <= rel_hi)
            if name == "waist" and measurements["torso"]["hips"]["cm"] is not None:
                hip_cm = measurements["torso"]["hips"]["cm"]
                if cm > hip_cm * 1.10:
                    tripped = True

            if tripped:
                flagged.append(name)
                old_cm = cm
                if name == "neck":
                    new_cm = (0.22 if gender == "male" else 0.20) * height
                elif name == "chest":
                    new_cm = TORSO_WIDTH_TO_DELTOID["chest"] * deltoid_cm * 2.3
                elif name == "waist":
                    hip_cm = measurements["torso"]["hips"]["cm"] or (0.58 * height)
                    new_cm = min(old_cm, hip_cm * 0.88)
                elif name == "hips":
                    new_cm = (0.58 if gender == "male" else 0.62) * height
                elif name == "shoulder_width":
                    new_cm = DELTOID_TO_HEIGHT_RATIO[gender] * height
                elif name == "thigh":
                    new_cm = THIGH_TO_HEIGHT_RATIO[gender] * height
                elif name == "inseam":
                    new_cm = 0.45 * height
                elif name == "arm_length":
                    new_cm = 0.34 * height
                elif name == "wrist":
                    new_cm = WRIST_TO_HEIGHT_RATIO[gender] * height
                else:
                    new_cm = old_cm

                val["cm"] = round(new_cm, 1)
                val["in"] = round(new_cm / 2.54, 1)
                val["source"] = "estimated"
                notes.append(
                    f"Implausible {name.replace('_', ' ')} ({old_cm:.1f} cm) tripped screening range ({lower}-{upper} cm); "
                    f"re-estimated as {new_cm:.1f} cm using anatomical fallback."
                )
    return flagged


def capture_checks(front, side):
    """Non-identifying geometric heuristics; cannot establish subject identity."""
    notes = []
    proportions = []
    for view in (front, side):
        shoulders = view.points[[11, 12]].mean(axis=0)
        hips = view.points[[23, 24]].mean(axis=0)
        torso = max(1.0, float(np.linalg.norm(shoulders-hips)))
        shoulder_width = abs(view.points[11, 0]-view.points[12, 0])
        ratio = shoulder_width/torso
        if (view.name == "front" and ratio < .35) or (view.name == "side" and ratio > .65):
            notes.append(f"{view.name.title()} orientation may be incorrect; use a straight front view and a 90-degree side view.")
        proportions.append(torso/view.body_px_height)
        if view.name == "front" and shoulder_width > 1:
            y = shoulders[1] + .4*(hips[1]-shoulders[1])
            span = mask_run(view, y, axis_x(view, y))
            if span and span[0]/shoulder_width > 1.6:
                notes.append("Front silhouette is very wide relative to the skeleton; loose clothing or merged arms may distort measurements.")
    if abs(proportions[0]-proportions[1])/max(proportions) > .25:
        notes.append("Front/side body proportions differ substantially; possible different subjects, posture or perspective. Use the same person in both photos; identity is not verified.")
    return notes


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


def measure(front_image, side_image, gender, options=None):
    """Only public measurement entry point. Inputs are validated BGR image arrays.

    Options: height_cm (optional known stature), debug (in-memory overlays).
    No files, masks, models or results are written by this function.
    """
    options = options or {}
    if gender not in HEAD_REF_CM:
        raise CaptureError("Select Male or Female.", 400)
    known_height = validate_height(options.get("height_cm"))
    for name, img in (("front", front_image), ("side", side_image)):
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 128:
            raise CaptureError(f"{name.title()} must be a uint8 BGR image at least 128 pixels on each side.", 400)
    notes = []
    front, side = prepare_view(front_image, "front", notes), prepare_view(side_image, "side", notes)
    capture_notes = notes.copy() + capture_checks(front, side)
    notes = capture_notes.copy()
    height, height_source = estimate_scale(front, side, gender, notes, known_height)
    if any("disagree" in note for note in notes):
        capture_notes.append("Scale disagreement")
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

    flagged = sanity_checks(height, measurements, gender, notes)
    flagged += enforce_plausibility_guards(measurements, height, gender, deltoid, notes)

    notes.append("Measured means user-supplied height or image-derived geometry scaled by it, not a validated tape measurement. Estimated values use priors or uncertain geometry. Clothing, hair and perspective can cause substantial error.")
    
    height_meta = getattr(front, "scale_meta", {}) or {}
    h_dict = metric(height, height_source)
    if known_height is None and "uncertainty_cm" in height_meta:
        h_dict["uncertainty_cm"] = height_meta["uncertainty_cm"]
        h_dict["uncertainty_band"] = height_meta["uncertainty_band"]
        h_dict["cues_used"] = height_meta["cues_used"]

    result = {"status": "ok", "height": h_dict, "measurements": measurements,
              "confidence": measurement_confidence(height_source, torso, (front, side), flagged, capture_notes),
              "scale_anchor": "user_height" if known_height is not None else "head_prior" if height_source == "estimated" else "adult_baseline",
              "warnings": list(dict.fromkeys(notes))}
    if options.get("debug", False):
        result["debug"] = {view.name: debug_overlay(view) for view in (front, side)}
    return result
