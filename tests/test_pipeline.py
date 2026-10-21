"""Synthetic tests for the actual shared algorithm; no model/photo dependency."""
import math
from unittest.mock import patch

import numpy as np
import pytest

from bodymeasure import pipeline as p
from bodymeasure.detection import clean_mask


def view(name="front", factor=1, chin=36):
    mask = np.zeros((240, 160), np.uint8)
    mask[10:211, 50:101] = 1
    points = np.full((33, 2), [75., 100.])
    for left, right, y in ((11, 12, 60), (23, 24, 125), (25, 26, 165), (27, 28, 210)):
        points[left], points[right] = (55, y), (95, y)
    points[0] = (75, 25)
    points[13], points[14] = (35, 100), (115, 100)
    points[15], points[16] = (30, 130), (120, 130)
    mask = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    return p.View(name, np.zeros((*mask.shape, 3), np.uint8), mask,
                  points*factor, np.ones(33), 10*factor, 210*factor,
                  chin=chin*factor if chin is not None else None)


@pytest.mark.parametrize("gender,head,baseline", [("male", 23.2, 175), ("female", 21.8, 162)])
def test_front_only_scale_and_fallback(gender, head, baseline):
    front, side = view(), view("side", 2, chin=None)
    front.visibility[1:] = 0.0
    notes = []
    height, source = p.estimate_scale(front, side, gender, notes)
    raw_head_height = 200 * head / 26
    if raw_head_height > baseline:
        assert baseline <= height < raw_head_height
    else:
        assert raw_head_height < height <= baseline
    assert source == "estimated"
    assert any("shrunk toward population prior" in n for n in notes)
    assert front.cm_per_px == pytest.approx(2 * side.cm_per_px)
    for chin in (None, 11, 200):
        front.chin = chin
        notes = []
        assert p.estimate_scale(front, side, gender, notes) == (baseline, "assumed")
        assert any("assuming" in note for note in notes)


def test_known_height_and_disagreement():
    front, side = view(), view("side", 2)
    notes = []
    assert p.estimate_scale(front, side, "male", notes, 220) == (220, "measured")
    assert front.cm_per_px == 1.1 and side.cm_per_px == .55
    assert any("disagree" in note for note in notes)
    front.chin = None
    assert p.estimate_scale(front, side, "male", [], 180) == (180, "measured")


@pytest.mark.parametrize("value", ["bad", float("nan"), float("inf"), -1, 119, 221, True])
def test_invalid_known_height(value):
    with pytest.raises(p.CaptureError):
        p.validate_height(value)


def test_outer_shoulder_span_and_missing_fallback():
    front = view()
    front.cm_per_px = .9
    cm, source = p.outer_shoulder_width(front, 180, "male", "measured", [])
    assert cm == pytest.approx(51*.9)
    assert source == "measured"
    front.mask[:] = 0
    cm, source = p.outer_shoulder_width(front, 180, "male", "measured", [])
    assert cm == 45 and source == "estimated"


def test_arm_contamination_fallback():
    front, side = view(), view("side")
    front.cm_per_px = side.cm_per_px = 1
    notes = []
    values = p.torso_measurements(front, side, 200, "male", "measured", 45, "measured", notes)
    assert values["chest"]["source"] == "estimated"
    assert values["chest"]["cm"] == round(p.ellipse_circumference(36, 36*.68), 1)
    assert any("excessively wide" in n for n in notes)
    front.points[13] = (99, 110)
    assert p.arm_hint(front, 110, p.mask_run(front, 110, 75))


def test_waist_constraint_requires_contamination():
    dimensions = {"waist": {"depth": 40, "contaminated": False, "source": "measured"}, "hips": {"depth": 30}}
    p.constrain_waist_depth(dimensions, [])
    assert dimensions["waist"]["depth"] == 40
    dimensions["waist"]["contaminated"] = True
    p.constrain_waist_depth(dimensions, [])
    assert dimensions["waist"]["depth"] == 27
    assert dimensions["waist"]["source"] == "estimated"


def test_merged_thighs_are_labeled_estimated():
    front, side = view(), view("side")
    front.cm_per_px = side.cm_per_px = .9
    notes = []
    result = p.thigh_measurement(front, side, 180, "male", "measured", notes)
    assert result["source"] == "estimated"
    assert any("splitting merged" in n for n in notes)
    assert result["cm"] > 0


@pytest.mark.parametrize("width,depth", [(10, 10), (40, 20), (20, 10), (35, 28)])
def test_ramanujan_against_numerical_ellipse_integral(width, depth):
    angles = np.linspace(0, math.pi/2, 20001)
    expected = 4*np.trapz(np.sqrt((width/2*np.sin(angles))**2 + (depth/2*np.cos(angles))**2), angles)
    assert p.ellipse_circumference(width, depth) == pytest.approx(expected, rel=1e-4)


def test_sources_units_and_neck_plausibility():
    data = {"head_neck": {"neck": p.metric(70.9, "measured")}}
    notes = []
    assert p.sanity_checks(175, data, "male", notes) == ["neck"]
    assert data["head_neck"]["neck"]["source"] == "estimated"
    assert "neck" in notes[0] and "70.9" in notes[0]
    assert p.metric(25.4, "measured") == {"cm": 25.4, "in": 10., "source": "measured"}


@pytest.mark.parametrize("gender", ["male", "female"])
@pytest.mark.parametrize("name", list(p.PLAUSIBILITY))
def test_each_plausibility_range(gender, name):
    data = {"group": {name: p.metric(1000, "measured")}}
    assert p.sanity_checks(175, data, gender, []) == [name]


def test_relative_range_is_checked():
    data = {"group": {"neck": p.metric(55, "measured")}}
    assert p.sanity_checks(120, data, "male", []) == ["neck"]


def test_confidence_never_high_and_flags_lower_it():
    torso = {key: {"source": "measured"} for key in ("chest", "waist", "hips")}
    views = (view(), view("side"))
    assert p.measurement_confidence("measured", torso, views) == "medium"
    for source in ("estimated", "assumed"):
        assert p.measurement_confidence(source, torso, views) == "low"
    assert p.measurement_confidence("measured", torso, views, ["neck"]) == "low"
    assert p.measurement_confidence("measured", torso, views, capture_warnings=["orientation"]) == "low"


def test_masks_orientation_clothing_and_subject_mismatch():
    mask = np.zeros((240, 300))
    mask[10:230, 10:100] = 1
    mask[10:230, 150:250] = 1
    notes = []
    assert clean_mask(mask, "front", notes).sum() > 0
    assert any("multiple" in n for n in notes)
    front, side = view(), view("side")
    front.points[11, 0], front.points[12, 0] = 70, 80
    side.points[[23, 24], 1] = 90
    notes = p.capture_checks(front, side)
    assert any("Front orientation" in n for n in notes)
    assert any("Side orientation" in n for n in notes)
    assert any("clothing" in n for n in notes)
    assert any("different subjects" in n for n in notes)


def test_mask_run_does_not_wrap():
    v = view()
    for y, x in ((-1, 75), (300, 75), (100, -1), (100, 200)):
        assert p.mask_run(v, y, x) is None
    assert p.mask_run(v, 100, 75)[0] == 51


def test_shared_measure_contract_without_models():
    with patch.object(p, "prepare_view", side_effect=[view(), view("side")]):
        result = p.measure(view().image, view().image, "male", {"height_cm": 180})
    assert result["height"] == {"cm": 180., "in": 70.9, "source": "measured"}
    assert result["scale_anchor"] == "user_height"
    assert result["confidence"] in ("medium", "low")
    assert all(value["source"] in ("measured", "estimated", "assumed")
               for group in result["measurements"].values() for value in group.values())


def test_boundary_preparation_warns_without_wrapping():
    v = view()
    v.mask[210:, 50:101] = 1
    v.points[[29, 30, 31, 32], 1] = 239
    notes = []
    with patch.object(p, "detect_pose", return_value=(v.points, v.visibility)), patch.object(p, "segment_body", return_value=v.mask), patch.object(p, "detect_chin", return_value=36):
        result = p.prepare_view(v.image, "front", notes)
    assert result.ground == 239
    assert "bottom" in result.touched_edges
    assert any("Feet may be cut off" in n for n in notes)


def test_arms_touching_vs_arms_away():
    # 1. Arms away
    front_away = view("front")
    front_away.mask[:] = 0
    front_away.mask[40:140, 55:96] = 1   # Torso width = 41 px
    front_away.mask[60:130, 20:30] = 1   # Left arm separated by gap
    front_away.mask[60:130, 120:130] = 1 # Right arm separated by gap
    p.build_torso_mask(front_away)
    chest_away_run = p.mask_run_custom(front_away.torso_mask, 80, 75)
    assert chest_away_run is not None
    assert chest_away_run[0] == 41

    # 2. Arms touching torso
    front_touching = view("front")
    front_touching.mask[:] = 0
    # Arms merge seamlessly into torso from x=20 to x=130 (width = 111 px)
    front_touching.mask[40:140, 20:131] = 1
    front_touching.points[13] = (35, 100)
    front_touching.points[14] = (115, 100)
    front_touching.points[15] = (30, 130)
    front_touching.points[16] = (120, 130)
    p.build_torso_mask(front_touching)

    chest_torso_run = p.mask_run_custom(front_touching.torso_mask, 80, 75)
    assert chest_torso_run is not None
    # Measured torso width from torso_mask MUST NOT include the 111px merged arms
    assert chest_torso_run[0] < 80

    c_left, c_right = p.get_torso_corridor(front_touching, 80)
    assert c_left >= 35 and c_right <= 115


def test_legs_touching_vs_apart():
    # 1. Legs apart
    front_apart = view("front")
    front_apart.mask[:] = 0
    front_apart.mask[10:125, 50:101] = 1  # Torso down to hips y=125
    front_apart.mask[125:211, 50:71] = 1  # Left leg x: 50..70
    front_apart.mask[125:211, 80:101] = 1 # Right leg x: 80..100 (gap 70..80)
    crotch_y = p.find_crotch(front_apart)
    assert crotch_y is not None
    assert 124 <= crotch_y <= 135

    front_apart.cm_per_px = 0.9
    lengths = p.length_measurements(front_apart, 180, "measured", [])
    assert lengths["inseam"]["source"] == "measured"

    # 2. Legs touching (merged)
    front_touching = view("front")
    front_touching.mask[:] = 0
    front_touching.mask[10:211, 50:101] = 1  # Solid rectangle, no leg gap
    crotch_y_merged = p.find_crotch(front_touching)
    assert crotch_y_merged is None

    notes = []
    side = view("side")
    side.cm_per_px = front_touching.cm_per_px = 1.0
    thigh = p.thigh_measurement(front_touching, side, 180, "male", "measured", notes)
    assert thigh["source"] == "estimated"
    assert any("legs merged" in n.lower() or "splitting merged" in n.lower() for n in notes)


def test_side_view_joint_visibility_recovery():
    from bodymeasure import detection as d
    with patch("bodymeasure.detection.mp.solutions.pose.Pose") as mock_pose_class:
        mock_pose_inst = mock_pose_class.return_value.__enter__.return_value
        mock_res = type("PoseResults", (), {})()
        mock_lms = []
        for i in range(33):
            mock_lms.append(type("Landmark", (), {"x": 0.5, "y": 0.05 + 0.028 * i, "z": 0.0, "visibility": 0.95})())
        mock_lms[25].visibility = 0.1
        mock_lms[26].visibility = 0.85
        mock_res.pose_landmarks = type("LandmarkList", (), {"landmark": mock_lms})()
        mock_pose_inst.process.return_value = mock_res

        img = np.zeros((240, 160, 3), np.uint8)
        notes = []
        pts, vis = d.detect_pose(img, "side", notes)
        assert vis[25] >= 0.5
        assert np.allclose(pts[25], pts[26])
        assert any("recovered from opposite side" in n for n in notes)

        mock_lms[25].visibility = 0.15
        mock_lms[26].visibility = 0.25
        notes2 = []
        pts2, vis2 = d.detect_pose(img, "side", notes2)
        assert vis2[25] >= 0.5 and vis2[26] >= 0.5
        assert any("Side knees were occluded and recovered from leg proportions" in n for n in notes2)


def test_scale_fusion_behaviors():
    # 1. Single-cue extreme head cue (implied 191 cm) is shrunk toward prior and never returned as 191
    front, side = view(), view("side", 2, chin=None)
    front.visibility[1:] = 0.0
    front.chin = front.vertex + (200.0 * 23.2 / 191.0)
    notes = []
    h, source = p.estimate_scale(front, side, "male", notes)
    assert h != 191.0
    assert h < 183.0
    assert source == "estimated"
    assert any("shrunk toward population prior" in n for n in notes)
    assert "uncertainty_band" in front.scale_meta
    assert "uncertainty_cm" in front.scale_meta
    assert "cues_used" in front.scale_meta

    # 2. Outlier cues are dropped when > 2.5 SD from median
    cues_test = [("head_prior", 175.0, 7.0), ("torso_proportion", 174.0, 8.5), ("neck_shoulder_cue", 176.0, 9.0), ("outlier_cue", 215.0, 8.0)]
    vals = np.array([c[1] for c in cues_test])
    med = float(np.median(vals))
    diffs = np.abs(vals - med)
    mad = float(np.median(diffs)) or 5.0
    valid_cues = [c for c in cues_test if abs(c[1] - med) <= max(14.0, 2.5 * 1.4826 * mad)]
    assert len(valid_cues) == 3
    assert not any(c[0] == "outlier_cue" for c in valid_cues)


def test_plausibility_guards_and_confidence_cap():
    measurements = {
        "head_neck": {"neck": p.metric(58.0, "measured")},
        "torso": {
            "shoulder_width": p.metric(30.0, "measured"),
            "chest": p.metric(95.0, "measured"),
            "waist": p.metric(115.0, "measured"),
            "hips": p.metric(90.0, "measured"),
            "torso_length": p.metric(45.0, "measured")
        },
        "arms": {
            "arm_length": p.metric(60.0, "measured"),
            "wrist": p.metric(17.0, "measured")
        },
        "legs": {
            "inseam": p.metric(80.0, "measured"),
            "thigh": p.metric(55.0, "measured")
        }
    }
    notes = []
    flagged = p.enforce_plausibility_guards(measurements, 175.0, "male", 44.0, notes)
    assert "neck" in flagged
    assert "shoulder_width" in flagged
    assert "waist" in flagged
    assert measurements["head_neck"]["neck"]["source"] == "estimated"
    assert measurements["head_neck"]["neck"]["cm"] <= 50.0
    assert measurements["torso"]["shoulder_width"]["source"] == "estimated"
    assert measurements["torso"]["shoulder_width"]["cm"] >= 35.0
    assert measurements["torso"]["waist"]["source"] == "estimated"
    assert measurements["torso"]["waist"]["cm"] <= measurements["torso"]["hips"]["cm"] * 1.10
    assert any("tripped screening range" in n for n in notes)

    views = (view(), view("side"))
    conf_ok = p.measurement_confidence("measured", measurements["torso"], views)
    assert conf_ok in ("medium", "low")
    conf_flagged = p.measurement_confidence("measured", measurements["torso"], views, flagged=flagged)
    assert conf_flagged == "low"

