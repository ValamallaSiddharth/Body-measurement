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
    height, source = p.estimate_scale(front, side, gender, [])
    assert height == pytest.approx(200*head/26)
    assert source == "estimated"
    assert front.cm_per_px == pytest.approx(2*side.cm_per_px)
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
