import math
import pytest
from scripts.evaluate import numeric_truth, error_summary


def test_ground_truth_is_not_invented():
    for value in ("", "86.0_tbr", "NaN", "inf", "-1", None):
        assert numeric_truth(value) is None
    assert numeric_truth("86.0") == 86


def test_summary_uses_signed_bias_and_correct_denominators():
    stats = error_summary([-6, -2, 0, 2, 5])
    assert stats["n"] == 5
    assert stats["mae_cm"] == 3
    assert stats["rmse_cm"] == pytest.approx(math.sqrt(69/5))
    assert stats["bias_cm"] == -.2
    assert stats["within_2cm_pct"] == 60
    assert stats["within_5cm_pct"] == 80
    assert error_summary([])["mae_cm"] == ""
