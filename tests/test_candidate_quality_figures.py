"""Protect pairing and zero-preserving ECDF construction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_candidate_quality import ecdf, paired_rows


def test_ecdf_keeps_zeros_and_duplicate_mass():
    x, y = ecdf([0, 0, 0.5, 1])
    np.testing.assert_array_equal(x, [0, 0, 0.5, 1, 1])
    np.testing.assert_array_equal(y, [0, 0.5, 0.75, 1, 1])


@pytest.mark.parametrize("values", [[], [float("nan")], [-0.01], [1.1]])
def test_invalid_quality_values_are_not_silently_filtered(values):
    with pytest.raises(ValueError):
        ecdf(values)


def test_pairing_by_identity_and_missing_gt_rejection():
    fixed = {
        "image": "x",
        "gt": 1,
        "size": "small",
        "mode": "fixed",
        "input_area": 5,
        "counts": {"after_conflict": 2, "weight_sum": 0.8},
    }
    adaptive = {
        **fixed,
        "mode": "adaptive",
        "counts": {"after_conflict": 5, "weight_sum": 1.2},
    }
    rows = paired_rows([adaptive, fixed])
    assert rows[0]["delta_count"] == 3
    assert rows[0]["delta_weight"] == pytest.approx(0.4)
    with pytest.raises(AssertionError):
        paired_rows([fixed])
