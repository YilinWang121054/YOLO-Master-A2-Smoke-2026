"""Small, independently checkable fixtures for the MATLAB-source audit port."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_visdrone_det_strict import (
    _eval_res,
    _filter_rows,
    evaluate,
    matlab_round_positive,
)


def test_visdrone_score_becomes_valid_ignore_flag():
    gt = np.array([[0, 0, 10, 10, 1, 1, 0, 0]])
    dt = np.array([[0, 0, 10, 10, 0.9, 1, -1, -1]])
    result = evaluate([gt], [dt], [100], [100])
    assert result["AP"] == pytest.approx(100)


def test_equal_score_matching_retains_input_order():
    gt = np.array([[0, 0, 10, 10, 0]])
    dt = np.array([[50, 50, 10, 10, 0.5], [0, 0, 10, 10, 0.5]])
    _, matched = _eval_res(gt, dt, 0.5)
    assert matched[:, 5].tolist() == [0, 1]


def test_ignore_overlap_uses_ignored_gt_geometry():
    gt = np.array([[50, 50, 2, 2, 0], [0, 0, 10, 10, 1]])
    dt = np.array([[1, 1, 2, 2, 0.5]])
    _, matched = _eval_res(gt, dt, 0.5)
    assert matched[0, 5] == -1


def test_matlab_half_value_rounding():
    assert matlab_round_positive(np.array([660.5, 12.5, 12.49])).tolist() == [
        661,
        13,
        12,
    ]


def test_ignore_integral_query_excludes_top_and_left_pixels():
    # Region pixels are 1..2, but the query for [1,1,2,2] only counts (2,2).
    gt = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])
    dt = np.array([[1.0, 1.0, 2.0, 2.0, 0.9, 1.0, -1.0, -1.0]])
    _, kept = _filter_rows(gt, dt, 10, 10)
    assert len(kept) == 1


def test_original_repeated_category_indexing():
    gt = [
        np.array([[0, 0, 10, 10, 1, 1, 0, 0], [20, 20, 10, 10, 1, 2, 0, 0]]),
        np.array([[0, 0, 10, 10, 1, 1, 0, 0]]),
    ]
    dt = [np.array([[0, 0, 10, 10, 0.9, 1, -1, -1]])] * 2
    result = evaluate(gt, dt, [100, 100], [100, 100])
    assert result["AP"] == pytest.approx(200 / 3)
    assert result["eval_classes"] == [0, 0, 1]
