"""Small, independently checkable fixtures for the MATLAB-source audit port."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_visdrone_det_strict import (
    _eval_res,
    _filter_rows,
    _voc_ap,
    comp_oas,
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


def _literal_integral_filter(gt, det, height, width):
    """Independent 1-based prefix-sum oracle for dropObjectsInIgr.m.

    This is another Python translation, NOT an executed MATLAB reference.
    Deliberately do not use the production slice-sum or round helper.
    """
    ignored = gt[gt[:, 5] == 0, :4]
    valid = gt[gt[:, 5] != 0].copy()
    valid[:, 4] = 1 - valid[:, 4]
    if not len(ignored):
        return valid, det
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    for row in ignored:
        x, y, w, h = np.maximum(1, row).astype(int)
        for iy in range(y, min(height, y + h) + 1):
            for ix in range(x, min(width, x + w) + 1):
                integral[iy, ix] = 1
    for iy in range(1, height + 1):
        for ix in range(1, width + 1):
            integral[iy, ix] += (
                integral[iy - 1, ix] + integral[iy, ix - 1] - integral[iy - 1, ix - 1]
            )

    def select(rows):
        indices = []
        for index, row in enumerate(rows):
            rounded = np.sign(row[:4]) * np.floor(np.abs(row[:4]) + 0.5)
            x, y, w, h = np.maximum(1, rounded).astype(int)
            x, y = min(width, x), min(height, y)
            right, bottom = min(width, x + w), min(height, y + h)
            covered = (
                integral[y, x] + integral[bottom, right]
                - integral[y, right] - integral[bottom, x]
            )
            if covered / (w * h) < 0.5:
                indices.append(index)
        return rows[indices]

    return select(valid), select(det)


@pytest.mark.parametrize("seed", range(12))
def test_slice_filter_matches_literal_matlab_integral_query(seed):
    rng = np.random.default_rng(seed)
    # Include negative/zero positions, borders, oversized boxes and half ties.
    gt = rng.integers(-3, 32, size=(32, 8))
    gt[:, 4] = rng.integers(0, 2, size=32)
    gt[:, 5] = rng.integers(0, 12, size=32)
    gt[0, 5] = 0
    dt = rng.integers(-3, 32, size=(80, 8)).astype(float)
    dt[:, :4] += 0.5
    expected = _literal_integral_filter(gt, dt, 17, 23)
    actual = _filter_rows(gt, dt, 17, 23)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_exact_half_ignore_coverage_is_removed():
    gt = np.array([[1, 1, 1, 1, 0, 0, 0, 0]])
    # Query counts one pixel; denominators are 2 and 3 respectively.
    dt = np.array([[1, 1, 2, 1, .9, 1, -1, -1], [1, 1, 3, 1, .8, 1, -1, -1]])
    _, kept = _filter_rows(gt, dt, 10, 10)
    np.testing.assert_array_equal(kept, dt[1:])


def test_clipped_query_retains_unclipped_area_denominator():
    gt = np.array([[1, 1, 9, 9, 0, 0, 0, 0]])
    dt = np.array([[8, 8, 10, 10, .9, 1, -1, -1]])
    assert len(_filter_rows(gt, dt, 10, 10)[1]) == 1  # 4/100, not 4/4.


def test_overlap_continuous_geometry_and_ignore_denominator():
    dt = np.array([[1, 1, 2, 2], [10, 0, 2, 2]], dtype=float)
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    overlap = comp_oas(dt, gt, np.array([False, True]))
    np.testing.assert_allclose(overlap, [[.04, 1], [0, 0]], rtol=0, atol=0)


def test_zero_extent_detections_have_zero_overlap_without_division():
    dt = np.array([[1, 1, 0, 2], [1, 1, 2, 0]], dtype=float)
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(comp_oas(dt, gt, np.array([False, True])), np.zeros((2, 2)))


def test_normal_gt_priority_and_single_use_with_reusable_ignore():
    gt = np.array([[0, 0, 10, 10, 1], [0, 0, 20, 10, 0]])
    dt = np.array([[0, 0, 10, 10, .9], [0, 0, 10, 10, .8], [0, 0, 10, 10, .7]])
    matched_gt, matched_dt = _eval_res(gt, dt, .5)
    assert matched_gt[:, 4].tolist() == [1, -1]
    assert matched_dt[:, 5].tolist() == [1, -1, -1]


def test_equal_gt_overlap_selects_last_gt_and_includes_exact_threshold():
    gt = np.array([[0, 0, 20, 10, 0], [0, 0, 20, 10, 0]])
    matched_gt, matched_dt = _eval_res(gt, [[0, 0, 10, 10, .9]], .5)
    assert matched_gt[:, 4].tolist() == [0, 1]
    assert matched_dt[0, 5] == 1
    assert _eval_res(gt, [[0, 0, 10, 10, .9]], np.nextafter(.5, 1))[1][0, 5] == 0


def test_empty_arrays_and_duplicate_detection():
    assert _eval_res([], [], .5)[1].shape == (0, 6)
    assert _eval_res([], [[0, 0, 10, 10, .9]], .5)[1][0, 5] == 0
    gt = [[0, 0, 10, 10, 0]]
    assert _eval_res(gt, [], .5)[0][0, 4] == 0
    dt = [[0, 0, 10, 10, .9], [0, 0, 10, 10, .8]]
    assert _eval_res(gt, dt, .5)[1][:, 5].tolist() == [1, 0]


def test_voc_envelope_not_coco_101_point_sampling():
    # Envelope is [.8,.8,.4] over recall increments [.25,.25,.5].
    assert _voc_ap(np.array([.25, .5, 1]), np.array([.5, .8, .4])) == pytest.approx(.6)
    assert _voc_ap(np.array([]), np.array([])) == 0


def test_maxdet_truncates_input_before_category_and_sorting():
    gt = np.array([[0, 0, 10, 10, 1, 1, 0, 0]])
    dt = np.array([[50, 50, 10, 10, .1, 2, -1, -1], [0, 0, 10, 10, .9, 1, -1, -1]])
    result = evaluate([gt], [dt], [100], [100])
    assert result["AR1"] == 0
    assert result["AR10"] == 100
    assert result["AP"] == 100


def test_cross_image_equal_scores_keep_image_then_row_order():
    gt = [np.array([[0, 0, 10, 10, 1, 1, 0, 0]])] * 2
    fp = np.array([[50, 50, 10, 10, .5, 1, -1, -1]])
    tp = np.array([[0, 0, 10, 10, .5, 1, -1, -1]])
    assert evaluate(gt, [fp, tp], [100] * 2, [100] * 2)["AP"] == 25
    assert evaluate(gt, [tp, fp], [100] * 2, [100] * 2)["AP"] == 50


def test_recall_denominator_keeps_ignored_gt_like_source():
    gt = np.array([[0, 0, 10, 10, 1, 1, 0, 0], [50, 50, 10, 10, 0, 1, 0, 0]])
    dt = np.array([[50, 50, 10, 10, .9, 1, -1, -1], [0, 0, 10, 10, .8, 1, -1, -1]])
    result = evaluate([gt], [dt], [100], [100])
    assert result["AP"] == 50
    assert result["AR500"] == 50
