"""Observer correctness gates for the independent fixed-checkpoint probe."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "YOLO-Master"))
from run_candidate_quality_probe import choose_images, describe, trace_assignment
from ultralytics.utils.tal import TaskAlignedAssigner


@pytest.mark.parametrize("mode", ["tal", "fixed", "adaptive"])
@pytest.mark.parametrize("empty", [False, True])
def test_exact_observer_and_conflicts(mode, empty):
    y, x = torch.meshgrid(
        torch.arange(8) * 8 + 4, torch.arange(8) * 8 + 4, indexing="ij"
    )
    points = torch.stack([x, y], -1).reshape(-1, 2).float()
    boxes = torch.tensor(
        [[[7.0, 7.0, 9.0, 9.0], [7.0, 7.0, 9.0, 9.0], [0.0, 0.0, 32.0, 32.0]]]
    )
    if empty:
        boxes = boxes[:, :0]
    inputs = (
        torch.linspace(0.1, 0.9, 64).reshape(1, 64, 1),
        torch.cat([points - 6, points + 6], -1).unsqueeze(0),
        points,
        torch.zeros(1, boxes.shape[1], 1),
        boxes,
        torch.ones(1, boxes.shape[1], 1, dtype=torch.bool),
    )
    trace, result = trace_assignment(
        TaskAlignedAssigner,
        {
            "topk": 10,
            "topk2": 10,
            "num_classes": 1,
            "alpha": 0.5,
            "beta": 6,
            "stal_mode": mode,
        },
        inputs,
    )
    assert trace.after.sum().item() == result[3].sum().item()
    assert not (trace.after & ~trace.candidates).any()


def test_selection_is_order_independent_one_frame_per_prefix():
    paths = [
        Path(f"{sequence:07d}_{frame:05d}_d_1.jpg")
        for sequence in range(20)
        for frame in range(3)
    ]
    selected = choose_images(paths, 16, 20260913)
    assert selected == choose_images(list(reversed(paths)), 16, 20260913)
    assert len({p.stem.split("_")[0] for p in selected}) == 16


def test_descriptive_empty_and_quantiles():
    assert describe([])["n"] == 0
    assert describe([0, 0, 3, 5]) == {
        "n": 4,
        "mean": 2.0,
        "p50": 0.0,
        "p90": 5.0,
        "zero_fraction": 0.5,
    }
    with pytest.raises(ValueError):
        describe([float("nan")])
