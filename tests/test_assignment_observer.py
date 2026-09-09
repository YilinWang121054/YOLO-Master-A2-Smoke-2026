"""Verify the observer preserves assignment and reports training coverage honestly."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "YOLO-Master-baseline"))
from assignment_observer import AssignmentObserver, area_bin, describe
from ultralytics.utils.tal import TaskAlignedAssigner


def test_area_boundaries_and_quantiles():
    assert [area_bin(x) for x in (1023, 1024, 9215, 9216)] == [
        "small",
        "medium",
        "medium",
        "large",
    ]
    summary = describe({0: 2, 3: 1, 5: 1})
    assert summary["mean"] == 2
    assert summary["zero_fraction"] == 0.5
    assert summary["p50"] == 0 and summary["p90"] == 5


def test_observer_preserves_outputs_rng_and_excludes_validation(tmp_path):
    axis = torch.arange(4).float() * 8 + 4
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([xx, yy], -1).reshape(-1, 2)
    boxes = torch.tensor([[[7.0, 7.0, 9.0, 9.0], [4.0, 4.0, 20.0, 20.0]]])
    args = (
        torch.linspace(0.2, 0.9, 16).reshape(1, 16, 1),
        torch.cat([points - 6, points + 6], -1).unsqueeze(0),
        points,
        torch.zeros(1, 2, 1),
        boxes,
        torch.ones(1, 2, 1, dtype=torch.bool),
    )
    assigner = TaskAlignedAssigner(topk=3, num_classes=1)
    expected = assigner(*args)
    before_rng = torch.get_rng_state().clone()
    trainer = SimpleNamespace(epoch=0, save_dir=tmp_path, train_loader=[1])
    observer = AssignmentObserver(TaskAlignedAssigner).install()
    try:
        observer.start_epoch(trainer)
        actual = assigner(*args)
        observer.batch_end(trainer)
        observer.end_training(trainer)
        assert (
            tmp_path / "assignment/epoch-001.json"
        ).exists()  # before validation/checkpoint
        assigner(*args)  # Validation call must not contribute observations.
        observer.save_epoch(trainer)
    finally:
        observer.close()
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)
    payload = json.loads((tmp_path / "assignment/epoch-001.json").read_text())
    assert payload["batches"] == 1
    assert payload["calls"] == {"topk3_topk23": 1}
    final = payload["branches"]["topk3_topk23"]["small"]["after_conflict"]
    assert final["n"] == 2
    assert sum(int(k) * v for k, v in final["histogram"].items()) == int(
        actual[3].sum()
    )


def test_empty_gt_is_counted_as_call_without_fake_gt():
    observer = AssignmentObserver(TaskAlignedAssigner).install()
    try:
        observer.enabled = True
        TaskAlignedAssigner(topk=3, num_classes=1)(
            torch.ones(1, 4, 1),
            torch.zeros(1, 4, 4),
            torch.zeros(4, 2),
            torch.zeros(1, 0, 1),
            torch.zeros(1, 0, 4),
            torch.zeros(1, 0, 1),
        )
        assert sum(observer.calls.values()) == 1 and not observer.histograms
    finally:
        observer.close()


def test_reject_silent_batch_change():
    observer = AssignmentObserver(TaskAlignedAssigner, expected_batch=4)
    with pytest.raises(RuntimeError, match="batch changed"):
        observer.start_epoch(SimpleNamespace(batch_size=2))
