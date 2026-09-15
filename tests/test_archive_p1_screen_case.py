"""Byte preservation and fail-closed guards for screening publication."""

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_p1_screen_case as archive
from archive_p1_screen_case import (
    METRICS,
    archive_file,
    contained,
    validate_evaluations,
)


@pytest.mark.parametrize("compress", [False, True])
def test_raw_bytes_round_trip(tmp_path, compress):
    source, target = tmp_path / "input", tmp_path / "out"
    original = b"\x1b[Kepoch 1\r\nnext\r" + bytes(range(256)) * 30
    source.write_bytes(original)
    result = archive_file(source, target, compress=compress)
    restored = gzip.decompress(target.read_bytes()) if compress else target.read_bytes()
    assert restored == original
    assert result["source_bytes"] == len(original)
    assert source.read_bytes() == original


def test_does_not_overwrite(tmp_path):
    source, target = tmp_path / "input", tmp_path / "out"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        archive_file(source, target)
    assert target.read_bytes() == b"old"


@pytest.mark.parametrize("relative", ["../outside", "."])
def test_path_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="escapes"):
        contained(tmp_path, relative)


def records():
    return [
        {
            "epoch": e,
            "images": 548,
            "device": "cpu",
            "metrics": dict.fromkeys(METRICS, 6.0),
        }
        for e in range(16, 21)
    ]


def test_all_five_required():
    validate_evaluations(records())
    with pytest.raises(ValueError, match="Exactly"):
        validate_evaluations(records()[:-1])


def test_missing_metric_rejected():
    rows = records()
    del rows[0]["metrics"]["APs"]
    with pytest.raises(ValueError, match="Missing required"):
        validate_evaluations(rows)


def test_incomplete_case_produces_no_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    monkeypatch.setattr(
        archive, "configuration", lambda: {"cases": [{"id": "s0-fixed"}]}
    )
    monkeypatch.setattr(archive, "verify_frozen", dict)

    def missing_result(_case):
        raise ValueError("Missing one of five evaluations")

    monkeypatch.setattr(archive, "summarize_case", missing_result)
    with pytest.raises(ValueError, match="Missing one"):
        archive.main(["--case", "s0-fixed"])
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("field,value", [("images", 547), ("device", "cuda")])
def test_incomplete_or_wrong_device(field, value):
    rows = records()
    rows[0][field] = value
    with pytest.raises(ValueError, match="unexpected"):
        validate_evaluations(rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 101])
def test_nonfinite_or_out_of_range(value):
    rows = records()
    rows[0]["metrics"]["APs"] = value
    with pytest.raises(ValueError, match="Invalid"):
        validate_evaluations(rows)
