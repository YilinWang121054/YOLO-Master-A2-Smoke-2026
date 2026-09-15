"""Guard descriptive statistics and immutable evidence copies."""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_completed_v2 as verify


def test_histogram_nearest_rank_includes_zero():
    result = verify.histogram_summary(Counter({0: 2, 3: 6, 10: 2}))
    assert result == {"n": 10, "mean": 3.8, "zero_fraction": 0.2, "p50": 3, "p90": 10}


def test_histogram_single_observation():
    assert verify.histogram_summary(Counter({0: 1})) == {
        "n": 1,
        "mean": 0.0,
        "zero_fraction": 1.0,
        "p50": 0,
        "p90": 0,
    }


def test_copy_exact_preserves_raw_bytes_and_rejects_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    source, target = tmp_path / "source.log", tmp_path / "out/log.txt"
    payload = b"\x1b[Kepoch\r\nresume\r"
    source.write_bytes(payload)
    record = verify.copy_exact(source, target)
    assert target.read_bytes() == payload
    assert record["sha256"] == verify.sha(source)
    assert verify.copy_exact(source, target) == record
    source.write_bytes(b"different")
    with pytest.raises(AssertionError, match="Refuse to replace"):
        verify.copy_exact(source, target)
    assert target.read_bytes() == payload


def test_save_new_refuses_existing_record(tmp_path):
    target = tmp_path / "record.json"
    verify.save_new(target, {"status": "historical"})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        verify.save_new(target, {"status": "rewritten"})
    assert target.read_bytes() == before
