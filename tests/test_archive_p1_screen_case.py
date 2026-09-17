"""Byte preservation and fail-closed guards for screening publication."""

import gzip
import json
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


def interrupted_logs(tmp_path):
    launch = tmp_path / "cpu-attempt.launch.json"
    event = {"worker": "cpu", "case": "s4", "pid": 123,
             "started_at": "2026-09-15T19:28:55+00:00"}
    launch.write_text(json.dumps(event), encoding="utf-8")
    out, err = tmp_path / "cpu-attempt.stdout.log", tmp_path / "cpu-attempt.stderr.log"
    out.write_bytes(b"incomplete\r\n\x1b[K")
    err.write_bytes(b"")
    receipt = {
        "schema_version": 1, "status": "interrupted_after_shutdown", **event,
        "exit_code": None, "child_present_at_observation": False,
        "observed_at": "2026-09-17T06:50:00+00:00",
        "shutdown_event": {"id": 1074, "provider": "User32", "record_id": 1,
                           "time": "2026-09-16T03:29:24+08:00"},
        "boot_event": {"id": 6005, "provider": "EventLog", "record_id": 2,
                       "time": "2026-09-16T03:31:20+08:00"},
        "original_files_sha256": {p.name: archive.sha(p) for p in (launch, out, err)},
    }
    sidecar = tmp_path / "cpu-attempt.interrupted.json"
    sidecar.write_text(json.dumps(receipt), encoding="utf-8")
    return sidecar, receipt


def test_interruption_preserves_originals_and_does_not_invent_exit(tmp_path):
    sidecar, _ = interrupted_logs(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert archive.validate_child_logs(tmp_path) == [sidecar.name]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert not list(tmp_path.glob("*.exit.json"))


@pytest.mark.parametrize("key,value", [
    ("exit_code", 0), ("pid", 999), ("status", "complete"),
    ("child_present_at_observation", True), ("original_files_sha256", {}),
    ("observed_at", "2026-09-15T19:29:00+00:00"),
    ("observed_at", "2026-09-17T06:50:00"), ("boot_event", {}),
])
def test_invalid_interruption_rejected(tmp_path, key, value):
    sidecar, receipt = interrupted_logs(tmp_path)
    receipt[key] = value
    sidecar.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        archive.validate_child_logs(tmp_path)


def test_unaccounted_missing_exit_still_rejected(tmp_path):
    sidecar, _ = interrupted_logs(tmp_path)
    sidecar.unlink()
    with pytest.raises(ValueError, match="no recorded exit"):
        archive.validate_child_logs(tmp_path)


def test_both_exit_and_interruption_rejected(tmp_path):
    interrupted_logs(tmp_path)
    (tmp_path / "cpu-attempt.exit.json").write_text('{"exit_code": 0}')
    with pytest.raises(ValueError, match="Ambiguous"):
        archive.validate_child_logs(tmp_path)


def test_normal_exit_still_supported(tmp_path):
    sidecar, _ = interrupted_logs(tmp_path)
    sidecar.unlink()
    (tmp_path / "cpu-attempt.exit.json").write_text('{"exit_code": 0}')
    assert archive.validate_child_logs(tmp_path) == []


@pytest.mark.parametrize("provider", ["Microsoft-Windows-Kernel-Power", "User32", None])
def test_unexpected_reboot_requires_correct_os_event(tmp_path, provider):
    sidecar, receipt = interrupted_logs(tmp_path)
    receipt["status"] = "interrupted_after_unexpected_reboot"
    receipt["unexpected_restart_event"] = {"id": 41, "provider": provider,
        "record_id": 10, "time": "2026-09-16T03:31:15+08:00"}
    del receipt["shutdown_event"]
    sidecar.write_text(json.dumps(receipt), encoding="utf-8")
    if provider == "Microsoft-Windows-Kernel-Power":
        assert archive.validate_child_logs(tmp_path) == [sidecar.name]
        assert not list(tmp_path.glob("*.exit.json"))
    else:
        with pytest.raises(ValueError, match="event evidence"):
            archive.validate_child_logs(tmp_path)
