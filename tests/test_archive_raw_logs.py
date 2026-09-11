"""Archive uninterrupted and resumed runs without inventing recovery evidence."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import archive_p0_raw_logs as archive


@pytest.fixture
def run_files(tmp_path, monkeypatch):
    root, project = tmp_path / "evidence", tmp_path / "runs"
    name = "p1-tal-s20260826"
    run = project / name
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "epoch\n" + "\n".join(str(epoch) for epoch in range(1, 121)) + "\n",
        encoding="utf-8",
    )
    initial = project / "recovery-logs/p1-seed3-chain" / name
    initial.mkdir(parents=True)
    (initial / "initial.stdout.log").write_bytes(b"\x1b[Kepoch 1\r\nepoch 120\r")
    (initial / "initial.stderr.log").write_bytes(b"")
    (initial / "last-launch.json").write_text('{"action":"fresh"}\n')
    (root / "results/closure-evaluation" / name).mkdir(parents=True)
    monkeypatch.setattr(archive, "ROOT", root)
    monkeypatch.setattr(archive, "PROJECT", project)
    return root, project, name, initial


def test_uninterrupted_run_preserves_bytes_and_records_absent_recovery(run_files):
    root, project, name, initial = run_files
    archive.main(["--run", name])
    manifest = json.loads((root / "results/closure-evaluation" / name / "training-log-manifest.json").read_text())
    assert len(manifest["records"]) == 3
    assert manifest["missing_online_assignment_epochs"] == list(range(1, 121))
    assert manifest["source_directories"][1]["status"] == "directory_absent"
    assert not (project / "recovery-logs" / name).exists()
    assert not (root / "logs/closure-evaluation" / name / "training-original/recovery").exists()
    for record in manifest["records"]:
        assert (root / record["archived"]).read_bytes() == Path(record["source"]).read_bytes()
        assert archive.sha256(root / record["archived"]) == record["sha256"]


def test_resumed_run_keeps_failure_logs(run_files):
    root, project, name, initial = run_files
    recovery = project / "recovery-logs" / name
    recovery.mkdir(parents=True)
    (recovery / "attempt.stderr.log").write_bytes(b"original failure\r\n")
    archive.main(["--run", name])
    copied = root / "logs/closure-evaluation" / name / "training-original/recovery/attempt.stderr.log"
    assert copied.read_bytes() == b"original failure\r\n"


@pytest.mark.parametrize("json_only", [False, True])
def test_existing_recovery_directory_without_logs_is_not_silently_skipped(run_files, json_only):
    root, project, name, initial = run_files
    recovery = project / "recovery-logs" / name
    recovery.mkdir(parents=True)
    if json_only:
        (recovery / "state.json").write_text("{}")
    with pytest.raises(ValueError, match="No original logs"):
        archive.main(["--run", name])
    assert not (root / "logs/closure-evaluation" / name).exists()


def test_missing_initial_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Required original log directory missing"):
        archive.original_sources(tmp_path, "p1-tal-s20260826")


def test_p0_still_requires_its_known_recovery_evidence(tmp_path):
    initial = tmp_path / "recovery-logs/p1-seed3-chain" / archive.NAME
    initial.mkdir(parents=True)
    (initial / "initial.stdout.log").write_bytes(b"original P0 log")
    with pytest.raises(ValueError, match="Required original log directory missing"):
        archive.original_sources(tmp_path, archive.NAME)


def test_incomplete_training_is_rejected(run_files):
    root, project, name, initial = run_files
    (project / name / "results.csv").write_text("epoch\n1\n")
    with pytest.raises(ValueError, match="120 completed"):
        archive.main(["--run", name])
    assert not (root / "logs/closure-evaluation" / name).exists()


def test_different_existing_archive_is_not_overwritten(run_files):
    root, project, name, initial = run_files
    copied = root / "logs/closure-evaluation" / name / "training-original/initial/initial.stdout.log"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(b"existing evidence must survive")
    with pytest.raises(ValueError, match="Refuse to replace"):
        archive.main(["--run", name])
    assert copied.read_bytes() == b"existing evidence must survive"
