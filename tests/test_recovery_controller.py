"""Regression tests for resume handoff and safe serial scheduling (no GPU)."""

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_successful_resume_emits_status_without_crashing(tmp_path, monkeypatch):
    module = load("resume_p1_training")
    executable = tmp_path / "yolo.exe"
    executable.touch()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "workspace": str(tmp_path),
                "run_dir": str(tmp_path),
                "recovery_log_dir": str(tmp_path / "logs"),
                "expected_epochs": 120,
                "experiment_id": "test-run",
                "expected_git_commit": "frozen",
                "yolo_executable": str(executable),
            }
        )
    )
    checkpoint = {"path": "last.pt", "completed_epochs": 19}
    monkeypatch.setattr(sys, "argv", ["resume", "--config", str(config)])
    monkeypatch.setattr(module, "read_results", lambda _: ([], [{}] * 19))
    monkeypatch.setattr(module, "select_checkpoint", lambda *_: (checkpoint, []))
    monkeypatch.setattr(module, "running_processes", lambda *_: [])
    monkeypatch.setattr(module, "current_commit", lambda *_: "frozen")
    monkeypatch.setattr(module.os, "chdir", lambda _: None)
    launches = []
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **kw: launches.append(a) or SimpleNamespace(pid=123),
    )
    assert module.main() == 0
    assert len(launches) == 1
    assert (
            json.loads((tmp_path / "logs/latest-check.json").read_text(encoding="utf-8"))["status"]
        == "resumed"
    )


def test_existing_incomplete_directory_is_never_fresh_started(tmp_path, monkeypatch):
    module = load("chain_p1_seed3")
    run = {
        "mode": "fixed",
        "token": "test",
        "config": tmp_path / "config",
        "run_dir": tmp_path / "run",
    }
    run["run_dir"].mkdir()
    monkeypatch.setattr(module, "CHAIN_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(module, "RUNS", [run])
    monkeypatch.setattr(module, "process_matches", lambda *_: [])
    monkeypatch.setattr(
        module,
        "launch_fresh",
        lambda *_: (_ for _ in ()).throw(AssertionError("duplicate fresh run")),
    )
    monkeypatch.setattr(module, "launch_resume", lambda *_: {"returncode": 2})
    assert module.run_once() == 2
    assert (
        json.loads((tmp_path / "logs/chain-state.json").read_text())["phase"]
        == "blocked"
    )


def test_p0_duplicate_guard_ignores_diagnostic_shell(monkeypatch):
    import psutil

    module = load("resume_p1_training")
    processes = [
        SimpleNamespace(
            info={
                "pid": 11,
                "name": "pwsh.exe",
                "cmdline": ["pwsh.exe", "Get-Content run_p0_baseline.py"],
            }
        ),
        SimpleNamespace(
            info={
                "pid": 22,
                "name": "python.exe",
                "cmdline": ["python.exe", "run_p0_baseline.py"],
            }
        ),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda *_: iter(processes))
    assert module.running_processes("p0-locked-s20260824-stats120", Path("p0-run")) == [
        22
    ]


def test_active_training_keeps_next_mode_queued(tmp_path, monkeypatch):
    module = load("chain_p1_seed3")
    run = {"token": "adaptive", "run_dir": tmp_path / "run"}
    monkeypatch.setattr(module, "CHAIN_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(module, "RUNS", [run, {"token": "tal"}])
    monkeypatch.setattr(module, "process_matches", lambda *_: [123])
    assert module.run_once() == 0
    assert (
        json.loads((tmp_path / "logs/chain-state.json").read_text())["run"]
        == "adaptive"
    )


def test_corrupt_last_falls_back_to_healthy(tmp_path, monkeypatch):
    module = load("resume_p1_training")
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("last.pt", "last_healthy.pt"):
        (weights / name).touch()

    def inspect(path, config):
        if path.name == "last.pt":
            raise pickle.UnpicklingError("interrupted write")
        return {"path": str(path), "completed_epochs": 5}

    monkeypatch.setattr(module, "inspect_checkpoint", inspect)
    checkpoint, errors = module.select_checkpoint(tmp_path, {})
    assert Path(checkpoint["path"]).name == "last_healthy.pt"
    assert errors[0]["error"].startswith("UnpicklingError")


def test_p0_complete_requires_all_online_epochs(tmp_path, monkeypatch):
    module = load("chain_p1_seed3")
    run = tmp_path / "p0-locked-s20260824-stats120"
    (run / "weights").mkdir(parents=True)
    (run / "weights/last.pt").touch()
    monkeypatch.setattr(module, "rows_in_results", lambda _: 120)
    assert not module.complete(run)
    (run / "assignment").mkdir()
    for epoch in range(1, 121):
        (run / "assignment" / f"epoch-{epoch:03d}.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "batches": 1618,
                    "expected_batches": 1618,
                    "calls": {"topk10": 1618},
                }
            )
        )
    assert module.complete(run)
