"""Archive the 2026-09-17 authorized recovery without rewriting any run evidence."""

import json
import subprocess
import sys
from pathlib import Path

import psutil

from archive_p1_screen_case import archive_file, validate_child_logs
from evaluate_p1_screen import verify_result
from p1_screen_contract import (
    PROJECT, ROOT, configuration, csv_rows, now, read, save, sha,
    verify_completion, verify_frozen,
)


def main():
    freeze = verify_frozen()
    state = read(PROJECT / "gpu-state.json")
    case = "s5-adaptive-compact-k8"
    assert state["state"] == "running" and state["case"] == case
    child = psutil.Process(state["pid"])
    command = child.cmdline()
    assert any("run_p1_screen.py" in s for s in command) and case in command
    completed = [c["id"] for c in configuration()["cases"] if verify_completion(c["id"])]
    assert len(completed) == 5
    cpu_logs = PROJECT / "logs/s4-adaptive-minextent"
    interruptions = validate_child_logs(cpu_logs)
    out = ROOT / "results/p1-screen-v2-20260917-recovery"
    out.mkdir(exist_ok=False)
    files = {}

    def take(source, relative):
        files[relative] = archive_file(source, out / relative)

    for source in sorted((PROJECT / "logs/recovery").glob("*.json")):
        take(source, "recovery/" + source.name)
    for suffix in ("launch.json", "stdout.log", "stderr.log", "interrupted.json"):
        name = "cpu-2026-09-15T192855060594+0000." + suffix
        take(cpu_logs / name, "interrupted-evaluation/" + name)
    for name in (
        "gpu-2026-09-17T064804433282+0000.launch.json",
        "runner-2026-09-17T064810005253+0000.json",
    ):
        take(PROJECT / "logs" / case / name, "resumed-training/" + name)
    for worker in ("gpu", "cpu"):
        take(PROJECT / f"{worker}-state.json", f"{worker}-state-snapshot.json")
    wrapper = "supervisor-gpu-20260917T0655019381806Z-d8b564d8494140138da54f97a4a12295"
    for suffix in ("stdout.log", "stderr.log"):
        take(PROJECT / "logs" / (wrapper + "." + suffix), "duplicate-guard-test/" + suffix)
    for name in (
        "archive_p1_screen_recovery.py", "archive_p1_screen_case.py",
        "start_p1_screen_worker.ps1", "install_p1_screen_recovery.ps1",
    ):
        take(ROOT / "scripts" / name, "executed-sources/" + name)
    take(ROOT / "tests/test_archive_p1_screen_case.py", "executed-sources/test_archive_p1_screen_case.py")
    test_command = [sys.executable, "-X", "utf8", "-m", "pytest",
                    "tests/test_archive_p1_screen_case.py", "tests/test_p1_screen_contract.py",
                    "-q", "-o", "addopts="]
    with (out / "tests.stdout.log").open("xb") as stdout, (out / "tests.stderr.log").open("xb") as stderr:
        result = subprocess.run(test_command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                timeout=120, check=False)
    assert result.returncode == 0
    for name in ("tests.stdout.log", "tests.stderr.log"):
        files[name] = {"archived_sha256": sha(out / name)}
    evaluations = sum(verify_result(c["id"], e) for c in configuration()["cases"] for e in range(16, 21))
    assert verify_frozen() == freeze
    save(out / "verification.json", {
        "observed_at": now(), "scope": "Recovery and provenance only, not a completed screening result or P1 acceptance",
        "training_pid_observed": child.pid, "training_command_observed": command,
        "completed_training_cases_hash_verified": completed,
        "completed_evaluations_hash_verified": evaluations,
        "s5_csv_epochs_at_observation": len(csv_rows(PROJECT / case)),
        "interrupted_cpu_receipts_validated": interruptions,
        "frozen_protocol_unchanged": True,
        "test_command": test_command, "tests_exit_code": result.returncode,
        "files": files,
    }, exclusive=True)
    print(json.dumps({"archive": str(out), "files": len(files), "evaluations": evaluations,
                      "tests_exit_code": result.returncode}), flush=True)


if __name__ == "__main__":
    main()
