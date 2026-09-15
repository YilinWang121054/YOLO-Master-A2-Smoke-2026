"""Separate reboot-safe GPU and CPU workers for the authorized finite screen."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import psutil
from p1_screen_contract import (
    CONFIG,
    CONTRACT_FILES,
    FROZEN,
    MODEL,
    PROJECT,
    ROOT,
    configuration,
    exclusive_lock,
    now,
    read,
    save,
    sha,
    verify_completion,
    verify_frozen,
    verify_source,
)


def freeze():
    verify_source()
    if FROZEN.exists():
        verify_frozen()
        print("Existing freeze verified; no changes", flush=True)
        return
    if PROJECT.exists() and any(
        (PROJECT / case["id"]).exists() for case in configuration()["cases"]
    ):
        raise ValueError("Cannot freeze a campaign after training has started")
    for case in configuration()["cases"]:
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(ROOT / "scripts/run_p1_screen.py"),
                "--case",
                case["id"],
                "--check-only",
            ],
            check=True,
            timeout=120,
        )
    if psutil.disk_usage("F:/").free < 15 * 1024**3:
        raise RuntimeError(
            "At least15GiB free disk required for checkpoints, predictions and logs"
        )
    save(
        FROZEN,
        {
            "frozen_at": now(),
            "config_sha256": sha(CONFIG),
            "model_yaml_sha256": sha(MODEL),
            "evidence_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "files_sha256": {
                p.relative_to(ROOT).as_posix(): sha(p) for p in CONTRACT_FILES
            },
        },
        exclusive=True,
    )
    print("Finite screen contract frozen before training", flush=True)


def status(worker, state, **extra):
    save(
        PROJECT / f"{worker}-state.json",
        {"observed_at": now(), "state": state, **extra},
    )


def paused():
    return (PROJECT / "PAUSE").exists()


def active_children(script, case_id):
    found = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        if (proc.info.get("name") or "").lower() not in {
            "python.exe",
            "pythonw.exe",
            "yolo.exe",
        }:
            continue
        command = proc.info.get("cmdline") or []
        if any(script in item for item in command) and case_id in command:
            found.append(proc.pid)
    return found


def await_memory(worker):
    minimum = configuration()["resources"][f"{worker}_min_free_ram_gib"]
    while True:
        available = psutil.virtual_memory().available / 1024**3
        gpu_settled = True
        if worker == "cpu":
            gpu_state_path = PROJECT / "gpu-state.json"
            gpu_state = read(gpu_state_path) if gpu_state_path.exists() else {}
            gpu_settled = gpu_state.get("state") == "complete"
            if gpu_state.get("state") == "running":
                try:
                    gpu_settled = (
                        time.time() - psutil.Process(gpu_state["pid"]).create_time()
                        >= 60
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    gpu_settled = False
        if available >= minimum and gpu_settled:
            return not paused()
        status(
            worker,
            "waiting_for_memory",
            minimum_free_gib=minimum,
            available_gib=available,
            gpu_startup_settled=gpu_settled,
        )
        if paused():
            return False
        time.sleep(30)


def run_child(worker, case_id, script, extra, timeout):
    if not await_memory(worker):
        status(worker, "paused")
        return False
    verify_frozen()
    stamp = now().replace(":", "").replace(".", "")
    directory = PROJECT / "logs" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(ROOT / "scripts" / script),
        "--case",
        case_id,
        *extra,
    ]
    with (
        (directory / f"{worker}-{stamp}.stdout.log").open("xb") as out,
        (directory / f"{worker}-{stamp}.stderr.log").open("xb") as err,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env={**os.environ, "PYTHONUTF8": "1"},
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        event = {
            "worker": worker,
            "case": case_id,
            "pid": process.pid,
            "command": command,
            "started_at": now(),
            "timeout_seconds": timeout,
        }
        save(directory / f"{worker}-{stamp}.launch.json", event, exclusive=True)
        status(worker, "running", **{k: v for k, v in event.items() if k != "worker"})
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Timeout; stopping only child PID {process.pid}", flush=True)
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            code = process.wait(timeout=30)
            event["timed_out"] = True
    save(
        directory / f"{worker}-{stamp}.exit.json",
        {**event, "exit_code": code, "ended_at": now()},
        exclusive=True,
    )
    if code == 75 and paused():
        status(worker, "paused_at_checkpoint", case=case_id)
        return False
    if code:
        save(
            PROJECT / f"{worker}-failure.json",
            {**event, "exit_code": code, "ended_at": now()},
            exclusive=True,
        )
        status(worker, "blocked_after_failure", case=case_id, exit_code=code)
        raise RuntimeError("Child failed; no automatic retry")
    return True


def worker_main(worker):
    verify_frozen()
    with exclusive_lock(PROJECT / "locks" / f"worker-{worker}.lock"):
        if (PROJECT / f"{worker}-failure.json").exists():
            status(worker, "blocked_after_failure")
            return
        if paused():
            status(worker, "paused")
            return
        from evaluate_p1_screen import verify_result

        for case in configuration()["cases"]:
            case_id = case["id"]
            script = "run_p1_screen.py" if worker == "gpu" else "evaluate_p1_screen.py"
            while active_children(script, case_id):
                status(
                    worker,
                    "waiting_for_existing_child",
                    case=case_id,
                    pids=active_children(script, case_id),
                )
                if paused():
                    return
                time.sleep(30)
            if worker == "gpu":
                if verify_completion(case_id):
                    continue
                # Do not compete with an older user-authorized training job.
                for proc in psutil.process_iter(["name", "cmdline"]):
                    if (proc.info.get("name") or "").lower() not in {
                        "python.exe",
                        "pythonw.exe",
                        "yolo.exe",
                    }:
                        continue
                    joined = " ".join(proc.info.get("cmdline") or [])
                    if any(
                        name in joined
                        for name in (
                            "run_p1_v2_fixed.py",
                            "run_p1_seed.py",
                            "run_p0_baseline.py",
                        )
                    ):
                        raise RuntimeError("Another training job is active")
                if not run_child(worker, case_id, "run_p1_screen.py", [], 24 * 3600):
                    return
                if not verify_completion(case_id):
                    raise RuntimeError("Training exited without complete evidence")
            else:
                while not verify_completion(case_id):
                    status(worker, "waiting_for_training", case=case_id)
                    if paused() or (PROJECT / "gpu-failure.json").exists():
                        return
                    time.sleep(30)
                for epoch in configuration()["evaluation_epochs"]:
                    if verify_result(case_id, epoch):
                        continue
                    if not run_child(
                        worker,
                        case_id,
                        "evaluate_p1_screen.py",
                        ["--epoch", str(epoch)],
                        7200,
                    ):
                        return
        if worker == "cpu" and not (PROJECT / "selection.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(ROOT / "scripts/select_p1_screen.py"),
                ],
                cwd=ROOT,
                check=True,
                timeout=120,
            )
        status(worker, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", choices=("gpu", "cpu"))
    args = parser.parse_args()
    if sum((args.freeze, args.launch, bool(args.worker))) != 1:
        raise ValueError("Choose one operation")
    if args.freeze:
        freeze()
    elif args.worker:
        try:
            worker_main(args.worker)
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error).startswith(
                "Already running:"
            ):
                print(str(error), flush=True)
                return
            failure = PROJECT / f"{args.worker}-failure.json"
            if not failure.exists():
                save(
                    failure,
                    {
                        "observed_at": now(),
                        "error": type(error).__name__,
                        "detail": str(error),
                    },
                    exclusive=True,
                )
            status(
                args.worker,
                "blocked_after_supervisor_error",
                error=type(error).__name__,
            )
            raise
    else:
        verify_frozen()
        for worker in ("gpu", "cpu"):
            stamp = now().replace(":", "").replace(".", "")
            log_dir = PROJECT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (
                (log_dir / f"supervisor-{worker}-{stamp}.stdout.log").open("xb") as out,
                (log_dir / f"supervisor-{worker}-{stamp}.stderr.log").open("xb") as err,
            ):
                command = [sys.executable, "-X", "utf8", __file__, "--worker", worker]
                proc = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=out,
                    stderr=err,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                print({"worker": worker, "pid": proc.pid}, flush=True)


if __name__ == "__main__":
    main()
