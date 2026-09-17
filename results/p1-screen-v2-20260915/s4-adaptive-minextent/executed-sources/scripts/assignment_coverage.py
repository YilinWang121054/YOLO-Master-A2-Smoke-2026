"""Check online assignment evidence separately from completed optimizer epochs."""

import json
from pathlib import Path


def missing_assignment_epochs(run_dir: Path, completed_epochs: int) -> list[int]:
    missing = []
    for epoch in range(1, completed_epochs + 1):
        try:
            record = json.loads(
                (run_dir / "assignment" / f"epoch-{epoch:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                record["epoch"] != epoch
                or record["batches"] <= 0
                or record["batches"] != record["expected_batches"]
                or not record["calls"]
            ):
                missing.append(epoch)
        except (OSError, ValueError, TypeError, KeyError):
            missing.append(epoch)
    return missing
