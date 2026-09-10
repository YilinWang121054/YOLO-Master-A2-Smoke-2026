"""Compare current lint diagnostics to the exact source base, without hiding failures."""

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "YOLO-Master-A2-final-review"
BASE = "0996b7da14dfaafae9d4488e960814ff19eb19ce"
FILES = (
    "ultralytics/cfg/__init__.py",
    "ultralytics/utils/loss.py",
    "ultralytics/utils/tal.py",
    "tests/test_stal_assignment.py",
)


def diagnostics(source, filename):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--output-format=json",
            "--stdin-filename",
            filename,
            "-",
        ],
        input=source,
        cwd=SOURCE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/closure-pr-lint-audit.json")
    args = parser.parse_args()
    base = subprocess.check_output(
        ["git", "-c", f"safe.directory={SOURCE.as_posix()}", "-C", str(SOURCE), "rev-parse", "--verify", f"{args.base}^{{commit}}"],
        text=True,
        encoding="utf-8",
    ).strip()
    report = {}
    for path in FILES:
        before = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={SOURCE.as_posix()}",
                "-C",
                str(SOURCE),
                "show",
                f"{base}:{path}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        baseline = diagnostics(before.stdout, path) if before.returncode == 0 else []
        current = diagnostics((SOURCE / path).read_text(encoding="utf-8"), path)
        key = lambda entry: (entry["code"], entry["message"])
        new = collections.Counter(map(key, current)) - collections.Counter(
            map(key, baseline)
        )
        report[path] = {
            "baseline_diagnostics": baseline,
            "current_diagnostics": current,
            "new_diagnostics": [
                {"code": k[0], "message": k[1], "count": n} for k, n in new.items()
            ],
        }
    out = args.output
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                path: {
                    "baseline": len(v["baseline_diagnostics"]),
                    "current": len(v["current_diagnostics"]),
                    "new": v["new_diagnostics"],
                }
                for path, v in report.items()
            }
        )
    )


if __name__ == "__main__":
    main()
