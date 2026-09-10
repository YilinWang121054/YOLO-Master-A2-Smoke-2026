"""Compare the two failing A2 CI shards with the exact upstream base."""

import ast
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT.parent / "YOLO-Master-A2-final-review"
BASE = "7bfbfd374a6b44720f98366e088f3a962e16321d"


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REVIEW.as_posix()}", "-C", str(REVIEW), *args],
        timeout=30,
    )


def main():
    proof = json.loads((ROOT / "results/github-a2-final-pr-publication-20260911.json").read_text(encoding="utf-8"))
    tested = proof["tested_local_commit"]
    source = ROOT / ".local/pr-274-ci-20260911"
    logs = ROOT / "logs/pr-274-ci-20260911"
    results = ROOT / "results/pr-274-ci-20260911"
    logs.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    records = {}
    for label, job_id, expected_head in (
        ("base-windows", 102860234472, BASE),
        ("pr-windows", 102984847852, proof["head_commit"]),
        ("base-ubuntu", 102860234422, BASE),
        ("pr-ubuntu", 102984847730, proof["head_commit"]),
    ):
        raw = (source / f"{job_id}.log").read_bytes()
        meta = json.loads((source / f"{job_id}.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(raw).hexdigest()
        if meta["job"]["head_sha"] != expected_head or meta["log_sha256"] != digest:
            raise ValueError(f"CI provenance mismatch: {label}")
        text = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8-sig"))
        failures = dict(re.findall(r"FAILED (tests/\S+) - ([^\r\n]+)", text))
        summaries = re.findall(r"(\d+ failed, \d+ passed, \d+ skipped, \d+ warnings[^\r\n]+)", text)
        if not failures or len(summaries) != 1:
            raise ValueError(f"Unexpected CI summary shape: {label}")
        records[label] = {"job_id": job_id, "head_sha": expected_head, "url": meta["job"]["html_url"],
                          "failures": failures, "summary": summaries[0], "log_sha256": digest,
                          "log_bytes": len(raw)}
        shutil.copyfile(source / f"{job_id}.log", logs / f"{job_id}.log")
        shutil.copyfile(source / f"{job_id}.json", results / f"{job_id}.json")
    for os_name in ("windows", "ubuntu"):
        if records[f"base-{os_name}"]["failures"] != records[f"pr-{os_name}"]["failures"]:
            raise ValueError(f"PR failure set/messages differ from upstream for {os_name}")
    unchanged = []
    for path in ("ultralytics/nn/foundation/offline.py", "ultralytics/nn/foundation_distill_model.py",
                 "tests/test_foundation_config.py", "tests/test_foundation_offline.py",
                 "tests/test_foundation_cache_training.py"):
        if git("show", f"{BASE}:{path}") != git("show", f"{tested}:{path}"):
            raise ValueError(f"Expected unchanged foundation file: {path}")
        unchanged.append(path)
    same_functions = []
    for path, name in (("ultralytics/utils/tal.py", "make_anchors"),
                       ("ultralytics/cfg/__init__.py", "validate_foundation_config")):
        versions = []
        for revision in (BASE, tested):
            tree = ast.parse(git("show", f"{revision}:{path}").decode("utf-8"))
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            versions.append(ast.dump(node, include_attributes=False))
        if versions[0] != versions[1]:
            raise ValueError(f"Function changed: {path}:{name}")
        same_functions.append(f"{path}:{name}")
    for kind in ("base", "pr"):
        for suffix in ("log", "xml"):
            target = logs if suffix == "log" else results
            shutil.copyfile(source / f"local-{kind}.{suffix}", target / f"local-{kind}.{suffix}")
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(), "pr": proof["url"], "base": BASE,
        "pr_head": proof["head_commit"], "local_tested_commit": tested,
        "records": records, "same_failure_sets_and_messages": True,
        "byte_identical_files": unchanged, "ast_identical_functions": same_functions,
        "local_reproduction": "Both Windows Python 3.11 CPU checkouts reproduce the config-regex and stride errors. "
        "The offline-key test reaches a denied E:\\abs mkdir in both; this differs from hosted Windows DID NOT RAISE. "
        "The sandbox was not elevated for this test and no root-directory write was authorized.",
        "scope": "These Windows/Ubuntu failures were already present at the exact base; not a claim that all CI is green.",
    }
    (results / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
