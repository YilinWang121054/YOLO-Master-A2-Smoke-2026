"""Read A2 PR CI jobs and preserve their original logs without rerunning CI."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, nargs="+", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / ".local/pr-274-ci-20260911"
    output.mkdir(parents=True, exist_ok=True)
    credentials = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(line.split("=", 1) for line in credentials.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if credentials.returncode or not token:
        raise RuntimeError("GitHub credential unavailable; credential output is not logged")
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "A2-CI-readonly-audit"}

    def get(path):
        response = requests.get("https://api.github.com" + path, headers=headers, timeout=45, allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"GitHub read returned HTTP {response.status_code} at {path}")
        return response.json()

    if get("/user")["login"].lower() != "yilinwang121054":
        raise ValueError("Credential identity differs from the authorized account")
    for job_id in args.jobs:
        base = "/repos/Tencent/YOLO-Master"
        job = get(f"{base}/actions/jobs/{job_id}")
        annotations = get(f"{base}/check-runs/{job_id}/annotations")
        response = requests.get(
            f"https://api.github.com{base}/actions/jobs/{job_id}/logs",
            headers=headers, timeout=45, allow_redirects=False,
        )
        if response.status_code == 302:
            location = response.headers["Location"]
            if urlparse(location).scheme != "https":
                raise ValueError("Refusing a non-HTTPS log download")
            # Signed artifact download gets no GitHub credentials; do not log its URL.
            response = requests.get(location, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"CI log download returned HTTP {response.status_code}")
        raw = response.content
        (output / f"{job_id}.log").write_bytes(raw)
        meta = {
            "observed_at": datetime.now(timezone.utc).isoformat(), "job": job,
            "annotations": annotations, "log_sha256": hashlib.sha256(raw).hexdigest(), "log_bytes": len(raw),
        }
        (output / f"{job_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"id": job_id, "name": job["name"], "conclusion": job["conclusion"],
                          "failed_steps": [x for x in job["steps"] if x["conclusion"] == "failure"],
                          "annotations": annotations, "log_bytes": len(raw)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        raise SystemExit("GitHub transport failed; credentials and signed artifact URLs are not logged") from None
