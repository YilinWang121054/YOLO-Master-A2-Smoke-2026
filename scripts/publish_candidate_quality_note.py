"""Publish the authorized candidate-figure update once to PR #274."""

import argparse
import base64
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "YilinWang121054/YOLO-Master-A2-Smoke-2026"
COMMIT = "2389a4b435dfaf7111037be24ee558b61a6a8a44"
PREFIX = "A2 候选质量诊断补充（2026-09-13）"
NOTE = ROOT / "docs/closure-review/PR候选诊断进度-20260913.md"
PROOF = ROOT / "results/github-candidate-figures-publication-20260913.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    credential = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(
        line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line
    )
    if credential.returncode or not fields.get("password"):
        raise RuntimeError("Credential unavailable; private output suppressed")
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": "Bearer " + fields["password"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )

    def api(path, body=None):
        response = session.request(
            "GET" if body is None else "POST",
            "https://api.github.com" + path,
            json=body,
            timeout=45,
            allow_redirects=False,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"GitHub HTTP {response.status_code}; private headers suppressed"
            )
        return response.json()

    if api("/user")["login"].lower() != "yilinwang121054":
        raise RuntimeError("Authorized identity mismatch")
    assert api(f"/repos/{REPO}")["private"] is False
    pr = api("/repos/Tencent/YOLO-Master/pulls/274")
    assert pr["state"] == "open" and not pr["merged"]
    assert pr["head"]["sha"] == "2efc4d91d7363d65b6f38e7d47c585231bb98158"
    paths = [
        "results/candidate-quality-20260913/figures-v1/README.md",
        "results/candidate-quality-20260913/figures-v1/verification.json",
        "results/candidate-quality-20260913/figures-v1/publication-manifest.json",
        "docs/closure-review/研究报告.md",
    ]
    for path in paths:
        remote = api(f"/repos/{REPO}/contents/{quote(path)}?ref={COMMIT}")
        assert base64.b64decode(remote["content"]) == (ROOT / path).read_bytes(), path
    body = NOTE.read_text(encoding="utf-8").strip()
    assert body.startswith(PREFIX) and COMMIT in body
    endpoint = "/repos/Tencent/YOLO-Master/issues/274/comments"

    def existing():
        matches = []
        for page in range(1, 6):
            comments = api(f"{endpoint}?per_page=100&page={page}")
            matches += [
                c
                for c in comments
                if c["user"]["login"].lower() == "yilinwang121054"
                and c["body"].startswith(PREFIX)
            ]
            if len(comments) < 100:
                return matches
        raise RuntimeError("Pagination limit; no write performed")

    matches = existing()
    assert len(matches) <= 1, "Duplicate notes; refusing further writes"
    if matches:
        assert matches[0]["body"].replace("\r\n", "\n") == body
    if not args.publish:
        print(
            json.dumps(
                {
                    "ready": True,
                    "existing": [c["html_url"] for c in matches],
                    "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "body": body,
                }
            )
        )
        return
    if matches:
        comment = matches[0]
    else:
        try:
            comment = api(endpoint, {"body": body})
        except (requests.RequestException, RuntimeError):
            matches = existing()
            if len(matches) != 1:
                raise RuntimeError(
                    "Uncertain publication; no blind POST retry"
                ) from None
            comment = matches[0]
    verified = api(f"/repos/Tencent/YOLO-Master/issues/comments/{comment['id']}")
    assert verified["user"]["login"].lower() == "yilinwang121054"
    assert verified["body"].replace("\r\n", "\n") == body
    proof = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "comment_url": verified["html_url"],
        "evidence_commit": COMMIT,
        "public_artifacts_verified": paths,
        "comment_read_back_verified": True,
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "body": body,
    }
    if PROOF.exists():
        old = json.loads(PROOF.read_text(encoding="utf-8"))
        assert (
            old["comment_url"] == proof["comment_url"]
            and old["body_sha256"] == proof["body_sha256"]
        )
    else:
        PROOF.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in proof.items() if key != "body"}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
