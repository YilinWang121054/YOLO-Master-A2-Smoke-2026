"""Publish one verified CI follow-up to the existing A2 PR; never merge it."""

import argparse
import base64
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    credential = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if credential.returncode or not token:
        raise RuntimeError("GitHub credential unavailable; credential output is not logged")
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "A2-CI-evidence-note"}

    def api(path, data=None):
        response = requests.request(
            "GET" if data is None else "POST", "https://api.github.com" + path,
            headers=headers, json=data, timeout=45, allow_redirects=False,
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(f"GitHub HTTP {response.status_code} at {path}")
        return response.json()

    if api("/user")["login"].lower() != "yilinwang121054":
        raise ValueError("Credential identity differs from the authorized account")
    evidence_repo = "YilinWang121054/YOLO-Master-A2-Smoke-2026"
    path = "results/pr-274-ci-20260911/comparison.json"
    remote = api(f"/repos/{evidence_repo}/contents/{path}?ref={args.evidence_commit}")
    raw = base64.b64decode(remote["content"])
    if raw != (root / path).read_bytes():
        raise ValueError("Public CI comparison differs from local evidence")
    comparison = json.loads(raw)
    pr_path = "/repos/Tencent/YOLO-Master/pulls/274"
    pr = api(pr_path)
    if pr["head"]["sha"] != comparison["pr_head"] or pr["base"]["sha"] != comparison["base"]:
        raise ValueError("PR changed after the CI comparison")
    note_url = f"https://github.com/{evidence_repo}/blob/{args.evidence_commit}/" + quote("docs/closure-review/PR-CI核对-20260911.md")
    body = (
        "CI follow-up (2026-09-11): I compared the failing Windows and Ubuntu Python 3.13 jobs "
        "against the exact upstream base, 7bfbfd3. Both shards have the same failing test names and error summaries as the base.\n\n"
        "- Windows: base 3 failed / 1947 passed; this PR 3 failed / 1965 passed.\n"
        "- Ubuntu x64: base 2 failed / 1950 passed; this PR 2 failed / 1968 passed.\n\n"
        "The shared failures are the foundation configuration error-message assertion and cached-response "
        "distillation's feature/stride count mismatch. Windows also has the offline-cache key validation failure. "
        "Although one traceback points to `tal.py`, `make_anchors` is unchanged; the foundation files and these test files are also unchanged.\n\n"
        f"[Comparison, original job links, preserved logs and local reproduction]({note_url}). "
        "The local Windows sandbox blocks the offline test's attempted root-directory write, so its final exception differs from hosted CI; this is documented separately.\n\n"
        "I have not skipped these tests or changed foundation behavior to make the checks green. "
        "This note explains the two checked failures; it does not claim that all CI passes."
    )
    comments_path = "/repos/Tencent/YOLO-Master/issues/274/comments"

    def existing():
        matches = []
        for page in range(1, 6):
            comments = api(f"{comments_path}?per_page=100&page={page}")
            matches.extend(c for c in comments if c["user"]["login"].lower() == "yilinwang121054"
                           and c["body"].startswith("CI follow-up (2026-09-11):"))
            if len(comments) < 100:
                return matches
        raise ValueError("Comment pagination limit reached; no write performed")

    matches = existing()
    if len(matches) > 1:
        raise ValueError("Multiple CI notes already exist; refusing a duplicate")
    if not args.publish:
        print(json.dumps({"ready": True, "existing": [c["html_url"] for c in matches], "body": body}))
        return
    if matches:
        comment = matches[0]
    else:
        try:
            comment = api(comments_path, {"body": body})
        except (requests.RequestException, RuntimeError):
            # Recover uncertain publication with a read, never a blind second POST.
            matches = existing()
            if len(matches) != 1:
                raise
            comment = matches[0]
    verified = api(f"/repos/Tencent/YOLO-Master/issues/comments/{comment['id']}")
    if verified["body"].replace("\r\n", "\n") != body:
        raise ValueError("Published note differs from the verified CI evidence body")
    result = {"observed_at": datetime.now(timezone.utc).isoformat(), "url": verified["html_url"],
              "evidence_commit": args.evidence_commit, "read_back_verified": True}
    (root / ".local/pr-274-ci-20260911/published-comment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
