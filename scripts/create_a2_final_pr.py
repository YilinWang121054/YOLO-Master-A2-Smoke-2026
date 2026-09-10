"""Publish the reviewed A2 PR after verifying public code and report bytes.

Credentials remain in memory. This script never merges a PR or changes a branch.
An uncertain creation response is recovered by looking up the existing head.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGIN = "YilinWang121054"
UPSTREAM = "Tencent/YOLO-Master"
FORK = LOGIN + "/YOLO-Master"
EVIDENCE = LOGIN + "/YOLO-Master-A2-Smoke-2026"
BRANCH = "review/a2-stal-final"
TITLE = "[犀牛鸟-A2]：Add configurable scale-aware TAL assignment for small objects"
REPORT = "docs/closure-review/研究报告.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    credentials = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(line.split("=", 1) for line in credentials.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if credentials.returncode or not token:
        raise RuntimeError("No GitHub credential available; credential output is not logged")

    def api(path, data=None):
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=None if data is None else json.dumps(data).encode("utf-8"),
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/vnd.github+json", "Content-Type": "application/json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "A2-reviewed-publication"},
            method="GET" if data is None else "POST",
        )
        attempts = 3 if data is None else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"GitHub HTTP {error.code} at {path}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt + 1 == attempts:
                    raise RuntimeError(f"GitHub transport failed at {path}; credentials not logged") from None
                time.sleep(2 * (attempt + 1))

    if api("/user")["login"].lower() != LOGIN.lower():
        raise ValueError("Credential identity differs from the authorized account")
    if api(f"/repos/{EVIDENCE}")["private"]:
        raise ValueError("Evidence repository is not public")
    evidence_head = api(f"/repos/{EVIDENCE}/git/ref/heads/main")["object"]["sha"]
    if evidence_head != args.evidence_commit:
        raise ValueError("Evidence main differs from the expected published commit")
    public_report = api(
        f"/repos/{EVIDENCE}/contents/{urllib.parse.quote(REPORT)}?ref={args.evidence_commit}"
    )
    content = base64.b64decode(public_report["content"])
    report_sha = hashlib.sha256(content).hexdigest()
    verification = json.loads((ROOT / "results/report-table-verification-20260911.json").read_text(encoding="utf-8"))
    if content != (ROOT / REPORT).read_bytes() or report_sha != verification["report_sha256"]:
        raise ValueError("Public report bytes differ from the verified local report")
    branch_proof = json.loads((ROOT / "results/github-a2-branch-publication-20260911.json").read_text(encoding="utf-8"))
    code_head = api(f"/repos/{FORK}/git/ref/heads/{BRANCH}")["object"]["sha"]
    code_tree = api(f"/repos/{FORK}/git/commits/{code_head}")["tree"]["sha"]
    if code_head != branch_proof["remote_commit"] or code_tree != branch_proof["tree"]:
        raise ValueError("Public code differs from the tested complete tree")
    if api(f"/repos/{UPSTREAM}/git/ref/heads/main")["object"]["sha"] != branch_proof["upstream_base"]:
        raise ValueError("Upstream main moved after testing; review before publication")
    root_url = f"https://github.com/{EVIDENCE}/blob/{args.evidence_commit}/"
    report_url = root_url + urllib.parse.quote(REPORT)
    evidence_url = f"https://github.com/{EVIDENCE}/tree/{args.evidence_commit}/"
    draft = (ROOT / "docs/closure-review/PR说明.md").read_text(encoding="utf-8")
    body = "## Summary" + draft.split("## Summary", 1)[1]
    placeholder = "The report and raw evidence will be linked to their verified commit before submission."
    if body.count(placeholder) != 1:
        raise ValueError("Expected one evidence-link placeholder in the reviewed draft")
    links = (
        f"[Research report (Chinese)]({report_url}) · "
        f"[P0 metrics, predictions and 120 assignment records]({evidence_url}results/closure-evaluation/p0-locked-s20260824-stats120) · "
        f"[Evidence index]({root_url}README.md) · "
        f"[Regression test log]({root_url}logs/final-pr-upstream-tests-20260911.log). "
        "These links are pinned to the verified evidence commit."
    )
    body = body.replace(placeholder, links)
    query = urllib.parse.urlencode({"state": "all", "head": LOGIN + ":" + BRANCH, "base": "main"})
    lookup_path = f"/repos/{UPSTREAM}/pulls?{query}"
    existing = api(lookup_path)
    if len(existing) > 1:
        raise ValueError("Multiple PRs use this head; refusing ambiguous publication")
    if not args.publish:
        print(json.dumps({"ready": True, "existing_pr": existing[0]["html_url"] if existing else None,
                          "title": TITLE, "report_sha256": report_sha, "evidence_commit": evidence_head}, ensure_ascii=False))
        return
    if existing:
        pr = existing[0]
    else:
        try:
            pr = api(f"/repos/{UPSTREAM}/pulls", {
                "title": TITLE, "head": LOGIN + ":" + BRANCH,
                "base": "main", "body": body, "draft": False,
            })
        except RuntimeError:
            # Never blindly repeat a POST after its outcome became uncertain.
            recovered = api(lookup_path)
            if len(recovered) != 1:
                raise
            pr = recovered[0]
    pr = api(f"/repos/{UPSTREAM}/pulls/{pr['number']}")
    if (pr["title"] != TITLE or pr["head"]["sha"] != code_head or pr["base"]["ref"] != "main"
            or pr["body"].replace("\r\n", "\n") != body.replace("\r\n", "\n")
            or pr["user"]["login"].lower() != LOGIN.lower()):
        raise ValueError("Published PR differs from reviewed title, head, body, base, or author")
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "url": pr["html_url"], "number": pr["number"], "title": pr["title"],
        "state": pr["state"], "draft": pr["draft"], "created_at": pr["created_at"],
        "author": pr["user"]["login"], "head_commit": code_head, "tree": code_tree,
        "tested_local_commit": branch_proof["tested_local_commit"],
        "evidence_commit": evidence_head, "report_url": report_url, "report_sha256": report_sha,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "public_report_bytes_verified": True, "pr_fields_read_back_verified": True,
        "authorization": "User authorized publication first, followed by their review; no merge requested or performed",
    }
    (ROOT / "results/github-a2-final-pr-publication-20260911.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
