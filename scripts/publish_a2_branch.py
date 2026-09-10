"""Publish the tested five-file A2 change through GitHub's Git Data API.

Used when Git HTTPS transport is unavailable. The remote commit is consolidated
onto the audited upstream base; its entire Git tree must equal the tested local
merge tree. Never updates main, force-updates a branch, or exposes credentials.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "YOLO-Master-A2-final-review"
LOGIN = "YilinWang121054"
FORK = LOGIN + "/YOLO-Master"
UPSTREAM = "Tencent/YOLO-Master"
BASE = "7bfbfd374a6b44720f98366e088f3a962e16321d"
TESTED = "6f558609882d9cc39379f88c912c9c3c5ec515d6"
BRANCH = "review/a2-stal-final"
FILES = sorted((
    "tests/test_stal_assignment.py",
    "ultralytics/cfg/__init__.py",
    "ultralytics/cfg/default.yaml",
    "ultralytics/utils/loss.py",
    "ultralytics/utils/tal.py",
))


def git(*args, text=True):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={SOURCE.as_posix()}", "-C", str(SOURCE), *args],
        text=text, encoding="utf-8" if text else None, timeout=30,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if git("rev-parse", "HEAD").strip() != TESTED or git("status", "--porcelain").strip():
        raise ValueError("Review checkout must be clean and at the tested commit")
    if sorted(git("diff", "--name-only", BASE, TESTED).splitlines()) != FILES:
        raise ValueError("Unexpected files in publication diff")
    expected_tree = git("rev-parse", "HEAD^{tree}").strip()
    credentials = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(line.split("=", 1) for line in credentials.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if credentials.returncode or not token:
        raise RuntimeError("No GitHub credential available; credential output is not logged")

    def api(path, data=None, missing_ok=False):
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=None if data is None else json.dumps(data).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "A2-reproducible-publication",
            },
            method="GET" if data is None else "POST",
        )
        # GET and content-addressed blob/tree creation can be replayed safely.
        # Ref/commit creation is not blindly replayed after an uncertain response.
        attempts = 3 if data is None or path.endswith(("/git/blobs", "/git/trees")) else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if missing_ok and error.code == 404:
                    return None
                raise RuntimeError(f"GitHub HTTP {error.code} at {path}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt + 1 == attempts:
                    raise RuntimeError(f"GitHub transport failed at {path}; credentials not logged") from None
                time.sleep(2 * (attempt + 1))

    if api("/user")["login"].lower() != LOGIN.lower():
        raise ValueError("Credential identity differs from the authorized account")
    if api(f"/repos/{UPSTREAM}/git/ref/heads/main")["object"]["sha"] != BASE:
        raise ValueError("Upstream main moved after testing; recheck before publishing")
    base_commit = api(f"/repos/{FORK}/git/commits/{BASE}")
    existing = api(f"/repos/{FORK}/git/ref/heads/{BRANCH}", missing_ok=True)
    commit_sha = None
    if existing:
        commit_sha = existing["object"]["sha"]
        if api(f"/repos/{FORK}/git/commits/{commit_sha}")["tree"]["sha"] != expected_tree:
            raise ValueError("Remote branch exists with different content; refusing to overwrite")
    if not args.publish:
        print(json.dumps({"ready": True, "local_tested_commit": TESTED, "tree": expected_tree, "existing_commit": commit_sha}))
        return
    if not commit_sha:
        entries = []
        for filename in FILES:
            content = git("show", f"{TESTED}:{filename}", text=False)
            expected_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            blob = api(f"/repos/{FORK}/git/blobs/{expected_blob}", missing_ok=True)
            if blob is None:
                blob = api(f"/repos/{FORK}/git/blobs", {
                    "content": base64.b64encode(content).decode("ascii"), "encoding": "base64",
                })
            if blob["sha"] != expected_blob:
                raise ValueError(f"Remote blob mismatch: {filename}")
            entries.append({"path": filename, "mode": "100644", "type": "blob", "sha": blob["sha"]})
            print(f"Verified blob: {filename}", flush=True)
        tree = api(f"/repos/{FORK}/git/trees", {"base_tree": base_commit["tree"]["sha"], "tree": entries})
        if tree["sha"] != expected_tree:
            raise ValueError("Remote publication tree differs from the tested local tree")
        commit = api(f"/repos/{FORK}/git/commits", {
            "message": "[犀牛鸟-A2] Add configurable adaptive STAL\n\nConsolidated onto upstream 7bfbfd3. The complete tree matches local tested merge 6f55860. Training evidence retains its original source commits.",
            "tree": expected_tree, "parents": [BASE],
        })
        commit_sha = commit["sha"]
        # A lost response is recovered by checking this ref on the next run.
        api(f"/repos/{FORK}/git/refs", {"ref": "refs/heads/" + BRANCH, "sha": commit_sha})
    verified = api(f"/repos/{FORK}/git/ref/heads/{BRANCH}")["object"]["sha"]
    verified_tree = api(f"/repos/{FORK}/git/commits/{verified}")["tree"]["sha"]
    if verified != commit_sha or verified_tree != expected_tree:
        raise ValueError("Final ref/tree verification failed")
    result = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "repository": FORK, "branch": BRANCH, "remote_commit": verified,
        "tested_local_commit": TESTED, "upstream_base": BASE,
        "tree": verified_tree, "full_tree_matches_tested_checkout": True,
        "changed_files": FILES,
        "url": f"https://github.com/{FORK}/tree/{BRANCH}",
        "method": "Git Data API; consolidated commit; no force update",
    }
    (ROOT / "results/github-a2-branch-publication-20260911.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError:
        raise SystemExit("GitHub API transport failed; no credential material logged") from None
