"""Apply the user-requested A2 prefix to two existing PRs; never publish new content."""

import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request


REPOSITORY = "Tencent/YOLO-Master"
AUTHOR = "YilinWang121054"
PREFIX = "[犀牛鸟-A2]："
NUMBERS = (235, 253)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Change titles only.")
    args = parser.parse_args()
    credential = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
        check=False,
    )
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if credential.returncode or not token:
        raise RuntimeError("No usable GitHub credential; credential output is not logged.")

    def api(path, payload=None):
        req = urllib.request.Request(
            "https://api.github.com" + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "A2-title-review",
            },
            method="GET" if payload is None else "PATCH",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    identity = api("/user")["login"]
    if identity.lower() != AUTHOR.lower():
        raise RuntimeError("Credential identity does not match the authorized account.")
    for number in NUMBERS:
        path = f"/repos/{REPOSITORY}/pulls/{number}"
        before = api(path)
        if before["user"]["login"].lower() != AUTHOR.lower():
            raise RuntimeError(f"PR {number} is not owned by the authorized account.")
        title = before["title"]
        expected = title if title.startswith(PREFIX) else PREFIX + title
        if args.apply and title != expected:
            api(path, {"title": expected})
        after = api(path) if args.apply else before
        if args.apply:
            if after["title"] != expected:
                raise RuntimeError(f"PR {number} title was not verified.")
            unchanged = ("body", "state", "merged_at")
            if any(before[k] != after[k] for k in unchanged):
                raise RuntimeError(f"PR {number} had a concurrent metadata change; review required.")
            if any(before[k]["sha"] != after[k]["sha"] for k in ("head", "base")):
                raise RuntimeError(f"PR {number} code changed concurrently; review required.")
        print(json.dumps({
            "identity": identity,
            "number": number,
            "url": before["html_url"],
            "old_title": title,
            "requested_title": expected,
            "verified_title": after["title"],
            "applied": args.apply,
            "state": after["state"],
            "merged_at": after["merged_at"],
            "body_sha256": hashlib.sha256((after["body"] or "").encode()).hexdigest(),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"GitHub HTTP {error.code}; no credential material logged.") from None
    except urllib.error.URLError:
        raise SystemExit("GitHub transport failed; no credential material logged.") from None
