"""Publish the authorized seed3 completion once, verifying public bytes first."""

import argparse
import base64
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "YilinWang121054/YOLO-Master-A2-Smoke-2026"
PREFIX = "A2 第三个seed结果补充（2026-09-13）"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    credential = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, timeout=30, cwd=ROOT,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    if credential.returncode or not fields.get("password"):
        raise RuntimeError("Credential unavailable; private output suppressed")
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + fields["password"],
                            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})

    def api(path, body=None):
        response = session.request("GET" if body is None else "POST", "https://api.github.com" + path,
                                   json=body, timeout=45, allow_redirects=False)
        if response.status_code not in (200, 201):
            raise RuntimeError(f"GitHub HTTP {response.status_code}; private headers suppressed")
        return response.json()

    if api("/user")["login"].lower() != "yilinwang121054":
        raise RuntimeError("Authorized identity mismatch")
    assert api(f"/repos/{REPO}")["private"] is False
    paths = ["results/closure-evaluation/p1-tal-s20260826/completion-verification.json",
             "results/closure-evaluation/p1-tal-s20260826/publication-sha256.txt",
             "results/report-table-verification-20260913-tal3.json", "docs/closure-review/研究报告.md"]
    for path in paths:
        remote = api(f"/repos/{REPO}/contents/{quote(path)}?ref={args.evidence_commit}")
        assert base64.b64decode(remote["content"]) == (ROOT / path).read_bytes(), path

    def link(path):
        return f"https://github.com/{REPO}/blob/{args.evidence_commit}/" + quote(path)

    body = (
        PREFIX + "\n\n"
        "补上seed 20260826的完整结果：fixed、adaptive、pure TAL均已完成120轮，"
        "主结果取第120轮last.pt，使用完整548张val。三组APs分别为12.6454、13.0384、12.6504；"
        "adaptive相对fixed增加0.3930个百分点，但总体AP减少0.0334、APl减少0.6630。\n\n"
        "这个seed有小幅正向变化，但不能据此认定P1达标。seed 20260824仍因optimizer协议差异列为探索性；"
        "现有三模式长训也缺完整的逐轮assigner统计，不会用离线分析补造训练历史。"
        "P0原始acce839基线的120份真实训练期统计另行保留。\n\n"
        f"- [更新后的研究报告，九组结果及限制]({link('docs/closure-review/研究报告.md')})\n"
        f"- [pure TAL完成核验、配置、原始日志与预测包]({link('results/closure-evaluation/p1-tal-s20260826/README.md')})\n"
        f"- [报告95个数值单元格核验]({link('results/report-table-verification-20260913-tal3.json')})\n\n"
        "总体指标仍采用已提交源码逐项对照的官方DET Python移植，未声称成功运行MATLAB；"
        "面积指标为官方ignore过滤后的COCO-style补充口径。关机前后的日志和checkpoint哈希均已保存。"
        "接下来补Mosaic交互与候选质量诊断，尚未完成的实验不计入结论。欢迎指出报告或实现中需要进一步核查的地方。"
    )
    endpoint = "/repos/Tencent/YOLO-Master/issues/274/comments"

    def existing():
        matches = []
        for page in range(1, 6):
            comments = api(f"{endpoint}?per_page=100&page={page}")
            matches += [c for c in comments if c["user"]["login"].lower() == "yilinwang121054"
                        and c["body"].startswith(PREFIX)]
            if len(comments) < 100:
                return matches
        raise RuntimeError("Pagination limit; no write performed")

    matches = existing()
    assert len(matches) <= 1, "Duplicate completion notes; refusing further writes"
    if not args.publish:
        print(json.dumps({"ready": True, "existing": [c["html_url"] for c in matches], "body": body}))
        return
    if matches:
        comment = matches[0]
    else:
        try:
            comment = api(endpoint, {"body": body})
        except (requests.RequestException, RuntimeError):
            matches = existing()
            if len(matches) != 1:
                raise RuntimeError("Uncertain publication; no blind POST retry") from None
            comment = matches[0]
    verified = api(f"/repos/Tencent/YOLO-Master/issues/comments/{comment['id']}")
    assert verified["body"].replace("\r\n", "\n") == body
    proof = {"observed_at": datetime.now(timezone.utc).isoformat(), "comment_url": verified["html_url"],
             "evidence_commit": args.evidence_commit, "public_artifact_bytes_verified": True,
             "comment_read_back_verified": True, "body": body}
    output = ROOT / "results/github-seed3-completion-publication-20260913.json"
    if output.exists():
        assert json.loads(output.read_text(encoding="utf-8"))["comment_url"] == verified["html_url"]
    else:
        output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in proof.items() if key != "body"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
