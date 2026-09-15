"""Publish the MATLAB success and authorized finite-screen start once."""

import publish_candidate_quality_note as publisher
import requests

publisher.COMMIT = "659aec9191c701b4c0ea4c72061c7032b3145ec6"
publisher.PREFIX = "A2 MATLAB 补评通过与第二轮筛选启动（2026-09-15）"
publisher.NOTE = publisher.ROOT / "docs/closure-review/PR补评与筛选启动-20260915.md"
publisher.PROOF = publisher.ROOT / "results/github-pr274-screen-launch-20260915.json"
publisher.EVIDENCE_PATHS = [
    "docs/closure-review/P1第二轮fixed-MATLAB补评成功-20260915.md",
    "docs/closure-review/P1第二轮筛选冻结-20260915.md",
    "results/p1-v2-fixed-matlab-official-20260915/comparison.json",
    "results/p1-screen-v2-20260915-launch/freeze-manifest.json",
    "results/p1-screen-v2-20260915-launch/verification.json",
]

if __name__ == "__main__":
    try:
        publisher.main()
    except requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
