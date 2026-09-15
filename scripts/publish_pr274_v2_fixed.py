"""Publish the authorized, completed fixed update with identity and byte checks."""

import publish_candidate_quality_note as publisher
import requests

publisher.COMMIT = "df0f51ad4a5767cf0dc952e86b2409f0c884d111"
publisher.PREFIX = "A2 第二轮 fixed 完成记录（2026-09-15）"
publisher.NOTE = publisher.ROOT / "docs/closure-review/PR第二轮fixed进度-20260915.md"
publisher.PROOF = publisher.ROOT / "results/github-pr274-v2-fixed-publication-20260915.json"
publisher.EVIDENCE_PATHS = [
    "docs/closure-review/P1第二轮fixed完成-20260915.md",
    "docs/closure-review/研究报告.md",
    "results/closure-evaluation/p1-v2-fixed-s20260825-stats120/completion-verification.json",
    "results/closure-evaluation/p1-v2-fixed-s20260825-stats120/publication-verification.json",
    "results/report-table-verification-20260915.json",
]

if __name__ == "__main__":
    try:
        publisher.main()
    except requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
