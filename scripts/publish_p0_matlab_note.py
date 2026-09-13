"""Publish the authorized P0 MATLAB completion note with public-byte verification."""

import publish_candidate_quality_note as publication

publication.COMMIT = "9abf90ac9dd1f76bf9a4e63cd6951fe7c8d6abb5"
publication.PREFIX = "A2 P0 官方 MATLAB 实测对齐补充（2026-09-13）"
publication.NOTE = publication.ROOT / "docs/closure-review/PR官方MATLAB对齐进度-20260913.md"
publication.PROOF = publication.ROOT / "results/github-p0-matlab-publication-20260913.json"
publication.EVIDENCE_PATHS = [
    "results/p0-matlab-official-20260913/comparison.json",
    "results/p0-matlab-official-20260913/verification.json",
    "results/p0-matlab-official-20260913/matlab-metrics.json",
    "results/p0-matlab-official-20260913/stdout.log",
    "results/p1-v2-amp-audit-20260913/summary.json",
    "docs/closure-review/研究报告.md",
]

if __name__ == "__main__":
    try:
        publication.main()
    except publication.requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
