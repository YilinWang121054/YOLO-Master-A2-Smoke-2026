"""Publish the authorized response to the default-compatibility and seed3 review."""

import publish_candidate_quality_note as publication

publication.COMMIT = "9d2e2ef4bc635f4ac4db68f688a003807b5edefd"
publication.PREFIX = "A2 默认兼容性复核与第三 seed 结果（2026-09-14）"
publication.NOTE = (
    publication.ROOT / "docs/closure-review/PR默认兼容与第三seed回复-20260914.md"
)
publication.PROOF = (
    publication.ROOT / "results/github-pr274-review-response-20260914.json"
)
publication.EVIDENCE_PATHS = [
    "results/pr274-review-checks-20260914/README.md",
    "results/pr274-review-checks-20260914/manifest.json",
    "results/pr274-review-checks-20260914/summary.json",
    "results/pr274-review-checks-20260914/stdout.log",
    "results/pr274-review-checks-20260914/pytest.xml",
    "results/pr274-review-checks-20260914/report-table-verification.json",
    "docs/closure-review/研究报告.md",
    "results/closure-evaluation/p1-tal-s20260826/completion-verification.json",
]

if __name__ == "__main__":
    try:
        publication.main()
    except publication.requests.RequestException:
        raise SystemExit("GitHub transport failed; no credentials logged") from None
