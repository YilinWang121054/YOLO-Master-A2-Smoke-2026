## A2 progress update: three-way TAL/STAL mechanism smoke

The mentor has confirmed the A2 protocol:

- COCO-style area bins are used only as an A2 supplement: small < 32^2, medium 32^2-96^2, large >= 96^2. These are **not** official VisDrone area bins.
- Evaluation bins use original validation-image GT boxes. Training assignment bins use the augmented/resized boxes entering the assigner.
- Formal P0/P1 conclusions require YOLO-Master v0.1-N, the full VisDrone train split, 548-image val split, imgsz=800, 120 epochs, patience=0, and maxDets=500.
- The primary acceptance metric is APs@[.50:.95], not AP50s. A +1.0 target means +1.0 absolute percentage point.
- Required controls are pure TAL, the repository's existing fixed-stride STAL, and a new adaptive STAL.

### Prototype

Public branch and commit:

- https://github.com/YilinWang121054/YOLO-Master/tree/feat/a2-adaptive-stal
- `52c2befa50706b9dff13b6e0813b19413d9f532d` based on upstream `0996b7d`

The branch adds explicit `stal_mode=tal|fixed|adaptive`. The default remains `fixed` to preserve current behavior. Adaptive mode combines small-GT candidate expansion, a pre-conflict minimum candidate guarantee, and area-adaptive top-k (13/10/10). Alpha=0.5, beta=6.0, and CIoU are unchanged. All experiment-affecting parameters are centralized in `default.yaml` with type/range/enum/relationship validation.

Focused tests cover TAL geometry equivalence, default fixed compatibility, area boundaries, empty GT, tiny boxes, overlapping/conflicting GTs, non-candidate conflict regression, and FP32/AMP mask/count/loss/gradient stability: **21 passed, 1 skipped** (Windows run with `PYTHONUTF8=1`).

### Fixed-subset mechanism smoke

All three runs used v0.1-N from YAML, 64 train / 32 val images, 1 epoch, imgsz=800, batch=4, seed=20260824, FP32, Mosaic off, and identical image-list hashes.

| mode | train small positives/GT | train zero-positive | val small positives/GT | val zero-positive |
| --- | ---: | ---: | ---: | ---: |
| pure TAL | 3.362 | 26.58% | 3.720 | 21.75% |
| fixed STAL | 3.721 | 17.91% | 4.040 | 16.03% |
| adaptive STAL | 6.473 | 5.88% | 6.557 | 7.66% |

Compared with current fixed STAL, adaptive STAL increased mean small-GT positives by 2.752/2.518 and reduced the train/val zero-positive rate by 12.03/8.37 percentage points. Medium/large assignment statistics were essentially unchanged.

All three 1-epoch runs had zero precision/recall/mAP. Therefore, this result supports only the preliminary claim that small-GT positive coverage improved. It is **not** a P0/P1 APs result and does not claim the +1.0 APs target.

### Bug found during smoke

An early adaptive diagnostic produced up to 86 positives for one GT despite top-k=13. The cause was conflict resolution taking `argmax` over all GT overlaps; in an all-zero-IoU tie, an anchor could be reassigned to a GT that had not proposed it. The fix masks non-candidate GT overlaps to `-inf` before conflict selection and adds a regression test. The faulty adaptive-r2 result was excluded. In the final r4 evidence, adaptive positives are bounded by top-k (maximum 13).

### Public evidence

- Full report: https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/docs/first-round-analysis.md
- Raw logs, per-GT CSVs, summaries, results.csv, subset manifest, checksums: https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/tree/main/results/mechanism-r4
- COCO-style evaluator and first sliced exploratory metrics: https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026

Next, I will use short runs only for parameter screening, then run the full 120-epoch three-way comparison and the required Mosaic interaction check before making any APs claim.
