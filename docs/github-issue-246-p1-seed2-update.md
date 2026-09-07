## A2 P1 seed 2 progress update

The fixed STAL, adaptive STAL, and pure TAL runs for seed `20260825` have all completed the full VisDrone2019-DET train/val protocol: 120/120 epochs, `imgsz=800`, `batch=4`, FP32, Mosaic on, `close_mosaic=10`, `pretrained=false`, explicit MuSGD, and `maxDets=500`. The primary checkpoint is epoch-120 `last.pt`; `best.pt` is supplementary. The runs are resumable after an interrupted machine shutdown, and no training process remains active.

Overall `AP/AP50/AP75/AR500` below use the VisDrone DET algorithm Python port. Area metrics use the A2 COCO-style bins on original validation GT boxes (`small < 32^2`, `medium 32^2 <= area < 96^2`, `large >= 96^2`). Before COCO evaluation, the formal supplementary view applies the VisDrone toolkit's 50% pixel-coverage ignore-region filter to valid GT and detections. These area bins are not official VisDrone area bins.

| mode | official AP | AP50 | AP75 | AR500 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed STAL | 21.7619 | 39.4406 | 20.7115 | 38.1235 | 13.0893 | 31.0922 | 39.4663 |
| adaptive STAL | 21.4915 | 38.9398 | 20.1943 | 37.7670 | 12.7558 | 30.4260 | 36.7081 |
| pure TAL | 21.5139 | 38.6953 | 20.3479 | 37.9989 | 12.7300 | 31.0032 | 42.0973 |

For seed `20260825`, adaptive versus fixed is `ΔAPs=-0.3335` percentage points. Combining the formal filtered supplementary results from seeds `20260824` and `20260825` gives mean `ΔAPs=-0.2486` pp; both completed seeds are non-positive. This is not a final P1 acceptance decision because the agreed rule requires three paired seeds, mean `ΔAPs >= +1.0` pp, and at least two positive seeds. The paired seed `20260826` chain was started on `2026-09-08 04:27:14 +08:00` with fixed STAL active first, followed by adaptive STAL and pure TAL; the run is reboot-resumable and remains pending.

Public evidence:

- [seed 2 comparison](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/docs/p1-seed2-comparison.md)
- [seed 2 summary](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/blob/main/results/p1-seed2-summary.json)
- [fixed seed 2 artifacts](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/tree/main/results/p1-fixed-s20260825)
- [adaptive seed 2 artifacts](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/tree/main/results/p1-adaptive-s20260825)
- [pure TAL seed 2 artifacts](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/tree/main/results/p1-tal-s20260825)
- [raw logs](https://github.com/YilinWang121054/YOLO-Master-A2-Smoke-2026/tree/main/logs)

The older `coco-style-metrics.json` files are retained as `crowd-per-class` approximations for auditability only; they are not used for the formal APs comparison.
