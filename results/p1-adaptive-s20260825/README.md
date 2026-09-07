# p1-adaptive-s20260825

Status: 120/120 epochs complete.

Checkpoint policy: primary result uses epoch-120 `last.pt`; `best.pt` is secondary.

Protocol:
- Dataset: VisDrone2019-DET val (548 images)
- Official VisDrone DET metrics: AP, AP50, AP75, AR1/10/100/500
- Supplementary COCO-style metrics: APs/APm/APl, ARs@500, AP50s
- COCO-style area split in this project: small `<32^2`, medium `32^2 <= area < 96^2`, large `>=96^2`
- COCO-style GT area is computed on original validation-image GT bbox pixels
- `ignore_regions=crowd-per-class` is a class-agnostic approximation used by the project script; it is not claimed to be the official VisDrone ignore-region evaluator

Key outputs:
- [official-det-metrics.json](official-det-metrics.json)
- [coco-style-metrics.json](coco-style-metrics.json)
- [coco-style-official-filter-metrics.json](coco-style-official-filter-metrics.json)
- [conversion-manifest.json](conversion-manifest.json)

Official DET metrics (pp):
- AP: 20.6613
- AP50: 36.1806
- AP75: 20.0618
- AR500: 37.3054

COCO-style metrics (pp):
- APs: 12.6596
- APm: 30.3815
- APl: 36.7854
- ARs@500: 27.8428
- AP50s: 27.3122

Seed pairing note:
- Compared with `p1-fixed-s20260825`, adaptive seed 2 changes `APs` by `-0.3335` pp under the formal official-filter supplementary protocol.

The formal supplementary values after applying the VisDrone DET toolkit's
50% pixel-coverage ignore-region filter are APs `12.7558`, APm `30.4260`,
APl `36.7081`, ARs@500 `27.9972`, and AP50s `27.4240`. The older
`coco-style-metrics.json` remains as a clearly labelled `crowd-per-class`
approximation for auditability and is not used for the formal comparison.
