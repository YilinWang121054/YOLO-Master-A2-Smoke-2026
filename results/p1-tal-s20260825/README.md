# p1-tal-s20260825

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
- AP: 20.9909
- AP50: 36.2530
- AP75: 20.5661
- AR500: 37.8859

COCO-style metrics (pp):
- APs: 12.6246
- APm: 30.9915
- APl: 42.1665
- ARs@500: 28.1226
- AP50s: 26.5627

Seed pairing note:
- Compared with `p1-fixed-s20260825`, pure TAL seed 2 changes `APs` by `-0.3593` pp under the formal official-filter supplementary protocol.

The formal supplementary values after applying the VisDrone DET toolkit's
50% pixel-coverage ignore-region filter are APs `12.7300`, APm `31.0032`,
APl `42.0973`, ARs@500 `28.3063`, and AP50s `26.7155`. The older
`coco-style-metrics.json` remains as a clearly labelled `crowd-per-class`
approximation for auditability and is not used for the formal comparison.
