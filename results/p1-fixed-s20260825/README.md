# A2 P1 fixed STAL — seed 2 evaluation

This directory records the completed evaluation of the fixed-stride STAL
120-epoch run `p1-fixed-s20260825`. The main checkpoint is `last.pt` from the
private experiment workspace; weights are not distributed in this evidence
repository.

## Protocol

- Dataset: VisDrone2019-DET validation split (548 images)
- Model input: `imgsz=800`, `max_det=500`, FP32
- Main checkpoint: epoch-120 `last.pt`
- Overall AP/AP50/AP75/AR use the official VisDrone DET algorithm Python port
- APs/APm/APl and ARs@500 use the project’s COCO-style supplementary split:
  small `<32²`, medium `32²–<96²`, large `>=96²`

## Results

| Metric | Value (percentage points) |
| --- | ---: |
| Official AP | 21.7619 |
| Official AP50 | 39.4406 |
| Official AP75 | 20.7115 |
| Official AR500 | 38.1235 |
| COCO-style APs | 12.9824 |
| COCO-style APm | 31.0875 |
| COCO-style APl | 39.5988 |
| COCO-style ARs@500 | 28.3227 |

`official-det-metrics.json` is the authoritative overall result. The
COCO-style file is supplementary and must not be described as an official
VisDrone area split.
