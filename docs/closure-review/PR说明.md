# PR 审阅稿

拟标题：`Add configurable scale-aware TAL assignment for small objects`

目标：`Tencent/YOLO-Master` 的 `main`。本地审阅分支为 `review/a2-stal-final`；训练仍使用原冻结分支。本文件尚未提交 GitHub。以下四节为建议 PR 正文，测试和待完成项在最终提交前更新。

## Summary

Small VisDrone objects can contain too few anchor centers for TAL, even with a larger top-k. This change adds an explicit `stal_mode=tal|fixed|adaptive` switch. Adaptive mode expands the candidate region of small GT boxes, adds nearby candidates when coverage is insufficient, and uses separate top-k values for small, medium, and large objects.

The configuration lives in `default.yaml`, with checks for types, finite values, ranges, and parameter relationships. The default remains fixed-stride expansion. Alpha, beta, and the IoU ranking metric are unchanged. The candidate-conflict fix is already merged in #253 and is a dependency of this work, not a second bug-fix contribution.

## Tests

The current local review checkout passes 31 CPU tests covering STAL, default configuration, and model configurations:

```text
python -m pytest tests/test_stal_assignment.py tests/test_default_config_integrity.py tests/test_master_model_configs.py -q -o addopts=''
31 passed
```

Coverage includes candidate geometry, area boundaries, empty GT, tiny boxes, overlapping targets, candidate conflicts, configuration injection and invalid parameters (including NaN/Inf). The existing autocast test exercises a small assignment/loss fixture; the current CPU run does not certify real-batch CUDA AMP parity. The full-file quality check still reports 15 existing base diagnostics, with no new lint diagnostics in the proposed files.

## Experiments

The full-data experiments use YOLO-Master v0.1-N, VisDrone2019-DET, imgsz=800, batch=4, FP32, Mosaic on, and 120 epochs. The primary checkpoint is `last.pt` from epoch 120. APs uses original-image GT area <32² and maxDets=500, after the VisDrone ignore-region filter. These area bins are an A2 supplement.

| Seed | fixed APs | adaptive APs | ΔAPs (pp) |
| --- | ---: | ---: | ---: |
| 20260824 | 13.5280 | 13.3655 | −0.1625 |
| 20260825 | 13.0766 | 12.7361 | −0.3405 |
| 20260826 | pending external evaluation | training | pending |

These results do not meet the A2 +1.0 APs target. Seed 20260824 also differs from later seeds in actual optimizer momentum (0.937 versus 0.9), so the rows are not presented as a uniform three-seed aggregate. A fixed-subset smoke reduced small-GT zero-positive rates, but this has not translated into higher APs in the completed pairs.

The report and raw evidence will be linked to their reviewed commit before submission. This PR does not claim an accuracy improvement or P1 acceptance.

## Limitations

The minimum candidate count is enforced before conflict resolution; it does not guarantee that every GT retains that many positives. Candidate counts also do not measure normalized target weights. Full online per-epoch statistics, the remaining paired runs and Mosaic interaction remain in progress. The original locked baseline is being prepared as a separate P0 run because the current fixed comparison includes #253.

The original MATLAB devkit could not run on this host because MATLAB R2024a reports MathWorks error 5201. The review package now includes a MATLAB-source audit port and concrete prediction-level counterexamples for the older Python port; its runtime parity remains pending until the original tool runs on a licensed host. P2 threshold/warmup scans and a second dataset/task have not been completed. Before opening this PR, the review branch must also be checked against current upstream main and the already merged #253.
