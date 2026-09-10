# PR 审阅稿

拟标题：`[犀牛鸟-A2]：Add configurable scale-aware TAL assignment for small objects`

提交时间：按用户9月10日转达的要求，PR提交和已有PR标题修改须在9月12日前完成，内部目标为北京时间9月11日内。用户随后授权核对完成后先提交GitHub，再由用户审核并更新，不再设置发布前确认门槛。不能把本地草稿或已合并的修复PR当作最终课题PR已提交。

目标：`Tencent/YOLO-Master` 的 `main`。本地审阅分支为 `review/a2-stal-final`，已合并上游7bfbfd3，当前提交6f55860；训练仍使用原冻结分支。本文件尚未提交 GitHub。以下四节为建议 PR 正文，证据链接在最终提交前回读核实。

## Summary

Small VisDrone objects can contain too few anchor centers for TAL, even with a larger top-k. This change adds an explicit `stal_mode=tal|fixed|adaptive` switch. Adaptive mode expands the candidate region of small GT boxes, adds nearby candidates when coverage is insufficient, and uses separate top-k values for small, medium, and large objects.

The configuration lives in `default.yaml`, with checks for types, finite values, ranges, and parameter relationships. The default remains fixed-stride expansion. Alpha, beta, and the IoU ranking metric are unchanged. The candidate-conflict fix is already merged in #253 and is a dependency of this work, not a second bug-fix contribution.

## Tests

The review checkout, merged with upstream main at 7bfbfd3, passes the STAL, default-configuration, model-configuration and TAL conflict tests on CPU:

```text
python -m pytest tests/test_stal_assignment.py tests/test_default_config_integrity.py tests/test_master_model_configs.py tests/test_tal_conflict_resolution.py tests/test_tal_mps_regression.py -q -o addopts=''
33 passed, 1 skipped
```

Coverage includes candidate geometry, area boundaries, empty GT, tiny boxes, overlapping targets, candidate conflicts, configuration injection and invalid parameters (including NaN/Inf). MPS is unavailable in this run. The autocast test exercises a small assignment/loss fixture; this CPU run does not certify real-batch CUDA AMP parity. The changed files retain 15 Ruff diagnostics already present in current main, with no new diagnostics; their formatting and spelling checks pass. Repository-wide quality checks are not clean, and their outputs are retained separately.

## Experiments

The full-data experiments use YOLO-Master v0.1-N, VisDrone2019-DET, imgsz=800, batch=4, FP32, Mosaic on, and 120 epochs. The primary checkpoint is `last.pt` from epoch 120. APs uses original-image GT area <32² and maxDets=500, after the VisDrone ignore-region filter. These area bins are an A2 supplement.

| Seed | fixed APs | adaptive APs | ΔAPs (pp) |
| --- | ---: | ---: | ---: |
| 20260824 | 13.5280 | 13.3655 | −0.1625 |
| 20260825 | 13.0766 | 12.7361 | −0.3405 |
| 20260826 | 12.6454 | training | pending |

These results do not meet the A2 +1.0 APs target. Seed 20260824 also differs from later seeds in actual optimizer momentum (0.937 versus 0.9), so the rows are not presented as a uniform three-seed aggregate. A fixed-subset smoke reduced small-GT zero-positive rates, but this has not translated into higher APs in the completed pairs.

The locked acce839 P0 baseline has also completed 120 epochs and full 548-image validation. Its supplemental APs/APm/APl are 12.5117/30.7556/39.7667. All 120 online assignment records were checked against the original files and cover 194,160 training batches. This baseline is recorded separately from the fixed comparison, which includes #253.

The report and raw evidence will be linked to their verified commit before submission. This PR does not claim an accuracy improvement or P1 acceptance.

## Limitations

The minimum candidate count is enforced before conflict resolution; it does not guarantee that every GT retains that many positives. Candidate counts also do not measure normalized target weights. Full online per-epoch statistics are available for the new P0 baseline, not retrospectively for the older TAL/fixed/adaptive runs. The remaining paired runs and Mosaic interaction are incomplete.

The original MATLAB devkit could not run on this host because MATLAB R2024a reports MathWorks error 5201. The review package includes a MATLAB-source audit port and concrete prediction-level counterexamples for the older Python port; its runtime parity remains pending until the original tool runs on a licensed host. P2 threshold/warmup scans and a second dataset/task have not been completed. The latest-main diff was checked to exclude the already merged #253 fix.
