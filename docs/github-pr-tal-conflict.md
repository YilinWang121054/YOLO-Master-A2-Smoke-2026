# 独立 TAL 冲突修复 PR

已创建 PR：<https://github.com/Tencent/YOLO-Master/pull/253>

对应 Issue 进展评论：<https://github.com/Tencent/YOLO-Master/issues/246#issuecomment-5482342694>

原始创建页面：<https://github.com/Tencent/YOLO-Master/compare/main...YilinWang121054:YOLO-Master:fix/tal-candidate-conflict?expand=1>

标题：

```text
Fix TAL conflict resolution for zero-IoU candidate ties
```

正文：

````markdown
## Summary

- restrict TAL conflict resolution to GTs that actually proposed the contested anchor
- prevent all-zero-IoU ties from assigning an anchor to a non-candidate GT
- add a focused regression test that verifies conflict resolution cannot increase any GT's candidate count

## Problem

`select_highest_overlaps()` currently calls `overlaps.argmax(1)` across every GT when an anchor is proposed by multiple GTs. Overlaps for non-candidate GT-anchor pairs are zero. If all candidate overlaps are also zero, `argmax` can select an unrelated lower-index GT that never proposed the anchor.

This was observed during the A2 VisDrone assignment smoke: one GT received more final positives than its pre-conflict top-k. The invalid exploratory run was excluded from the reported result.

The fix masks non-candidate overlap entries to `-inf` before `argmax`, so the selected GT is always one of the current candidates.

## Tests

```
python -m pytest tests/test_tal_conflict_resolution.py tests/test_tal_mps_regression.py -q
2 passed, 1 skipped
```

Related progress report: #246
````

分支 commit：[`9e77308`](https://github.com/YilinWang121054/YOLO-Master/commit/9e77308)
