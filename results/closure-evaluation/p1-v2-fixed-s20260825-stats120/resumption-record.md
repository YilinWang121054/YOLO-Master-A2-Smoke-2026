# P1 v2 fixed 续训记录

北京时间 2026-09-14 18:49:13，按用户“任务及实验 继续”的指示，从完整第 64 轮恢复训练，目标仍为 120 轮。上午主动关机暂停前未完成的第 65 轮重新运行，不从中间 batch 接续。未改变模型、数据、seed、优化器、batch、精度或增强配置。

## 断点与过程检查

- 源码版本：`52c2befa50706b9dff13b6e0813b19413d9f532d`，已跟踪文件无改动。
- `last.pt` SHA-256：`b62b2875eb20e67f0126a41ff86fcb80d8baed4ab841155dcf7a90d76b5cd1c9`。checkpoint 内 epoch=63（零起点），优化器及 scaler 状态有效。
- `last.pt`、`last_healthy.pt`、`args.yaml`、`results.csv` 均与上午 pause-manifest 中的原件及备份 SHA 一致。
- CSV 为 64 个完整 epoch，真实增强后 assignment 记录连续覆盖 1–64，每份均为 1618 batches。
- `resume_p1_training.py --check-only` 返回 `ready`；只调用一次正式恢复，无 CSV 回退，无重复训练进程。
- 日志确认 `Resuming ... from epoch 65 to 120 total epochs`。18:49:56 运行至 47/1618 batches，loss 有限，stderr 为空。这只证明续训已正常推进，不是实验完成或 P1 达标证明。
- 仅重新启用 Windows 登录恢复任务 `YOLO-Master-A2-P1-V2-Fixed-Resume`；已完成的旧队列保持禁用。

## 调用和输出

工作目录：`E:/desktop/保研+工作/就业/实践/腾讯犀牛鸟/YOLO-Master-A2-Smoke-2026`。

```powershell
$env:PYTHONUTF8='1'
& 'F:/conda-envs/yolo-master/python.exe' -X utf8 scripts/resume_p1_training.py --config configs/p1-v2-fixed-s20260825.resume.json --check-only
& 'F:/conda-envs/yolo-master/python.exe' -X utf8 scripts/resume_p1_training.py --config configs/p1-v2-fixed-s20260825.resume.json
```

- 启动回执：本目录 `latest-resume.json`，本次 PID 36780（以后须重新检查，不能依赖旧 PID）。
- 新日志：`resume-20260914-184913.stdout.log`、`resume-20260914-184913.stderr.log`。
- 原日志及暂停备份不覆盖；备份目录：`F:/YOLO-Master-A2-P1/user-pauses/20260914-0947-p1-v2-fixed-s20260825-stats120`。
- GitHub 检查：证据仓库本地/远端 main 均为 `aba828baa44f6015fd05affcfcaee892034989cb`；PR #274 仍待 review，无新评论。此次续训不重复发布旧成果。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-14T18:49:56+08:00
- Verification Status: ANALYZED — checkpoint consistency and live progress checked; final results pending
- Version Label: resume_20260914_184913
- Execution Status: RUNNING
- Skill contribution: resume checks and provenance recording only; no scientific parameters or acceptance criteria changed
