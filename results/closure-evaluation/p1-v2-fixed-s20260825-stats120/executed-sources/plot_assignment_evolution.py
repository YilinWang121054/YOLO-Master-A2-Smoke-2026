"""Plot the observed per-epoch P0 assignment trajectory, with source data.

Figure contract: descriptive quantitative chart, one training run, augmented
GT observations (not independent replicates). The three size bins share one
axis. No smoothing, extrapolation or confidence intervals. Report export:
183 x 95 mm, editable SVG/PDF text, all glyphs >=7 pt, plus 600 dpi PNG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assignment_coverage import missing_assignment_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--label", default="P0 locked baseline")
    args = parser.parse_args()
    with (args.run_dir / "results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        completed = [int(row["epoch"]) for row in csv.DictReader(stream)]
    if completed != list(range(1, len(completed) + 1)) or not completed:
        raise ValueError("Training epoch sequence is missing or inconsistent")
    if not args.allow_partial and len(completed) != 120:
        raise ValueError("Final figure requires all 120 completed epochs")
    missing = missing_assignment_epochs(args.run_dir, len(completed))
    if missing:
        raise ValueError(f"Online assignment coverage is incomplete: {missing}")
    records, hashes = [], {}
    for epoch in completed:
        path = args.run_dir / "assignment" / f"epoch-{epoch:03d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for branch, bins in payload["branches"].items():
            for group, stages in bins.items():
                for stage, stats in stages.items():
                    records.append(
                        {
                            "epoch": epoch,
                            "branch": branch,
                            "area_bin": group,
                            "stage": stage,
                            "gt_observations": stats["n"],
                            "mean": stats["mean"],
                            "p50": stats["p50"],
                            "p90": stats["p90"],
                            "zero_fraction": stats["zero_fraction"],
                            "batches": payload["batches"],
                        }
                    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "assignment-source-data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )
    branches = sorted({r["branch"] for r in records})
    files = []
    for branch in branches:
        fig, ax = plt.subplots(figsize=(7.2047244, 3.7401575))  # 183 x 95 mm
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.2, top=0.8)
        for group, color, marker in (
            ("small", "#0072B2", "o"),
            ("medium", "#D55E00", "s"),
            ("large", "#009E73", "^"),
        ):
            series = [
                r
                for r in records
                if r["branch"] == branch
                and r["area_bin"] == group
                and r["stage"] == "after_conflict"
            ]
            assert [r["epoch"] for r in series] == completed
            values = [float("nan") if r["mean"] is None else r["mean"] for r in series]
            ax.plot(
                completed,
                values,
                label=group.capitalize(),
                color=color,
                marker=marker,
                markersize=3,
                markevery=max(1, len(completed) // 20),
                linewidth=1.2,
            )
        ax.set(
            xlabel="Completed training epoch",
            ylabel="Mean final positive anchors per GT",
            ylim=(0, None),
        )
        ax.set_xlim(max(0, completed[0] - 0.3), completed[-1] + 0.3)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, frameon=False)
        fig.suptitle(
            f"{args.label}: assignment evolution ({len(completed)}/120 epochs)",
            fontsize=10,
            y=0.97,
        )
        fig.text(
            0.12,
            0.055,
            "One training run; all augmented training batches. No extrapolation or uncertainty estimates.",
            fontsize=7,
        )
        stem = out / f"positive-count-evolution-{branch}"
        for suffix in (".svg", ".pdf", ".png"):
            path = stem.with_suffix(suffix)
            fig.savefig(path, dpi=600, facecolor="white")
            files.append(path.name)
        plt.close(fig)
    metadata = {
        "run": args.run_dir.name,
        "display_label": args.label,
        "completed_epochs": len(completed),
        "planned_epochs": 120,
        "status": "interim"
        if len(completed) < 120
        else "complete_trajectory_pending_author_review",
        "unit": "GT observations after training augmentation; objects may repeat across epochs",
        "summary": "arithmetic mean of after-conflict positive counts within each epoch and area bin",
        "uncertainty": "none; one run, no independent seed error bars",
        "exclusions": "uncompleted epochs are absent; bins with no GT show a gap rather than a fabricated zero",
        "source_files_sha256": hashes,
        "exports": files,
    }
    (out / "figure-manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    caption = (
        f"{args.label} 正样本演化，已完成 {len(completed)}/120 epoch。曲线为各轮训练增强后 small/medium/large GT "
        "在冲突消解后的平均正样本数。分档按进入 assigner 的面积，小于1024、[1024,9216)、大于等于9216。"
        "所有完成训练batch均计入；对象可跨epoch重复，不能把这些观测当作独立seed。"
        "无平滑、外推或置信区间。均值、P50/P90、零正样本比例及候选扩展前后数据见同目录CSV。\n"
    )
    (out / "caption.md").write_text(caption, encoding="utf-8")
    print(
        json.dumps(
            {"output": str(out), "epochs": len(completed), "files": files},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
