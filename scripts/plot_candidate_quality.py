"""Render source-bound candidate diagnostics, without rerunning the model.

Figure contract: paired differences plus candidate-quality decomposition;
one fixed checkpoint, 16 train images, all 1321 small GT observations.
No inferential statistics, smoothing, jitter or confidence intervals.
Exports: 183x130 and 183x86 mm, >=7 pt text, editable SVG/PDF, 600 dpi PNG.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ecdf(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("ECDF needs nonempty finite observations")
    if not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Quality metric outside [0,1]")
    x, counts = np.unique(values, return_counts=True)
    y = counts.cumsum() / len(values)
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
    # Repeated observations contribute their full mass; keep the jump at zero.
    return np.r_[0, x, 1], np.r_[0, y, 1]


def paired_rows(records):
    by_mode = {mode: {} for mode in ("fixed", "adaptive")}
    for row in records:
        if row["size"] == "small" and row["mode"] in by_mode:
            key = row["image"], row["gt"]
            if key in by_mode[row["mode"]]:
                raise ValueError("Duplicate GT identity")
            by_mode[row["mode"]][key] = row
    assert by_mode["fixed"].keys() == by_mode["adaptive"].keys()
    result = []
    for (image, index), fixed in sorted(by_mode["fixed"].items()):
        adaptive = by_mode["adaptive"][(image, index)]
        assert fixed["input_area"] == adaptive["input_area"]
        f, a = fixed["counts"], adaptive["counts"]
        result.append(
            {
                "image": image,
                "gt": index,
                "fixed_count": f["after_conflict"],
                "adaptive_count": a["after_conflict"],
                "delta_count": a["after_conflict"] - f["after_conflict"],
                "fixed_weight": f["weight_sum"],
                "adaptive_weight": a["weight_sum"],
                "delta_weight": a["weight_sum"] - f["weight_sum"],
            }
        )
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, stem, panels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text().strip():
            bounds = item.get_window_extent(renderer)
            if (
                bounds.x0 < -1
                or bounds.y0 < -1
                or bounds.x1 > fig.bbox.x1 + 1
                or bounds.y1 > fig.bbox.y1 + 1
            ):
                outside.append(item.get_text())
    if outside:
        raise ValueError(f"Text extends beyond the export canvas: {outside}")
    fig.savefig(output / f"{stem}.svg", dpi=600)
    fig.savefig(output / f"{stem}.pdf", dpi=600)
    fig.savefig(output / f"{stem}.png", dpi=600)
    for letter, ax in panels.items():
        extent = (
            ax.get_tightbbox(renderer)
            .transformed(fig.dpi_scale_trans.inverted())
            .expanded(1.04, 1.04)
        )
        fig.savefig(output / f"{stem}-panel-{letter}.png", dpi=300, bbox_inches=extent)
    size = (fig.get_size_inches() * 25.4).tolist()
    plt.close(fig)
    return {
        "physical_size_mm": size,
        "text_outside_canvas": outside,
        "panels": list(panels),
    }


def panel_title(ax, letter, title):
    ax.set_title(title, loc="left", fontsize=8, pad=9)
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")


def quality_figure(output, paired, selected):
    fig = plt.figure(figsize=(183 / 25.4, 130 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1],
        left=0.11,
        right=0.975,
        bottom=0.20,
        top=0.85,
        wspace=0.55,
        hspace=0.80,
    )
    a, b, c = (
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    )
    a.scatter(
        [r["delta_count"] for r in paired],
        [r["delta_weight"] for r in paired],
        s=10,
        color="#31708E",
        alpha=0.32,
        edgecolors="none",
        rasterized=True,
    )
    a.axhline(0, color="#868686", linewidth=0.7, zorder=0)
    a.axvline(0, color="#868686", linewidth=0.7, zorder=0)
    a.set_xlabel("Change in positive count per GT")
    a.set_ylabel("Change in target-score sum per GT")
    a.margins(0.07)
    panel_title(a, "a", "Paired changes: adaptive − fixed")
    for ax, metric, letter, title, xlabel in (
        (b, "target_weight", "b", "Supervision weight", "Normalized target score"),
        (c, "iou", "c", "Localization quality", "Ordinary IoU"),
    ):
        for inside, color, style in ((True, "#666666", "-"), (False, "#C97725", "--")):
            values = [
                float(r[metric])
                for r in selected
                if bool(int(r["inside_fixed"])) == inside
            ]
            x, y = ecdf(values)
            ax.step(x, y, where="post", color=color, linestyle=style, linewidth=1.35)
        ax.set(
            xlim=(0, 1), ylim=(0, 1.025), xlabel=xlabel, ylabel="Cumulative fraction"
        )
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.5, 1])
        panel_title(ax, letter, title)
    fig.suptitle(
        "Extra positives and supervision under identical predictions",
        x=0.04,
        y=0.975,
        ha="left",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.926,
        "One fixed checkpoint · 16 train images · 1,321 small GTs · LetterBox only",
        fontsize=7,
    )
    handles = [
        Line2D([], [], color="#666666", label="Inside fixed candidates (n = 5,530)"),
        Line2D(
            [],
            [],
            color="#C97725",
            linestyle="--",
            label="New vs fixed candidates (n = 2,874)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.063),
        ncol=2,
        fontsize=7,
    )
    fig.text(
        0.04,
        0.025,
        "a: All paired GTs; b–c: all 8,404 selected adaptive positives. Descriptive only; no independent seed repeats.",
        fontsize=7,
    )
    return save_figure(fig, output, "candidate-quality", {"a": a, "b": b, "c": c})


def geometry_figure(output, focus, candidates):
    image, index = focus["image"], focus["gt"]
    rows = [r for r in candidates if r["image"] == image and int(r["gt"]) == index]
    bbox = np.asarray(focus["input_bbox"], dtype=np.float32)
    xy = np.asarray([[float(r["x"]), float(r["y"])] for r in rows])
    low = np.minimum(xy.min(0), bbox[:2])
    high = np.maximum(xy.max(0), bbox[2:])
    side = float((high - low).max() + 12)
    low = np.maximum((high + low) / 2 - side / 2, 0)
    low = np.minimum(low, 800 - side)
    high = low + side
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 86 / 25.4))
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.24, top=0.76, wspace=0.32)
    stats = {}
    for ax, mode, label, letter in zip(
        axes,
        ("tal", "fixed", "adaptive"),
        ("Pure TAL", "Fixed STAL", "Adaptive STAL"),
        "abc",
    ):
        subset = [r for r in rows if r["mode"] == mode]
        positive = [r for r in subset if int(r["after_conflict"])]
        lost = [
            r
            for r in subset
            if int(r["before_conflict"]) and not int(r["after_conflict"])
        ]
        ax.scatter(
            [float(r["x"]) for r in subset],
            [float(r["y"]) for r in subset],
            s=20,
            facecolors="none",
            edgecolors="#A0A0A0",
            linewidths=0.65,
        )
        points = ax.scatter(
            [float(r["x"]) for r in positive],
            [float(r["y"]) for r in positive],
            c=[float(r["target_weight"]) for r in positive],
            cmap="cividis",
            vmin=0,
            vmax=1,
            s=28,
            edgecolors="#252525",
            linewidths=0.25,
        )
        if lost:
            ax.scatter(
                [float(r["x"]) for r in lost],
                [float(r["y"]) for r in lost],
                marker="x",
                color="#AE5574",
                s=25,
            )
        ax.add_patch(
            Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="#242424",
                linewidth=1,
                linestyle="--",
            )
        )
        ax.set(
            xlim=(low[0], high[0]),
            ylim=(high[1], low[1]),
            aspect="equal",
            xlabel="Input x (px)",
        )
        panel_title(
            ax, letter, f"{label}\n{len(subset)} candidates · {len(positive)} positives"
        )
        stats[mode] = {
            "candidates": len(subset),
            "selected": len(positive),
            "lost_after_preselection": len(lost),
        }
    axes[0].set_ylabel("Input y (px)")
    cax = fig.add_axes([0.935, 0.285, 0.013, 0.41])
    bar = fig.colorbar(points, cax=cax, ticks=[0, 0.5, 1])
    bar.ax.set_title("Target\nscore", fontsize=7, pad=7)
    fig.suptitle(
        "Candidate geometry for a preselected small GT",
        x=0.04,
        y=0.97,
        ha="left",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.91,
        "First listed image, first small GT · Same predictions in all modes · No image pixels displayed",
        fontsize=7,
    )
    handles = [
        Line2D([], [], color="#242424", linestyle="--", label="Original GT"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            color="#A0A0A0",
            label="Candidate",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#555555",
            label="Selected (color = weight)",
        ),
        Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            color="#AE5574",
            label="Lost after preselection",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.045),
        ncol=2,
        fontsize=7,
    )
    result = save_figure(fig, output, "candidate-geometry", dict(zip("abc", axes)))
    result.update(
        image=image,
        gt=index,
        actual_bbox_fp32=bbox.tolist(),
        counts=stats,
        example_selection="First input image; smallest small-GT index, not selected by effect",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    summary = json.loads((args.source / "summary.json").read_text())
    for name, expected in summary["artifacts_sha256"].items():
        assert sha(args.source / name) == expected
    gt = [
        json.loads(line)
        for line in (args.source / "gt-records.jsonl").read_text().splitlines()
    ]
    with gzip.open(
        args.source / "candidate-records.csv.gz", "rt", newline="", encoding="utf-8"
    ) as stream:
        candidates = list(csv.DictReader(stream))
    paired = paired_rows(gt)
    selected = [
        r
        for r in candidates
        if r["mode"] == "adaptive" and r["size"] == "small" and int(r["after_conflict"])
    ]
    assert len(paired) == 1321 and len(selected) == 8404
    assert sum(int(r["inside_fixed"]) for r in selected) == 5530
    first = manifest["inputs"][0]["image"]
    focus = min(
        (
            r
            for r in gt
            if r["image"] == first and r["mode"] == "tal" and r["size"] == "small"
        ),
        key=lambda r: r["gt"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    write_csv(args.output / "paired-small-gt.csv", paired)
    write_csv(args.output / "selected-small-candidates.csv", selected)
    write_csv(
        args.output / "geometry-example-candidates.csv",
        [r for r in candidates if r["image"] == first and int(r["gt"]) == focus["gt"]],
    )
    quality = quality_figure(args.output, paired, selected)
    geometry = geometry_figure(args.output, focus, candidates)
    result = {
        "backend": "Python/matplotlib",
        "matplotlib_version": matplotlib.__version__,
        "script_sha256": sha(Path(__file__)),
        "source_data_sha256": summary["artifacts_sha256"],
        "source_summary_sha256": sha(args.source / "summary.json"),
        "selection_counts": {
            "unique_gt_before": len(gt) // 3,
            "small_gt_plotted": len(paired),
            "other_gt_not_plotted": len(gt) // 3 - len(paired),
            "reason": "A2 small-object diagnostic; all other sizes retained in source tables",
            "adaptive_small_selected": len(selected),
            "ecdf_zero_values_retained": True,
        },
        "quality": quality,
        "geometry": geometry,
        "outputs_sha256": {
            p.name: sha(p) for p in args.output.iterdir() if p.is_file()
        },
        "scope": "Rendered existing records; no model inference or training, no causal or acceptance verdict",
    }
    (args.output / "figure-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "figure_count": 2,
                "small_gt": len(paired),
                "selected_pairs": len(selected),
            }
        )
    )


if __name__ == "__main__":
    main()
