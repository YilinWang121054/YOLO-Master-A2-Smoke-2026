"""Apply predeclared screening guards without starting a new experiment."""

from __future__ import annotations

import json
import statistics

from evaluate_p1_screen import verify_result
from p1_screen_contract import (
    CONFIG,
    PROJECT,
    configuration,
    now,
    read,
    save,
    sha,
    train_args,
    verify_completion,
    verify_frozen,
)


def eligible(delta, zero_delta, rules):
    return (
        delta["APs"] > rules["minimum_delta_APs"]
        and all(delta[key] >= rules[f"min_delta_{key}"] for key in ("AP", "APm", "APl"))
        and all(
            value <= rules["max_delta_zero_fraction"] + 1e-12
            for value in zero_delta.values()
        )
    )


def summarize_case(case_id):
    config = configuration()
    if not verify_completion(case_id):
        raise ValueError("Incomplete training")
    rows, histograms = [], {}
    for epoch in config["evaluation_epochs"]:
        if not verify_result(case_id, epoch):
            raise ValueError("Missing one of five full validation evaluations")
        rows.append(
            read(PROJECT / "evaluation" / case_id / f"epoch-{epoch:03d}/complete.json")[
                "metrics"
            ]
        )
        branches = read(PROJECT / case_id / "assignment" / f"epoch-{epoch:03d}.json")[
            "branches"
        ]
        for branch, bins in branches.items():
            for size, stages in bins.items():
                item = stages["after_conflict"]
                key = f"{branch}/{size}"
                count, zero = histograms.get(key, (0, 0))
                histograms[key] = (
                    count + item["n"],
                    zero + item["histogram"].get("0", 0),
                )
    if any(n == 0 for n, _ in histograms.values()):
        raise ValueError("No observations for a required area bin")
    return {
        "case": case_id,
        "epochs": config["evaluation_epochs"],
        "per_epoch": rows,
        "mean": {key: statistics.mean(row[key] for row in rows) for key in rows[0]},
        "pooled_zero_fraction": {
            key: zero / n for key, (n, zero) in histograms.items()
        },
    }


def main():
    verify_frozen()
    config = configuration()
    rows = [summarize_case(case["id"]) for case in config["cases"]]
    reference, candidates = rows[0], []
    for row in rows[1:]:
        delta = {
            key: value - reference["mean"][key] for key, value in row["mean"].items()
        }
        zero_delta = {
            key: value - reference["pooled_zero_fraction"][key]
            for key, value in row["pooled_zero_fraction"].items()
        }
        row.update(
            delta=delta,
            zero_delta=zero_delta,
            eligible=eligible(delta, zero_delta, config["selection"]),
        )
        if row["eligible"]:
            candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -round(row["mean"]["APs"], 9),
            train_args(row["case"])["stal_candidate_scale"],
            train_args(row["case"])["stal_topk_small"],
            row["case"],
        )
    )
    result = {
        "observed_at": now(),
        "config_sha256": sha(CONFIG),
        "results": rows,
        "selected": candidates[0]["case"] if candidates else None,
        "status": "screening_complete_pending_review",
        "formal_training_started": False,
        "scope": "Single-seed 20e parameter selection, not independent seeds or P1 acceptance",
    }
    save(PROJECT / "selection.json", result, exclusive=True)
    print(
        json.dumps({"selected": result["selected"], "formal_training_started": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
