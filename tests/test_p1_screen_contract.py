"""Keep the new screening budget, selection boundaries, and recovery honest."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p1_screen_contract as contract
from select_p1_screen import eligible


def test_frozen_design_has_one_reference_and_five_candidates():
    config = contract.configuration()
    assert len(config["cases"]) * config["train"]["epochs"] == 120
    assert [
        contract.train_args(case["id"])["stal_mode"] for case in config["cases"]
    ] == ["fixed"] + ["adaptive"] * 5


@pytest.mark.parametrize(
    "mutation", ["budget", "epochs", "window", "duplicate", "other_parameter", "mode"]
)
def test_reject_invalid_design(mutation):
    config = copy.deepcopy(contract.configuration())
    if mutation == "budget":
        config["training_epoch_budget"] = 121
    elif mutation == "epochs":
        config["train"]["epochs"] = 21
    elif mutation == "window":
        config["evaluation_epochs"] = [20]
    elif mutation == "duplicate":
        config["cases"][1]["id"] = config["cases"][0]["id"]
    elif mutation == "other_parameter":
        config["cases"][1]["overrides"]["batch"] = 2
    else:
        config["cases"][0]["overrides"]["stal_mode"] = "adaptive"
    with pytest.raises(ValueError):
        contract.validate_config(config)


def test_guard_boundaries_and_zero_worsening():
    rules = contract.configuration()["selection"]
    delta = {"APs": 0.1, "AP": -0.3, "APm": -0.5, "APl": -0.5}
    assert eligible(delta, {"small": 0.0}, rules)
    assert not eligible({**delta, "APs": 0.0}, {"small": 0.0}, rules)
    assert not eligible({**delta, "AP": -0.30001}, {"small": 0.0}, rules)
    assert not eligible(delta, {"small": 0.00001}, rules)


def test_exclusive_record_is_atomic_and_cannot_replace(tmp_path):
    target = tmp_path / "record.json"
    contract.save(target, {"original": True}, exclusive=True)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        contract.save(target, {"original": False}, exclusive=True)
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_lock_releases_without_pid_cleanup(tmp_path):
    target = tmp_path / "worker.lock"
    with contract.exclusive_lock(target):  # noqa: SIM117 - hold the first lock while testing re-entry
        with (
            pytest.raises(RuntimeError, match="Already running"),
            contract.exclusive_lock(target),
        ):
            pass
    with contract.exclusive_lock(target):
        pass


@pytest.mark.parametrize(
    "text", ["epoch,loss\n2,1\n", "epoch,loss\n1,nan\n", "epoch,loss\n1,1\n1,1\n"]
)
def test_reject_broken_csv(tmp_path, text):
    (tmp_path / "results.csv").write_text(text)
    with pytest.raises(ValueError):
        contract.csv_rows(tmp_path)
