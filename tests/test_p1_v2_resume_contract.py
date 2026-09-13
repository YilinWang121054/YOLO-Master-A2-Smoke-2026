"""Accept only the trainer's known device serialization, never protocol changes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p1_v2_fixed import validate_train_args


def test_device_serialization_is_not_protocol_drift():
    validate_train_args({"device": 0, "batch": 4}, {"device": "0", "batch": 4})


@pytest.mark.parametrize("record", [{"device": "1", "batch": 4}, {"device": "0", "batch": 2}, {"device": "0"}])
def test_reject_changed_or_missing_protocol(record):
    with pytest.raises(ValueError):
        validate_train_args({"device": 0, "batch": 4}, record)
