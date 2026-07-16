from __future__ import annotations

import sys

import pytest

from scripts import run_experiment


def test_full_profile_cannot_run_in_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_prepare(*args: object, **kwargs: object) -> object:
        pytest.fail("foreground full run reached experiment preparation")

    monkeypatch.setattr(
        run_experiment,
        "prepare_experiment",
        unexpected_prepare,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "run",
            "config/experiments/chronos_a4_combined.json",
            "--profile",
            "full",
        ],
    )

    with pytest.raises(SystemExit, match="use .* launch .* --profile full"):
        run_experiment.main()
