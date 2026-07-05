from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_forecast.tuning import recency_sample_weights, suggest_catboost_params


def test_tuning_recency_sample_weights_support_floor() -> None:
    frame = pd.DataFrame(
        {
            "forecast_origin_utc": [
                pd.Timestamp("2024-01-10T00:00:00Z"),
                pd.Timestamp("2024-01-05T00:00:00Z"),
            ]
        }
    )

    weights = recency_sample_weights(
        frame,
        reference_origin=pd.Timestamp("2024-01-10T00:00:00Z"),
        half_life_days=5,
        floor=0.2,
    )

    assert weights is not None
    assert weights.tolist() == pytest.approx([1.0, 0.6])


def test_conservative_catboost_search_profile_uses_stricter_ranges() -> None:
    trial = RecordingTrial(bootstrap_type="Bernoulli")

    params = suggest_catboost_params(
        trial,
        feature_count=8,
        random_seed=7,
        max_iterations=500,
        has_time=True,
        search_profile="conservative",
    )

    assert trial.int_calls["depth"] == (3, 6, 1)
    assert trial.float_calls["learning_rate"] == (0.01, 0.08, True)
    assert trial.float_calls["l2_leaf_reg"] == (10.0, 200.0, True)
    assert trial.float_calls["random_strength"] == (2.0, 20.0, False)
    assert trial.float_calls["rsm"] == (0.60, 0.95, False)
    assert trial.categorical_calls["border_count"] == [32, 64, 128]
    assert params["has_time"] is True
    assert params["subsample"] == pytest.approx(0.65)
    assert params["rsm"] == pytest.approx(0.60)


def test_catboost_search_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="search_profile"):
        suggest_catboost_params(
            RecordingTrial(),
            feature_count=1,
            random_seed=1,
            max_iterations=500,
            search_profile="wild",
        )


def test_catboost_search_requires_iteration_budget_that_matches_space() -> None:
    with pytest.raises(ValueError, match="at least 300"):
        suggest_catboost_params(
            RecordingTrial(),
            feature_count=1,
            random_seed=1,
            max_iterations=200,
        )


class RecordingTrial:
    def __init__(self, *, bootstrap_type: str = "Bayesian") -> None:
        self.bootstrap_type = bootstrap_type
        self.int_calls: dict[str, tuple[int, int, int]] = {}
        self.float_calls: dict[str, tuple[float, float, bool]] = {}
        self.categorical_calls: dict[str, list[object]] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.categorical_calls[name] = list(choices)
        if name == "bootstrap_type":
            return self.bootstrap_type
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        self.int_calls[name] = (low, high, step)
        return low

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        self.float_calls[name] = (low, high, log)
        return low
