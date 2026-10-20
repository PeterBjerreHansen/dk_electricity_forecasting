from __future__ import annotations

import json
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

import dkenergy_databricks
import dkenergy_forecast
from dkenergy_forecast.models.catboost_experimental import (
    FittedCatBoost,
    load_catboost,
    predict_catboost_regressor,
)


class DayAheadPriceModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper around a fitted CatBoost day-ahead price model.

    Input is the target-free feature frame from ``build_price_feature_frame``;
    output is one ``y_pred`` in DKK/MWh per row. Residual-mode models add the
    weighted-median baseline back inside ``predict``.
    """

    def load_context(self, context: Any) -> None:
        contract = json.loads(Path(context.artifacts["contract"]).read_text(encoding="utf-8"))
        CatBoostRegressor, _ = load_catboost()
        model = CatBoostRegressor()
        model.load_model(context.artifacts["catboost_model"])
        self._fitted = FittedCatBoost(model=model, **contract)

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        prediction = predict_catboost_regressor(self._fitted, model_input)
        return pd.DataFrame({"y_pred": prediction.to_numpy()})


def model_input_columns(fitted: FittedCatBoost) -> list[str]:
    columns = list(fitted.feature_columns)
    if fitted.target_mode == "residual_baseline" and fitted.residual_baseline_column not in columns:
        columns.append(fitted.residual_baseline_column)
    return columns


def log_day_ahead_model(
    fitted: FittedCatBoost,
    *,
    input_example: pd.DataFrame,
    registered_model_name: str | None = None,
) -> Any:
    """Log the model to the active MLflow run, optionally registering it."""

    columns = model_input_columns(fitted)
    example = input_example[columns].head(48).reset_index(drop=True)
    output_example = pd.DataFrame({"y_pred": predict_catboost_regressor(fitted, example).to_numpy()})
    contract = {
        "feature_columns": fitted.feature_columns,
        "cat_features": fitted.cat_features,
        "target_mode": fitted.target_mode,
        "residual_baseline_column": fitted.residual_baseline_column,
        "training_rows": fitted.training_rows,
    }
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.cbm"
        contract_path = Path(tmp) / "contract.json"
        fitted.model.save_model(str(model_path))
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=DayAheadPriceModel(),
            artifacts={"catboost_model": str(model_path), "contract": str(contract_path)},
            code_paths=[
                str(Path(dkenergy_forecast.__file__).parent),
                str(Path(dkenergy_databricks.__file__).parent),
            ],
            pip_requirements=_pip_requirements(),
            signature=infer_signature(example, output_example),
            input_example=example,
            registered_model_name=registered_model_name,
        )


def _pip_requirements() -> list[str]:
    requirements = [
        f"{package}=={metadata.version(package)}"
        for package in ["catboost", "pandas", "pyarrow"]
    ]
    for mlflow_distribution in ["mlflow", "mlflow-skinny"]:
        try:
            requirements.append(f"{mlflow_distribution}=={metadata.version(mlflow_distribution)}")
            break
        except metadata.PackageNotFoundError:
            continue
    return requirements
