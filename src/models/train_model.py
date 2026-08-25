"""Baseline model training and evaluation for RUL prediction (FD001 subset)."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.make_dataset import load_rul_file


def get_last_cycle_per_unit(test_df: pd.DataFrame) -> pd.DataFrame:
    """Select only the last recorded cycle for each engine in the test set.

    This matches the evaluation protocol of the official C-MAPSS RUL ground truth file.
    """
    return test_df.sort_values("time_in_cycles").groupby("unit_number").tail(1)


def train_baseline_model(train_df: pd.DataFrame, feature_columns: list[str]) -> LinearRegression:
    """Train a simple Linear Regression baseline on the engineered training features."""
    X_train = train_df[feature_columns]
    y_train = train_df["RUL"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, test_last_cycle: pd.DataFrame, feature_columns: list[str], rul_ground_truth: pd.DataFrame):
    """Evaluate the model against the official C-MAPSS test ground truth."""
    X_test = test_last_cycle[feature_columns]
    y_pred = model.predict(X_test)

    merged = test_last_cycle[["unit_number"]].copy()
    merged["predicted_RUL"] = y_pred
    merged = merged.merge(rul_ground_truth, on="unit_number")

    rmse = np.sqrt(mean_squared_error(merged["RUL"], merged["predicted_RUL"]))
    mae = mean_absolute_error(merged["RUL"], merged["predicted_RUL"])

    return rmse, mae, merged


if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed")
    RAW_DIR = Path("data/raw")
    MODELS_DIR = Path("models")

    train_df = pd.read_csv(PROCESSED_DIR / "train_fd001_processed.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_fd001_processed.csv")
    rul_ground_truth = load_rul_file(RAW_DIR / "RUL_FD001.txt")

    feature_columns = [c for c in train_df.columns if c not in ("unit_number", "time_in_cycles", "RUL")]

    model = train_baseline_model(train_df, feature_columns)

    test_last_cycle = get_last_cycle_per_unit(test_df)
    rmse, mae, results_df = evaluate_model(model, test_last_cycle, feature_columns, rul_ground_truth)

    print(f"Baseline Linear Regression — RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    joblib.dump(model, MODELS_DIR / "baseline_linear_regression.joblib")
    results_df.to_csv(PROCESSED_DIR / "baseline_predictions.csv", index=False)