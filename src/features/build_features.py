"""Feature engineering pipeline: RUL target calculation, sensor filtering, and scaling."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sensors identified as near-constant during EDA — adjust based on your own findings
LOW_VARIANCE_SENSORS = [
    "sensor_measurement_1",
    "sensor_measurement_5",
    "sensor_measurement_6",
    "sensor_measurement_10",
    "sensor_measurement_16",
    "sensor_measurement_18",
    "sensor_measurement_19",
]

RUL_CLIP_CEILING = 125


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Remaining Useful Life (RUL) target for each row in the training set."""
    max_cycle_per_unit = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle_per_unit - df["time_in_cycles"]
    return df


def apply_piecewise_rul(df: pd.DataFrame, ceiling: int = RUL_CLIP_CEILING) -> pd.DataFrame:
    """Clip the RUL target at a fixed ceiling (piecewise linear degradation assumption)."""
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=ceiling)
    return df


def drop_low_variance_sensors(df: pd.DataFrame, sensors_to_drop: list[str] = LOW_VARIANCE_SENSORS) -> pd.DataFrame:
    """Remove sensors identified as non-informative (near-zero variance) during EDA."""
    return df.drop(columns=sensors_to_drop, errors="ignore")


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]):
    """Fit a MinMaxScaler on training data and apply it to both train and test sets."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    return train_df, test_df, scaler


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Full feature engineering pipeline for the FD001 subset."""
    train_df = compute_rul(train_df)
    train_df = apply_piecewise_rul(train_df)

    train_df = drop_low_variance_sensors(train_df)
    test_df = drop_low_variance_sensors(test_df)

    feature_columns = [
        c for c in train_df.columns
        if c not in ("unit_number", "time_in_cycles", "RUL")
    ]

    train_df, test_df, scaler = scale_features(train_df, test_df, feature_columns)

    return train_df, test_df, scaler, feature_columns


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data.make_dataset import load_fd001

    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    PROCESSED_DIR.mkdir(exist_ok=True)

    data = load_fd001(RAW_DIR)
    train_processed, test_processed, scaler, feature_cols = build_features(data["train"], data["test"])

    train_processed.to_csv(PROCESSED_DIR / "train_fd001_processed.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test_fd001_processed.csv", index=False)
    joblib.dump(scaler, "models/scaler_fd001.joblib")

    print(f"Processed train shape: {train_processed.shape}")
    print(f"RUL range after clipping: {train_processed['RUL'].min()} to {train_processed['RUL'].max()}")
    print(f"Features used: {feature_cols}")
