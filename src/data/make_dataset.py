"""Data ingestion module for the NASA C-MAPSS turbofan dataset (FD001 subset)."""

from pathlib import Path
import pandas as pd

# Column names as documented by NASA for the C-MAPSS dataset
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = [f"operational_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_measurement_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_raw_file(file_path: Path) -> pd.DataFrame:
    """Load a single raw C-MAPSS .txt file into a labeled DataFrame.

    The raw files are whitespace-separated and have no header row.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # drop trailing empty columns from double spaces
    df.columns = COLUMN_NAMES
    return df


def load_rul_file(file_path: Path) -> pd.DataFrame:
    """Load the RUL ground-truth file (one RUL value per test engine)."""
    rul = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    rul.columns = ["RUL"]
    rul["unit_number"] = rul.index + 1
    return rul


def load_fd001(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load train, test, and RUL data for the FD001 subset.

    Returns a dict with keys: 'train', 'test', 'rul'.
    """
    train_df = load_raw_file(raw_dir / "train_FD001.txt")
    test_df = load_raw_file(raw_dir / "test_FD001.txt")
    rul_df = load_rul_file(raw_dir / "RUL_FD001.txt")
    return {"train": train_df, "test": test_df, "rul": rul_df}


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    data = load_fd001(RAW_DIR)

    print(f"Train shape: {data['train'].shape}")
    print(f"Test shape: {data['test'].shape}")
    print(f"RUL shape: {data['rul'].shape}")
    print(data["train"].head())