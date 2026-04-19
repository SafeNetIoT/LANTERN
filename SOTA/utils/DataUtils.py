# SOTA/utils/DataUtils.py
import os
import pandas as pd
from datetime import datetime
from typing import List, Generator

def list_csv_files(folder: str) -> List[str]:
    """Return all .csv files (full paths) sorted alphabetically."""
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".csv")
    ]
    return sorted(files)


def list_sequence_folders(base_dir: str, start_date: str, end_date: str) -> List[str]:
    """
    List sub-folders (in YYYY-MM-DD format) between start_date and end_date.
    Returns the absolute paths sorted chronologically.
    """
    dirs = []
    for d in sorted(os.listdir(base_dir)):
        d_path = os.path.join(base_dir, d)
        if not os.path.isdir(d_path):
            continue
        if not (start_date <= d <= end_date):
            continue
        dirs.append(d_path)
    return dirs


def stream_chronological_files(base_dir: str) -> Generator[str, None, None]:
    """
    Yield all CSV files under base_dir chronologically by date and file name.
    Assumes structure like: base_dir/YYYY-MM-DD/window_00_00_01_00.csv
    """
    date_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    for day in date_folders:
        day_path = os.path.join(base_dir, day)
        csv_files = sorted([
            os.path.join(day_path, f)
            for f in os.listdir(day_path)
            if f.endswith(".csv")
        ])
        for f in csv_files:
            yield f


def load_csv_safely(path: str) -> pd.DataFrame:
    """Load a CSV safely, returning an empty DataFrame if unreadable."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Skipping file {path}: {e}")
        return pd.DataFrame()


# ============================================================
# DATA PREPARATION UTILITIES
# ============================================================

def concatenate_blocks(file_list: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files into a single DataFrame.
    Skips broken or empty files automatically.
    """
    dfs = []
    for f in file_list:
        df = load_csv_safely(f)
        if df.empty:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def preprocess_labels(df: pd.DataFrame, label_col="category") -> pd.DataFrame:
    """
    Clean label column — removes empty or NaN category values.
    Returns a filtered copy of df.
    """
    df = df.copy()
    if label_col not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""]
    return df


def window_block_loader(base_dir: str, start_index: int, end_index: int) -> pd.DataFrame:
    """
    Load consecutive blocks by index (0-based) for a specified range.
    Useful for training on sliding windows of size N.
    """
    all_files = list(stream_chronological_files(base_dir))
    if start_index >= len(all_files):
        raise IndexError("start_index exceeds available blocks.")
    selected_files = all_files[start_index:end_index]
    return concatenate_blocks(selected_files)


# ============================================================
# COMPOSITE PIPELINES
# ============================================================

def get_initial_training_window(base_dir: str, num_blocks: int) -> pd.DataFrame:
    """Load the first num_blocks as the training dataset."""
    files = []
    for i, f in enumerate(stream_chronological_files(base_dir)):
        if i >= num_blocks:
            break
        files.append(f)
    print(f"[DataUtils] Loaded {len(files)} files for initial training window.")
    return concatenate_blocks(files)


def get_test_blocks_after(base_dir: str, start_block_index: int) -> List[str]:
    """Return all remaining block paths after start_block_index."""
    all_files = list(stream_chronological_files(base_dir))
    if start_block_index >= len(all_files):
        return []
    return all_files[start_block_index:]


# ============================================================
# ============================================================
# ============================================================
# For the Kyoto Dataset Test
# ============================================================
import os
import glob
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# existing Kyoto feature definitions can stay the same
FEATURE_NUMERIC = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "same_srv_rate",
    "serror_rate",
    "srv_serror_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_src_port_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
]

FEATURE_CATEGORICAL = [
    "service",
    "flag",
    "protocol",
]


def list_kyoto_daily_csv_files(base_dir: str, year: str = "2015") -> list[str]:
    pattern = os.path.join(base_dir, year, "*", "*.csv")
    return sorted(glob.glob(pattern))


def get_kyoto_day_slice(file_list: list[str], start_idx: int, end_idx: int) -> list[str]:
    return file_list[start_idx:end_idx]


def load_kyoto_csv_day(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    return df


def load_kyoto_csv_range(file_list: list[str], verbose: bool = True) -> pd.DataFrame:
    dfs = []
    total = len(file_list)

    for i, fp in enumerate(file_list, start=1):
        if verbose:
            print(f"[LOAD-CSV] {i}/{total}: {fp}")
        dfs.append(load_kyoto_csv_day(fp))

    if len(dfs) == 0:
        raise ValueError("No Kyoto CSV files were loaded.")

    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

from typing import Optional
def take_random_contiguous_slice(
        df: pd.DataFrame, 
        n: int = 10000, 
        seed: Optional[int] = None
        ) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    start_idx = int(rng.integers(0, len(df) - n + 1))
    return df.iloc[start_idx:start_idx + n].reset_index(drop=True)


def build_kyoto_preprocessor() -> ColumnTransformer:
    try:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), FEATURE_NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CATEGORICAL),
            ]
        )
    except TypeError:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), FEATURE_NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), FEATURE_CATEGORICAL),
            ]
        )


def build_kyoto_xy(df: pd.DataFrame, preprocessor: ColumnTransformer):
    X = preprocessor.transform(df).astype(np.float32)
    y = df["group_label"].values.astype(object)
    return X, y


def print_kyoto_day_summary(df: pd.DataFrame, name: str = "dataset") -> None:
    day_min = df["day"].min() if "day" in df.columns else None
    day_max = df["day"].max() if "day" in df.columns else None
    print(f"\n[{name}] shape = {df.shape}")
    print(f"[{name}] day range = {day_min} -> {day_max}")
    print(df["group_label"].value_counts(dropna=False))