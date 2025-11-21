# utils/DataUtils.py
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
