import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

KYOTO_COLUMNS = [
    "duration",
    "service",
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
    "flag",
    "ids_detection",
    "malware_detection",
    "ashula_detection",
    "label",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "start_time",
    "protocol",
]

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

AUX_NUMERIC = [
    "src_port",
    "dst_port",
    "label",
]

CATEGORICAL_COLS = [
    "service",
    "flag",
    "protocol",
]

RAW_BASE = "../../data/public_datasets/kyoto"
OUT_BASE = "../../data/public_datasets/kyoto_csv"
YEAR = "2015"


def list_kyoto_txt_files(base_dir: str, year: str = "2015") -> list[str]:
    pattern = os.path.join(base_dir, year, "*", "*.txt")
    return sorted(glob.glob(pattern))


def read_kyoto_txt(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=KYOTO_COLUMNS,
        low_memory=False,
    )


def clean_kyoto_df(df: pd.DataFrame, day_str: str) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = FEATURE_NUMERIC + AUX_NUMERIC
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_NUMERIC + CATEGORICAL_COLS + ["label"]).reset_index(drop=True)

    df["group_label"] = df["label"].map(lambda x: "benign" if x == 1 else "malicious")
    df["day"] = pd.to_datetime(day_str, format="%Y%m%d", errors="coerce")

    return df


def txt_to_csv_path(txt_path: str, raw_base: str, out_base: str) -> str:
    rel = os.path.relpath(txt_path, raw_base)
    csv_rel = os.path.splitext(rel)[0] + ".csv"
    return os.path.join(out_base, csv_rel)


def convert_one_file(txt_path: str, raw_base: str, out_base: str) -> tuple[str, int]:
    day_str = os.path.splitext(os.path.basename(txt_path))[0]

    df = read_kyoto_txt(txt_path)
    df = clean_kyoto_df(df, day_str)

    out_path = txt_to_csv_path(txt_path, raw_base, out_base)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    return out_path, len(df)


def main():
    txt_files = list_kyoto_txt_files(RAW_BASE, YEAR)
    print(f"[INFO] Found {len(txt_files)} txt files")

    rows = []

    for txt_path in tqdm(txt_files, desc="Converting Kyoto", unit="file", dynamic_ncols=True):
        try:
            out_path, n_rows = convert_one_file(txt_path, RAW_BASE, OUT_BASE)
            rows.append({
                "txt_path": txt_path,
                "csv_path": out_path,
                "rows": n_rows,
                "status": "ok",
            })
        except Exception as e:
            rows.append({
                "txt_path": txt_path,
                "csv_path": "",
                "rows": 0,
                "status": f"error: {e}",
            })

    log_df = pd.DataFrame(rows)
    log_path = os.path.join(OUT_BASE, f"conversion_log_{YEAR}.csv")
    os.makedirs(OUT_BASE, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print(f"[INFO] Saved conversion log to {log_path}")
    print(log_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()