import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
import argparse
from datetime import datetime, timedelta

from utils import DataUtils
from utils.ModelUtils import TFIDFTextEncoder, ContrastiveModelTrainer
from utils.DriftUtils import (
    compute_entropy_reference, compute_entropy_score, detect_entropy_drift,
    compute_lmt_reference, compute_lmt_block
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def save_json_safely(obj, path):
    def make_safe(o):
        if isinstance(o, dict):
            return {str(k): make_safe(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [make_safe(v) for v in o]
        elif isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, (np.int32, np.int64)):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return o

    obj_safe = make_safe(obj)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj_safe, f, indent=4)
    print(f"[SAVE] JSON saved -> {path}")


def list_files_in_range(data_dir, start_date, end_date):
    folders = DataUtils.list_sequence_folders(data_dir, start_date, end_date)
    all_files = []
    for folder in folders:
        all_files.extend(DataUtils.list_csv_files(folder))
    return sorted(all_files)


def compute_train_end_date(train_start_date, window_days):
    start_dt = datetime.strptime(train_start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=window_days - 1)
    return end_dt.strftime("%Y-%m-%d")


def train_window_from_files(train_files, max_features, retrain_id, output_csv):
    df_train = DataUtils.concatenate_blocks(train_files)
    df_train = DataUtils.preprocess_labels(df_train)
    if df_train.empty:
        raise RuntimeError("Empty training set")

    encoder = TFIDFTextEncoder(max_features=max_features)
    X_train = encoder.fit_transform(df_train)
    y_train = df_train["category"].values

    model_trainer = ContrastiveModelTrainer()
    model_trainer.fit(X_train, y_train)

    print(f"[TRAIN] Model trained on {len(df_train)} samples from {len(train_files)} blocks")

    entropy_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

    entropy_dir = os.path.join(os.path.dirname(output_csv), "entropy_refs")
    entropy_path = os.path.join(entropy_dir, f"entropy_ref_{retrain_id:03d}.json")
    save_json_safely(entropy_ref, entropy_path)

    lmt_dir = os.path.join(os.path.dirname(output_csv), "lmt_refs")
    lmt_path = os.path.join(lmt_dir, f"lmt_ref_{retrain_id:03d}.json")
    save_json_safely(lmt_ref, lmt_path)

    return encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref


def evaluate_static_window_sensitivity(
    data_dir,
    output_csv,
    train_start_date="2025-03-16",
    test_start_date="2025-04-15",
    test_end_date="2025-06-15",
    window_days=8,
    max_features=2000,
):
    retrain_id = 0
    metric_log = []

    train_end_date = compute_train_end_date(train_start_date, window_days)

    print("=" * 80)
    print(f"[CONFIG] window_days     = {window_days}")
    print(f"[CONFIG] train_start_date = {train_start_date}")
    print(f"[CONFIG] train_end_date   = {train_end_date}")
    print(f"[CONFIG] test_start_date  = {test_start_date}")
    print(f"[CONFIG] test_end_date    = {test_end_date}")
    print("=" * 80)

    # -------------------------------------------------
    # 1. Load training files only from the requested range
    # -------------------------------------------------
    train_files = list_files_in_range(data_dir, train_start_date, train_end_date)
    if len(train_files) == 0:
        raise RuntimeError(f"No training files found for {train_start_date} -> {train_end_date}")

    encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref = train_window_from_files(
        train_files=train_files,
        max_features=max_features,
        retrain_id=retrain_id,
        output_csv=output_csv
    )

    # -------------------------------------------------
    # 2. Load testing files from the SAME fixed test range
    # -------------------------------------------------
    test_files = list_files_in_range(data_dir, test_start_date, test_end_date)
    if len(test_files) == 0:
        raise RuntimeError(f"No test files found for {test_start_date} -> {test_end_date}")

    print(f"[TEST] Evaluating on {len(test_files)} blocks")

    recent_lmt_decisions = deque(maxlen=10)

    for block_index, seq_file in enumerate(tqdm(test_files, desc=f"Testing {window_days}d"), start=1):
        df_block = DataUtils.concatenate_blocks([seq_file])
        df_block = DataUtils.preprocess_labels(df_block)
        if df_block.empty:
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values

        metrics, recon_errors = model_trainer.evaluate(X_block, y_block)
        if len(recon_errors) == 0:
            continue

        # ------------------------------
        # Entropy
        # ------------------------------
        entropy_score, entropy_drift_ratio, _ = compute_entropy_score(
            model_trainer, X_block, entropy_ref, use_mad=False
        )
        entropy_mon, entropy_dec = detect_entropy_drift(
            entropy_score, entropy_ref, use_mad=False
        )

        # ------------------------------
        # LMT
        # ------------------------------
        mean_lmt_score, lmt_sample_ratio, per_class_scores, lmt_mon, lmt_dec = compute_lmt_block(
            model_trainer, X_block, y_block, baseline_shapes, lmt_ref, use_mad=True
        )

        recent_lmt_decisions.append(lmt_dec)
        lmt_count = sum(recent_lmt_decisions)

        trigger_entropy_lmt = entropy_dec and lmt_dec
        trigger_lmt_stable = (lmt_count >= 8)

        # static case
        decision_any = 0

        if trigger_entropy_lmt and not trigger_lmt_stable:
            retrain_trigger = "entropy_lmt_both"
        elif trigger_lmt_stable and not trigger_entropy_lmt:
            retrain_trigger = f"lmt_stable({lmt_count}/10)"
        elif trigger_entropy_lmt and trigger_lmt_stable:
            retrain_trigger = f"both_conditions({lmt_count}/10)"
        else:
            retrain_trigger = "none"

        metrics.update({
            "window_days": window_days,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "test_start_date": test_start_date,
            "test_end_date": test_end_date,
            "block_index": block_index,
            "file": seq_file,
            "retrain_id": retrain_id,

            "entropy_score": entropy_score,
            "entropy_drift_ratio": entropy_drift_ratio,
            "entropy_monitor": entropy_mon,
            "entropy_decision": entropy_dec,

            "lmt_mean_score": mean_lmt_score,
            "lmt_sample_drift_ratio": lmt_sample_ratio,
            **{f"lmt_{c}_score": v for c, v in per_class_scores.items()},
            "lmt_monitor": lmt_mon,
            "lmt_decision": lmt_dec,

            "trigger_entropy_lmt": int(trigger_entropy_lmt),
            "trigger_lmt_stable": int(trigger_lmt_stable),
            "retrain_trigger_source": retrain_trigger,
            "decision_any": int(decision_any),
        })

        metric_log.append(metrics)

    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    metric_df.to_csv(output_csv, index=False)
    print(f"[SAVE] Metrics saved -> {output_csv}")

    return metric_df


def run_window_sensitivity_suite(
    data_dir,
    out_dir,
    train_start_date="2025-03-16",
    test_start_date="2025-04-15",
    test_end_date="2025-06-15",
    window_days_list=None,
    max_features=2000,
):
    if window_days_list is None:
        window_days_list = [2, 4, 8, 16, 30]

    all_results = []

    for wd in window_days_list:
        output_csv = os.path.join(out_dir, f"window_{wd}d_metrics.csv")
        df = evaluate_static_window_sensitivity(
            data_dir=data_dir,
            output_csv=output_csv,
            train_start_date=train_start_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            window_days=wd,
            max_features=max_features,
        )
        all_results.append(df)

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(out_dir, "window_sensitivity_all_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVE] Combined metrics saved -> {summary_path}")

    # block-level summary for quick comparison
    agg_cols = {
        "accuracy": "mean",
        "f1": "mean",
        "precision": "mean",
        "recall": "mean",
        "entropy_score": "mean",
        "entropy_drift_ratio": "mean",
        "lmt_mean_score": "mean",
        "lmt_sample_drift_ratio": "mean",
        "entropy_monitor": "mean",
        "entropy_decision": "mean",
        "lmt_monitor": "mean",
        "lmt_decision": "mean",
    }

    existing_agg_cols = {k: v for k, v in agg_cols.items() if k in summary_df.columns}

    quick_summary = (
        summary_df.groupby("window_days", as_index=False)
        .agg(existing_agg_cols)
        .sort_values("window_days")
    )

    quick_summary_path = os.path.join(out_dir, "window_sensitivity_summary.csv")
    quick_summary.to_csv(quick_summary_path, index=False)
    print(f"[SAVE] Summary saved -> {quick_summary_path}")

    return summary_df, quick_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static window sensitivity experiment")
    parser.add_argument("--data", type=str, default="../data/sequences", help="Base data directory")
    parser.add_argument("--train_start", type=str, default="2025-03-16", help="Training start date")
    parser.add_argument("--test_start", type=str, default="2025-04-15", help="Testing start date")
    parser.add_argument("--test_end", type=str, default="2025-06-15", help="Testing end date")
    parser.add_argument("--out_dir", type=str, default="../data/res/window_sensitivity", help="Output directory")
    parser.add_argument("--days_list", type=str, default="2,4,8,16,30", help="Comma-separated window sizes")
    parser.add_argument("--max_features", type=int, default=2000, help="TF-IDF max features")
    args = parser.parse_args()

    window_days_list = [int(x.strip()) for x in args.days_list.split(",") if x.strip()]

    run_window_sensitivity_suite(
        data_dir=args.data,
        out_dir=args.out_dir,
        train_start_date=args.train_start,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        window_days_list=window_days_list,
        max_features=args.max_features,
    )