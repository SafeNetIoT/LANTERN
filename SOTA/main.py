import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# SOTA Implementations
import json
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from collections import deque

from utils import DataUtils
from utils.ModelUtils import (
    TFIDFTextEncoder,
    AutoencoderClassifierTrainer,
    RFModelTrainer,
    ContrastiveModelTrainer
)

from utils.DriftUtils import (
    run_drift_detection,
    compute_cade_reference, cade_fast_detect,
    _owad_fit_calibrator, owad_run,
    compute_chen_reference, chen_fast_detect
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# =========================================================
# Static Experiment Pipeline
# =========================================================
def run_static_experiment(
    data_dir,
    output_csv,
    model_trainer_cls,
    start_date=None,
    end_date=None,
    window_days=8,
    blocks_per_day=24,
    max_features=2000,
    drift_methods=None,
):

    N_BLOCKS = window_days * blocks_per_day
    metric_log = []

    # -----------------------------------------------------
    # Load files
    # -----------------------------------------------------
    if start_date and end_date:
        print(f"Restricting to date range {start_date} → {end_date}")
        all_folders = DataUtils.list_sequence_folders(data_dir, start_date, end_date)
        all_files = []
        for f in all_folders:
            all_files.extend(DataUtils.list_csv_files(f))
        file_stream = iter(sorted(all_files))
    else:
        print("Using all chronological files...")
        file_stream = DataUtils.stream_chronological_files(data_dir)

    window_files = deque(maxlen=N_BLOCKS)
    print(f"Initializing {N_BLOCKS}-block training window...")

    for _ in tqdm(range(N_BLOCKS)):
        try:
            window_files.append(next(file_stream))
        except StopIteration:
            raise RuntimeError("Not enough blocks in range")

    # -----------------------------------------------------
    # Load training window
    # -----------------------------------------------------
    dfs_train = []
    for f in window_files:
        try:
            df_tmp = pd.read_csv(f, engine="c", on_bad_lines="skip")
            dfs_train.append(df_tmp)
        except Exception:
            try:
                df_tmp = pd.read_csv(f, engine="python", on_bad_lines="skip")
                dfs_train.append(df_tmp)
            except Exception:
                print(f"[WARN] Skipping malformed file {f}")

    if not dfs_train:
        raise RuntimeError("Training data empty")

    df_train = pd.concat(dfs_train, ignore_index=True)
    df_train = df_train.dropna(subset=["category"])
    df_train = df_train[df_train["category"].astype(str).str.strip() != ""]
    df_train["category"] = df_train["category"].astype(str)

    # -----------------------------------------------------
    # Train base model
    # -----------------------------------------------------
    encoder = TFIDFTextEncoder(max_features=max_features)
    X_train = encoder.fit_transform(df_train)
    y_train = df_train["category"].values

    model_trainer = model_trainer_cls()
    model_trainer.fit(X_train, y_train)

    metrics_train, train_recon_errors = model_trainer.evaluate(X_train, y_train)
    train_recon_errors = np.array(train_recon_errors)

    # =========================================================
    # Build Drift Baselines
    # =========================================================

    # CADE baseline
    ref_stats_cade = None
    if "cade" in drift_methods:
        print("[CADE] Building reference...")
        z_train = model_trainer.encode(X_train)
        if hasattr(model_trainer, "encode_labels"):
            y_enc = model_trainer.encode_labels(y_train)
        else:
            _, y_enc = np.unique(y_train, return_inverse=True)
        ref_stats_cade = compute_cade_reference(z_train, y_enc)

    # OWAD baseline
    owad_calibrator = None
    p_ctrl = None
    owad_ctrl_stats = {}

    if "owad" in drift_methods:
        print("[OWAD] Fitting calibrator...")
        owad_calibrator = _owad_fit_calibrator(train_recon_errors)
        p_ctrl = owad_calibrator(train_recon_errors)
        owad_ctrl_stats = {
            "owad_ctrl_mean": p_ctrl.mean(),
            "owad_ctrl_std": p_ctrl.std()
        }

    # Chen baseline
    chen_ref_stats = None
    if "chen" in drift_methods:
        print("[CHEN] Building reference pseudo-loss...")
        chen_ref_stats = compute_chen_reference(model_trainer, X_train, y_train)

    # =========================================================
    # Evaluate future blocks
    # =========================================================
    block_index = N_BLOCKS + 1

    for seq_file in tqdm(file_stream, desc="Evaluating"):
        try:
            df_test = pd.read_csv(seq_file, low_memory=False)
        except Exception as e:
            print(f"[WARN] Skipping {seq_file}: {e}")
            continue

        df_test = df_test.dropna(subset=["category"])
        df_test = df_test[df_test["category"].astype(str).str.strip() != ""]
        if df_test.empty:
            continue
        df_test["category"] = df_test["category"].astype(str)

        X_block = encoder.transform(df_test)
        y_block = df_test["category"].values

        metrics, recon_errors = model_trainer.evaluate(X_block, y_block)
        recon_errors = np.array(recon_errors)
        if len(recon_errors) == 0:
            continue

        # -----------------------------------------------------
        # CADE
        # -----------------------------------------------------
        if "cade" in drift_methods:
            z_test = model_trainer.encode(X_block)
            drift, score = cade_fast_detect(z_test, ref_stats_cade)
            metrics["drift_cade_detected"] = drift
            metrics["drift_cade_score"] = score

        # -----------------------------------------------------
        # OWAD
        # -----------------------------------------------------
        if "owad" in drift_methods:
            drift, pval = owad_run(train_recon_errors, recon_errors)
            metrics["drift_owad_detected"] = drift
            metrics["drift_owad_score"] = pval
            metrics.update(owad_ctrl_stats)

        # -----------------------------------------------------
        # Chen
        # -----------------------------------------------------
        if "chen" in drift_methods:
            drift, score = chen_fast_detect(model_trainer, X_block, y_block, chen_ref_stats)
            metrics["drift_chen_detected"] = drift
            metrics["drift_chen_score"] = score

        # -----------------------------------------------------
        # KL & Mateen via unified interface
        # -----------------------------------------------------
        other_methods = [m for m in drift_methods if m not in ["cade", "owad", "chen"]]
        if other_methods:
            results = run_drift_detection(train_recon_errors, recon_errors, other_methods)
            for m, (drift, score) in results.items():
                metrics[f"drift_{m}_detected"] = drift
                metrics[f"drift_{m}_score"] = score

        metrics.update({
            "block_index": block_index,
            "file": seq_file,
            "model": model_trainer_cls.__name__
        })
        metric_log.append(metrics)
        block_index += 1

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    metric_df.to_csv(output_csv, index=False)
    print(f"Saved metrics → {output_csv}")
    return metric_df


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run static TF-IDF experiment.")
    parser.add_argument("--start", type=str, default="2025-03-15")
    parser.add_argument("--end", type=str, default="2025-10-15")
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--model", type=str, default="cae")
    parser.add_argument("--out", type=str, default="../data/res/static.csv")
    parser.add_argument("--drift_methods", type=str, default="cade")

    args = parser.parse_args()

    model_map = {
        "rf": RFModelTrainer,
        "ae": AutoencoderClassifierTrainer,
        "cae": ContrastiveModelTrainer
    }

    model_cls = model_map.get(args.model.lower(), RFModelTrainer)
    drift_methods = [m.strip() for m in args.drift_methods.split(",") if m.strip()]

    run_static_experiment(
        data_dir="../data/sequences",
        output_csv=args.out,
        model_trainer_cls=model_cls,
        start_date=args.start,
        end_date=args.end,
        window_days=args.days,
        drift_methods=drift_methods,
    )
