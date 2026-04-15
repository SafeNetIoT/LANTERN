import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
import argparse
import gc

from utils import DataUtils
from utils.ModelUtils import TFIDFTextEncoder, ContrastiveModelTrainer
from utils.DriftUtils import (
    compute_entropy_reference,
    compute_lmt_reference,
    compute_reference_block_scores,
    compute_reference_z_scores,
    compute_nu,
    compute_block_drift_evidence,
    update_sequential_state,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# =========================================================
# Helper to safely store JSON with numpy types
# =========================================================
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
    print(f"[SAVE] Reference saved → {path}")


def safe_delete(local_vars, *names):
    for name in names:
        if name in local_vars:
            del local_vars[name]


def cleanup_memory(tag=None):
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if tag:
        print(f"[MEM] Cleanup: {tag}")


def log_ram(tag):
    if psutil is None:
        return
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[RAM] {tag}: RSS={rss_gb:.2f} GiB")


# =========================================================
# Main Experiment
# =========================================================
def run_experiment_periodic(
    data_dir,
    output_csv,
    start_date=None,
    end_date=None,
    window_days=8,
    blocks_per_day=24,
    max_features=2000,
    h_value=8.0,
    nu_method="3sigma",
    periodic_retrain_days=30,
):
    n_blocks = window_days * blocks_per_day
    periodic_retrain_blocks = periodic_retrain_days * blocks_per_day

    reference_log = []
    test_log = []
    retrain_id = 0

    # --- Chronological stream ---
    if start_date and end_date:
        print(f"[INIT] Restricting to date range {start_date} → {end_date}")
        all_folders = DataUtils.list_sequence_folders(data_dir, start_date, end_date)
        all_files = []
        for folder in all_folders:
            all_files.extend(DataUtils.list_csv_files(folder))
        file_stream = iter(sorted(all_files))
    else:
        print("[INIT] Using all chronological files in data directory")
        file_stream = DataUtils.stream_chronological_files(data_dir)

    # --- Fill initial window ---
    window_files = deque(maxlen=n_blocks)
    history_files = deque(maxlen=n_blocks)

    print(f"[INIT] Filling {n_blocks}-block initial window...")
    for _ in tqdm(range(n_blocks)):
        try:
            f = next(file_stream)
            window_files.append(f)
            history_files.append(f)
        except StopIteration:
            raise RuntimeError("Not enough blocks to initialize window!")

    # --- Initial training ---
    (
        encoder,
        model_trainer,
        entropy_ref,
        baseline_shapes,
        lmt_ref,
        ref_pe_scores,
        ref_lmt_scores,
        ref_z_scores,
        nu,
    ) = train_window(
        window_files,
        max_features,
        retrain_id,
        output_csv,
        reference_log=reference_log,
        max_train_rows=3000000,
        log_memory=True,
        nu_method=nu_method,
    )

    block_index = n_blocks + 1
    g_stat = 0.0
    h = float(h_value)
    blocks_since_retrain = 0

    print(f"[SEQ] h = {h:.3f}")
    print(f"[SEQ] nu = {nu:.6f}")
    print(f"[PERIODIC] Retrain every {periodic_retrain_days} days ({periodic_retrain_blocks} blocks)")

    # --- Process stream ---
    for seq_file in tqdm(file_stream, desc="Evaluating"):
        history_files.append(seq_file)

        df_block = DataUtils.concatenate_blocks([seq_file])
        df_block = DataUtils.preprocess_labels(df_block)
        if df_block.empty:
            block_index += 1
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values
        metrics, recon_errors = model_trainer.evaluate(X_block, y_block)
        if len(recon_errors) == 0:
            block_index += 1
            continue

        drift_info = compute_block_drift_evidence(
            model_trainer=model_trainer,
            X_block=X_block,
            y_block=y_block,
            entropy_ref=entropy_ref,
            baseline_shapes=baseline_shapes,
            lmt_ref=lmt_ref,
            ref_pe_scores=ref_pe_scores,
            ref_lmt_scores=ref_lmt_scores,
            use_mad_lmt=True,
        )

        g_prev = g_stat
        g_stat, decision_any = update_sequential_state(
            prev_g=g_stat,
            z_t=drift_info["z_evidence"],
            nu=nu,
            h=h,
        )

        print(
            f"[SEQ] block={block_index} | "
            f"Z={drift_info['z_evidence']:.6f} | "
            f"G_prev={g_prev:.6f} | "
            f"G={g_stat:.6f} | "
            f"nu={nu:.6f} | "
            f"h={h:.6f} | "
            f"trigger={int(decision_any)}"
        )

        blocks_since_retrain += 1
        periodic_due = int(blocks_since_retrain >= periodic_retrain_blocks)

        metrics.update({
            "block_index": block_index,
            "file": seq_file,
            "retrain_id": retrain_id,
            "data_split": "test",
            "entropy_score": drift_info["entropy_score"],
            "entropy_drift_ratio": drift_info["entropy_drift_ratio"],
            "lmt_mean_score": drift_info["lmt_mean_score"],
            "lmt_sample_drift_ratio": drift_info["lmt_sample_drift_ratio"],
            "p_pe": drift_info["p_pe"],
            "p_lmt": drift_info["p_lmt"],
            "z_evidence": drift_info["z_evidence"],
            "g_stat": g_stat,
            "nu": nu,
            "h": h,
            "trigger_decision": int(decision_any),
            "retrain_policy": f"periodic_{periodic_retrain_days}d",
            "periodic_retrain_due": periodic_due,
            **{f"lmt_{c}_score": v for c, v in drift_info["per_class_scores"].items()},
            "nu_method": nu_method,
        })
        test_log.append(metrics)

        if blocks_since_retrain >= periodic_retrain_blocks:
            retrain_id += 1
            print(f"[Periodic Retrain {retrain_id}] at block {block_index}")

            window_files = deque(history_files, maxlen=n_blocks)

            print(f"[Periodic Retrain {retrain_id}] Rebuilt window covers latest {len(window_files)} blocks")
            print(f"[Periodic Retrain {retrain_id}] First file: {window_files[0]}")
            print(f"[Periodic Retrain {retrain_id}] Last file:  {window_files[-1]}")

            safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors", "drift_info")
            cleanup_memory(f"Before periodic retrain {retrain_id}")

            if psutil is not None:
                log_ram(f"Before periodic retrain {retrain_id}")

            (
                encoder,
                model_trainer,
                entropy_ref,
                baseline_shapes,
                lmt_ref,
                ref_pe_scores,
                ref_lmt_scores,
                ref_z_scores,
                nu,
            ) = train_window(
                window_files,
                max_features,
                retrain_id,
                output_csv,
                reference_log=reference_log,
                max_train_rows=3000000,
                log_memory=True,
                nu_method=nu_method,
            )

            g_stat = 0.0
            blocks_since_retrain = 0

        safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors", "drift_info")
        cleanup_memory(f"End of block {block_index}")
        block_index += 1

    # --- Save metrics separately ---
    out_dir = os.path.dirname(output_csv)
    os.makedirs(out_dir, exist_ok=True)

    reference_csv = os.path.join(out_dir, "reference_blocks.csv")
    test_csv = os.path.join(out_dir, "test_blocks.csv")

    reference_df = pd.DataFrame(reference_log)
    test_df = pd.DataFrame(test_log)

    reference_df.to_csv(reference_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Reference metrics saved → {reference_csv}")
    print(f"Test metrics saved → {test_csv}")

    return reference_df, test_df


def train_window(
    window_files,
    max_features,
    retrain_id,
    output_csv,
    reference_log=None,
    max_train_rows=None,
    log_memory=True,
    nu_method="3sigma",
):
    if log_memory:
        log_ram(f"train_window {retrain_id} start")

    df_train = DataUtils.concatenate_blocks(list(window_files))
    df_train = DataUtils.preprocess_labels(df_train)
    if df_train.empty:
        raise RuntimeError("Empty training window")

    if max_train_rows is not None and len(df_train) > max_train_rows:
        df_train = df_train.tail(max_train_rows)
        print(f"[TRAIN] Capped training data to last {max_train_rows} rows")

    if log_memory:
        log_ram(f"train_window {retrain_id} after loading data")

    encoder = TFIDFTextEncoder(max_features=max_features)
    X_train = encoder.fit_transform(df_train)
    y_train = df_train["category"].values

    model_trainer = ContrastiveModelTrainer()
    model_trainer.fit(X_train, y_train)
    print(f"[TRAIN] Model trained on {len(df_train)} samples ({len(window_files)} blocks)")

    if log_memory:
        log_ram(f"train_window {retrain_id} after fit")

    entropy_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

    if log_memory:
        log_ram(f"train_window {retrain_id} after computing references")

    ref_pe_scores, ref_lmt_scores = compute_reference_block_scores(
        window_files=window_files,
        data_utils=DataUtils,
        encoder=encoder,
        model_trainer=model_trainer,
        entropy_ref=entropy_ref,
        baseline_shapes=baseline_shapes,
        lmt_ref=lmt_ref,
        use_mad_lmt=True,
    )

    ref_z_scores = compute_reference_z_scores(ref_pe_scores, ref_lmt_scores)
    nu = compute_nu(ref_z_scores, method=nu_method)

    if reference_log is not None:
        for i, ref_file in enumerate(window_files):
            ref_df_block = DataUtils.concatenate_blocks([ref_file])
            ref_df_block = DataUtils.preprocess_labels(ref_df_block)
            if ref_df_block.empty:
                continue

            X_ref_block = encoder.transform(ref_df_block)
            y_ref_block = ref_df_block["category"].values
            ref_metrics, _ = model_trainer.evaluate(X_ref_block, y_ref_block)

            drift_info_ref = compute_block_drift_evidence(
                model_trainer=model_trainer,
                X_block=X_ref_block,
                y_block=y_ref_block,
                entropy_ref=entropy_ref,
                baseline_shapes=baseline_shapes,
                lmt_ref=lmt_ref,
                ref_pe_scores=ref_pe_scores,
                ref_lmt_scores=ref_lmt_scores,
                use_mad_lmt=True,
            )

            ref_metrics.update({
                "reference_block_pos": i + 1,
                "file": ref_file,
                "retrain_id": retrain_id,
                "data_split": "reference",
                "entropy_score": drift_info_ref["entropy_score"],
                "entropy_drift_ratio": drift_info_ref["entropy_drift_ratio"],
                "lmt_mean_score": drift_info_ref["lmt_mean_score"],
                "lmt_sample_drift_ratio": drift_info_ref["lmt_sample_drift_ratio"],
                "p_pe": drift_info_ref["p_pe"],
                "p_lmt": drift_info_ref["p_lmt"],
                "z_evidence": drift_info_ref["z_evidence"],
                "g_stat": np.nan,
                "nu": nu,
                "h": np.nan,
                "trigger_decision": 0,
                "retrain_policy": "reference",
                "periodic_retrain_due": 0,
                **{f"lmt_{c}_score": v for c, v in drift_info_ref["per_class_scores"].items()},
                "nu_method": nu_method,
            })
            reference_log.append(ref_metrics)

            del ref_df_block, X_ref_block, y_ref_block, ref_metrics, drift_info_ref
            cleanup_memory(f"Reference block logging retrain_id={retrain_id}, idx={i}")

    if log_memory:
        log_ram(f"train_window {retrain_id} after reference block scores")

    baseline_dir = os.path.join(os.path.dirname(output_csv), "entropy_refs")
    baseline_path = os.path.join(baseline_dir, f"entropy_ref_{retrain_id:03d}.json")
    save_json_safely(entropy_ref, baseline_path)

    baseline_dir = os.path.join(os.path.dirname(output_csv), "lmt_refs")
    baseline_path = os.path.join(baseline_dir, f"lmt_ref_{retrain_id:03d}.json")
    save_json_safely(lmt_ref, baseline_path)

    del df_train, X_train, y_train
    cleanup_memory(f"train_window {retrain_id} end")

    if log_memory:
        log_ram(f"train_window {retrain_id} final")

    return (
        encoder,
        model_trainer,
        entropy_ref,
        baseline_shapes,
        lmt_ref,
        ref_pe_scores,
        ref_lmt_scores,
        ref_z_scores,
        nu,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periodic retraining baseline")
    parser.add_argument("--data", type=str, default="../data/sequences", help="Base data directory")
    parser.add_argument("--start", type=str, default="2025-03-16", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-03-25", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=8, help="Window length in days")
    parser.add_argument("--out", type=str, default="../data/rq3/periodic30d/output_anchor.csv", help="Output CSV path")
    parser.add_argument("--periodic_days", type=int, default=30, help="Periodic retraining interval in days")
    parser.add_argument("--h", type=float, default=8.0, help="Sequential threshold h for logging")
    parser.add_argument(
        "--nu_method",
        type=str,
        default="3sigma",
        choices=["median", "mean", "q75", "1sigma", "2sigma", "3sigma"],
        help="Method to compute nu"
    )
    args = parser.parse_args()

    run_experiment_periodic(
        data_dir=args.data,
        output_csv=args.out,
        start_date=args.start,
        end_date=args.end,
        window_days=args.days,
        max_features=2000,
        h_value=args.h,
        nu_method=args.nu_method,
        periodic_retrain_days=args.periodic_days,
    )