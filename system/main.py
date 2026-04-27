import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
import argparse

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

import gc
import time

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # keep logical order
os.environ["CUDA_VISIBLE_DEVICES"] = "0"           # GPU #1


# =========================================================
# Helper to safely store JSON with numpy types
# =========================================================
def save_json_safely(obj, path):
    """Convert NumPy types recursively to Python native and save JSON."""
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
    print(f"[SAVE] JSON saved → {path}")

def safe_delete(local_vars, *names):
    for name in names:
        if name in local_vars:
            del local_vars[name]


# =========================================================
# Memory Helpers
# =========================================================
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

def get_rss_gb():
    if psutil is None:
        return np.nan
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

def get_gpu_mem_allocated_gb():
    if torch is None or not torch.cuda.is_available():
        return np.nan
    return torch.cuda.memory_allocated() / (1024 ** 3)

def get_gpu_mem_reserved_gb():
    if torch is None or not torch.cuda.is_available():
        return np.nan
    return torch.cuda.memory_reserved() / (1024 ** 3)

def get_gpu_peak_mem_allocated_gb():
    if torch is None or not torch.cuda.is_available():
        return np.nan
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def reset_gpu_peak_stats():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()



# =========================================================
# Main Experiment
# =========================================================

def run_experiment_lantern(
    data_dir,
    output_csv,
    start_date=None,
    end_date=None,
    window_days=8,
    blocks_per_day=24,
    max_features=2000,
    h_value=8.0,
    nu_method="3sigma",
    static_calibration=True,
):
    N_BLOCKS = window_days * blocks_per_day
    reference_log = []
    test_log = []
    retrain_id = 0

    # Memory record
    block_times_sec = []
    train_times_sec = []
    train_peak_ram_gb = []
    train_peak_gpu_gb = []
    train_records = []

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
    window_files = deque(maxlen=N_BLOCKS)
    history_files = deque(maxlen=N_BLOCKS)  

    print(f"[INIT] Filling {N_BLOCKS}-block initial window...")
    for _ in tqdm(range(N_BLOCKS)):
        try:
            f = next(file_stream)
            window_files.append(f)
            history_files.append(f)
        except StopIteration:
            raise RuntimeError("Not enough blocks to initialize window!")

    # --- Initial training, trainer and entropy references ---
    init_train_t0 = time.perf_counter()
    init_ram_before = get_rss_gb()
    reset_gpu_peak_stats()
    peak_ram_this_train = init_ram_before
    
    (encoder,
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
    reference_log = reference_log,
    max_train_rows=300000000,
    log_memory=True,
    nu_method=nu_method,
    )

    init_train_time_sec = time.perf_counter() - init_train_t0
    peak_ram_this_train = max(peak_ram_this_train, get_rss_gb())
    peak_gpu_this_train = get_gpu_peak_mem_allocated_gb()

    train_times_sec.append(init_train_time_sec)
    train_peak_ram_gb.append(peak_ram_this_train)
    train_peak_gpu_gb.append(peak_gpu_this_train)
    train_records.append({
        "train_id": 0,
        "train_stage": "initial",
        "train_time_sec": init_train_time_sec,
        "peak_ram_gb": peak_ram_this_train,
        "peak_gpu_gb": peak_gpu_this_train,
        "window_days": window_days,
        "n_blocks": N_BLOCKS,
    })

    print(
        f"[COST] Train initial | "
        f"time={init_train_time_sec:.2f}s | "
        f"peak_ram={peak_ram_this_train:.2f} GiB | "
        f"peak_gpu={peak_gpu_this_train:.2f} GiB"
    )

    block_index = N_BLOCKS + 1
    
    # sequential state
    g_stat = 0.0
    h = float(h_value)
    print(f"[SEQ] h = {h:.3f}")
    print(f"[SEQ] nu = {nu:.6f}")
    


    # --- Process stream ---
    for seq_file in tqdm(file_stream, desc="Evaluating"):
        block_t0 = time.perf_counter()
        #always update chronological history
        history_files.append(seq_file)

        df_block = DataUtils.concatenate_blocks([seq_file])
        df_block = DataUtils.preprocess_labels(df_block)
        if df_block.empty:
            #block_index += 1
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values
        metrics, recon_errors = model_trainer.evaluate(X_block, y_block)
        if len(recon_errors) == 0:
            continue
        
        # --- Drift detection ---
        # --- New drift pipeline: conformal calibration + fused evidence + sequential accumulation ---
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
        
        block_time_sec = time.perf_counter() - block_t0
        block_times_sec.append(block_time_sec)

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
            **{f"lmt_{c}_score": v for c, v in drift_info["per_class_scores"].items()},
            "nu_method": nu_method,
            "block_time_sec": block_time_sec,
        })
        test_log.append(metrics)

        
        # --- Adaptive retraining ---
        if decision_any and not static_calibration:
            retrain_id += 1
            print(f"[Retrain {retrain_id}] Triggered at block {block_index}")
            print(f"  ↳ PE score = {drift_info['entropy_score']:.6f}, p_pe = {drift_info['p_pe']:.6f}")
            print(f"  ↳ LMT score = {drift_info['lmt_mean_score']:.6f}, p_lmt = {drift_info['p_lmt']:.6f}")
            print(f"  ↳ Z_t = {drift_info['z_evidence']:.6f}, G_t = {g_stat:.6f}, nu = {nu:.6f}, h = {h:.6f}")

            # rebuild Wref as the most recent N_BLOCKS ending at current block
            window_files = deque(history_files, maxlen=N_BLOCKS)

            print(f"[Retrain {retrain_id}] Rebuilt window covers latest {len(window_files)} blocks")
            print(f"[Retrain {retrain_id}] First file: {window_files[0]}")
            print(f"[Retrain {retrain_id}] Last file:  {window_files[-1]}")

            safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors", "drift_info")
            cleanup_memory(f"Before retrain {retrain_id}")

            if psutil is not None:
                log_ram(f"Before retrain {retrain_id}")

            update_train_t0 = time.perf_counter()
            update_ram_before = get_rss_gb()
            reset_gpu_peak_stats()

            peak_ram_this_train = update_ram_before

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
                reference_log = reference_log,
                max_train_rows=300000000,
                log_memory=True,
                nu_method=nu_method,
            )

            update_train_time_sec = time.perf_counter() - update_train_t0
            peak_ram_this_train = max(peak_ram_this_train, get_rss_gb())
            peak_gpu_this_train = get_gpu_peak_mem_allocated_gb()

            train_times_sec.append(update_train_time_sec)
            train_peak_ram_gb.append(peak_ram_this_train)
            train_peak_gpu_gb.append(peak_gpu_this_train)
            train_records.append({
                "train_id": retrain_id,
                "train_stage": "update",
                "train_time_sec": update_train_time_sec,
                "peak_ram_gb": peak_ram_this_train,
                "peak_gpu_gb": peak_gpu_this_train,
                "window_days": window_days,
                "n_blocks": N_BLOCKS,
            })

            print(
                f"[COST] Train update {retrain_id} | "
                f"time={update_train_time_sec:.2f}s | "
                f"peak_ram={peak_ram_this_train:.2f} GiB | "
                f"peak_gpu={peak_gpu_this_train:.2f} GiB"
            )

            # reset sequential statistic after update
            g_stat = 0.0


        if decision_any and static_calibration:
            print(f"[STATIC TRIGGER] block {block_index} | "
                  f"PE={drift_info['entropy_score']:.6f}, "
                  f"LMT={drift_info['lmt_mean_score']:.6f}, "
                  f"Z_t={drift_info['z_evidence']:.6f}, "
                  f"G_t={g_stat:.6f}, nu={nu:.6f}, h={h:.6f}")

        safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors")
        #del df_block, X_block, y_block, metrics, recon_errors
        cleanup_memory(f"End of block {block_index}")
        block_index += 1

    # --- Save metrics seperately ---
    out_dir = os.path.dirname(output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    reference_csv = os.path.join(out_dir, "reference_blocks.csv")
    test_csv = os.path.join(out_dir, "test_blocks.csv")

    reference_df = pd.DataFrame(reference_log)
    test_df = pd.DataFrame(test_log)

    reference_df.to_csv(reference_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Reference metrics saved → {reference_csv}")
    print(f"Test metrics saved → {test_csv}")

    initial_train_times = [r["train_time_sec"] for r in train_records if r["train_stage"] == "initial"]
    update_train_times = [r["train_time_sec"] for r in train_records if r["train_stage"] == "update"]

    cost_summary = {
        "window_days": int(window_days),
        "n_test_blocks": int(len(test_df)),
        "avg_block_time_sec": float(np.mean(block_times_sec)) if block_times_sec else None,
        "median_block_time_sec": float(np.median(block_times_sec)) if block_times_sec else None,
        "p95_block_time_sec": float(np.quantile(block_times_sec, 0.95)) if block_times_sec else None,
        "n_total_trains": int(len(train_records)),
        "n_update_trains": int(len(update_train_times)),
        "initial_train_time_sec": float(initial_train_times[0]) if initial_train_times else None,
        "avg_update_train_time_sec": float(np.mean(update_train_times)) if update_train_times else None,
        "total_train_time_sec": float(np.sum(train_times_sec)) if train_times_sec else 0.0,
        "peak_ram_gb_during_train": float(np.max(train_peak_ram_gb)) if train_peak_ram_gb else None,
        "peak_gpu_gb_during_train": float(np.max(train_peak_gpu_gb)) if train_peak_gpu_gb else None,
    }

    cost_json = os.path.join(out_dir, "cost_summary.json")
    save_json_safely(cost_summary, cost_json)
    
    print("\n=== COST SUMMARY ===")
    train_cost_csv = os.path.join(out_dir, "train_cost_records.csv")
    pd.DataFrame(train_records).to_csv(train_cost_csv, index=False)
    print(f"Training cost records saved → {train_cost_csv}")

    for k, v in cost_summary.items():
        print(f"{k}: {v}")

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
    """Load and train model on the most recent window."""
    if log_memory:
        log_ram(f"train_window {retrain_id} start")

    df_train = DataUtils.concatenate_blocks(list(window_files))
    df_train = DataUtils.preprocess_labels(df_train)
    if df_train.empty:
        raise RuntimeError("Empty training window")

    # optional safety cap
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

    # ---------------------------------------------------------
    # Store reference block data into the reference log
    # ---------------------------------------------------------
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
                **{f"lmt_{c}_score": v for c, v in drift_info_ref["per_class_scores"].items()},
                "nu_method": nu_method,
            })
            reference_log.append(ref_metrics)

            del ref_df_block, X_ref_block, y_ref_block, ref_metrics, drift_info_ref
            cleanup_memory(f"Reference block logging retrain_id={retrain_id}, idx={i}")

            
    if log_memory:
        log_ram(f"train_window {retrain_id} after reference block scores")


    # save to file
    baseline_dir = os.path.join(os.path.dirname(output_csv), "entropy_refs")
    baseline_path = os.path.join(baseline_dir, f"entropy_ref_{retrain_id:03d}.json")
    save_json_safely(entropy_ref, baseline_path)

    # Save LMT reference for traceability
    baseline_dir = os.path.join(os.path.dirname(output_csv), "lmt_refs")
    baseline_path = os.path.join(baseline_dir, f"lmt_ref_{retrain_id:03d}.json")
    save_json_safely(lmt_ref, baseline_path)

    # free large temporary training objects
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


# --- CLI entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive experiment with dummy drift (always retrain).")
    parser.add_argument("--data", type=str, default="../data/sequences", help="Base data directory")
    parser.add_argument("--start", type=str, default="2025-03-16", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-03-25", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=8, help="Window length in days (default=8)")
    parser.add_argument("--out", type=str, default="../data/res/rq3/h8/output_anchor.csv", help="Output CSV path")


    parser.add_argument("--h", type=float, default=8.0, help="Sequential threshold h")
    parser.add_argument("--static", action="store_true", help="Run static calibration (no retraining)")
    parser.add_argument(
        "--nu_method",
        type=str,
        default="3sigma",
        choices=["median", "mean", "q75", "1sigma", "2sigma", "3sigma"],
        help="Method to compute nu"
    )
    args = parser.parse_args()

    run_experiment_lantern(
        data_dir=args.data,
        output_csv=args.out,
        start_date=args.start,
        end_date=args.end,
        window_days=args.days,
        max_features=2000,
        h_value=args.h,
        nu_method=args.nu_method,
        static_calibration=args.static,
    )
