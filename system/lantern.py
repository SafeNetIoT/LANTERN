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
    compute_entropy_reference, compute_entropy_score, detect_entropy_drift,
    compute_lmt_reference, compute_lmt_block
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # keep logical order
os.environ["CUDA_VISIBLE_DEVICES"] = "3"           # GPU #1


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
    print(f"[SAVE] Entropy reference saved → {path}")

# =========================================================
# Main Experiment
# =========================================================

def run_experiment_entropy_lmt(
    data_dir,
    output_csv,
    start_date=None,
    end_date=None,
    window_days=8,
    blocks_per_day=24,
    max_features=2000,
):
    N_BLOCKS = window_days * blocks_per_day
    metric_log = []
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
    window_files = deque(maxlen=N_BLOCKS)
    print(f"[INIT] Filling {N_BLOCKS}-block initial window...")
    for _ in tqdm(range(N_BLOCKS)):
        try:
            f = next(file_stream)
            window_files.append(f)
        except StopIteration:
            raise RuntimeError("Not enough blocks to initialize window!")

    # --- Initial training, trainer and entropy references ---
    encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref = train_window(window_files, max_features, retrain_id, output_csv)

    block_index = N_BLOCKS + 1
    
    # --- Initialize memory for temporal consistency ---
    recent_lmt_decisions = deque(maxlen=10)


    # --- Process stream ---
    for seq_file in tqdm(file_stream, desc="Evaluating"):
        df_block = DataUtils.concatenate_blocks([seq_file])
        df_block = DataUtils.preprocess_labels(df_block)
        if df_block.empty:
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values
        metrics, recon_errors = model_trainer.evaluate(X_block, y_block)
        if len(recon_errors) == 0:
            continue
        
        # --- Drift detection ---
        entropy_score, drift_ratio, _ = compute_entropy_score(model_trainer, X_block, entropy_ref, use_mad=False)
        entropy_mon, entropy_dec = detect_entropy_drift(entropy_score, entropy_ref, use_mad=False)

        mean_lmt_score, lmt_sample_ratio, per_class_scores, lmt_mon, lmt_dec = compute_lmt_block(
            model_trainer, X_block, y_block, baseline_shapes, lmt_ref, use_mad=True
        )
        

      # --- Update temporal LMT memory ---
        recent_lmt_decisions.append(lmt_dec)
        lmt_count = sum(recent_lmt_decisions)

        # --- Combined retrain rule ---
        trigger_entropy_lmt = entropy_dec and lmt_dec
        trigger_lmt_stable = (lmt_count >= 8)
        decision_any = trigger_entropy_lmt or trigger_lmt_stable
        # decision_any = 0   # For static case

        # --- Identify which condition triggered ---
        if trigger_entropy_lmt and not trigger_lmt_stable:
            retrain_trigger = "entropy_lmt_both" 
        elif trigger_lmt_stable and not trigger_entropy_lmt:
            retrain_trigger = f"lmt_stable({lmt_count}/10)"
        elif trigger_entropy_lmt and trigger_lmt_stable:
            retrain_trigger = f"both_conditions({lmt_count}/10)"
        else:
            retrain_trigger = "none"


        metrics.update({
            "block_index": block_index,
            "file": seq_file,
            "retrain_id": retrain_id,
            "entropy_score": entropy_score,
            "entropy_drift_ratio": drift_ratio,
            "entropy_monitor": entropy_mon,
            "entropy_decision": entropy_dec,
            "lmt_mean_score": mean_lmt_score,
            "lmt_sample_drift_ratio": lmt_sample_ratio,
            **{f"lmt_{c}_score": v for c, v in per_class_scores.items()},
            "lmt_monitor": lmt_mon,
            "lmt_decision": lmt_dec,
            "retrain_trigger_source": retrain_trigger,

        })
        metric_log.append(metrics)

        # --- Adaptive retraining ---
        if decision_any:
            retrain_id += 1
            print(f"[Retrain {retrain_id}] Triggered at block {block_index}")
            if entropy_dec:
                print(f"  ↳ Entropy score = {entropy_score:.4f} (ratio={drift_ratio:.4f})")
            if lmt_dec:
                print(f"  ↳ LMT score = {mean_lmt_score:.4f} (ratio={lmt_sample_ratio:.4f})")

            window_files.append(seq_file)
            encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref = train_window(window_files, max_features, retrain_id, output_csv)

        block_index += 1

    # --- Save metrics ---
    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    metric_df.to_csv(output_csv, index=False)
    print(f"Metrics saved → {output_csv}")
    return metric_df


def train_window(window_files, max_features, retrain_id, output_csv):
    """Load and train model on the most recent window."""
    df_train = DataUtils.concatenate_blocks(list(window_files))
    df_train = DataUtils.preprocess_labels(df_train)
    if df_train.empty:
        raise RuntimeError("Empty training window")

    encoder = TFIDFTextEncoder(max_features=max_features)
    X_train = encoder.fit_transform(df_train)
    y_train = df_train["category"].values

    model_trainer = ContrastiveModelTrainer()
    model_trainer.fit(X_train, y_train)
    print(f"[TRAIN] Model trained on {len(df_train)} samples ({len(window_files)} blocks)")

    entropy_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

    
    # save to file
    baseline_dir = os.path.join(os.path.dirname(output_csv), "entropy_refs")
    baseline_path = os.path.join(baseline_dir, f"entropy_ref_{retrain_id:03d}.json")
    save_json_safely(entropy_ref, baseline_path)

    # Save LMT reference for traceability
    baseline_dir = os.path.join(os.path.dirname(output_csv), "lmt_refs")
    baseline_path = os.path.join(baseline_dir, f"lmt_ref_{retrain_id:03d}.json")
    save_json_safely(lmt_ref, baseline_path)

    return encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref


# --- CLI entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive experiment with dummy drift (always retrain).")
    parser.add_argument("--data", type=str, default="../data/sequences", help="Base data directory")
    parser.add_argument("--start", type=str, default="2025-03-15", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-10-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=8, help="Window length in days (default=8)")
    parser.add_argument("--out", type=str, default="../data/res/dynamictest/dynamictest.csv", help="Output CSV path")
    args = parser.parse_args()

    run_experiment_entropy_lmt(
        data_dir=args.data,
        output_csv=args.out,
        start_date=args.start,
        end_date=args.end,
        window_days=args.days,
    )
