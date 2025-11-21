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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def build_block_stream(main_dir, xpot_dir, start_date=None, end_date=None):
    if start_date and end_date:
        main_folders = DataUtils.list_sequence_folders(main_dir, start_date, end_date)
        main_files = []
        for folder in main_folders:
            main_files.extend(DataUtils.list_csv_files(folder))
        main_files = sorted(main_files)
    else:
        main_files = list(DataUtils.stream_chronological_files(main_dir))

    blocks = []
    for mf in main_files:
        block_file = os.path.basename(mf)
        block_dir = os.path.basename(os.path.dirname(mf))
        xp = os.path.join(xpot_dir, block_dir, block_file)
        if os.path.exists(xp):
            blocks.append([mf, xp])
        else:
            blocks.append([mf])
    return blocks



def train_window(window_blocks, max_features, retrain_id, output_csv):
    all_files = [f for block in window_blocks for f in block]
    df_train = DataUtils.concatenate_blocks(all_files)
    df_train = DataUtils.preprocess_labels(df_train)
    if df_train.empty:
        raise RuntimeError("Empty training window")

    encoder = TFIDFTextEncoder(max_features=max_features)
    X_train = encoder.fit_transform(df_train)
    y_train = df_train["category"].values

    model_trainer = ContrastiveModelTrainer()
    model_trainer.fit(X_train, y_train)

    entropy_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

    base_dir = os.path.join(os.path.dirname(output_csv), "entropy_refs")
    base_path = os.path.join(base_dir, f"entropy_ref_{retrain_id:03d}.json")
    save_json_safely(entropy_ref, base_path)

    base_dir = os.path.join(os.path.dirname(output_csv), "lmt_refs")
    base_path = os.path.join(base_dir, f"lmt_ref_{retrain_id:03d}.json")
    save_json_safely(lmt_ref, base_path)

    return encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref


def run_experiment_entropy_lmt(
    data_dir,
    xpot_dir,
    output_csv,
    start_date=None,
    end_date=None,
    window_days=8,
    blocks_per_day=24,
    max_features=2000,
):

    N_BLOCKS = window_days * blocks_per_day

    main_dir = data_dir

    block_stream = build_block_stream(main_dir, xpot_dir, start_date, end_date)

    if len(block_stream) < N_BLOCKS:
        raise RuntimeError("Insufficient blocks to initialize window")

    window_blocks = deque(maxlen=N_BLOCKS)
    for b in block_stream[:N_BLOCKS]:
        window_blocks.append(b)

    retrain_id = 0
    metric_log = []

    encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref = train_window(
        window_blocks, max_features, retrain_id, output_csv
    )

    recent_lmt_decisions = deque(maxlen=10)
    block_index = N_BLOCKS

    for block_files in tqdm(block_stream[N_BLOCKS:], desc="Evaluating"):
        df_block = DataUtils.concatenate_blocks(block_files)
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

        entropy_score, drift_ratio, _ = compute_entropy_score(
            model_trainer, X_block, entropy_ref, use_mad=False
        )
        entropy_mon, entropy_dec = detect_entropy_drift(
            entropy_score, entropy_ref, use_mad=False
        )

        mean_lmt_score, lmt_ratio, per_class_scores, lmt_mon, lmt_dec = compute_lmt_block(
            model_trainer, X_block, y_block, baseline_shapes, lmt_ref, use_mad=True
        )

        recent_lmt_decisions.append(lmt_dec)
        lmt_count = sum(recent_lmt_decisions)

        trigger_entropy_lmt = entropy_dec and lmt_dec
        trigger_lmt_stable = (lmt_count >= 8)
        # decision_any = trigger_entropy_lmt or trigger_lmt_stable
        decision_any = False
        if trigger_entropy_lmt and not trigger_lmt_stable:
            retrigger = "entropy_lmt_both"
        elif trigger_lmt_stable and not trigger_entropy_lmt:
            retrigger = f"lmt_stable_{lmt_count}"
        elif trigger_entropy_lmt and trigger_lmt_stable:
            retrigger = f"both_conditions_{lmt_count}"
        else:
            retrigger = "none"

        main_file = block_files[0]


        sources = []
        for f in block_files:
            if "xpot" in f:
                sources.append("xpot")
            elif "sequences" in f:
                sources.append("main")


        metrics.update({
            "block_index": block_index,
            "file": ";".join(block_files),
            "retrain_id": retrain_id,
            "entropy_score": entropy_score,
            "entropy_drift_ratio": drift_ratio,
            "entropy_monitor": entropy_mon,
            "entropy_decision": entropy_dec,
            "lmt_mean_score": mean_lmt_score,
            "lmt_sample_drift_ratio": lmt_ratio,
            **{f"lmt_{c}_score": v for c, v in per_class_scores.items()},
            "lmt_monitor": lmt_mon,
            "lmt_decision": lmt_dec,
            "retrain_trigger_source": retrigger,
            "sources": ";".join(sorted(set(sources))),
        })
        metric_log.append(metrics)

        if decision_any:
            retrain_id += 1
            window_blocks.append(block_files)
            encoder, model_trainer, entropy_ref, baseline_shapes, lmt_ref = train_window(
                window_blocks, max_features, retrain_id, output_csv
            )

        block_index += 1

    metric_df = pd.DataFrame(metric_log)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    metric_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    return metric_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive experiment with dummy drift (always retrain).")
    parser.add_argument("--data", type=str, default="../data/sequences", help="Base data directory")
    parser.add_argument("--xpot", type=str, default="../data/xpot/sequences")
    parser.add_argument("--start", type=str, default="2025-03-15", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-05-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=8, help="Window length in days (default=8)")
    parser.add_argument("--out", type=str, default="../data/res/hetero/static.csv", help="Output CSV path")
    args = parser.parse_args()

    run_experiment_entropy_lmt(
        data_dir=args.data,
        xpot_dir=args.xpot,
        output_csv=args.out,
        start_date=args.start,
        end_date=args.end,
        window_days=args.days
    )

