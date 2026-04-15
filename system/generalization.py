import sys, os, json, gc
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
    compute_entropy_score,
    compute_lmt_reference,
    compute_lmt_block,
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


def flatten_block_files(blocks):
    return [f for block in blocks for f in block]


def infer_sources(block_files):
    sources = []
    for f in block_files:
        f_lower = f.lower()
        if "xpot" in f_lower:
            sources.append("xpot")
        elif "sequences" in f_lower:
            sources.append("main")
        else:
            sources.append("unknown")
    return ";".join(sorted(set(sources)))


def compute_reference_block_scores_hetero(
    window_blocks,
    encoder,
    model_trainer,
    entropy_ref,
    baseline_shapes,
    lmt_ref,
):
    ref_pe_scores = []
    ref_lmt_scores = []

    for i, block_files in enumerate(window_blocks):
        df_block = DataUtils.concatenate_blocks(block_files)
        df_block = DataUtils.preprocess_labels(df_block)
        if df_block.empty:
            print(f"[REF] Skip empty reference block at pos={i+1}")
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values

        entropy_score, _, _ = compute_entropy_score(
            model_trainer,
            X_block,
            entropy_ref,
            use_mad=False,
        )

        lmt_mean_score, _, _, _, _ = compute_lmt_block(
            model_trainer,
            X_block,
            y_block,
            baseline_shapes,
            lmt_ref,
            use_mad=True,
        )

        ref_pe_scores.append(entropy_score)
        ref_lmt_scores.append(lmt_mean_score)

        del df_block, X_block, y_block
        cleanup_memory(f"reference hetero block {i+1}")

    return ref_pe_scores, ref_lmt_scores


def train_window(
    window_blocks,
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

    all_files = flatten_block_files(window_blocks)
    df_train = DataUtils.concatenate_blocks(all_files)
    df_train = DataUtils.preprocess_labels(df_train)
    if df_train.empty:
        raise RuntimeError("Empty training window")

    if max_train_rows is not None and len(df_train) > max_train_rows:
        df_train = df_train.tail(max_train_rows)
        print(f"[TRAIN] Capped training data to last {max_train_rows} rows")

    if log_memory:
        log_ram(f"train_window {retrain_id} after loading")

    encoder = TFIDFTextEncoder(max_features=max_features)
    X_train = encoder.fit_transform(df_train)
    y_train = df_train["category"].values

    model_trainer = ContrastiveModelTrainer()
    model_trainer.fit(X_train, y_train)
    print(f"[TRAIN] Model trained on {len(df_train)} samples ({len(window_blocks)} mixed blocks)")

    if log_memory:
        log_ram(f"train_window {retrain_id} after fit")

    entropy_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

    if log_memory:
        log_ram(f"train_window {retrain_id} after reference fit")

    ref_pe_scores, ref_lmt_scores = compute_reference_block_scores_hetero(
        window_blocks=window_blocks,
        encoder=encoder,
        model_trainer=model_trainer,
        entropy_ref=entropy_ref,
        baseline_shapes=baseline_shapes,
        lmt_ref=lmt_ref,
    )

    ref_z_scores = compute_reference_z_scores(ref_pe_scores, ref_lmt_scores)
    nu = compute_nu(ref_z_scores, method=nu_method)

    if reference_log is not None:
        for i, block_files in enumerate(window_blocks):
            df_block = DataUtils.concatenate_blocks(block_files)
            df_block = DataUtils.preprocess_labels(df_block)
            if df_block.empty:
                continue

            X_block = encoder.transform(df_block)
            y_block = df_block["category"].values
            ref_metrics, recon_errors = model_trainer.evaluate(X_block, y_block)

            if len(recon_errors) == 0:
                del df_block, X_block, y_block, ref_metrics, recon_errors
                cleanup_memory(f"Reference block logging retrain_id={retrain_id}, idx={i}")
                continue

            drift_info_ref = compute_block_drift_evidence(
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

            ref_metrics.update({
                "reference_block_pos": i + 1,
                "file": ";".join(block_files),
                "retrain_id": retrain_id,
                "data_split": "reference",
                "sources": infer_sources(block_files),
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

            del df_block, X_block, y_block, ref_metrics, recon_errors, drift_info_ref
            cleanup_memory(f"Reference block logging retrain_id={retrain_id}, idx={i}")

    if log_memory:
        log_ram(f"train_window {retrain_id} after reference block scoring")

    base_dir = os.path.join(os.path.dirname(output_csv), "entropy_refs")
    base_path = os.path.join(base_dir, f"entropy_ref_{retrain_id:03d}.json")
    save_json_safely(entropy_ref, base_path)

    base_dir = os.path.join(os.path.dirname(output_csv), "lmt_refs")
    base_path = os.path.join(base_dir, f"lmt_ref_{retrain_id:03d}.json")
    save_json_safely(lmt_ref, base_path)

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


def run_experiment_entropy_lmt(
    data_dir,
    xpot_dir,
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
    n_blocks = window_days * blocks_per_day

    block_stream = build_block_stream(data_dir, xpot_dir, start_date, end_date)

    if len(block_stream) < n_blocks:
        raise RuntimeError("Insufficient blocks to initialize window")

    window_blocks = deque(maxlen=n_blocks)
    history_blocks = deque(maxlen=n_blocks)

    for b in block_stream[:n_blocks]:
        window_blocks.append(b)
        history_blocks.append(b)

    retrain_id = 0
    reference_log = []
    test_log = []

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
        window_blocks=window_blocks,
        max_features=max_features,
        retrain_id=retrain_id,
        output_csv=output_csv,
        reference_log=reference_log,
        max_train_rows=3000000,
        log_memory=True,
        nu_method=nu_method,
    )

    g_stat = 0.0
    h = float(h_value)
    block_index = n_blocks + 1

    print(f"[SEQ] h = {h:.3f}")
    print(f"[SEQ] nu = {nu:.6f}")

    for block_files in tqdm(block_stream[n_blocks:], desc="Evaluating"):
        history_blocks.append(block_files)

        df_block = DataUtils.concatenate_blocks(block_files)
        df_block = DataUtils.preprocess_labels(df_block)
        if df_block.empty:
            print(f"[SKIP] block={block_index} | reason=empty_block | file={';'.join(block_files)}")
            cleanup_memory(f"Skipped block {block_index}")
            block_index += 1
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values
        metrics, recon_errors = model_trainer.evaluate(X_block, y_block)
        if len(recon_errors) == 0:
            print(f"[SKIP] block={block_index} | reason=empty_recon | file={';'.join(block_files)}")
            safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors")
            cleanup_memory(f"Skipped block {block_index}")
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

        metrics.update({
            "block_index": block_index,
            "file": ";".join(block_files),
            "retrain_id": retrain_id,
            "data_split": "test",
            "sources": infer_sources(block_files),
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
        })
        test_log.append(metrics)

        if decision_any and not static_calibration:
            retrain_id += 1
            print(f"[Retrain {retrain_id}] Triggered at block {block_index}")
            print(f"  ↳ sources = {infer_sources(block_files)}")
            print(f"  ↳ PE score = {drift_info['entropy_score']:.6f}, p_pe = {drift_info['p_pe']:.6f}")
            print(f"  ↳ LMT score = {drift_info['lmt_mean_score']:.6f}, p_lmt = {drift_info['p_lmt']:.6f}")
            print(f"  ↳ Z_t = {drift_info['z_evidence']:.6f}, G_t = {g_stat:.6f}, nu = {nu:.6f}, h = {h:.6f}")

            window_blocks = deque(history_blocks, maxlen=n_blocks)

            print(f"[Retrain {retrain_id}] Rebuilt window covers latest {len(window_blocks)} blocks")
            print(f"[Retrain {retrain_id}] First block: {';'.join(window_blocks[0])}")
            print(f"[Retrain {retrain_id}] Last block:  {';'.join(window_blocks[-1])}")

            safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors", "drift_info")
            cleanup_memory(f"Before retrain {retrain_id}")
            log_ram(f"Before retrain {retrain_id}")

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
                window_blocks=window_blocks,
                max_features=max_features,
                retrain_id=retrain_id,
                output_csv=output_csv,
                reference_log=reference_log,
                max_train_rows=3000000,
                log_memory=True,
                nu_method=nu_method,
            )

            g_stat = 0.0

        elif decision_any and static_calibration:
            print(
                f"[STATIC TRIGGER] block {block_index} | "
                f"sources={infer_sources(block_files)} | "
                f"PE={drift_info['entropy_score']:.6f}, "
                f"LMT={drift_info['lmt_mean_score']:.6f}, "
                f"Z_t={drift_info['z_evidence']:.6f}, "
                f"G_t={g_stat:.6f}, nu={nu:.6f}, h={h:.6f}"
            )

        safe_delete(locals(), "df_block", "X_block", "y_block", "metrics", "recon_errors", "drift_info")
        cleanup_memory(f"End of block {block_index}")
        block_index += 1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heterogeneous source experiment with fused sequential drift evidence.")
    parser.add_argument("--data", type=str, default="../data/sequences", help="Base data directory")
    parser.add_argument("--xpot", type=str, default="../data/xpot/sequences", help="Aligned XPOT directory")
    parser.add_argument("--start", type=str, default="2025-03-16", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-05-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=8, help="Window length in days")
    parser.add_argument("--out", type=str, default="../data/res/generalisation_static/output_anchor.csv", help="Output CSV path")

    parser.add_argument("--h", type=float, default=8.0, help="Sequential threshold h")
    parser.add_argument("--static", action="store_true", help="Run static calibration only")
    parser.add_argument(
        "--nu_method",
        type=str,
        default="3sigma",
        choices=["median", "mean", "q75", "1sigma", "2sigma", "3sigma"],
        help="Method to compute nu from reference z scores",
    )

    args = parser.parse_args()

    run_experiment_entropy_lmt(
        data_dir=args.data,
        xpot_dir=args.xpot,
        output_csv=args.out,
        start_date=args.start,
        end_date=args.end,
        window_days=args.days,
        max_features=2000,
        h_value=args.h,
        nu_method=args.nu_method,
        static_calibration=args.static,
    )