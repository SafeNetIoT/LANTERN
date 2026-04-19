import os
import glob
import random
import warnings
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.ModelUtils import (
    AutoencoderClassifierTrainer,
    RFModelTrainer,
    ContrastiveModelTrainer,
)

from utils.DriftUtils import (
    compute_cade_class_reference,
    compute_chen_reference,
    compute_gidx_reference,
    compute_entropy_reference,
    compute_lmt_reference,
    cade_score_min_over_classes,
    chen_fast_detect,
    gidx_fast_detect,
    pe_fast_detect,
)

warnings.filterwarnings("ignore")


# =========================================================
# Config
# =========================================================
BASE_DIR = "../../data/public_datasets/ids17/MachineLearningCVE"
OUT_CSV = "ids17_portscan_stream_scores.csv"

MODEL_NAME = "cae"   # choices: "rf", "ae", "cae"
RANDOM_STATE = 42

BLOCK_SIZE = 10000

# =========================================================
# File order
# =========================================================
ORDERED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

MONDAY_FILE = "Monday-WorkingHours.pcap_ISCX.csv"
PORTSCAN_FILE = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
PORTSCAN_LABEL = "PortScan"


# =========================================================
# Helpers
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_model_cls(name: str):
    mapping = {
        "rf": RFModelTrainer,
        "ae": AutoencoderClassifierTrainer,
        "cae": ContrastiveModelTrainer,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported model name: {name}")
    return mapping[name]


def load_ids17_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "Label" not in df.columns:
        raise ValueError(f"'Label' column missing in {path}")

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.replace("\ufffd", "-", regex=False)
        .str.replace("�", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame):
    exclude = {"Label", "group_label"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def map_binary_label(x: str) -> str:
    return "benign" if x == "BENIGN" else "malicious"


def build_xy(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["group_label"].values.astype(object)
    return X, y


def get_probs_and_preds(model_trainer, X):
    z = model_trainer.encode(X)
    probs = model_trainer.classifier.predict_proba(z)
    probs = np.asarray(probs, dtype=float)

    pred_idx = np.argmax(probs, axis=1)

    if hasattr(model_trainer, "label_encoder"):
        preds = model_trainer.label_encoder.inverse_transform(pred_idx)
    else:
        if hasattr(model_trainer.classifier, "classes_"):
            preds = np.asarray(model_trainer.classifier.classes_)[pred_idx]
        else:
            preds = pred_idx

    return probs, preds


def build_blocks(df: pd.DataFrame, block_size=None):
    if block_size is None:
        return [df.reset_index(drop=True)]

    blocks = []
    n = len(df)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = df.iloc[start:end].reset_index(drop=True)
        if len(block) > 0:
            blocks.append(block)

    return blocks


# =========================================================
# LMT scoring
# =========================================================
def lmt_score_with_predicted_class(model_trainer, X_block, baseline_shapes):
    z = model_trainer.encode(X_block)
    _, preds = get_probs_and_preds(model_trainer, X_block)

    preds = np.asarray(preds, dtype=object)
    all_scores = []

    for c, (mu, Sigma) in baseline_shapes.items():
        mask = (preds == c)
        Zc = z[mask]

        if len(Zc) == 0:
            continue

        eps = 1e-6
        Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0])

        inv_sigma = np.linalg.inv(Sigma_reg)
        diffs = Zc - mu
        d2 = np.sum(diffs @ inv_sigma * diffs, axis=1)

        all_scores.extend(d2.tolist())

    if len(all_scores) == 0:
        return np.nan

    return float(np.median(all_scores))


# =========================================================
# Reference computation
# =========================================================
def compute_references(model_trainer, X_train, y_train):
    print("\n[INFO] Computing reference statistics...")

    t0 = time.time()
    print("[REF] 1/5 CADE reference...")
    cade_ref = compute_cade_class_reference(model_trainer, X_train, y_train)
    print(f"[REF] CADE done in {time.time() - t0:.2f}s")

    t1 = time.time()
    print("[REF] 2/5 CHEN reference...")
    chen_ref = compute_chen_reference(model_trainer, X_train, y_train)
    print(f"[REF] CHEN done in {time.time() - t1:.2f}s")

    t2 = time.time()
    print("[REF] 3/5 GIDX reference...")
    gidx_ref = compute_gidx_reference(model_trainer, X_train, y_train)
    print(f"[REF] GIDX done in {time.time() - t2:.2f}s")

    t3 = time.time()
    print("[REF] 4/5 PE reference...")
    pe_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    print(f"[REF] PE done in {time.time() - t3:.2f}s")

    t4 = time.time()
    print("[REF] 5/5 LMT reference...")
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)
    print(f"[REF] LMT done in {time.time() - t4:.2f}s")

    print("[INFO] Reference stats ready")

    return {
        "cade": cade_ref,
        "chen": chen_ref,
        "gidx": gidx_ref,
        "pe": pe_ref,
        "lmt_shapes": baseline_shapes,
        "lmt_ref": lmt_ref,
    }


# =========================================================
# Block scoring
# =========================================================
def compute_block_scores(model_trainer, X_block, y_block, refs):
    y_block = np.asarray(y_block, dtype=object)

    benign_mask = (y_block == "benign")
    mal_mask = (y_block == "malicious")

    X_b = X_block[benign_mask]
    y_b = y_block[benign_mask]

    X_m = X_block[mal_mask]
    y_m = y_block[mal_mask]

    def score_subset(method, X, y):
        if len(y) == 0:
            return np.nan

        if method == "cade":
            z = model_trainer.encode(X)
            return float(cade_score_min_over_classes(z, refs["cade"]))
        
        elif method == "chen":
            _, score = chen_fast_detect(model_trainer, X, y, refs["chen"])
            return float(score)

        elif method == "gidx":
            _, score = gidx_fast_detect(model_trainer, X, y, refs["gidx"])
            return float(score)

        elif method == "pe":
            _, score = pe_fast_detect(model_trainer, X, refs["pe"])
            return float(score)

        elif method == "lmt":
            return float(
                lmt_score_with_predicted_class(
                    model_trainer,
                    X,
                    refs["lmt_shapes"],
                )
            )

        else:
            raise ValueError(f"Unsupported method: {method}")
        
    mal_ratio = len(y_m) / len(y_block) if len(y_block) > 0 else 0


    out = {
        "n_total": int(len(y_block)),
        "n_benign": int(len(y_b)),
        "n_malicious": int(len(y_m)),
        "malicious_ratio": float(mal_ratio),
    }

    methods = ["cade","chen" ,"gidx", "pe", "lmt"]

    for m in methods:
        out[f"{m}_score_benign"] = score_subset(m, X_b, y_b)
        out[f"{m}_score_malicious"] = score_subset(m, X_m, y_m)

    return out


# =========================================================
# Main evaluation loop
# =========================================================
def run_stream_scoring(model_trainer, scaler, refs, ordered_files, feature_cols):
    rows = []

    pbar = tqdm(ordered_files, desc="Evaluating IDS17 stream", unit="file", dynamic_ncols=True)

    global_block_id = 0

    for file_idx, fname in enumerate(pbar, start=1):
        fp = os.path.join(BASE_DIR, fname)
        df_day = load_ids17_file(fp)
        df_day["group_label"] = df_day["Label"].map(map_binary_label)

        blocks = build_blocks(df_day, BLOCK_SIZE)

        for block_idx, block in enumerate(blocks):
            X_raw = block[feature_cols].values.astype(np.float32)
            y_block = block["group_label"].values.astype(object)

            X_block = scaler.transform(X_raw).astype(np.float32)

            _, y_pred = get_probs_and_preds(model_trainer, X_block)

            acc = accuracy_score(y_block, y_pred)
            f1 = f1_score(y_block, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_block, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_block, y_pred, average="macro", zero_division=0)


            # per-class accuracy
            benign_acc = np.mean(y_pred[y_block == "benign"] == "benign") if np.any(y_block == "benign") else np.nan
            mal_acc = np.mean(y_pred[y_block == "malicious"] == "malicious") if np.any(y_block == "malicious") else np.nan

            
            scores = compute_block_scores(
                model_trainer=model_trainer,
                X_block=X_block,
                y_block=y_block,
                refs=refs,
            )

            row = {
                "global_block_id": int(global_block_id),
                "file_order": int(file_idx),
                "file_name": fname,
                "block_id_in_file": int(block_idx),
                "mode": "full_block",
                "samples_used": int(len(block)),
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "acc_benign": float(benign_acc),
                "acc_malicious": float(mal_acc),
            }
            row.update(scores)
            rows.append(row)

            global_block_id += 1

        pbar.set_postfix({
            "file": fname[:20],
            "blocks": len(blocks),
            "rows": len(df_day),
        })

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
def main():
    set_seed(RANDOM_STATE)

    print("[INFO] Loading Monday BENIGN for training...")
    df_monday = load_ids17_file(os.path.join(BASE_DIR, MONDAY_FILE))
    df_monday = df_monday[df_monday["Label"] == "BENIGN"].reset_index(drop=True)

    print("[INFO] Loading Friday PortScan for training...")
    df_port = load_ids17_file(os.path.join(BASE_DIR, PORTSCAN_FILE))
    df_port = df_port[df_port["Label"] == PORTSCAN_LABEL].reset_index(drop=True)

    print(f"[INFO] Monday benign rows: {len(df_monday)}")
    print(f"[INFO] PortScan rows: {len(df_port)}")

    # sample balanced training set
    df_train_b = df_monday.copy()
    df_train_m = df_port.copy()

    df_train_b["group_label"] = "benign"
    df_train_m["group_label"] = "malicious"

    df_train = pd.concat([df_train_b, df_train_m], ignore_index=True)
    df_train = df_train.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


    print("\n[INFO] Training set summary:")
    print(df_train["group_label"].value_counts())


    feature_cols = get_feature_columns(df_train)

    print(f"[INFO] Number of feature columns: {len(feature_cols)}")
    print(f"[INFO] First 10 feature columns: {feature_cols[:10]}")

    X_train_raw, y_train = build_xy(df_train, feature_cols)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)

    print(f"\n[INFO] X_train shape = {X_train.shape}")

    model_cls = get_model_cls(MODEL_NAME)
    model_trainer = model_cls()

    print("\n[INFO] Training model...")
    model_trainer.fit(X_train, y_train)

    # optional quick sanity test on training set
    print("[INFO] Sanity evaluation on train set...")
    _, y_pred_train = get_probs_and_preds(model_trainer, X_train)

    acc = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
    prec = precision_score(y_train, y_pred_train, average="macro", zero_division=0)
    rec = recall_score(y_train, y_pred_train, average="macro", zero_division=0)

    print("\n=== TRAIN SUMMARY ===")
    print(f"Train samples: {len(y_train)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    refs = compute_references(model_trainer, X_train, y_train)

    print("\n[INFO] Running chronological stream evaluation...")
    df_results = run_stream_scoring(
        model_trainer=model_trainer,
        scaler=scaler,
        refs=refs,
        ordered_files=ORDERED_FILES,
        feature_cols=feature_cols,
    )

    df_results.to_csv(OUT_CSV, index=False)

    print(f"\n[INFO] Saved results to {OUT_CSV}")
    print(df_results.head())


if __name__ == "__main__":
    main()