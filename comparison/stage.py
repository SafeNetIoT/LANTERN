import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ModelUtils import (
    AutoencoderClassifierTrainer,
    RFModelTrainer,
    ContrastiveModelTrainer,
)

from utils.DriftUtils import (
    compute_cade_reference,
    cade_fast_detect,
    compute_cade_class_reference,
    cade_score_min_over_classes,
    compute_chen_reference,
    chen_fast_detect,
    mateen_ks_test,
    compute_gidx_reference,
    gidx_fast_detect,
    compute_entropy_reference,
    pe_fast_detect,
    compute_lmt_reference,
    lmt_fast_detect,
)

warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset: str
    data_dir: str
    out_dir: str
    class_order: list
    files: Optional[dict]
    drop_cols: Optional[list]
    model_name: str = "cae"
    random_state: int = 42
    per_class_total: int = 5000
    train_ratio: float = 0.80
    unseen_test_samples: int = 400
    seen_test_total: int = 1600
    methods: tuple = ("cade", "chen", "mateen", "gidx", "pe", "lmt")


def get_config(dataset: str = "ciciot2023") -> ExperimentConfig:
    if dataset == "ciciot2023":
        return ExperimentConfig(
            dataset="ciciot2023",
            data_dir="../data/public_datasets/CICIOT2023",
            out_dir="../data/public_datasets/resiot",
            files={
                "benign": "BenignTraffic.pcap.csv",
                "SYN Flood": "DoS-SYN_Flood3.pcap.csv",
                "HTTP Flood": "DDoS-HTTP_Flood.pcap.csv",
                "DNS_Spoofing": "DNS_Spoofing.pcap.csv",
                "Dictionary Brute Force": "DictionaryBruteForce.pcap.csv",
                "OS Scan": "Recon-OSScan.pcap.csv",
                "Command Injection": "CommandInjection.pcap.csv",
                "BrowserHijacking": "BrowserHijacking.pcap.csv",
                "SQL Injection": "SqlInjection.pcap.csv",
                "XSS": "XSS.pcap.csv",
            },
            class_order=[
                "benign",
                "HTTP Flood",
                "OS Scan",
                "Command Injection",
                "BrowserHijacking",
                "SQL Injection",
                "XSS",
            ],
            drop_cols=None,
        )

    if dataset == "android":
        return ExperimentConfig(
            dataset="android",
            data_dir="../data/public_datasets/andriod_mal",
            out_dir="../data/public_datasets/res_android_mal",
            files=None,
            class_order=[
                "benign",
                "dowgin",
                "fakeapp",
                "simplelocker",
                "plankton",
                "svpeng",
                "youmi",
            ],
            drop_cols=[
                "Flow ID",
                "Source IP",
                "Source Port",
                "Destination IP",
                "Destination Port",
                "Protocol",
                "Timestamp",
                "Label",
            ],
        )

    raise ValueError(f"Unknown dataset: {dataset}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ciciot2023",
        choices=["ciciot2023", "android"],
        help="Dataset to run: ciciot2023 or android",
    )
    return parser.parse_args()


ARGS = parse_args()
CFG = get_config(ARGS.dataset)

DATA_DIR = CFG.data_dir
OUT_DIR = CFG.out_dir
FILES = CFG.files
CLASS_ORDER = CFG.class_order
DROP_COLS = CFG.drop_cols

MODEL_NAME = CFG.model_name
RANDOM_STATE = CFG.random_state
PER_CLASS_TOTAL = CFG.per_class_total
TRAIN_RATIO = CFG.train_ratio
UNSEEN_TEST_SAMPLES = CFG.unseen_test_samples
SEEN_TEST_TOTAL = CFG.seen_test_total
METHODS = list(CFG.methods)

os.makedirs(OUT_DIR, exist_ok=True)
    
# =========================================================
# General helpers
# =========================================================
def load_and_prepare_data():
    if CFG.dataset == "ciciot2023":
        return load_flat_file_dataset()

    if CFG.dataset == "android":
        return load_folder_dataset()

    raise ValueError(f"Unsupported dataset: {CFG.dataset}")


def load_flat_file_dataset():
    data_train = {}
    data_test = {}

    for i, cls in enumerate(CLASS_ORDER):
        path = os.path.join(DATA_DIR, FILES[cls])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = safe_read_csv(path)
        df = clean_dataframe(df)

        sampled = sample_dataframe(df, PER_CLASS_TOTAL, RANDOM_STATE + i)
        df_train, df_test = train_test_split_per_class(
            sampled,
            train_ratio=TRAIN_RATIO,
            seed=RANDOM_STATE + i,
        )

        data_train[cls] = df_train
        data_test[cls] = df_test

        print(f"[INFO] {cls}: train={len(df_train)}, test={len(df_test)}")

    return data_train, data_test


def load_folder_dataset():
    data_train = {}
    data_test = {}

    for i, cls in enumerate(CLASS_ORDER):
        class_dir = os.path.join(DATA_DIR, cls)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        csv_files = sorted(
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.endswith(".csv")
        )

        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in: {class_dir}")

        df_list = []

        for fp in csv_files:
            df = safe_read_csv(fp)
            df = clean_dataframe(df)
            df = df.loc[:, ~df.columns.duplicated()].copy()

            drop_cols = [c for c in DROP_COLS if c in df.columns]
            df = df.drop(columns=drop_cols)

            df_list.append(df)

        df_all = pd.concat(df_list, ignore_index=True)
        df_all = clean_dataframe(df_all)

        sampled = sample_dataframe(df_all, PER_CLASS_TOTAL, RANDOM_STATE + i)
        df_train, df_test = train_test_split_per_class(
            sampled,
            train_ratio=TRAIN_RATIO,
            seed=RANDOM_STATE + i,
        )

        data_train[cls] = df_train
        data_test[cls] = df_test

        print(f"[INFO] {cls}: total={len(df_all)}, train={len(df_train)}, test={len(df_test)}")

    return data_train, data_test



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def minmax_norm(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    vmin = values.min()
    vmax = values.max()
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def get_model_cls(name: str):
    mapping = {
        "rf": RFModelTrainer,
        "ae": AutoencoderClassifierTrainer,
        "cae": ContrastiveModelTrainer,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported model name: {name}")
    return mapping[name]


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index(drop=True)
    return df


def infer_feature_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise RuntimeError("No numeric feature columns found.")
    return numeric_cols


def sample_dataframe(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def train_test_split_per_class(df: pd.DataFrame, train_ratio: float, seed: int):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_ratio)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test = df.iloc[n_train:].reset_index(drop=True)
    return df_train, df_test


def build_xy(df: pd.DataFrame, feature_cols, label_name: str):
    X = df[feature_cols].values.astype(np.float32)
    y = np.array([label_name] * len(df), dtype=object)
    return X, y


def build_multiclass_xy(df_by_class, classes, feature_cols):
    xs = []
    ys = []
    for cls in classes:
        df = df_by_class[cls]
        X, y = build_xy(df, feature_cols, cls)
        xs.append(X)
        ys.append(y)
    X_all = np.vstack(xs)
    y_all = np.concatenate(ys)
    return X_all, y_all


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


def extract_recon_errors(model_trainer, X, y):
    _, recon_errors = model_trainer.evaluate(X, y)
    return np.asarray(recon_errors, dtype=float)


def lmt_score_with_predicted_class(model_trainer, X_block, baseline_shapes):
    """
    LMT score for benchmark use.

    Instead of using true labels, assign each sample to the predicted seen class,
    then compute Mahalanobis distance to that class reference shape.
    This makes LMT valid for unseen-class probing.
    """
    z_block = model_trainer.encode(X_block)

    probs, preds = get_probs_and_preds(model_trainer, X_block)
    preds = np.asarray(preds, dtype=object)

    all_scores = []

    for c, (mu_c, Sigma_c) in baseline_shapes.items():
        mask = (preds == c)
        Zc = z_block[mask]
        if len(Zc) == 0:
            continue

        inv_sigma = np.linalg.inv(Sigma_c)
        diffs = Zc - mu_c
        d2 = np.sum(diffs @ inv_sigma * diffs, axis=1)
        all_scores.extend(d2.tolist())

    if len(all_scores) == 0:
        return np.nan

    return float(np.mean(all_scores))

'''
def lmt_score_with_predicted_class(model_trainer, X_block, baseline_shapes):
    # Standardised Cov
    z = model_trainer.encode(X_block)
    probs, preds = get_probs_and_preds(model_trainer, X_block)

    preds = np.asarray(preds, dtype=object)

    all_scores = []

    for c, (mu, Sigma) in baseline_shapes.items():
        mask = (preds == c)
        Zc = z[mask]

        if len(Zc) == 0:
            continue

        eps = 1e-6
        Sigma_c_reg = Sigma + eps * np.eye(Sigma.shape[0])

        inv_sigma = np.linalg.inv(Sigma_c_reg)
        diffs = Zc - mu
        d2 = np.sum(diffs @ inv_sigma * diffs, axis=1)

        all_scores.extend(d2.tolist())

    if len(all_scores) == 0:
        return np.nan

    return float(np.median(all_scores))
'''

def compute_conditional_scores(
    model_trainer,
    X_test,
    y_test,
    seen_classes,
    unseen_class,
    cade_ref,
    chen_ref,
    gidx_ref,
    pe_ref,
    baseline_shapes,
    train_recon_errors,
):
    """
    Compute class conditional drift scores for:
    - benign
    - seen attack classes mean (excluding benign)
    - unseen class
    """
    y_test = np.asarray(y_test, dtype=object)

    def _subset_score(method_name, X_sub, y_sub):
        if len(y_sub) == 0:
            return np.nan

        if method_name == "cade":
            z_sub = model_trainer.encode(X_sub)
            score = cade_score_min_over_classes(z_sub, cade_ref)
            return float(score)

        elif method_name == "chen":
            _, score = chen_fast_detect(model_trainer, X_sub, y_sub, chen_ref)
            return float(score)

        elif method_name == "mateen":
            test_recon_sub = extract_recon_errors(model_trainer, X_sub, y_sub)
            _, score = mateen_ks_test(train_recon_errors, test_recon_sub)
            return float(score)

        elif method_name == "gidx":
            _, score = gidx_fast_detect(model_trainer, X_sub, y_sub, gidx_ref)
            return float(score)

        elif method_name == "pe":
            _, score = pe_fast_detect(model_trainer, X_sub, pe_ref)
            return float(score)

        elif method_name == "lmt":
            score = lmt_score_with_predicted_class(
                model_trainer,
                X_sub,
                baseline_shapes,
            )
            return float(score)

        else:
            raise ValueError(f"Unsupported method: {method_name}")

    out = {}
    cond_methods = ["cade", "chen", "mateen", "gidx", "pe", "lmt"]

    # benign
    benign_mask = (y_test == "benign")
    X_benign = X_test[benign_mask]
    y_benign = y_test[benign_mask]

    for method in cond_methods:
        out[f"{method}_score_benign"] = _subset_score(method, X_benign, y_benign)

    # seen mean
    seen_attack_classes = [c for c in seen_classes if c != "benign"]

    for method in cond_methods:
        seen_scores = []
        for cls in seen_attack_classes:
            cls_mask = (y_test == cls)
            X_cls = X_test[cls_mask]
            y_cls = y_test[cls_mask]
            if len(y_cls) == 0:
                continue
            seen_scores.append(_subset_score(method, X_cls, y_cls))

        out[f"{method}_score_seen_mean"] = (
            float(np.mean(seen_scores)) if len(seen_scores) > 0 else np.nan
        )

    # unseen
    if unseen_class is None:
        for method in cond_methods:
            out[f"{method}_score_unseen"] = np.nan
    else:
        unseen_mask = (y_test == unseen_class)
        X_unseen = X_test[unseen_mask]
        y_unseen = y_test[unseen_mask]

        for method in cond_methods:
            out[f"{method}_score_unseen"] = _subset_score(method, X_unseen, y_unseen)

    return out




# =========================================================
# Main experiment
# =========================================================
def main():
    set_seed(RANDOM_STATE)

    print("[INFO] Loading and splitting data...")
    train_splits, test_splits = load_and_prepare_data()

    feature_cols = infer_feature_columns(train_splits["benign"])
    print(f"[INFO] Number of numeric features: {len(feature_cols)}")

    model_cls = get_model_cls(MODEL_NAME)
    print(f"[INFO] Model: {MODEL_NAME}")

    rows = []

    # ==================================================
    # Baseline stage (no unseen class)
    # ==================================================
    print("\n" + "=" * 70)
    print("[INFO] Baseline Stage (seen only)")
    seen_classes = CLASS_ORDER[:2]
    unseen_class = None

    # --- train set ---
    X_train_raw, y_train = build_multiclass_xy(
        train_splits,
        seen_classes,
        feature_cols,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)

    model_trainer = model_cls()
    model_trainer.fit(X_train, y_train)

    # --- references ---
    _, train_recon_errors = model_trainer.evaluate(X_train, y_train)
    train_recon_errors = np.asarray(train_recon_errors, dtype=float)

    cade_ref = compute_cade_class_reference(model_trainer, X_train, y_train)
    chen_ref = compute_chen_reference(model_trainer, X_train, y_train)
    gidx_ref = compute_gidx_reference(model_trainer, X_train, y_train)
    pe_ref = compute_entropy_reference(model_trainer, X_train, y_train)
    baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

    # --- test set (seen only) ---
    per_class_seen_df_list = []
    k = len(seen_classes)
    base = SEEN_TEST_TOTAL // k
    rem = SEEN_TEST_TOTAL % k

    for i, cls in enumerate(seen_classes):
        take_n = base + (1 if i < rem else 0)
        src = test_splits[cls]
        take_n = min(take_n, len(src))
        df_take = src.sample(n=take_n, random_state=RANDOM_STATE + 1000 + i).copy()
        df_take["__label__"] = cls
        per_class_seen_df_list.append(df_take)

    test_df = pd.concat(per_class_seen_df_list, ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=RANDOM_STATE + 1999).reset_index(drop=True)

    X_test_raw = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["__label__"].values.astype(object)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # --- performance ---
    probs, preds = get_probs_and_preds(model_trainer, X_test)
    accuracy = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)

    test_recon_errors = extract_recon_errors(model_trainer, X_test, y_test)

    # --- drift ---
    z_test = model_trainer.encode(X_test)

    cade_score = cade_score_min_over_classes(z_test, cade_ref)
    chen_detected, chen_score = chen_fast_detect(model_trainer, X_test, y_test, chen_ref)
    mateen_detected, mateen_score = mateen_ks_test(train_recon_errors, test_recon_errors)
    gidx_detected, gidx_score = gidx_fast_detect(model_trainer, X_test, y_test, gidx_ref)
    pe_detected, pe_score = pe_fast_detect(model_trainer, X_test, pe_ref)

    lmt_score = lmt_score_with_predicted_class(model_trainer, X_test, baseline_shapes)

    # --- conditional ---
    conditional_scores = compute_conditional_scores(
        model_trainer=model_trainer,
        X_test=X_test,
        y_test=y_test,
        seen_classes=seen_classes,
        unseen_class=unseen_class,
        cade_ref=cade_ref,
        chen_ref=chen_ref,
        gidx_ref=gidx_ref,
        pe_ref=pe_ref,
        baseline_shapes=baseline_shapes,
        train_recon_errors=train_recon_errors,
    )
    row = {
        "stage_id": 0,
        "seen_classes": " | ".join(seen_classes),
        "probe_class": "None",

        "n_train": int(len(X_train)),
        "n_test_seen": int(len(test_df)),
        "n_test_unseen": 0,
        "test_unseen_ratio": 0.0,

        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),

        "cade_detected": np.nan,
        "cade_score": float(cade_score),

        "chen_detected": bool(chen_detected),
        "chen_score": float(chen_score),

        "mateen_detected": bool(mateen_detected),
        "mateen_score": float(mateen_score),

        "gidx_detected": bool(gidx_detected),
        "gidx_score": float(gidx_score),

        "pe_detected": bool(pe_detected),
        "pe_score": float(pe_score),

        "lmt_detected": np.nan,
        "lmt_score": float(lmt_score),

        "gidx_ref_median": float(gidx_ref["median"]),
        "gidx_ref_mad": float(gidx_ref["mad"]),

        "pe_ref_median": float(pe_ref["median"]),
        "pe_ref_mad": float(pe_ref["mad"]),
        "pe_ref_mean": float(pe_ref["mean"]),
        "pe_ref_std": float(pe_ref["std"]),

        "lmt_ref_median": float(lmt_ref["median"]),
        "lmt_ref_mad": float(lmt_ref["mad"]),
    }

    row.update(conditional_scores)
    rows.append(row)


    # ================================
    # seen progress
    # ================================
    for n_seen in range(2, len(CLASS_ORDER)):
        seen_classes = CLASS_ORDER[:n_seen]
        unseen_class = CLASS_ORDER[n_seen]

        print()
        print("=" * 70)
        print(f"[INFO] Stage {n_seen - 1}")
        print(f"[INFO] Seen classes: {seen_classes}")
        print(f"[INFO] Probe class: {unseen_class}")

        # -------------------------------------------------
        # Build training set from seen classes only
        # -------------------------------------------------
        X_train_raw, y_train = build_multiclass_xy(
            train_splits,
            seen_classes,
            feature_cols,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)

        # -------------------------------------------------
        # Train a fresh model at every stage
        # -------------------------------------------------
        print("[INFO] Training model...")
        model_trainer = model_cls()
        model_trainer.fit(X_train, y_train)

        # -------------------------------------------------
        # Build references on current training distribution
        # -------------------------------------------------
        print("[INFO] Building references...")
        _, train_recon_errors = model_trainer.evaluate(X_train, y_train)
        train_recon_errors = np.asarray(train_recon_errors, dtype=float)

        '''
        z_train = model_trainer.encode(X_train)
        if hasattr(model_trainer, "encode_labels"):
            y_train_enc = model_trainer.encode_labels(y_train)
        else:
            _, y_train_enc = np.unique(y_train, return_inverse=True)
        '''
        
        #cade_ref = compute_cade_reference(z_train, y_train_enc)
        cade_ref = compute_cade_class_reference(model_trainer, X_train, y_train)
        #print("CADE ref structure:")
        #for k, v in cade_ref.items():
        #    print(k, type(v), list(v.keys()))
        chen_ref = compute_chen_reference(model_trainer, X_train, y_train)
        gidx_ref = compute_gidx_reference(model_trainer, X_train, y_train)
        pe_ref = compute_entropy_reference(model_trainer, X_train, y_train)
        baseline_shapes, lmt_ref = compute_lmt_reference(model_trainer, X_train, y_train)

        # -------------------------------------------------
        # Build mixed test set with preserved labels
        # -------------------------------------------------
        per_class_seen_df_list = []
        k = len(seen_classes)
        base = SEEN_TEST_TOTAL // k
        rem = SEEN_TEST_TOTAL % k

        for i, cls in enumerate(seen_classes):
            take_n = base + (1 if i < rem else 0)
            src = test_splits[cls]
            take_n = min(take_n, len(src))
            df_take = src.sample(n=take_n, random_state=RANDOM_STATE + 500 + n_seen + i).copy()
            df_take["__label__"] = cls
            per_class_seen_df_list.append(df_take)

        seen_test_df = pd.concat(per_class_seen_df_list, ignore_index=True)

        unseen_test_df = test_splits[unseen_class].sample(
            n=min(UNSEEN_TEST_SAMPLES, len(test_splits[unseen_class])),
            random_state=RANDOM_STATE + 100 + n_seen,
        ).copy()
        unseen_test_df["__label__"] = unseen_class

        test_df = pd.concat([seen_test_df, unseen_test_df], ignore_index=True)
        test_df = test_df.sample(frac=1.0, random_state=RANDOM_STATE + 999 + n_seen).reset_index(drop=True)

        X_test_raw = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["__label__"].values.astype(object)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        # -------------------------------------------------
        # Evaluate performance
        # -------------------------------------------------
        probs, preds = get_probs_and_preds(model_trainer, X_test)
        accuracy = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)

        test_recon_errors = extract_recon_errors(model_trainer, X_test, y_test)

        # -------------------------------------------------
        # Overall drift scores
        # -------------------------------------------------
        z_test = model_trainer.encode(X_test)

        cade_score = cade_score_min_over_classes(z_test, cade_ref)
        cade_thr = cade_ref["_global"]["median"] + 3.0 * cade_ref["_global"]["mad"]
        cade_detected = bool(cade_score > cade_thr)

        chen_detected, chen_score = chen_fast_detect(model_trainer, X_test, y_test, chen_ref)
        mateen_detected, mateen_score = mateen_ks_test(train_recon_errors, test_recon_errors)

        # If you use the original supervised G-idx, keep this call.
        # If you switched to the unsupervised version, remove y_test here.
        gidx_detected, gidx_score = gidx_fast_detect(model_trainer, X_test, y_test, gidx_ref)

        pe_detected, pe_score = pe_fast_detect(model_trainer, X_test, pe_ref)
        lmt_score = lmt_score_with_predicted_class(
            model_trainer,
            X_test,
            baseline_shapes,
        )

        lmt_thr = lmt_ref["median"] + 3.0 * lmt_ref["mad"]
        lmt_detected = bool(lmt_score > lmt_thr)

        conditional_scores = compute_conditional_scores(
            model_trainer=model_trainer,
            X_test=X_test,
            y_test=y_test,
            seen_classes=seen_classes,
            unseen_class=unseen_class,
            cade_ref=cade_ref,
            chen_ref=chen_ref,
            gidx_ref=gidx_ref,
            pe_ref=pe_ref,
            baseline_shapes=baseline_shapes,
            train_recon_errors=train_recon_errors,
            #lmt_ref=lmt_ref,
        )

        row = {
            "stage_id": int(n_seen - 1),
            "seen_classes": " | ".join(seen_classes),
            "probe_class": unseen_class,
            "n_train": int(len(X_train)),
            "n_test_seen": int(len(seen_test_df)),
            "n_test_unseen": int(len(unseen_test_df)),
            "test_unseen_ratio": float(len(unseen_test_df) / len(test_df)),

            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),

            "cade_detected": bool(cade_detected),
            "cade_score": float(cade_score),

            "chen_detected": bool(chen_detected),
            "chen_score": float(chen_score),

            "mateen_detected": bool(mateen_detected),
            "mateen_score": float(mateen_score),

            "gidx_detected": bool(gidx_detected),
            "gidx_score": float(gidx_score),

            "pe_detected": bool(pe_detected),
            "pe_score": float(pe_score),

            "lmt_detected": bool(lmt_detected),
            "lmt_score": float(lmt_score),

            "gidx_ref_median": float(gidx_ref["median"]),
            "gidx_ref_mad": float(gidx_ref["mad"]),

            "pe_ref_median": float(pe_ref["median"]),
            "pe_ref_mad": float(pe_ref["mad"]),
            "pe_ref_mean": float(pe_ref["mean"]),
            "pe_ref_std": float(pe_ref["std"]),

            "lmt_ref_median": float(lmt_ref["median"]),
            "lmt_ref_mad": float(lmt_ref["mad"]),
        }

        row.update(conditional_scores)
        rows.append(row)

    df_results = pd.DataFrame(rows)

    # -----------------------------------------------------
    # Normalize scores for plotting
    # -----------------------------------------------------
    for method in METHODS:
        raw_col = f"{method}_score"
        norm_col = f"{method}_score_norm"
        df_results[norm_col] = minmax_norm(df_results[raw_col].values)

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------
    csv_path = os.path.join(OUT_DIR, f"public_progressive_{MODEL_NAME}.csv")
    df_results.to_csv(csv_path, index=False)

    print()
    print("[INFO] Saved:")
    print(csv_path)


    print()
    print("[INFO] Final results:")
    print(
        df_results[
            [
                "seen_classes",
                "probe_class",
                "n_train",
                "n_test_seen",
                "n_test_unseen",
                "accuracy",
                "f1_macro",

                "cade_score",
                "cade_score_benign",
                "cade_score_seen_mean",
                "cade_score_unseen",

                "chen_score",
                "chen_score_benign",
                "chen_score_seen_mean",
                "chen_score_unseen",

                "mateen_score",
                "mateen_score_benign",
                "mateen_score_seen_mean",
                "mateen_score_unseen",

                "gidx_score",
                "gidx_score_benign",
                "gidx_score_seen_mean",
                "gidx_score_unseen",

                "pe_score",
                "pe_score_benign",
                "pe_score_seen_mean",
                "pe_score_unseen",

                "lmt_score",
                "lmt_score_benign",
                "lmt_score_seen_mean",
                "lmt_score_unseen",
            ]
        ]
    )


if __name__ == "__main__":
    main()