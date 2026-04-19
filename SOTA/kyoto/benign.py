import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.ModelUtils import (
    AutoencoderClassifierTrainer,
    RFModelTrainer,
    ContrastiveModelTrainer,
)

from utils.DataUtils import (
    list_kyoto_daily_csv_files,
    get_kyoto_day_slice,
    load_kyoto_csv_day,
    load_kyoto_csv_range,
    take_random_contiguous_slice,
    build_kyoto_preprocessor,
    build_kyoto_xy,
    print_kyoto_day_summary,
)


warnings.filterwarnings("ignore")

BASE_DIR = "../../data/public_datasets/kyoto_csv"
YEAR = "2015"
MODEL_NAME = "cae"
RANDOM_STATE = 42

USE_FAST_DEBUG = False
DEBUG_SLICE_SIZE = 10000


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


#================================
# Drift References
#================================
from utils.DriftUtils import (
    compute_cade_class_reference,
    compute_chen_reference,
    compute_gidx_reference,
    compute_entropy_reference,
    compute_lmt_reference,
)


def compute_references(model_trainer, X_train, y_train):
    print("\n[INFO] Computing reference statistics...")

    t0 = time.time()
    print("[REF] 1/5 CADE reference...")
    cade_ref = compute_cade_class_reference(model_trainer, X_train, y_train)
    print(f"[REF] CADE done in {time.time() - t0:.2f}s")

    '''
    t1 = time.time()
    print("[REF] 2/5 CHEN reference...")
    chen_ref = compute_chen_reference(model_trainer, X_train, y_train)
    print(f"[REF] CHEN done in {time.time() - t1:.2f}s")
    '''

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
        #"chen": chen_ref,
        "gidx": gidx_ref,
        "pe": pe_ref,
        "lmt_shapes": baseline_shapes,
        "lmt_ref": lmt_ref,
    }

#======================
# Block Scoring
#======================
from utils.DriftUtils import (
    cade_score_min_over_classes,
    chen_fast_detect,
    gidx_fast_detect,
    pe_fast_detect,
)


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

def maybe_slice_day(df_day, day_index):
    """
    Use a random contiguous slice only in debug mode.
    Otherwise keep the full day.
    """
    if not USE_FAST_DEBUG:
        return df_day.reset_index(drop=True)

    return take_random_contiguous_slice(
        df_day,
        n=DEBUG_SLICE_SIZE,
        seed=RANDOM_STATE + day_index,
    )

def compute_block_scores(
    model_trainer,
    X_block,
    y_block,
    refs,
):
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

        
        #elif method == "chen":
        #    _, score = chen_fast_detect(model_trainer, X, y, refs["chen"])
        #    return float(score)
        

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

    out = {
        "n_total": len(y_block),
        "n_benign": len(y_b),
        "n_malicious": len(y_m),
    }

    METHODS = ["cade", "gidx", "pe", "lmt"]

    for m in METHODS:
        out[f"{m}_score_benign"] = score_subset(m, X_b, y_b)
        out[f"{m}_score_malicious"] = score_subset(m, X_m, y_m)

    return out


def run_daily_scoring(
    model_trainer,
    preprocessor,
    refs,
    eval_files,
):
    rows = []

    pbar = tqdm(
        eval_files,
        desc="Evaluating",
        unit="day",
        dynamic_ncols=True
    )

    for i, fp in enumerate(pbar, start=1):
        day_name = os.path.splitext(os.path.basename(fp))[0]
        t0 = time.time()

        df_day = load_kyoto_csv_day(fp)
        df_day = maybe_slice_day(df_day, i)

        X_block = preprocessor.transform(df_day).astype(np.float32)
        y_block = df_day["group_label"].values

        _, y_pred = get_probs_and_preds(model_trainer, X_block)

        acc = accuracy_score(y_block, y_pred)
        f1 = f1_score(y_block, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_block, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_block, y_pred, average="macro", zero_division=0)

        scores = compute_block_scores(
            model_trainer,
            X_block,
            y_block,
            refs,
        )

        row = {
            "block_start": df_day["day"].iloc[0],
            "mode": "debug_slice" if USE_FAST_DEBUG else "full_day",
            "samples_used": int(len(df_day)),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        }

        row.update(scores)
        rows.append(row)

        elapsed = time.time() - t0
        pbar.set_postfix({
            "day": day_name,
            "n": int(len(df_day)),
            "benign": int((df_day["group_label"] == "benign").sum()),
            "mal": int((df_day["group_label"] == "malicious").sum()),
            "f1": f"{f1:.3f}",
            "t": f"{elapsed:.2f}s",
        })

    return pd.DataFrame(rows)

def main():
    set_seed(RANDOM_STATE)

    daily_files = list_kyoto_daily_csv_files(BASE_DIR, year=YEAR)
    print(f"[INFO] Found {len(daily_files)} daily files")

    # 8 day setting
    # train: first 6 days
    # test: next 2 days
    train_files = get_kyoto_day_slice(daily_files, 0, 8)
    test_files = get_kyoto_day_slice(daily_files, 8, 9)

    print(f"[INFO] Train days: {len(train_files)}")
    print(f"[INFO] Test days: {len(test_files)}")

    df_train = load_kyoto_csv_range(train_files, verbose=True)
    df_test = load_kyoto_csv_range(test_files, verbose=True)

    print_kyoto_day_summary(df_train, name="train")
    print_kyoto_day_summary(df_test, name="test")

    preprocessor = build_kyoto_preprocessor()
    preprocessor.fit(df_train)

    X_train, y_train = build_kyoto_xy(df_train, preprocessor)
    X_test, y_test = build_kyoto_xy(df_test, preprocessor)

    print(f"\n[INFO] X_train shape = {X_train.shape}")
    print(f"[INFO] X_test shape  = {X_test.shape}")

    model_cls = get_model_cls(MODEL_NAME)
    model_trainer = model_cls()

    print("\n[INFO] Training model...")
    model_trainer.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    _, y_pred = get_probs_and_preds(model_trainer, X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== 8-DAY KYOTO TRAIN/TEST SUMMARY ===")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples : {len(y_test)}")
    print("\nTrain class counts:")
    print(df_train["group_label"].value_counts())
    print("\nTest class counts:")
    print(df_test["group_label"].value_counts())
    print("\nPerformance:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    # Compute references only after model is trained
    refs = compute_references(model_trainer, X_train, y_train)
    

    # evaluation files after first 8 days
    eval_files = get_kyoto_day_slice(daily_files, 8, len(daily_files))  #len(daily_files)

    df_results = run_daily_scoring(
        model_trainer,
        preprocessor,
        refs,
        eval_files,
    )

    df_results.to_csv("kyoto_daily_scores.csv", index=False)

    print("\n[INFO] Saved kyoto_daily_scores.csv")
    print(df_results.head())


if __name__ == "__main__":
    main()