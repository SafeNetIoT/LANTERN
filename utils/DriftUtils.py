# utils/DriftUtils.py
import numpy as np

# ======================================================
# (1) Core Computations
# ======================================================

def empirical_p_value(current_score, reference_scores, larger_is_more_abnormal=True):
    """
    Conformal style empirical p value.

    current_score: scalar score of current block
    reference_scores: list or np.ndarray of reference block scores
    """
    ref = np.asarray(reference_scores, dtype=float)

    if larger_is_more_abnormal:
        count = np.sum(ref >= current_score)
    else:
        count = np.sum(ref <= current_score)

    return (count + 1.0) / (len(ref) + 1.0)


def fuse_drift_evidence(p_lmt, p_pe, eps=1e-12):
    """
    Z_t = -log p_lmt - log p_pe
    """
    p_lmt = max(float(p_lmt), eps)
    p_pe = max(float(p_pe), eps)
    return -np.log(p_lmt) - np.log(p_pe)


def sequential_accumulation(prev_g, z_t, nu):
    """
    G_t = max(0, G_{t-1} + Z_t - nu)
    """
    return max(0.0, float(prev_g) + float(z_t) - float(nu))

# ======================================================
# (2) LMT Indicator
# ======================================================
def compute_baseline_shapes(Z_base, Y_base, classes, regularization=1e-6):
    shapes = {}
    for c in classes:
        Zc = Z_base[Y_base == c]
        if len(Zc) == 0:
            continue
        mu_c = np.mean(Zc, axis=0)
        cov_est = LedoitWolf().fit(Zc)
        Sigma_c = cov_est.covariance_ + regularization * np.eye(Zc.shape[1])
        shapes[c] = (mu_c, Sigma_c)
    return shapes


def compute_lmt_scores(Z_block, Y_pred, baseline_shapes):
    all_scores = []
    per_class_scores = {}
    for c, (mu_c, Sigma_c) in baseline_shapes.items():
        Zc = Z_block[Y_pred == c]
        if len(Zc) == 0:
            continue
        inv_Sigma = np.linalg.inv(Sigma_c)
        diffs = Zc - mu_c
        d2 = np.sum(diffs @ inv_Sigma * diffs, axis=1)
        per_class_scores[c] = np.mean(d2)
        all_scores.extend(d2)
    return per_class_scores, all_scores

# compute_lmt_block: see later part

# ======================================================
# (3) PE Indicator
# ======================================================
def compute_entropy_score(model_trainer, X_block, entropy_ref, use_mad=False):
    """
    Compute mean predictive entropy and drifted sample ratio for one block.

    Returns:
        H_block: mean entropy over all samples
        drift_ratio: fraction of samples exceeding decision threshold
        H_i: per-sample entropy array
    """
    z_block = model_trainer.encode(X_block)
    probs = model_trainer.classifier.predict_proba(z_block)
    probs = np.asarray(probs, dtype=float)

    noise = np.random.uniform(0.001, 0.005, size=probs.shape)
    probs = probs + noise
    probs = (probs + 1e-3) / (1 + 1e-3 * probs.shape[1])
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs_safe = np.clip(probs, 1e-12, 1.0)

    H_i = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    H_block = float(np.mean(H_i))

    # threshold for individual samples
    if use_mad:
        thr_individual = entropy_ref["thr_mad_decision"]
    else:
        thr_individual = entropy_ref["thr_std_decision"]

    drift_ratio = float(np.mean(H_i > thr_individual))
    print(f"[Entropy] Block mean entropy={H_block:.4f}, drift_ratio={drift_ratio:.4f}")
    return H_block, drift_ratio, H_i



# ======================================================
# (4) CONFORMAL + FUSED EVIDENCE + SEQUENTIAL HELPERS
# ======================================================

def compute_reference_block_scores(
    window_files,
    data_utils,
    encoder,
    model_trainer,
    entropy_ref,
    baseline_shapes,
    lmt_ref,
    use_mad_lmt=True,
):
    """
    Compute block level PE and LMT scores for every block in the current
    reference window. These scores are used for conformal calibration.

    Returns
    -------
    ref_pe_scores : list[float]
    ref_lmt_scores : list[float]
    """
    ref_pe_scores = []
    ref_lmt_scores = []

    for seq_file in window_files:
        try:
            df_block = data_utils.concatenate_blocks([seq_file])
            df_block = data_utils.preprocess_labels(df_block)
        except Exception as e:
            print(f"[WARN] Skipping reference block {seq_file}: {e}")
            continue

        if df_block.empty:
            continue

        X_block = encoder.transform(df_block)
        y_block = df_block["category"].values

        pe_score, _, _ = compute_entropy_score(
            model_trainer, X_block, entropy_ref, use_mad=False
        )

        lmt_score, _, _, _, _ = compute_lmt_block(
            model_trainer,
            X_block,
            y_block,
            baseline_shapes,
            lmt_ref,
            use_mad=use_mad_lmt,
        )

        ref_pe_scores.append(float(pe_score))
        ref_lmt_scores.append(float(lmt_score))

    return ref_pe_scores, ref_lmt_scores


def compute_reference_z_scores(ref_pe_scores, ref_lmt_scores):
    """
    Compute reference fused evidence values Z_t from reference block scores.
    """
    ref_z_scores = []

    for pe_s, lmt_s in zip(ref_pe_scores, ref_lmt_scores):
        p_pe = empirical_p_value(pe_s, ref_pe_scores, larger_is_more_abnormal=True)
        p_lmt = empirical_p_value(lmt_s, ref_lmt_scores, larger_is_more_abnormal=True)
        z_t = fuse_drift_evidence(p_lmt, p_pe)
        ref_z_scores.append(float(z_t))

    return ref_z_scores

def compute_nu(ref_z_scores, method="median"):
    """
    Compute sequential baseline allowance nu from reference Z scores.
    """
    ref_z_scores = np.asarray(ref_z_scores, dtype=float)

    if len(ref_z_scores) == 0:
        raise RuntimeError("[SEQ] Empty reference Z scores, cannot compute nu")

    if method == "median":
        return float(np.median(ref_z_scores))
    elif method == "mean":
        return float(np.mean(ref_z_scores))
    elif method == "q75":
        return float(np.quantile(ref_z_scores, 0.75))
    else:
        raise ValueError(f"[SEQ] Unsupported nu method: {method}")

def compute_block_drift_evidence(
    model_trainer,
    X_block,
    y_block,
    entropy_ref,
    baseline_shapes,
    lmt_ref,
    ref_pe_scores,
    ref_lmt_scores,
    use_mad_lmt=True,
):
    """
    Full per block drift pipeline.

    Returns
    -------
    drift_info : dict containing
        entropy_score
        entropy_drift_ratio
        lmt_mean_score
        lmt_sample_drift_ratio
        per_class_scores
        p_pe
        p_lmt
        z_evidence
    """
    entropy_score, entropy_drift_ratio, _ = compute_entropy_score(
        model_trainer, X_block, entropy_ref, use_mad=False
    )

    lmt_mean_score, lmt_sample_drift_ratio, per_class_scores, _, _ = compute_lmt_block(
        model_trainer,
        X_block,
        y_block,
        baseline_shapes,
        lmt_ref,
        use_mad=use_mad_lmt,
    )

    p_pe = empirical_p_value(
        entropy_score, ref_pe_scores, larger_is_more_abnormal=True
    )
    p_lmt = empirical_p_value(
        lmt_mean_score, ref_lmt_scores, larger_is_more_abnormal=True
    )

    z_evidence = fuse_drift_evidence(p_lmt, p_pe)

    return {
        "entropy_score": float(entropy_score),
        "entropy_drift_ratio": float(entropy_drift_ratio),
        "lmt_mean_score": float(lmt_mean_score),
        "lmt_sample_drift_ratio": float(lmt_sample_drift_ratio),
        "per_class_scores": per_class_scores,
        "p_pe": float(p_pe),
        "p_lmt": float(p_lmt),
        "z_evidence": float(z_evidence),
    }

def update_sequential_state(prev_g, z_t, nu, h):
    """
    Update sequential statistic and return new state plus trigger flag.
    """
    g_t = sequential_accumulation(prev_g, z_t, nu)
    decision = bool(g_t > h)
    return g_t, decision


# ======================================================
# ======================================================
# ======================================================
# ======================================================
# ======================================================
# ======================================================
# ======================================================
# (1) ENTROPY REFERENCE (training window)
# ======================================================

def compute_entropy_reference(model_trainer, X_train, y_train):
    """
    Build baseline entropy statistics from training window.
    Returns dict with median, MAD, mean, std, and both thresholds.
    """
    z_train = model_trainer.encode(X_train)
    probs = model_trainer.classifier.predict_proba(z_train)
    probs = np.asarray(probs, dtype=float)

    # smoothing to avoid log(0)
    probs = (probs + 1e-3) / (1 + 1e-3 * probs.shape[1])
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs_safe = np.clip(probs, 1e-12, 1.0)

    entropy_vals = -np.sum(probs_safe * np.log(probs_safe), axis=1)

    med = np.median(entropy_vals)
    mad = 1.4826 * np.median(np.abs(entropy_vals - med))
    mu  = np.mean(entropy_vals)
    sigma = np.std(entropy_vals)

    thr_mad_monitor  = med + 2.0 * mad
    thr_mad_decision = med + 3.0 * mad
    thr_std_monitor  = mu + 2.0 * sigma
    thr_std_decision = mu + 3.0 * sigma

    print(f"[EntropyRef] median={med:.4f}, MAD={mad:.4f}, mean={mu:.4f}, std={sigma:.4f}")
    #print(f"[EntropyRef] thr_mad_monitor={thr_mad_monitor:.4f}, thr_mad_decision={thr_mad_decision:.4f}")
    #print(f"[EntropyRef] thr_std_monitor={thr_std_monitor:.4f}, thr_std_decision={thr_std_decision:.4f}")

    return {
        "entropy_median": med,
        "entropy_mad": mad,
        "entropy_mean": mu,
        "entropy_std": sigma,
        "thr_mad_monitor": thr_mad_monitor,
        "thr_mad_decision": thr_mad_decision,
        "thr_std_monitor": thr_std_monitor,
        "thr_std_decision": thr_std_decision,
    }


# ======================================================
# (2) ENTROPY DRIFT DETECTION (per block)
# ======================================================

def compute_entropy_score(model_trainer, X_block, entropy_ref, use_mad=False):
    """
    Compute mean predictive entropy and drifted sample ratio for one block.

    Returns:
        H_block: mean entropy over all samples
        drift_ratio: fraction of samples exceeding decision threshold
        H_i: per-sample entropy array
    """
    z_block = model_trainer.encode(X_block)
    probs = model_trainer.classifier.predict_proba(z_block)
    probs = np.asarray(probs, dtype=float)

    noise = np.random.uniform(0.001, 0.005, size=probs.shape)
    probs = probs + noise
    probs = (probs + 1e-3) / (1 + 1e-3 * probs.shape[1])
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs_safe = np.clip(probs, 1e-12, 1.0)

    H_i = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    H_block = float(np.mean(H_i))

    # threshold for individual samples
    if use_mad:
        thr_individual = entropy_ref["thr_mad_decision"]
    else:
        thr_individual = entropy_ref["thr_std_decision"]

    drift_ratio = float(np.mean(H_i > thr_individual))
    print(f"[Entropy] Block mean entropy={H_block:.4f}, drift_ratio={drift_ratio:.4f}")
    return H_block, drift_ratio, H_i


def detect_entropy_drift(entropy_score, entropy_ref, use_mad=False):
    """
    Compare block entropy against reference thresholds.
    Returns monitor_flag, decision_flag.
    """
    if use_mad:
        mon = entropy_score > entropy_ref["thr_mad_monitor"]
        dec = entropy_score > entropy_ref["thr_mad_decision"]
    else:
        mon = entropy_score > entropy_ref["thr_std_monitor"]
        dec = entropy_score > entropy_ref["thr_std_decision"]

    return bool(mon), bool(dec)


# ======================================================
# (3) LMT REFERENCE AND DETECTION
# ======================================================
import numpy as np
from sklearn.covariance import LedoitWolf


# ---------- Phase 1: Baseline reference ----------
from sklearn.covariance import LedoitWolf
import numpy as np

def compute_lmt_reference(
    model_trainer,
    X_train,
    y_train,
    n_blocks=20,
    k_monitor=1.5,
    k_decision=3.0,
):
    """
    Build LMT baseline reference.

    Returns
    -------
    baseline_shapes : dict[class] -> (mu, Sigma)
    lmt_ref : dict with
        per_class thresholds
        global stats (mean of per class medians etc)
        global thresholds from global median and MAD or mean and std
    """

    print("[LMT] Computing baseline latent shapes...")
    Z_train = model_trainer.encode(X_train)
    classes = np.unique(y_train)

    baseline_shapes = {}
    for c in classes:
        Zc = Z_train[y_train == c]
        if len(Zc) == 0:
            continue
        mu_c = np.mean(Zc, axis=0)
        cov_est = LedoitWolf().fit(Zc)
        Sigma_c = cov_est.covariance_ + 1e-6 * np.eye(Zc.shape[1])
        baseline_shapes[c] = (mu_c, Sigma_c)

    print("[LMT] Estimating baseline thresholds from pseudo blocks...")
    history_scores = {c: [] for c in classes}
    X_chunks = np.array_split(X_train, n_blocks)
    y_chunks = np.array_split(y_train, n_blocks)

    for Xi, yi in zip(X_chunks, y_chunks):
        Zi = model_trainer.encode(Xi)
        per_class_scores, _ = compute_lmt_scores(Zi, yi, baseline_shapes)
        for c, v in per_class_scores.items():
            history_scores[c].append(v)

    # per class stats and thresholds
    lmt_thresholds = {}
    for c, vals in history_scores.items():
        vals = np.array(vals)
        if len(vals) == 0:
            continue

        med_c = np.median(vals)
        mad_c = 1.4826 * np.median(np.abs(vals - med_c))
        mu_c  = np.mean(vals)
        std_c = np.std(vals)

        thr_mad_monitor_c  = med_c + k_monitor * mad_c
        thr_mad_decision_c = med_c + k_decision * mad_c
        thr_std_monitor_c  = mu_c + k_monitor * std_c
        thr_std_decision_c = mu_c + k_decision * std_c

        lmt_thresholds[c] = {
            "median": med_c,
            "mad": mad_c,
            "mean": mu_c,
            "std": std_c,
            "thr_mad_monitor": thr_mad_monitor_c,
            "thr_mad_decision": thr_mad_decision_c,
            "thr_std_monitor": thr_std_monitor_c,
            "thr_std_decision": thr_std_decision_c,
        }

    valid_classes = list(lmt_thresholds.keys())
    if not valid_classes:
        raise RuntimeError("[LMT] No valid class thresholds found in baseline")

    # global stats are means of per class stats
    global_median = float(np.mean([lmt_thresholds[c]["median"] for c in valid_classes]))
    global_mad    = float(np.mean([lmt_thresholds[c]["mad"]    for c in valid_classes]))
    global_mean   = float(np.mean([lmt_thresholds[c]["mean"]   for c in valid_classes]))
    global_std    = float(np.mean([lmt_thresholds[c]["std"]    for c in valid_classes]))

    thr_mad_monitor_global  = global_median + k_monitor * global_mad
    thr_mad_decision_global = global_median + k_decision * global_mad
    thr_std_monitor_global  = global_mean   + k_monitor * global_std
    thr_std_decision_global = global_mean   + k_decision * global_std

    lmt_ref = {
        "per_class": lmt_thresholds,
        "global_median": global_median,
        "global_mad": global_mad,
        "global_mean": global_mean,
        "global_std": global_std,
        "thr_mad_monitor": thr_mad_monitor_global,
        "thr_mad_decision": thr_mad_decision_global,
        "thr_std_monitor": thr_std_monitor_global,
        "thr_std_decision": thr_std_decision_global,
    }

    print(
        "[LMT Ref] global median {:.4f}, MAD {:.4f}, "
        "thr_mad_decision {:.4f}".format(
            global_median, global_mad, thr_mad_decision_global
        )
    )

    return baseline_shapes, lmt_ref


# ---------- Phase 2: Per-block detection ----------
def compute_lmt_block(
    model_trainer,
    X_block,
    y_block,
    baseline_shapes,
    lmt_ref,
    use_mad=True,
):
    """
    Compute LMT metrics for one block (Corrected with Weighted Average).

    Returns
    -------
    mean_lmt_score : float
        Weighted mean of LMT scores in this block (sum of all d2 / total samples)
    drift_ratio_samples : float
        Fraction of samples whose LMT distance exceeds their class decision threshold
    per_class_scores : dict
        class label -> mean LMT score in this block
    lmt_monitor : bool
        Global monitoring flag based on mean_lmt_score
    lmt_decision : bool
        Global decision flag based on mean_lmt_score
    """
    # latent vectors for this block
    Z_block = model_trainer.encode(X_block)
    y_pred = y_block     # or model_trainer.predict(X_block) if desired

    per_class_scores = {}
    drifted_samples = 0
    total_samples = 0
    total_lmt_sum = 0.0  # [NEW] Track sum of all distances for weighted average

    per_class_thr = lmt_ref["per_class"]

    for c, (mu_c, Sigma_c) in baseline_shapes.items():
        mask_c = (y_pred == c)
        Zc = Z_block[mask_c]
        if Zc.shape[0] == 0:
            continue

        inv_Sigma = np.linalg.inv(Sigma_c)
        diffs = Zc - mu_c
        d2 = np.sum(diffs @ inv_Sigma * diffs, axis=1)  # Mahalanobis squared

        # mean score for THIS CLASS in this block
        mean_c = float(np.mean(d2))
        per_class_scores[c] = mean_c

        # [NEW] Accumulate total sum for weighted global average later
        total_lmt_sum += np.sum(d2)

        # sample level drift count for this class
        if c in per_class_thr:
            thr_info = per_class_thr[c]
            thr_dec_c = thr_info["thr_mad_decision"] if use_mad else thr_info["thr_std_decision"]
            drifted_samples += int(np.sum(d2 > thr_dec_c))
        
        total_samples += Zc.shape[0]

    if total_samples == 0:
        return np.nan, 0.0, {}, False, False

    # [CORRECTED] Weighted mean across all samples in the block
    # Old buggy line: mean_lmt_score = float(np.mean(list(per_class_scores.values())))
    mean_lmt_score = float(total_lmt_sum / total_samples)

    # sample level drift ratio
    drift_ratio_samples = float(drifted_samples / total_samples)

    # global thresholds from lmt_ref
    if use_mad:
        lmt_mon = mean_lmt_score > lmt_ref["thr_mad_monitor"]
        lmt_dec = mean_lmt_score > lmt_ref["thr_mad_decision"]
    else:
        lmt_mon = mean_lmt_score > lmt_ref["thr_std_monitor"]
        lmt_dec = mean_lmt_score > lmt_ref["thr_std_decision"]

    print(
        f"[LMT] Block mean (weighted)={mean_lmt_score:.4f}, "
        f"sample_ratio={drift_ratio_samples:.4f}, "
        f"monitor={lmt_mon}, decision={lmt_dec}"
    )

    return mean_lmt_score, drift_ratio_samples, per_class_scores, lmt_mon, lmt_dec



def compute_lmt_thresholds(history_scores, k_monitor=3.0, k_decision=3.0):
    thresholds = {}
    for c, vals in history_scores.items():
        vals = np.asarray(vals)
        if len(vals) == 0:
            continue

        median_ref = np.median(vals)
        mad_ref = 1.4826 * np.median(np.abs(vals - median_ref))
        mean_ref = np.mean(vals)
        std_ref = np.std(vals)

        mad_ref = max(mad_ref, 1e-9)
        std_ref = max(std_ref, 1e-9)

        thresholds[c] = {
            "median": median_ref,
            "mad": mad_ref,
            "mean": mean_ref,
            "std": std_ref,
            "thr_mad_monitor": median_ref + k_monitor * mad_ref,
            "thr_mad_decision": median_ref + k_decision * mad_ref,
            "thr_std_monitor": mean_ref + k_monitor * std_ref,
            "thr_std_decision": mean_ref + k_decision * std_ref,
        }
    return thresholds