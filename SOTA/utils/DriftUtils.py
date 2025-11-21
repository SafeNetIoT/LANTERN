# SOTA/utils/DriftUtils.py
import numpy as np
import torch
from scipy.stats import ks_2samp
from scipy.special import rel_entr
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors


# =========================================================
# KL Divergence Drift Test
# =========================================================
def _kl_divergence(p, q, eps=1e-8):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(rel_entr(p, q)))


def kl_drift_test(ref_errors, test_errors, bins=50, k_mad=3.0):
    ref_hist, bin_edges = np.histogram(ref_errors, bins=bins, density=True)
    test_hist, _ = np.histogram(test_errors, bins=bin_edges, density=True)

    kl_value = _kl_divergence(test_hist, ref_hist)

    kl_ref_vals = []
    for _ in range(10):
        idx = np.random.choice(len(ref_errors), size=len(test_errors), replace=False)
        sub = ref_errors[idx]
        h, _ = np.histogram(sub, bins=bin_edges, density=True)
        kl_ref_vals.append(_kl_divergence(h, ref_hist))

    kl_ref_vals = np.asarray(kl_ref_vals)
    med = np.median(kl_ref_vals)
    mad = 1.4826 * np.median(np.abs(kl_ref_vals - med))
    mad = max(mad, 1e-9)

    thr = med + k_mad * mad
    return kl_value > thr, kl_value


# =========================================================
# Mateen Drift Test (KS Test)
# =========================================================
def mateen_ks_test(ref_errors, test_errors, alpha=0.05):
    stat, p = ks_2samp(ref_errors, test_errors)
    return (p < alpha), float(p)


# =========================================================
# OWAD Calibrator
# =========================================================
def _owad_fit_calibrator(control_scores, is_mal_conf=True):
    x = np.asarray(control_scores, float)
    xs = np.sort(x)
    ranks = np.arange(len(xs), dtype=float) / max(len(xs), 1)

    iso = IsotonicRegression(
        y_min=0.0, y_max=1.0,
        increasing=True, out_of_bounds="clip"
    )
    iso.fit(xs, ranks)

    def calib(v):
        p = iso.predict(np.asarray(v, float))
        return 1.0 - p if is_mal_conf else p

    return calib


# =========================================================
# OWAD Permutation G Test
# =========================================================
def _owad_gtest(p_ctrl, p_test, bins=5):
    h_ctrl, _ = np.histogram(p_ctrl, bins=bins, range=(0, 1))
    h_test, _ = np.histogram(p_test, bins=bins, range=(0, 1))

    eps = 1e-10
    E = h_ctrl / max(h_ctrl.sum(), 1)
    E = E * h_test.sum()

    O = h_test.astype(float) + eps
    E = E.astype(float) + eps

    return float(np.sum(O * (np.log(O) - np.log(E))))


def owad_run(control_scores, test_scores, bins=5, rounds=100, alpha=0.05):
    calib = _owad_fit_calibrator(control_scores)
    p_ctrl = calib(control_scores)
    p_test = calib(test_scores)

    s_obs = _owad_gtest(p_ctrl, p_test, bins=bins)

    z = np.concatenate([p_ctrl, p_test])
    n = len(p_test)
    cnt = 0
    rng = np.random.default_rng(42)

    for _ in range(rounds):
        rng.shuffle(z)
        zx, zy = z[:n], z[n:]
        if _owad_gtest(zy, zx, bins=bins) >= s_obs:
            cnt += 1

    pval = (cnt + 1) / (rounds + 1)
    return pval < alpha, float(pval)


# =========================================================
# Chen Pseudo-Loss Drift Detection
# =========================================================
def _chen_ce_loss(probs):
    m = np.max(probs, axis=1)
    return -np.log(np.clip(m, 1e-8, 1))


def _chen_hc_loss(z_ref, y_ref, z_test, y_pred, k=20, margin=1.0):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(z_ref)), metric="euclidean").fit(z_ref)
    dists, idxs = nbrs.kneighbors(z_test)

    losses = []
    for i, neigh in enumerate(idxs):
        labels = y_ref[neigh]
        d = dists[i]

        P = d[labels == y_pred[i]]
        N = d[labels != y_pred[i]]

        t1 = np.mean(np.maximum(0, P - margin)) if len(P) else 0
        t2 = np.mean(np.maximum(0, 2 * margin - N)) if len(N) else 0
        losses.append(t1 + t2)
    return np.asarray(losses)


def compute_chen_reference(model, X_train, y_train):
    z = model.encode(X_train)
    probs = model.classifier.predict_proba(z)
    y_pred = np.argmax(probs, axis=1)

    L_ce = _chen_ce_loss(probs)
    L_hc = _chen_hc_loss(z, y_train, z, y_pred)

    scores = L_ce + L_hc
    med = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - med))
    mad = max(mad, 1e-9)

    return {
        "z_ref": z,
        "y_ref": y_train,
        "median": float(med),
        "mad": float(mad),
    }


def chen_fast_detect(model, X_block, y_block, ref_stats, k_mad=3.0):
    z_ref = ref_stats["z_ref"]
    y_ref = ref_stats["y_ref"]

    z = model.encode(X_block)
    probs = model.classifier.predict_proba(z)
    y_pred = np.argmax(probs, axis=1)

    L_ce = _chen_ce_loss(probs)
    L_hc = _chen_hc_loss(z_ref, y_ref, z, y_pred)

    score = float(np.mean(L_ce + L_hc))

    thr = ref_stats["median"] + k_mad * ref_stats["mad"]
    return score > thr, score


# =========================================================
# CADE Drift Detection
# =========================================================
def compute_cade_reference(z_train, y_train_enc):
    classes = np.unique(y_train_enc)
    stats = {}
    norms = []

    for c in classes:
        Zc = z_train[y_train_enc == c]
        if len(Zc) < 2:
            continue

        mu = Zc.mean(axis=0)
        d = np.linalg.norm(Zc - mu, axis=1)
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        mad = max(mad, 1e-9)

        norm = np.abs(d - med) / mad
        norms.extend(norm)

        stats[c] = {"centroid": mu, "median": med, "mad": mad}

    norms = np.asarray(norms)
    gmed = np.median(norms)
    gmad = 1.4826 * np.median(np.abs(norms - gmed))
    gmad = max(gmad, 1e-9)

    stats["_global"] = {"median": gmed, "mad": gmad}
    return stats


def cade_fast_detect(z_block, ref_stats, sigma_k=3.0):
    gmed = ref_stats["_global"]["median"]
    gmad = ref_stats["_global"]["mad"]
    thr = gmed + sigma_k * gmad

    scores = []
    for z in z_block:
        vals = []
        for c, s in ref_stats.items():
            if c == "_global":
                continue
            d = np.linalg.norm(z - s["centroid"])
            vals.append(abs(d - s["median"]) / s["mad"])
        if vals:
            scores.append(min(vals))

    if not scores:
        return False, 0.0

    score = float(np.mean(scores))
    return score > thr, score


# =========================================================
# Unified Drift Detection Wrapper
# =========================================================
def run_drift_detection(ref_errors, test_errors, methods, alpha=0.05):
    results = {}

    if "kl" in methods:
        results["kl"] = kl_drift_test(ref_errors, test_errors)

    if "mateen" in methods:
        results["mateen"] = mateen_ks_test(ref_errors, test_errors, alpha)

    if "owad" in methods:
        results["owad"] = owad_run(ref_errors, test_errors, alpha=alpha)

    return results
