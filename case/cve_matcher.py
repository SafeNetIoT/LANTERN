import pandas as pd
import re
import os
import sys
from collections import defaultdict
import time
from typing import Set, List, Dict, Tuple

# ============================================================
#             1. CONFIGURATION
# ============================================================
MIN_TOKEN_HITS = 2
TOP_N_STOP_WORDS = 20

CVE_CORPUS_PATH = "../data/cve_corpus.csv"
OUTPUT_DIR = "output_cve_analysis"
DAILY_DATA_BASE_PATH = "../data/daily/http_day_"

# FIX: Ensure dates match file structure. The code below will handle the hyphen replacement.
DRIFT_DAYS = [
    "2025 04 20",
    "2025 05 27",
    "2025 06 06",
    "2025 09 28",
    "2025 10 10"
]



# ------------------------------------------------------------
#               2. TOKEN EXTRACTION (FIXED)
# ------------------------------------------------------------
def extract_tokens(s: str) -> Set[str]:
    """Tokenize URI or body into meaningful tokens."""
    s = str(s).lower()
    # FIX: Ensure '-' is handled, and non-token characters are replaced with space.
    # Keep: a-z, 0-9, _, %
    s = re.sub(r"[^a-z0-9_%\-]+", " ", s) 
    # Split by common delimiters (now including hyphen)
    parts = re.split(r"[\/\?\=\&\.\-\_]", s)
    return set(t for t in parts if t and len(t) > 2)

# ------------------------------------------------------------
#               3. CVE PREPROCESSING
# ------------------------------------------------------------
def prepare_cve_corpus(cve_df: pd.DataFrame):
    """Prepares the CVE index (runs only once)."""
    cve_df["path"] = cve_df["path"].fillna("").astype(str).str.lower().str.strip()
    cve_df["request"] = cve_df["request"].fillna("").astype(str).str.lower().str.strip()
    cve_df["full_signature"] = cve_df["path"] + " " + cve_df["request"]

    cve_df["tokens"] = cve_df["full_signature"].apply(lambda x: list(extract_tokens(x)))

    all_tokens = [t for sub in cve_df["tokens"] for t in sub]
    counts = pd.Series(all_tokens).value_counts()
    stop_words = set(counts.head(TOP_N_STOP_WORDS).index)

    inverted = defaultdict(list)
    for idx, row in cve_df.iterrows():
        for tok in row["tokens"]:
            if tok not in stop_words:
                inverted[tok].append(idx)

    paths = cve_df["path"][cve_df["path"].str.len() > 0].tolist()
    reqs = cve_df["request"][cve_df["request"].str.len() > 0].tolist()
    strong_pattern = "|".join(map(re.escape, paths + reqs))

    cve_df["tokens"] = cve_df["tokens"].apply(set)

    return cve_df, inverted, strong_pattern

# ------------------------------------------------------------
#               4. STRONG MATCH MAPPING (BATCH) - ACCURACY
# ------------------------------------------------------------
def batch_strong_match(strong_df: pd.DataFrame, cve_paths: List[str], cve_reqs: List[str], cve_ids: List[str]) -> List[List[str]]:
    """Precisely maps strong hits to CVE IDs (slow loop, run only on subset)."""
    out = []

    # Use zip to iterate over the strong hits subset
    for uri, body in zip(strong_df["http_uri"], strong_df["http_body"]):
        uri_low = uri.lower()
        body_low = body.lower()
        hits = []

        # Brute force search against the full CVE list (for accuracy)
        for p, r, cid in zip(cve_paths, cve_reqs, cve_ids):
            ok = False
            # Check path in URI
            if p and p in uri_low:
                ok = True
            # Check request in Body
            elif r and r in body_low:
                ok = True
            if ok:
                hits.append(cid)

        out.append(list(set(hits)))

    return out

# ------------------------------------------------------------
#               5. WEAK MATCH (BATCH) - SPEED
# ------------------------------------------------------------
def batch_fast_match(token_df: pd.DataFrame, cve_tokens_list: List[Set[str]], cve_ids: List[str], inverted_idx: Dict[str, List[int]]) -> List[List[str]]:
    """Index-accelerated weak match via token overlap."""
    all_hits = []

    # Use zip to iterate over the weak hits subset
    for uri_t, body_t in zip(token_df["uri_tokens"], token_df["body_tokens"]):
        req_tokens = uri_t | body_t

        # 1. Candidate Generation (using Inverted Index)
        cand = set()
        for tok in req_tokens:
            if tok in inverted_idx:
                cand.update(inverted_idx[tok])

        if not cand:
            all_hits.append([])
            continue

        # 2. Token Overlap Check (only on candidates)
        row_hits = []
        for cid in cand:
            # cid is the index (integer) into the CVE lists
            overlap = len(req_tokens.intersection(cve_tokens_list[cid]))
            if overlap >= MIN_TOKEN_HITS:
                row_hits.append(cve_ids[cid])

        all_hits.append(row_hits)

    return all_hits

# ------------------------------------------------------------
#               DAILY PROCESS (FIXED VERSION)
# ------------------------------------------------------------
def process_and_export_day(path, day, cve_df, inverted_idx, strong_pattern, out_dir):

    start = time.time()

    try:
        drift_df = pd.read_csv(path)
    except:
        print(f"day {day} skipped (file not found or error)")
        return

    # Data preparation
    drift_df["http_uri"] = drift_df["http_uri"].fillna("").astype(str)
    drift_df["http_body"] = drift_df["http_body"].fillna("").astype(str)
    
    # Pre-calculate tokens
    drift_df["uri_tokens"] = drift_df["http_uri"].apply(extract_tokens)
    drift_df["body_tokens"] = drift_df["http_body"].apply(extract_tokens)
    
    drift_df["cve_hits"] = [[] for _ in range(len(drift_df))]

    # Get necessary lists from CVE corpus
    cve_paths = cve_df["path"].tolist()
    cve_reqs = cve_df["request"].tolist()
    cve_ids = cve_df["cve_ids"].tolist()
    cve_tokens_list = cve_df["tokens"].tolist()

    # --- PHASE A: Split traffic by Strong Match Mask ---
    uri_hit = drift_df["http_uri"].str.contains(strong_pattern, case=False, na=False, regex=True)
    body_hit = drift_df["http_body"].str.contains(strong_pattern, case=False, na=False, regex=True)
    strong_mask = uri_hit | body_hit

    strong_df = drift_df[strong_mask].copy()
    weak_df = drift_df[~strong_mask].copy()

    # --- PHASE B: Run Matches and FIX Value Alignment ---
    if len(strong_df) > 0:
        # 1. Accurate Strong Match Mapping
        strong_hits = batch_strong_match(strong_df, cve_paths, cve_reqs, cve_ids)
        # FIX: Convert to Series using strong_df.index to ensure alignment
        strong_hits_series = pd.Series(strong_hits, index=strong_df.index)
        drift_df.loc[strong_df.index, "cve_hits"] = strong_hits_series

    if len(weak_df) > 0:
        # 2. Index-Accelerated Weak Match
        weak_hits = batch_fast_match(weak_df, cve_tokens_list, cve_ids, inverted_idx)
        # FIX: Convert to Series using weak_df.index to ensure alignment
        weak_hits_series = pd.Series(weak_hits, index=weak_df.index)
        drift_df.loc[weak_df.index, "cve_hits"] = weak_hits_series

    # --- PHASE C: Finalize and Export ---
    matched = drift_df[drift_df["cve_hits"].apply(len) > 0].copy()

    if len(matched) == 0:
        print(f"day {day} done (no hit)")
        return

    # FIX: Create hashable key for deduplication
    matched['hit_key'] = matched["cve_hits"].apply(lambda x: ",".join(sorted(x)))
    
    # Columns to check for full deduplication
    keep_cols = [c for c in matched.columns if c not in ["cve_hits", "uri_tokens", "body_tokens", "hit_key"]]

    # Deduplicate the row-level data
    matched_final = matched.drop_duplicates(subset=keep_cols + ["hit_key"], keep="first")
    
    # Clean up columns for the final export CSV
    matched_final = matched_final.drop(columns=["uri_tokens", "body_tokens", "hit_key"])

    # Aggregate report 
    report = (
        matched.explode("cve_hits")
        .groupby(["cve_hits", "category"])
        .size()
        .reset_index(name="count")
    )

    pivot = report.pivot_table(
        index="cve_hits",
        columns="category",
        values="count",
        fill_value=0,
        aggfunc="sum"
    )
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)

    os.makedirs(out_dir, exist_ok=True)

    matched_out = os.path.join(out_dir, f"matched_{day}.csv")
    pivot_out = os.path.join(out_dir, f"report_{day}.csv")

    matched_final.to_csv(matched_out, index=False)
    pivot.to_csv(pivot_out, index=True)

    end = time.time()
    print(f"day {day} done in {end - start:.2f} sec")
    
# ------------------------------------------------------------
#                       MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    start_all = time.time()

    try:
        cve_df = pd.read_csv(CVE_CORPUS_PATH)
        cve_df, inverted_idx, strong_pattern = prepare_cve_corpus(cve_df)
    except:
        print("fatal cve load")
        sys.exit(1)

    print(f"cve index ready {len(cve_df)} entries")

    for d in DRIFT_DAYS:
        # FIX: Correctly format the date string for file path access
        d_str = d.replace(" ", "-") 
        path = f"{DAILY_DATA_BASE_PATH}{d_str}.csv"
        print(f"start {d_str}")
        process_and_export_day(path, d_str, cve_df, inverted_idx, strong_pattern, OUTPUT_DIR)

    end_all = time.time()
    # FIX: Correct calculation order for print statement
    print(f"all complete in {end_all - start_all:.2f} sec")