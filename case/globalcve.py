import os
import sys
import time
import pandas as pd

from cve_matcher import (
    prepare_cve_corpus,
    extract_tokens,
    batch_fast_match,
    batch_strong_match,
)

# ============================================================
# CONFIGURATION
# ============================================================
DAILY_DIR = "../data/daily"
CVE_CORPUS_PATH = "../data/cve/cve_corpus.csv"
OUTPUT_DIR = "../data/cve/global_context_daily"

ALLOWED_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def list_daily_files_for_months(base_dir: str, allowed_months):
    files = []
    for fn in os.listdir(base_dir):
        if not fn.endswith(".csv"):
            continue
        if not fn.startswith("http_day_"):
            continue

        day_str = fn.replace("http_day_", "").replace(".csv", "")
        month_str = day_str[:7]

        if month_str in allowed_months:
            files.append(os.path.join(base_dir, fn))

    return sorted(files)


def extract_day_from_path(path: str):
    fn = os.path.basename(path)
    return fn.replace("http_day_", "").replace(".csv", "")


def extract_month_from_day(day_str: str):
    return day_str[:7]


# ============================================================
# MAIN
# ============================================================
def main():
    start_all_time = time.time()

    try:
        cve_df = pd.read_csv(CVE_CORPUS_PATH)
        cve_df, inverted_idx, strong_pattern = prepare_cve_corpus(cve_df)
    except Exception as e:
        print(f"FATAL ERROR: Could not load or prepare CVE corpus. {e}")
        sys.exit(1)

    cve_paths = cve_df["path"].tolist()
    cve_reqs = cve_df["request"].tolist()
    cve_ids = cve_df["cve_ids"].tolist()
    cve_tokens_list = cve_df["tokens"].tolist()

    print(f"CVE index ready with {len(cve_df)} entries.")
    print("selected months:", ALLOWED_MONTHS)

    daily_files = list_daily_files_for_months(DAILY_DIR, ALLOWED_MONTHS)

    if len(daily_files) == 0:
        print("No daily files found for the selected months.")
        sys.exit(0)

    print(f"Found {len(daily_files)} daily files.")

    all_records = []

    for path in daily_files:
        day_start_time = time.time()

        day_str = extract_day_from_path(path)
        month_str = extract_month_from_day(day_str)

        print(f"\nProcessing {day_str}")

        try:
            raw = pd.read_csv(path)
        except Exception as e:
            print(f"skip day {day_str}: failed to read {path} ({e})")
            continue

        raw["http_uri"] = raw["http_uri"].fillna("").astype(str)
        raw["http_body"] = raw["http_body"].fillna("").astype(str)

        raw["uri_tokens"] = raw["http_uri"].apply(extract_tokens)
        raw["body_tokens"] = raw["http_body"].apply(extract_tokens)
        raw["cve_hits"] = [[] for _ in range(len(raw))]

        s_uri = raw["http_uri"].str.contains(strong_pattern, case=False, na=False, regex=True)
        s_body = raw["http_body"].str.contains(strong_pattern, case=False, na=False, regex=True)
        s_mask = s_uri | s_body

        strong_df = raw[s_mask].copy()
        weak_df = raw[~s_mask].copy()

        if len(strong_df) > 0:
            strong_hits = batch_strong_match(strong_df, cve_paths, cve_reqs, cve_ids)
            raw.loc[strong_df.index, "cve_hits"] = pd.Series(strong_hits, index=strong_df.index)

        if len(weak_df) > 0:
            weak_hits = batch_fast_match(weak_df, cve_tokens_list, cve_ids, inverted_idx)
            raw.loc[weak_df.index, "cve_hits"] = pd.Series(weak_hits, index=weak_df.index)

        match_rows = raw.explode("cve_hits")
        match_rows = match_rows[match_rows["cve_hits"].notnull()]
        match_rows = match_rows[match_rows["cve_hits"] != ""]

        if len(match_rows) == 0:
            print(f"No CVE hits found for {day_str}.")
            continue

        cnt = match_rows["cve_hits"].value_counts().reset_index()
        cnt.columns = ["cve_id", "count"]
        cnt["day"] = day_str
        cnt["month"] = month_str
        cnt["file"] = path

        all_records.append(cnt)

        daily_out = os.path.join(OUTPUT_DIR, f"{day_str}_daily_summary.csv")
        cnt.to_csv(daily_out, index=False)

        day_end_time = time.time()
        print(f"Saved daily summary to {daily_out}")
        print(f"Finished {day_str} in {day_end_time - day_start_time:.2f} seconds.")

    if len(all_records) == 0:
        print("No CVE hits found across all selected daily files.")
        sys.exit(0)

    all_df = pd.concat(all_records, ignore_index=True)

    all_daily_out = os.path.join(OUTPUT_DIR, "all_daily_block_summary.csv")
    all_df.to_csv(all_daily_out, index=False)
    print(f"\nSaved combined daily summary to {all_daily_out}")

    monthly = (
        all_df.groupby(["month", "cve_id"])
        .agg({"count": "sum"})
        .reset_index()
        .sort_values(["month", "count"], ascending=[True, False])
    )

    monthly_out = os.path.join(OUTPUT_DIR, "monthly_summary.csv")
    monthly.to_csv(monthly_out, index=False)
    print(f"Saved monthly summary to {monthly_out}")

    end_all_time = time.time()
    print(f"\nAll analyses complete in {end_all_time - start_all_time:.2f} seconds.")


if __name__ == "__main__":
    main()