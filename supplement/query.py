import json
import hashlib
import os
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

USERNAME = "aide-ucl"
PASSWORD = "63Ed1068a7aa4ab248!"
OPENSEARCH_HOST = "https://os.gcaaide.org"
INDEX_PATTERN = "proxypot-2"

START_DATE = "2025-03-15"
END_DATE = "2026-03-15"

HEADERS = {"Content-Type": "application/json"}
OUTPUT_DIR = "../data/honeypot/http_https_monthly_patterns/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_FIELDS = [
    "clientIP",
    "startTime",
    "endTime",
    "protocol",
    "session",
    "state",
    "sessionLength",
    "sessionTimeout",
    "category",
    "credentials",
    "allUserNames",
    "allUserPasswords",
    "urls",
    "httpRequests",
]

def ensure_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]

def normalize_http_requests(reqs):
    reqs = ensure_list(reqs)
    norm = []
    for r in reqs:
        if isinstance(r, dict):
            norm.append({
                "method": r.get("method"),
                "status": r.get("status"),
                "uri": r.get("uri") if "uri" in r else r.get("uris"),
                "body": r.get("body"),
                "cookies": r.get("cookies"),
                "headers": r.get("headers"),
            })
        else:
            norm.append(r)
    return norm

def normalize_behaviour(src):
    return {
        "category": src.get("category"),
        "credentials": ensure_list(src.get("credentials")),
        "allUserNames": ensure_list(src.get("allUserNames")),
        "allUserPasswords": ensure_list(src.get("allUserPasswords")),
        "urls": ensure_list(src.get("urls")),
        "httpRequests": normalize_http_requests(src.get("httpRequests")),
    }

def make_signature(behaviour_dict):
    payload = json.dumps(behaviour_dict, sort_keys=True, ensure_ascii=False)
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return signature, payload

def init_record(month, protocol, behaviour_dict, signature, payload, start_time, end_time, client_ip):
    rec = {
        "month": month,
        "protocol": protocol,
        "signature": signature,
        #"pattern_json": payload,
        "category": behaviour_dict["category"],
        "credentials": behaviour_dict["credentials"],
        "allUserNames": behaviour_dict["allUserNames"],
        "allUserPasswords": behaviour_dict["allUserPasswords"],
        "urls": behaviour_dict["urls"],
        "httpRequests": behaviour_dict["httpRequests"],
        "first_seen": start_time,
        "last_seen": start_time,
        "last_end_time": end_time,
        "count": 1,
        "unique_ips": set(),
        "days_seen": set(),
    }

    if client_ip:
        rec["unique_ips"].add(client_ip)
    if pd.notna(start_time):
        rec["days_seen"].add(start_time.strftime("%Y-%m-%d"))

    return rec

def update_record(rec, client_ip, start_time, end_time):
    rec["count"] += 1

    if client_ip:
        rec["unique_ips"].add(client_ip)

    if pd.notna(start_time):
        if rec["first_seen"] is None or start_time < rec["first_seen"]:
            rec["first_seen"] = start_time
        if rec["last_seen"] is None or start_time > rec["last_seen"]:
            rec["last_seen"] = start_time
        rec["days_seen"].add(start_time.strftime("%Y-%m-%d"))

    if pd.notna(end_time):
        if rec["last_end_time"] is None or end_time > rec["last_end_time"]:
            rec["last_end_time"] = end_time

def finalize_record(rec):
    active_span_days = None
    if rec["first_seen"] is not None and rec["last_seen"] is not None:
        active_span_days = (rec["last_seen"] - rec["first_seen"]).total_seconds() / 86400.0

    return {
        "month": rec["month"],
        "protocol": rec["protocol"],
        "signature": rec["signature"],
        "count": rec["count"],
        "unique_ips": len(rec["unique_ips"]),
        "first_seen": rec["first_seen"],
        "last_seen": rec["last_seen"],
        "last_end_time": rec["last_end_time"],
        "active_span_days": active_span_days,
        "days_seen": len(rec["days_seen"]),
        "category": rec["category"],
        "credentials": json.dumps(rec["credentials"], ensure_ascii=False),
        "allUserNames": json.dumps(rec["allUserNames"], ensure_ascii=False),
        "allUserPasswords": json.dumps(rec["allUserPasswords"], ensure_ascii=False),
        "urls": json.dumps(rec["urls"], ensure_ascii=False),
        "httpRequests": json.dumps(rec["httpRequests"], ensure_ascii=False),
        #"pattern_json": rec["pattern_json"],
    }

def month_starts(start_date, end_date):
    start = pd.to_datetime(start_date).to_period("M").to_timestamp()
    end = pd.to_datetime(end_date).to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS")

def fetch_grouped_for_one_month(month_start):
    month_start = pd.Timestamp(month_start)
    month_end = (month_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(seconds=1)

    global_start = pd.Timestamp(START_DATE)
    global_end = pd.Timestamp(END_DATE) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    query_start = max(month_start, global_start)
    query_end = min(month_end, global_end)

    if query_start > query_end:
        return pd.DataFrame()

    month_str = query_start.strftime("%Y-%m")
    url = f"{OPENSEARCH_HOST}/{INDEX_PATTERN}/_search"
    search_after = None
    grouped = {}
    total_hits = 0

    print(f"\nProcessing month {month_str}: {query_start} to {query_end}")

    while True:
        query = {
            "size": 2000,
            "_source": SOURCE_FIELDS,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "startTime": {
                                    "gte": query_start.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "lte": query_end.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "format": "yyyy-MM-dd'T'HH:mm:ss"
                                }
                            }
                        },
                        {
                            "terms": {
                                "protocol.keyword": ["http", "https", "HTTP", "HTTPS"]
                            }
                        }
                    ]
                }
            },
            "sort": [
                {"startTime": "asc"},
                {"_id": "asc"}
            ]
        }

        if search_after is not None:
            query["search_after"] = search_after

        resp = requests.post(
            url,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            headers=HEADERS,
            json=query,
            timeout=120
        )

        if resp.status_code != 200:
            raise RuntimeError(f"OpenSearch error for {month_str}: {resp.status_code}: {resp.text}")

        data = resp.json()
        hits = data["hits"]["hits"]

        if not hits:
            break

        for h in hits:
            src = h.get("_source", {})
            client_ip = src.get("clientIP")
            start_time = pd.to_datetime(src.get("startTime"), utc=True, errors="coerce")
            end_time = pd.to_datetime(src.get("endTime"), utc=True, errors="coerce")
            protocol = src.get("protocol")

            if pd.isna(start_time):
                continue

            behaviour = normalize_behaviour(src)
            signature, payload = make_signature(behaviour)

            key = (month_str, protocol, signature)

            if key not in grouped:
                grouped[key] = init_record(
                    month=month_str,
                    protocol=protocol,
                    behaviour_dict=behaviour,
                    signature=signature,
                    payload=payload,
                    start_time=start_time,
                    end_time=end_time,
                    client_ip=client_ip,
                )
            else:
                update_record(grouped[key], client_ip, start_time, end_time)

        total_hits += len(hits)
        print(f"{month_str}: processed {total_hits} raw records so far...")

        search_after = hits[-1]["sort"]

        if len(hits) < query["size"]:
            break

    rows = [finalize_record(rec) for rec in grouped.values()]
    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["count", "first_seen"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return df

if __name__ == "__main__":
    all_summary_rows = []

    for ms in month_starts(START_DATE, END_DATE):
        month_str = pd.Timestamp(ms).strftime("%Y-%m")
        out_patterns = os.path.join(
            OUTPUT_DIR,
            f"http_patterns_{month_str}.parquet"
        )
        out_summary = os.path.join(
            OUTPUT_DIR,
            f"http_summary_{month_str}.parquet"
        )

        if os.path.exists(out_patterns):
            print(f"Skip {month_str}, already exists: {out_patterns}")
            continue

        df_month = fetch_grouped_for_one_month(ms)

        df_month.to_csv(out_patterns, index=False)
        print(f"{month_str}: rows={len(df_month)} saved -> {out_patterns}")

        if not df_month.empty:
            summary = (
                df_month.groupby("month", as_index=False)
                .agg(
                    total_sessions=("count", "sum"),
                    unique_patterns=("signature", "nunique"),
                    unique_ips=("unique_ips", "sum")
                )
            )
            summary["reuse_ratio"] = (
                summary["total_sessions"] / summary["unique_patterns"]
            )
        else:
            summary = pd.DataFrame([{
                "month": month_str,
                "total_sessions": 0,
                "unique_patterns": 0,
                "unique_ips": 0,
                "reuse_ratio": 0.0,
            }])

        summary.to_csv(out_summary, index=False)
        print(f"{month_str}: summary saved -> {out_summary}")

        all_summary_rows.append(summary)

    if all_summary_rows:
        final_summary = pd.concat(all_summary_rows, ignore_index=True)
        final_summary = final_summary.sort_values("month").reset_index(drop=True)

        final_summary_out = os.path.join(
            OUTPUT_DIR,
            f"http_summary_{START_DATE}_to_{END_DATE}.parquet"
        )
        final_summary.to_csv(final_summary_out, index=False)
        print(f"\nCombined summary saved -> {final_summary_out}")