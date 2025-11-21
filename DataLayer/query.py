from datetime import datetime, timedelta
import requests
import json
import pandas as pd
from requests.auth import HTTPBasicAuth
import csv

# === Config ===
USERNAME = "xxx"   # Replace with your username
PASSWORD = "xxx"     # Replace with your password
OPENSEARCH_HOST = "xxx" # Replace with your OpenSearch host
INDEX_PATTERN = "xxx"
HEADERS = {"Content-Type": "application/json"}

START_DATE = "2025-03-15"
END_DATE = "2025-07-15"

# === Daily date iterator ===
def daterange(start, end):
    current = start
    while current <= end:
        yield current, current
        current += timedelta(days=1)

start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

for day_idx, (start_day, end_day) in enumerate(daterange(start_dt, end_dt), 1):
    print(f"\nQuerying Day {start_day.date()}")

    query = {
        "size": 10000,
        "_source": [
            "protocol", "category", "_id", "httpRequests.headers",
            "httpRequests.method", "httpRequests.status", "httpRequests.uri", "httpRequests.body",
            "httpRequests.headers.Accept", "httpRequests.headers.Accept-Encoding", "httpRequests.headers.User-Agent",
            "httpRequests.headers.Content-Type", "httpRequests.headers.Referer", "httpRequests.headers.Connection",
            "startTime", "endTime", "sessionTimeout", "sessionLength",
            "clientIP", "clientGeo.asn", "clientGeo.as_org", "clientGeo.country_name",
            "hostIP", "hostGeo.asn", "hostGeo.as_org", "hostGeo.country_name"
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "startTime": {
                                "gte": f"{start_day.strftime('%Y-%m-%d')}T00:00:00",
                                "lte": f"{end_day.strftime('%Y-%m-%d')}T23:59:59",
                                "format": "yyyy-MM-dd'T'HH:mm:ss"
                            }
                        }
                    },
                    {"terms": {"protocol": ["http", "https"]}}
                ]
            }
        }
    }

    # Scroll request
    url = f"{OPENSEARCH_HOST}/{INDEX_PATTERN}/_search?scroll=2m"
    response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), headers=HEADERS, json=query)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        continue

    data = response.json()
    scroll_id = data.get("_scroll_id")
    all_records = []

    while True:
        hits = data["hits"]["hits"]
        if not hits:
            break

        for hit in hits:
            http_requests = hit["_source"].get("httpRequests", [])
            if not isinstance(http_requests, list):
                http_requests = [http_requests]

            for i, request in enumerate(http_requests):
                all_records.append({
                    "protocol": hit["_source"].get("protocol", ""),
                    "category": hit["_source"].get("category", ""),
                    "id": f"{hit.get('_id', '')}#{i}",
                    "http_headers": request.get("headers", {}),
                    "http_method": request.get("method", ""),
                    "http_status": request.get("status", ""),
                    "http_uri": request.get("uri", ""),
                    "http_body": request.get("body", ""),
                    "http_accept": request.get("headers", {}).get("Accept", ""),
                    "http_accept_encoding": request.get("headers", {}).get("Accept-Encoding", ""),
                    "http_user_agent": request.get("headers", {}).get("User-Agent", ""),
                    "http_content_type": request.get("headers", {}).get("Content-Type", ""),
                    "http_referer": request.get("headers", {}).get("Referer", ""),
                    "http_connection": request.get("headers", {}).get("Connection", ""),
                    "sessionTimeout": hit["_source"].get("sessionTimeout", ""),
                    "sessionLength": hit["_source"].get("sessionLength", ""),
                    "startTime": hit["_source"].get("startTime", ""),
                    "endTime": hit["_source"].get("endTime", ""),
                    "clientIP": hit["_source"].get("clientIP", ""),
                    "client_asn": hit["_source"].get("clientGeo", {}).get("asn", ""),
                    "client_as_org": hit["_source"].get("clientGeo", {}).get("as_org", ""),
                    "client_country": hit["_source"].get("clientGeo", {}).get("country_name", ""),
                    "hostIP": hit["_source"].get("hostIP", ""),
                    "host_asn": hit["_source"].get("hostGeo", {}).get("asn", ""),
                    "host_as_org": hit["_source"].get("hostGeo", {}).get("as_org", ""),
                    "host_country": hit["_source"].get("hostGeo", {}).get("country_name", "")
                })

        print(f"Retrieved {len(hits)} records. Total so far: {len(all_records)}")

        scroll_query = {"scroll": "2m", "scroll_id": scroll_id}
        response = requests.post(f"{OPENSEARCH_HOST}/_search/scroll", auth=HTTPBasicAuth(USERNAME, PASSWORD), headers=HEADERS, json=scroll_query)
        if response.status_code != 200:
            print(f"Scroll error {response.status_code}")
            break
        data = response.json()
        scroll_id = data.get("_scroll_id")

    # Clear scroll context
    requests.delete(f"{OPENSEARCH_HOST}/_search/scroll", auth=HTTPBasicAuth(USERNAME, PASSWORD),
                    headers=HEADERS, json={"scroll_id": scroll_id})
    print("Scroll context cleared.")

    # Save to CSV
    df = pd.DataFrame(all_records)
    df = df.drop(columns=["http_headers"])
    file_name = f"../data/daily/http_day_{start_day.date()}.csv"
    df.to_csv(file_name, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', lineterminator='\n')
    print(f"Day {start_day.date()} saved → {file_name} ({len(df)} records)")