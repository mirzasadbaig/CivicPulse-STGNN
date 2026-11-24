#!/usr/bin/env python3
"""Read-only NYC 311 category/agency investigation for July--September 2026.
Uses the official erm2-nwe9 API. Writes transparent audit outputs; does not change data.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
MONTHS = (
    ("2026-07", "2026-07-01", "2026-08-01"),
    ("2026-08", "2026-08-01", "2026-09-01"),
    ("2026-09", "2026-09-01", "2026-10-01"),
)
TERMS = ("water", "sewer", "drain", "leak", "flood", "hydrant", "catch basin", "plumb")


def fetch(params: dict, token: str | None):
    req_url = URL + "?" + urlencode(params)
    headers = {"User-Agent": "CivicPulse-Academic-Coverage-Audit/1.0"}
    if token:
        headers["X-App-Token"] = token
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(Request(req_url, headers=headers), timeout=90) as resp:
                data = json.load(resp)
            if not isinstance(data, list):
                raise ValueError(f"Unexpected API response: {str(data)[:250]}")
            return data
        except (HTTPError, URLError, TimeoutError, ValueError) as err:
            last_error = err
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Official API query failed: {last_error}")


def write_csv(file: Path, rows: list[dict], fieldnames: list[str]):
    with file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/2026_data_audit/category_change_check")
    parser.add_argument("--token", default=None, help="Optional Socrata App Token")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for month, start, end in MONTHS:
            print(month, "queries complaint_type and agency grouped by count, date window", start, end)
        return

    categories = []
    agencies = []
    audit_lines = ["CivicPulse official API category investigation", f"Dataset: {URL}",
                   "Month windows: July--September 2026, start inclusive, end exclusive", ""]

    for month, start, end in MONTHS:
        where = f"created_date >= '{start}T00:00:00' AND created_date < '{end}T00:00:00'"
        print(f"Checking {month} via NYC Open Data...", flush=True)
        for field, sink in (("complaint_type", categories), ("agency", agencies)):
            query = {
                "$select": f"{field}, count(*) as n",
                "$where": where,
                "$group": field,
                "$order": "n DESC",
                "$limit": 2000,
            }
            data = fetch(query, args.token)
            if len(data) >= 2000:
                raise RuntimeError(f"{month} {field} list reached 2000; increase limit to avoid truncation")
            for d in data:
                sink.append({"month": month, field: str(d.get(field, "<missing>")), "count": int(d["n"])})
            print(f"  {field}: {len(data)} unique values, {sum(int(d['n']) for d in data):,} requests")
        month_cats = [r for r in categories if r["month"] == month]
        matches = [r for r in month_cats if any(t in r["complaint_type"].lower() for t in TERMS)]
        audit_lines.append(f"=== {month} ===")
        audit_lines.append(f"Total requests: {sum(r['count'] for r in month_cats):,}")
        audit_lines.append(f"Distinct complaint types: {len(month_cats)}")
        audit_lines.append("Exact old categories:")
        for old in ["Water System", "Sewer", "HEAT/HOT WATER", "Street Condition"]:
            n = sum(r["count"] for r in month_cats if r["complaint_type"] == old)
            audit_lines.append(f"  {old}: {n:,}")
        audit_lines.append("Other water/sewer-like category names (not proof of equivalence):")
        for r in sorted(matches, key=lambda x: -x["count"])[:40]:
            audit_lines.append(f"  {r['complaint_type']}: {r['count']:,}")
        if not matches:
            audit_lines.append("  None")
        audit_lines.append("Top 12 complaint types:")
        for r in month_cats[:12]:
            audit_lines.append(f"  {r['complaint_type']}: {r['count']:,}")
        audit_lines.append("Agency DEP count:")
        dep = sum(r["count"] for r in agencies if r["month"] == month and r["agency"].strip().upper() == "DEP")
        audit_lines.append(f"  DEP: {dep:,}")
        audit_lines.append("")

    write_csv(outdir / "official_complaint_types_by_month.csv", categories, ["month", "complaint_type", "count"])
    write_csv(outdir / "official_agencies_by_month.csv", agencies, ["month", "agency", "count"])
    (outdir / "audit_summary.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print("\n".join(audit_lines))
    print(f"Saved 3 files to {outdir}")
    print("Note: classification of changed categories requires NYC metadata or agency confirmation; don't auto-remap.")

if __name__ == "__main__":
    main()
