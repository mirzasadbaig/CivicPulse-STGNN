"""Download daily, geographically aggregated *real* NYC 311 records.

This avoids downloading millions of individual requests. Each download covers
one calendar month and is paginated if necessary. The API source is NYC Open
Data dataset erm2-nwe9 (2020-present), queried via Socrata SoQL v2.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import time
from pathlib import Path
import pandas as pd
import requests

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
DEFAULT_CATEGORIES = ["HEAT/HOT WATER", "Water System", "Sewer", "Street Condition"]


def _months(start: str, end: str):
    cursor = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if cursor >= last:
        raise ValueError("--start must be before --end")
    while cursor < last:
        month_end = (cursor.to_period("M") + 1).to_timestamp()
        nxt = min(month_end, last)
        yield cursor.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cursor = nxt


def download(start: str, end: str, output: str, categories=None, borough=None, page_size=30000):
    categories = list(categories or DEFAULT_CATEGORIES)
    if any("'" in c for c in categories):
        raise ValueError("Invalid complaint category (apostrophe unsupported in query)")
    if borough and borough.upper() not in {"BRONX","BROOKLYN","MANHATTAN","QUEENS","STATEN ISLAND"}:
        raise ValueError("Unrecognized borough")
    path=Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    signature=hashlib.sha256(repr((categories,borough,start,end)).encode()).hexdigest()[:10]
    cache_dir=path.parent/f'{path.stem}_monthly_{signature}'
    cache_dir.mkdir(parents=True,exist_ok=True)
    headers={"User-Agent":"CivicPulse-STGNN educational research (NYC public data)"}
    if os.getenv("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"]=os.getenv("SOCRATA_APP_TOKEN")
    for lo,hi in _months(start,end):
        cache_file=cache_dir/f'{lo}_{hi}.csv'
        if cache_file.exists():
            cached=pd.read_csv(cache_file,dtype={'incident_zip':str})
            all_rows.extend(cached.to_dict('records'))
            print(f'{lo} to {hi}: reused {len(cached):,} cached grouped rows',flush=True)
            continue
        where=[f"created_date >= '{lo}T00:00:00'",f"created_date < '{hi}T00:00:00'",
               "incident_zip IS NOT NULL", "latitude IS NOT NULL", "longitude IS NOT NULL",
               "complaint_type in ("+",".join("'"+c+"'" for c in categories)+")"]
        if borough: where.append(f"borough = '{borough.upper()}'")
        offset=0;rows=[]
        while True:
            query={"$select":"date_trunc_ymd(created_date) as day, incident_zip, complaint_type, "
                              "count(*) as n, avg(latitude) as lat, avg(longitude) as lon",
                   "$where":" AND ".join(where), "$group":"day,incident_zip,complaint_type",
                   "$order":"day,incident_zip,complaint_type", "$limit":page_size,"$offset":offset}
            for attempt in range(5):
                try:
                    r=requests.get(URL,params=query,headers=headers,timeout=120)
                    r.raise_for_status()
                    batch=r.json()
                    if not isinstance(batch,list): raise ValueError(f"Unexpected API response: {str(batch)[:300]}")
                    break
                except (requests.RequestException,ValueError) as exc:
                    if attempt==4:
                        raise RuntimeError(f"Failed month {lo}--{hi}: {exc}. Inspect API query/connection.") from exc
                    time.sleep(2**attempt)
            rows.extend(batch)
            if len(batch)<page_size: break
            offset+=page_size
            if offset>1000000: raise RuntimeError("Unexpectedly many grouped records in one month")
        if rows:
            pd.DataFrame(rows).to_csv(cache_file,index=False)
        print(f"{lo} to {hi}: {len(rows):,} daily ZIP-category groups",flush=True)
        all_rows.extend(rows)
    if not all_rows: raise RuntimeError("No records returned. Check category spellings/dates/network.")
    df=pd.DataFrame(all_rows)
    need={'day','incident_zip','complaint_type','n','lat','lon'}
    if not need.issubset(df): raise RuntimeError(f"Unexpected source columns: {list(df.columns)}")
    df['day']=pd.to_datetime(df['day'],errors='coerce').dt.strftime('%Y-%m-%d')
    df['n']=pd.to_numeric(df['n'],errors='coerce')
    for k in ('lat','lon'): df[k]=pd.to_numeric(df[k],errors='coerce')
    df['incident_zip']=df['incident_zip'].astype(str).str.zfill(5)
    df=df.dropna(subset=list(need)).sort_values(['day','incident_zip','complaint_type'])
    df['source']='NYC_311_PUBLIC_AGGREGATED'
    if df.empty: raise RuntimeError("No valid rows returned after cleaning")
    df.to_csv(path,index=False)
    print(f"Saved {len(df):,} groups, representing {int(df.n.sum()):,} complaint records to {path.resolve()}")
    return df


def main():
    p=argparse.ArgumentParser(description='Fetch daily ZIP/category counts from official NYC 311 API')
    p.add_argument('--start',default='2024-01-01',help='inclusive YYYY-MM-DD')
    p.add_argument('--end',default='2026-01-01',help='exclusive YYYY-MM-DD')
    p.add_argument('--borough',default=None)
    p.add_argument('--categories',nargs='+',default=DEFAULT_CATEGORIES)
    p.add_argument('--output',default='data/nyc_daily_2024_2025.csv')
    a=p.parse_args()
    download(a.start,a.end,a.output,a.categories,a.borough)
if __name__=='__main__':main()
