#!/usr/bin/env python3
"""Verify recorded evidence and optionally SHA-256 of local (gitignored) CSV inputs.

This script DOES NOT reproduce model fitting or attest scientific independence.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'past_sha256': '5f98e176da8168a4b5dbc409c958b5c58322d835d43ad8f96cd289110d012fec',
    'future_sha256': 'e442a3de8c0dd7652418816a28b0c118220b8921be910521cba163e8c25f8afe',
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--past',type=Path,help='Optional local 2024-2025 aggregate CSV')
    p.add_argument('--future',type=Path,help='Optional local 2026 Jan-Sep aggregate CSV')
    a=p.parse_args()
    manifest=json.loads((ROOT/'evidence/benchmark_controls/experiment_manifest.json').read_text())
    for k,ex in EXPECTED.items():
        assert manifest[k]==ex, f'{k}: manifest mismatch'
    assert manifest['model_seeds']==[7,17,27,37,42]
    assert manifest['windows']['Jan-Jul 2026']['cases']==16800
    assert manifest['windows']['Jan-Sep 2026']['cases']==21680
    with (ROOT/'evidence/benchmark_controls/model_summary.csv').open(newline='') as f:
        rows={(r['period'],r['model']):r for r in csv.DictReader(f)}
    expected={
        ('Jan-Jul 2026','Persistence previous 48h'):(.5827761700429099,.6127699783474381),
        ('Jan-Jul 2026','GCN-GRU spatial'):(.47885858992914176,.6040015761652181),
        ('Jan-Jul 2026','GCN-GRU shuffled graph'):(.49529773671636307,.5975055086555893),
        ('Jan-Jul 2026','GRU identity adjacency'):(.4706265012612277,.5884333467857475),
        ('Jan-Sep 2026','Persistence previous 48h'):(.5744836410162676,.5950010152403934),
        ('Jan-Sep 2026','GCN-GRU spatial'):(.4513039686129375,.5663504406693628),
    }
    for key,(f1,ap) in expected.items():
        if key not in rows: raise AssertionError('Missing '+str(key))
        row=rows[key]
        assert math.isclose(float(row['mean_f1']),f1,abs_tol=1e-10)
        assert math.isclose(float(row['mean_ap']),ap,abs_tol=1e-10)
    for flag,key in [(a.past,'past_sha256'),(a.future,'future_sha256')]:
        if flag:
            if not flag.exists(): raise FileNotFoundError(str(flag))
            digest=sha256(flag)
            if digest!=EXPECTED[key]: raise SystemExit(f'ERROR: {key} hash mismatch: {digest}')
            print(f'OK: {flag} matches recorded {key}')
    print('OK: evidence manifest and selected reported metrics match (not a model rerun).')
    if a.past is None and a.future is None:
        print('No input file hash verified in this invocation; pass --past and --future for exact CSV checks.')

if __name__=='__main__':
    main()
