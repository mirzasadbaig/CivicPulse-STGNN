#!/usr/bin/env python3
"""Audit saved NYC 311 aggregate exports, without training or modifying any model.

This diagnostic is NOT an argument that the source data are wrong. It compares
raw per-month coverage and category composition so the researcher can decide
whether a new download and a distinct holdout evaluation are warranted.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REQUIRED = {'day', 'incident_zip', 'complaint_type', 'n'}


def load_csv(path, year):
    df = pd.read_csv(path, dtype={'incident_zip': 'string'}, low_memory=False)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    df['day'] = pd.to_datetime(df['day'], errors='coerce').dt.normalize()
    df['n'] = pd.to_numeric(df['n'], errors='coerce')
    df = df.dropna(subset=['day', 'incident_zip', 'complaint_type', 'n'])
    df = df[(df['n'] >= 0) & (df['day'].dt.year == year)].copy()
    df['month'] = df['day'].dt.month
    df['month_name'] = df['day'].dt.strftime('%Y-%m')
    if df.empty:
        raise ValueError(f'No valid rows for year {year} in {path}')
    return df


def totals(df, year):
    out = df.groupby('month').agg(grouped_rows=('n', 'size'), complaints=('n', 'sum'), days_present=('day','nunique'))
    out = out.reindex(range(1,10), fill_value=0)
    out.columns = [f'{year}_{name}' for name in out.columns]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--historical', required=True, help='Original 2024-2025 NYC aggregated CSV')
    ap.add_argument('--holdout', required=True, help='2026 Jan-Sep aggregated CSV')
    ap.add_argument('--output', default='results/2026_data_audit')
    ap.add_argument('--node-metrics', default=None, help='Optional 2025 metrics.json with frozen node keys')
    args = ap.parse_args()
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    hist = load_csv(args.historical,2025)
    hold = load_csv(args.holdout,2026)
    comparison = totals(hist,2025).join(totals(hold,2026))
    for yr in (2025,2026):
        comparison[f'{yr}_requests_per_day'] = comparison[f'{yr}_complaints'] / comparison[f'{yr}_days_present'].replace(0,float('nan'))
        comparison[f'{yr}_groups_per_day'] = comparison[f'{yr}_grouped_rows'] / comparison[f'{yr}_days_present'].replace(0,float('nan'))
    comparison['2026_vs_2025_requests_pct'] = 100 * comparison['2026_complaints'] / comparison['2025_complaints'].replace(0,float('nan'))
    comparison['2026_vs_2025_groups_pct'] = 100 * comparison['2026_grouped_rows'] / comparison['2025_grouped_rows'].replace(0,float('nan'))
    comparison.index.name='month'
    comparison.to_csv(outdir/'monthly_coverage_comparison.csv', float_format='%.3f')
    # Monthly breakdown makes shifts in just one complaint category visible.
    breakdown = pd.concat([
        hist.groupby(['month','complaint_type'])['n'].sum().rename('complaints_2025'),
        hold.groupby(['month','complaint_type'])['n'].sum().rename('complaints_2026')], axis=1).fillna(0)
    breakdown.to_csv(outdir/'category_monthly_comparison.csv')
    # A day-level report can expose missing calendar days or unexplained gaps.
    a = hist.groupby('day').agg(complaints_2025=('n','sum'), groups_2025=('n','size'))
    b = hold.groupby('day').agg(complaints_2026=('n','sum'), groups_2026=('n','size'))
    daily = pd.concat([a,b],axis=0).sort_index().fillna(0)
    daily.to_csv(outdir/'daily_coverage.csv')
    duplicate_count = {str(y):int(d.duplicated(['day','incident_zip','complaint_type']).sum()) for y,d in [(2025,hist),(2026,hold)]}
    source_summary={}
    for y,d in [(2025,hist),(2026,hold)]:
        source_summary[str(y)] = sorted(map(str,d.source.dropna().unique())) if 'source' in d else []
    node_report={}
    if args.node_metrics:
        old=json.load(open(args.node_metrics))
        keys={(str(n['zip']).zfill(5),str(n['complaint_type'])) for n in old['nodes']}
        for y,d in [(2025,hist),(2026,hold)]:
            pick = [(str(zipcode).zfill(5),str(category)) in keys for zipcode,category in zip(d.incident_zip,d.complaint_type)]
            f = d.loc[pick]
            node_report[str(y)]={'selected_complaints':int(f.n.sum()),'selected_grouped_rows':int(len(f)), 'monthly_selected_complaints':{str(k):int(v) for k,v in f.groupby('month').n.sum().items()}}
            f.groupby('month').n.sum().to_csv(outdir/f'node_selected_{y}.csv',header=['complaints'])
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    x=range(1,10)
    for ax,metric,label in [(axes[0],'requests_per_day','Requests per observed day'),(axes[1],'groups_per_day','ZIP/category groups per observed day')]:
        ax.plot(x,comparison[f'2025_{metric}'],marker='o',label='2025')
        ax.plot(x,comparison[f'2026_{metric}'],marker='o',label='2026')
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
    axes[0].set_title('NYC 311 export coverage — same calendar months')
    axes[-1].set_xticks(list(x),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'])
    fig.tight_layout();fig.savefig(outdir/'monthly_data_coverage.png',dpi=180);plt.close(fig)
    problems=[]
    if duplicate_count['2026']:
        problems.append(f"2026 contains {duplicate_count['2026']} duplicate day/ZIP/category rows; inspect aggregation")
    for mo in range(1,10):
        need=pd.Period(f'2026-{mo:02d}').days_in_month
        seen=int(comparison.loc[mo,'2026_days_present'])
        if seen<need: problems.append(f'2026-{mo:02d} has {seen}/{need} calendar days with any matching rows')
    for mo in range(2,10):
        before=float(comparison.loc[mo-1,'2026_groups_per_day']);now=float(comparison.loc[mo,'2026_groups_per_day'])
        if before>0 and now<.7*before:
            problems.append(f'2026-{mo:02d}: >30% drop in daily ZIP/category groups compared with preceding month')
    text = [
        'CivicPulse NYC 311 — data coverage audit (not a model evaluation)',
        f"2025 compared rows: {len(hist):,}; requests: {int(hist.n.sum()):,}",
        f"2026 compared rows: {len(hold):,}; requests: {int(hold.n.sum()):,}",
        f"Duplicate grouped keys: {duplicate_count}",
        f"Source labels: {source_summary}",
        '', 'Monthly comparison (2025 vs 2026; year-specific counts):',
        comparison.to_string(float_format=lambda x:f'{x:.1f}'),
        '', 'Flagged checks:', *(['- '+p for p in problems] if problems else ['- No automatic flags']),
        '', 'Interpretation: a decline can reflect seasonality, changing reporting or incomplete exports. This audit does not distinguish those causes by itself.',
        'Before publication, check category_monthly_comparison.csv against official raw source counts for the same months.',
    ]
    if node_report: text.extend(['', 'Frozen node counts:', json.dumps(node_report,indent=2)])
    (outdir/'audit_notes.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:9]));print('Wrote audit files to:',outdir.resolve())

if __name__=='__main__':main()
