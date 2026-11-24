#!/usr/bin/env python3
"""Strict external 2026 holdout for the *corrected* CivicPulse spatial GCN-GRU.

Place this script next to the existing `src` package. It trains on the pre-2026
training anchors, selects checkpoint and classification thresholds on 2025
validation anchors, and evaluates a pre-specified January-July 2026 diagnostic subset.

It deliberately does NOT use load_data() on the concatenated files, because
that would reselect nodes, fit normalization and learn targets using 2026 data.

2025 Sep-Dec is NOT used for training, selection or scoring here. This makes
results directly comparable to the controlled five-seed validation protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
from src.data import load_data, windows
from src.experiment import device_auto, predict, run_gru, score, seed_all, val_threshold

FEATURES = ['log count', 'log 7d mean', 'log 14d mean', 'log earlier 7d mean',
            '7d/14d ratio', 'same weekday count', 'weekday sin', 'weekday cos',
            'year sin', 'year cos']


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def no_graph_feature(ds):
    col = ds.feature_names.index('neighbor count')
    ds.x = np.delete(ds.x, col, axis=-1).copy()
    ds.feature_names = [name for i, name in enumerate(ds.feature_names) if i != col]
    assert ds.feature_names == FEATURES, f'Unexpected feature schema: {ds.feature_names}'
    return ds


def read_future_counts(csv, frozen_ds, allow_synthetic=False):
    frame = pd.read_csv(csv, dtype={'incident_zip': str})
    required = {'day', 'incident_zip', 'complaint_type', 'n', 'lat', 'lon', 'source'}
    if not required.issubset(frame):
        raise ValueError(f'Missing 2026 columns: {required-set(frame)}')
    source_set = set(frame.source.dropna().unique())
    if source_set != {frozen_ds.source}:
        raise ValueError(f'Source mismatch: 2025={frozen_ds.source}, 2026={source_set}')
    if frozen_ds.source != 'NYC_311_PUBLIC_AGGREGATED' and not allow_synthetic:
        raise ValueError('Synthetic data require --allow-synthetic-test; never report these as NYC metrics')
    frame['day'] = pd.to_datetime(frame.day, errors='coerce').dt.normalize()
    frame['incident_zip'] = frame.incident_zip.astype(str).str.strip().str.zfill(5)
    for col in ['n', 'lat', 'lon']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.dropna(subset=['day', 'incident_zip', 'complaint_type', 'n', 'lat', 'lon'])
    frame = frame[(frame.incident_zip.str.fullmatch(r'\d{5}')) & (frame.n >= 0) &
                  frame.lat.between(40,42) & frame.lon.between(-75,-72)].copy()
    if frame.empty:
        raise ValueError('No valid rows in external holdout')
    first = pd.Timestamp('2026-01-01')
    # July 31 is a SOURCE-CONTINUITY cutoff, not a chosen performance cutoff.
    # August-September are excluded because source categories Water System and
    # Sewer disappear while other DEP maintenance categories continue.
    full_export_end = pd.Timestamp('2026-09-30')
    final = pd.Timestamp('2026-07-31')
    if frame.day.min() < first or frame.day.max() > full_export_end:
        raise ValueError('Expected original 2026 Jan-Sep source CSV')
    frame = frame[frame.day.between(first, final)].copy()
    if frame.day.min() != first or frame.day.max() != final:
        raise ValueError('January 1 or July 31 missing in exported source')
    days = pd.date_range(first, final, freq='D')
    # Prevent zero-filling a wholly absent selected category within any month.
    expected_cats = {str(cat) for _, cat in frozen_ds.nodes}
    for month, group in frame.groupby(frame.day.dt.to_period('M')):
        if group.day.nunique() != month.days_in_month:
            raise ValueError(f'{month}: at least one calendar day missing')
        missing = expected_cats - set(group.complaint_type.unique())
        if missing:
            raise ValueError(f'{month}: selected complaint types missing: {sorted(missing)}')
    nodemap = {tuple(node): i for i, node in enumerate(frozen_ds.nodes)}
    new_counts = np.zeros((len(days), len(nodemap)), dtype=np.float32)
    tmap = {day: i for i, day in enumerate(days)}
    grouped = frame.groupby(['day', 'incident_zip', 'complaint_type'], sort=False).n.sum()
    selected_count = 0
    for (date, zip5, category), count in grouped.items():
        node_id = nodemap.get((zip5, category))
        if node_id is not None:
            new_counts[tmap[date], node_id] += count
            selected_count += count
    monthly = frame.assign(month=frame.day.dt.strftime('%Y-%m')).groupby('month').agg(
        grouped_rows=('n','size'), complaints=('n','sum'), days_present=('day','nunique'))
    return days, new_counts, monthly, int(frame.n.sum()), int(selected_count)


def raw_features(counts, dates):
    """Reproduce original src.data.load_data features, omitting neighbor count.

    IMPORTANT: x(t) uses only counts at dates <=t. No 2026 labels are used.
    """
    all_features = []
    for t, d in enumerate(dates):
        current = counts[t]
        past7 = counts[max(0,t-6):t+1]
        past14 = counts[max(0,t-13):t+1]
        prev7 = counts[max(0,t-13):max(0,t-6)]
        weekday_ago = counts[t-7] if t >= 7 else np.zeros_like(current)
        values = np.stack([
            np.log1p(current), np.log1p(past7.mean(axis=0)),
            np.log1p(past14.mean(axis=0)),
            np.log1p(prev7.mean(axis=0)) if len(prev7) else np.zeros_like(current),
            np.clip(past7.mean(axis=0) / (past14.mean(axis=0)+1), 0, 5),
            np.log1p(weekday_ago),
            np.full_like(current, np.sin(2*np.pi*d.dayofweek/7)),
            np.full_like(current, np.cos(2*np.pi*d.dayofweek/7)),
            np.full_like(current, np.sin(2*np.pi*d.dayofyear/365.25)),
            np.full_like(current, np.cos(2*np.pi*d.dayofyear/365.25))],axis=1)
        all_features.append(values)
    return np.asarray(all_features, dtype=np.float32)


def build_external_features(ds, days2026, counts2026):
    if ds.dates[-1] != pd.Timestamp('2025-12-31') or ds.dates[0] != pd.Timestamp('2024-01-01'):
        raise ValueError('Original CSV must span 2024-01-01 through 2025-12-31')
    if days2026[0] != ds.dates[-1] + pd.Timedelta(days=1):
        raise ValueError('Missing daily continuity between 2025 and 2026')
    dates = ds.dates.append(days2026)
    counts = np.concatenate([ds.counts, counts2026],axis=0)
    f = raw_features(counts, dates)
    # Precisely the original training range, not all 2024-25 and never 2026.
    cut_train = int(.70*len(ds.dates))
    scaler = StandardScaler()
    scaler.fit(f[:cut_train].reshape(-1,len(FEATURES)))
    xs = scaler.transform(f.reshape(-1,len(FEATURES))).reshape(f.shape).astype(np.float32)
    xs = np.clip(xs,-8,8)
    dev = float(np.max(np.abs(xs[:len(ds.dates)] - ds.x)))
    if dev > 1e-4:
        raise AssertionError(f'Preprocessing mismatch with original training code: max abs={dev:.7g}')
    return dates, counts, xs, dev


def read_2026_target(ds, counts, dates):
    first = int(np.searchsorted(dates, np.datetime64('2026-01-01')))
    # Two-day future target must have full coverage through July 31.
    end = len(dates)-2
    anchors = np.arange(first, end, dtype=np.int64)
    future = counts[anchors+1] + counts[anchors+2]
    y = (future > ds.threshold_counts[None,:]).astype(np.float32)
    if len(anchors) != 210:
        raise AssertionError(f'Expected 210 fully observed anchor days, got {len(anchors)}')
    if not np.any(y) or np.all(y):
        raise ValueError('External holdout has only one target class; classification scores invalid')
    return anchors, future, y


def train_logistic(ds):
    def xx(indices, all_x):
        av = np.stack([all_x[max(0,i-6):i+1].mean(0) for i in indices])
        return np.concatenate([all_x[indices], av],axis=-1).reshape(-1,2*ds.x.shape[-1])
    model = LogisticRegression(class_weight='balanced', max_iter=1500, random_state=0)
    model.fit(xx(ds.idx_train, ds.x), ds.y[ds.idx_train].ravel().astype(int))
    probs = model.predict_proba(xx(ds.idx_val,ds.x))[:,1].reshape(ds.y[ds.idx_val].shape)
    threshold = val_threshold(ds.y[ds.idx_val], probs)
    return model, threshold, xx


def main():
    parser = argparse.ArgumentParser(description='2026 Jan-Jul source-continuity sensitivity, post-hoc cutoff; never tunes on 2026')
    parser.add_argument('--past', required=True, help='2024-2025 daily aggregated CSV')
    parser.add_argument('--future', required=True, help='Original 2026 Jan-Sep daily aggregated CSV; script evaluates only Jan-Jul')
    parser.add_argument('--output', default='results/strict_2026_holdout')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7,17,27,37,42])
    parser.add_argument('--lookback', type=int, default=28)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--allow-synthetic-test', action='store_true')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Seed list contains duplicates')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_data(args.past, lookback=args.lookback, max_zips=20,
                   max_nodes=80, quantile=.75, graph='spatial')
    ds = no_graph_feature(ds)
    if ds.source != 'NYC_311_PUBLIC_AGGREGATED' and not args.allow_synthetic_test:
        raise ValueError('Expected NYC data; synthetic mode requires --allow-synthetic-test')
    d26,c26, monthly, total, selected = read_future_counts(args.future,ds,args.allow_synthetic_test)
    dates,counts,all_x,match_error = build_external_features(ds,d26,c26)
    anchors,actual_future,y26 = read_2026_target(ds,counts,dates)
    print(f'2026: {len(d26)} calendar days, {len(anchors)} forecast anchors, '
          f'{len(ds.nodes)} frozen nodes, {len(y26)*len(ds.nodes)} classification cases')
    print(f'Original preprocessing reproduced, maximum difference {match_error:.2g}')
    print(f'2026 complaint totals: source={total:,}, matching frozen nodes={selected:,}')
    print(f'2026 positive prevalence: {y26.mean():.3%}; fixed 2024/25 label thresholds')
    print('Monthly data coverage (all original ZIP/category groups):')
    print(monthly.to_string())
    for (prevmonth,prev),(month,current) in zip(list(monthly.iterrows()), list(monthly.iloc[1:].iterrows())):
        if float(current.grouped_rows)<.6*float(prev.grouped_rows):
            print(f'CAUTION: monthly grouped rows fell >40%: {prevmonth} to {month}. '
                  'Check source completeness/category changes before interpretation.')
    device = torch.device('cpu') if args.cpu else device_auto()
    # Uses the exact same validation-only model-selection procedure as fair_ablation.py
    test_windows = windows(all_x,anchors,args.lookback)
    results=[]; predictions=[]
    for seed in args.seeds:
        for name,adj_numpy in [
                ('GCN-GRU spatial', ds.adj),
                ('GRU identity adjacency', np.eye(len(ds.nodes),dtype=np.float32))]:
            seed_all(seed)
            val_metrics, model, _, _ = run_gru(
                name,ds,adj_numpy,args.lookback,args.epochs,args.patience,args.lr,
                args.batch_size,args.hidden,seed,device,evaluate_test=False)
            adj_tensor = torch.tensor(adj_numpy,device=device)
            probabilities = predict(model,test_windows,adj_tensor,device,args.batch_size)
            threshold = float(val_metrics['validation_f1_threshold'])
            metrics_26 = score(y26,probabilities,threshold)
            results.append({'seed':seed,'model':name,'validation_best_AP':val_metrics['validation_best_AP'],
                            'best_epoch':val_metrics['best_epoch'],
                            'validation_threshold':threshold,'external_2026':metrics_26})
            print(f'EXTERNAL 2026: seed={seed} {name}: F1={metrics_26["f1"]:.3f}, '
                  f'AP={metrics_26["average_precision_pr_auc"]:.3f}, '
                  f'precision={metrics_26["precision"]:.3f}, recall={metrics_26["recall"]:.3f}')
            if seed == 42 and name == 'GCN-GRU spatial':
                torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'feature_names':ds.feature_names,'frozen_nodes':ds.nodes,
                            'frozen_graph_adj':ds.adj,'frozen_threshold_counts':ds.threshold_counts,
                            'seed':42,'validation_probability_threshold':threshold,
                            'training_end_anchor':str(ds.dates[ds.idx_train[-1]].date()),
                            'validation_start_anchor':str(ds.dates[ds.idx_val[0]].date())},out/'frozen_spatial_seed42.pt')
                for ti, anchor in enumerate(anchors):
                    for j, (zipcode, category) in enumerate(ds.nodes):
                        predictions.append({'anchor_date':str(dates[anchor].date()),'zip':zipcode,
                                            'category':category,'observed_future_48h_count':float(actual_future[ti,j]),
                                            'frozen_train_75th_pct_threshold':float(ds.threshold_counts[j]),
                                            'actual_elevated_demand':int(y26[ti,j]),
                                            'probability':float(probabilities[ti,j]),
                                            'predicted_elevated_demand':int(probabilities[ti,j]>=threshold)})
    # Deterministic logistic baseline, trained using same graph-free features and original training anchors.
    logit,logit_threshold,tabular=train_logistic(ds)
    logit_input=tabular(anchors,all_x)
    logit_prob=logit.predict_proba(logit_input)[:,1].reshape(y26.shape)
    logistic_metrics=score(y26,logit_prob,logit_threshold)
    print(f'EXTERNAL 2026: Logistic baseline: F1={logistic_metrics["f1"]:.3f}, '
          f'AP={logistic_metrics["average_precision_pr_auc"]:.3f}')
    summaries=[]
    for model in sorted(set(entry['model'] for entry in results)):
        selected_m=[entry['external_2026'] for entry in results if entry['model']==model]
        summaries.append({'model':model,'n_seeds':len(selected_m),
                          'mean_F1':float(np.mean([x['f1'] for x in selected_m])),
                          'sd_F1':float(np.std([x['f1'] for x in selected_m],ddof=1)) if len(selected_m)>1 else None,
                          'mean_PR_AUC':float(np.mean([x['average_precision_pr_auc'] for x in selected_m])),
                          'sd_PR_AUC':float(np.std([x['average_precision_pr_auc'] for x in selected_m],ddof=1)) if len(selected_m)>1 else None})
    summaries.append({'model':'Logistic (graph-free)', 'n_seeds':1,'mean_F1':logistic_metrics['f1'],
                      'sd_F1':None,'mean_PR_AUC':logistic_metrics['average_precision_pr_auc'],
                      'sd_PR_AUC':None})
    payload={'status':'SOURCE_CONTINUITY_SENSITIVITY_2026_JAN_JUL' if ds.source=='NYC_311_PUBLIC_AGGREGATED' else 'SYNTHETIC_SMOKE_TEST_DO_NOT_REPORT',
             'historical_csv':args.past,'historical_sha256':sha256_file(args.past),
             'holdout_csv':args.future,'holdout_sha256':sha256_file(args.future),
             'data_source':ds.source,'train_source_period':'2024-01-01 through 2025-12-31',
             'training_anchors_end':str(ds.dates[ds.idx_train[-1]].date()),
             'validation_anchors_start':str(ds.dates[ds.idx_val[0]].date()),
             'external_2026_anchor_start':str(dates[anchors[0]].date()),
             'external_2026_anchor_end':str(dates[anchors[-1]].date()),
             'note':'2025 test block not used in this script. All nodes, scaling, target and graph frozen to pre-2026 training data. No hyperparameter tuning on 2026.',
             'architecture':'src.model.GCNGRU with spatial adjacency or identity adjacency',
             'lookback':args.lookback,'horizon_days':2,'quantile':.75,
             'frozen_nodes':len(ds.nodes),'cases':int(y26.size),
             'external_positive_rate':float(y26.mean()),'source_complaints_2026':total,
             'source_complaints_matching_frozen_nodes':selected,
             'source_monthly_coverage':monthly.reset_index().to_dict(orient='records'),
             'evaluation_scope':'2026 Jan-Jul only, excludes source-discontinuous Aug-Sep',
             'cutoff_selected_for_data_integrity_not_model_score':True,
             'seeds':args.seeds,'individual_runs':results,'summary':summaries,
             'logistic_external_2026':logistic_metrics}
    (out/'2026_holdout_metrics.json').write_text(json.dumps(payload,indent=2))
    pd.DataFrame(summaries).to_csv(out/'2026_holdout_summary.csv',index=False)
    if predictions:
        pd.DataFrame(predictions).to_csv(out/'2026_spatial_seed42_predictions.csv',index=False)
    print('\nSUMMARY (2026 Jan-Jul post-hoc source-continuity sensitivity):')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Saved:',out.resolve())

if __name__ == '__main__':
    main()
