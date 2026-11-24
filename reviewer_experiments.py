#!/usr/bin/env python3
"""CivicPulse-STGNN: reviewer-requested controls and uncertainty.

Requirements: run from original CivicPulse_STGNN_NYC project (with src/), with
frozen_2026_eval.py placed beside this script. 2026 is never used to fit models,
choose checkpoints or calibrate decision thresholds.

Uses prior project code and training/validation definitions unchanged. The
shuffled graph is a RANDOM NODE RELABELING of the original spatial graph,
with fixed permutation independent of model seed. It preserves edge weights
and global spectrum but not individual nodes' degrees. The target and features
are unchanged. More randomizations are desirable for a conclusive study.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import torch

from src.data import load_data, windows
from src.experiment import run_gru, predict, score, seed_all, device_auto
from frozen_2026_eval import no_graph_feature, read_future_counts, build_external_features, read_2026_target, sha256_file

NAMES = ['GCN-GRU spatial', 'GCN-GRU shuffled graph', 'GRU identity adjacency']
MONTHS = ['Jan-Jul 2026', 'Jan-Sep 2026']


def normalize_raw(raw):
    a = np.asarray(raw, dtype=np.float32) + np.eye(len(raw), dtype=np.float32)
    inv = np.maximum(a.sum(axis=1), 1e-6) ** (-0.5)
    return np.asarray(inv[:, None] * a * inv[None, :], dtype=np.float32)


def shuffled_graph(raw, shuffle_seed):
    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(raw.shape[0])
    if np.array_equal(perm, np.arange(len(perm))):
        raise ValueError('Degenerate permutation; change --shuffle-seed')
    perm_raw = raw[np.ix_(perm, perm)].copy()
    # Raw undirected edges and their weights are preserved as a global multiset.
    np.testing.assert_allclose(np.sort(perm_raw.ravel()), np.sort(raw.ravel()))
    return normalize_raw(perm_raw), perm


def naive_score(counts, anchors, thresholds, kind):
    if kind == 'recent48h':
        prev = counts[anchors-1] + counts[anchors]
    elif kind == 'previous_week_same_days':
        prev = counts[anchors-6] + counts[anchors-5]
    else:
        raise ValueError(kind)
    # Strictly monotone count/exceedance margin, not calibrated probability.
    return np.log1p(prev) - np.log1p(thresholds[None, :])


def fit_score_threshold(y, continuous):
    """Optimize F1 on 2025 validation only; ties are handled deterministically."""
    flat = continuous.ravel()
    qs = np.linspace(0, 1, 401)
    options = np.unique(np.concatenate([np.quantile(flat, qs), [flat.min()-1e-5, flat.max()+1e-5]]))
    yy = y.ravel().astype(np.uint8)
    vals = [f1_score(yy, flat > t, zero_division=0) for t in options]
    return float(options[np.argmax(vals)])


def metric_scores(y, continuous, threshold, *, inclusive=True):
    yt = y.ravel().astype(np.uint8)
    cp = continuous.ravel()
    yp = cp >= threshold if inclusive else cp > threshold
    return {'n': int(yt.size), 'positives': int(yt.sum()),
            'prevalence': float(yt.mean()),
            'f1': float(f1_score(yt, yp, zero_division=0)),
            'ap': float(average_precision_score(yt, cp)) if yt.sum() else float('nan'),
            'precision': float(precision_score(yt, yp, zero_division=0)),
            'recall': float(recall_score(yt, yp, zero_division=0)),
            'threshold': float(threshold)}


def grouped_scores(y, pred, dates, cats, threshold, *, inclusive=True):
    """One row per calendar month/category, preserving per-date node clustering."""
    out=[]
    masks=[(str(m), np.asarray(dates.to_period('M')==m)) for m in dates.to_period('M').unique()]
    for label, dm in masks:
        ys=y[dm]
        ps=pred[dm]
        mm=metric_scores(ys,ps,threshold,inclusive=inclusive)
        out.append({'grouping':'month','group':label,**mm})
    for cat in sorted(set(cats)):
        cm=np.asarray([c==cat for c in cats])
        mm=metric_scores(y[:,cm],pred[:,cm],threshold,inclusive=inclusive)
        out.append({'grouping':'category','group':cat,**mm})
    return out


def moving_block_indices(n_days, block_size, rng):
    if block_size > n_days:
        raise ValueError('block size exceeds evaluated days')
    starts = rng.integers(0, n_days-block_size+1, size=int(np.ceil(n_days/block_size)))
    return np.concatenate([np.arange(s,s+block_size) for s in starts])[:n_days]


def paired_bootstrap(y, model_predictions, seed_list, thresholds, *, n_boot=300, block=28, seed=20261124):
    """Resample dates in contiguous paired blocks, keeping all 80 nodes together.

    CI conditions on the set of fitted seeds and on this study's ZIP/category set.
    It is NOT a spatial generalization CI or proof of causality.
    """
    rng=np.random.default_rng(seed)
    n_days=len(y)
    seeds=list(seed_list)
    def one_delta(ii):
        yy=y[ii].ravel().astype(np.uint8)
        if yy.sum()==0 or yy.sum()==len(yy):
            return None
        ap=[]; f1=[]
        for s in seeds:
            ps=model_predictions[('GCN-GRU spatial',s)][ii].ravel()
            pg=model_predictions[('GRU identity adjacency',s)][ii].ravel()
            ap.append(average_precision_score(yy,ps)-average_precision_score(yy,pg))
            f1.append(f1_score(yy,ps>=thresholds[('GCN-GRU spatial',s)],zero_division=0)-
                      f1_score(yy,pg>=thresholds[('GRU identity adjacency',s)],zero_division=0))
        return float(np.mean(ap)),float(np.mean(f1))
    original=one_delta(np.arange(n_days))
    replicates=[]
    for _ in range(n_boot):
        res=one_delta(moving_block_indices(n_days,block,rng))
        if res is not None:replicates.append(res)
    arr=np.array(replicates)
    if not len(arr):raise RuntimeError('No valid bootstrap replicates')
    return {'method':'paired moving-block bootstrap over consecutive anchor dates',
            'block_days':block,'requested_replicates':n_boot,'valid_replicates':len(arr),
            'seed':seed,'graph_seeds':seeds,'fixed_thresholds_selected_on_2025_validation':True,
            'conditional_on':'fitted model seeds, selected ZIPs, and evaluated 2026 period',
            'delta_AP_graph_minus_identity':{'point':original[0], 'percentile_95_CI':list(map(float,np.quantile(arr[:,0],[.025,.975]))),
                                             'fraction_bootstrap_nonpositive':float(np.mean(arr[:,0]<=0))},
            'delta_F1_graph_minus_identity':{'point':original[1], 'percentile_95_CI':list(map(float,np.quantile(arr[:,1],[.025,.975]))),
                                             'fraction_bootstrap_nonpositive':float(np.mean(arr[:,1]<=0))}}


def year_over_year_counts(ds, dates_all, counts_all, output):
    rows=[]
    arr=[(pd.DatetimeIndex(ds.dates), ds.counts, '2025'),
         (pd.DatetimeIndex(dates_all), counts_all, '2026')]
    for dates,counts,tag in arr:
        for m in range(1,8):
            ix=np.asarray((dates.year==int(tag))&(dates.month==m))
            if ix.sum()==0:continue
            for category in sorted(set(c for _,c in ds.nodes)):
                ni=[k for k,(z,c) in enumerate(ds.nodes) if c==category]
                rows.append({'year':tag,'month':m,'category':category,
                             'selected_zip_count':len(set(z for z,c in ds.nodes if c==category)),
                             'observed_days':int(ix.sum()),'complaints_in_selected_nodes':float(counts[np.ix_(ix,ni)].sum())})
    pd.DataFrame(rows).to_csv(output,index=False)


def main():
    a=argparse.ArgumentParser(description='Reviewer controls: naive baselines, shuffled graph, block CIs, subgroups')
    a.add_argument('--past',required=True)
    a.add_argument('--future',required=True)
    a.add_argument('--output',default='results/reviewer_controls')
    a.add_argument('--seeds',nargs='+',type=int,default=[7,17,27,37,42])
    a.add_argument('--epochs',type=int,default=35)
    a.add_argument('--lookback',type=int,default=28)
    a.add_argument('--hidden',type=int,default=32)
    a.add_argument('--patience',type=int,default=8)
    a.add_argument('--batch-size',type=int,default=16)
    a.add_argument('--lr',type=float,default=.001)
    a.add_argument('--shuffle-seed',type=int,default=20261124)
    a.add_argument('--bootstrap',type=int,default=300)
    a.add_argument('--block-days',type=int,default=28)
    a.add_argument('--cpu',action='store_true')
    a.add_argument('--synthetic-smoke',action='store_true',help='ONLY for software testing; outputs are not research metrics')
    args=a.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Duplicate model seed')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ds=no_graph_feature(load_data(args.past,graph='spatial',lookback=args.lookback,max_zips=20,max_nodes=80,quantile=.75))
    if ds.source!='NYC_311_PUBLIC_AGGREGATED' and not args.synthetic_smoke:
        raise ValueError('Expected real NYC public data; for a smoke test use --synthetic-smoke')
    days26,counts26,monthly,source_totals,matched=read_future_counts(args.future,ds,args.synthetic_smoke)
    dates_all,counts_all,all_x,matching_error=build_external_features(ds,days26,counts26)
    anchor_all,counts_future,y_all=read_2026_target(ds,counts_all,dates_all)
    dates_26=pd.DatetimeIndex(dates_all[anchor_all])
    index_sets={'Jan-Jul 2026':np.flatnonzero(dates_26 <= pd.Timestamp('2026-07-29')),
                'Jan-Sep 2026':np.arange(len(anchor_all))}
    assert len(index_sets['Jan-Jul 2026'])==210 and len(index_sets['Jan-Sep 2026'])==271
    dev=torch.device('cpu') if args.cpu else device_auto()
    xwin=windows(all_x,anchor_all,args.lookback)
    adj_shuffled,perm=shuffled_graph(ds.raw_adj,args.shuffle_seed)
    print('Inputs verified; frozen node count:',len(ds.nodes),' training scaler mismatch:',matching_error,
          'on',str(dev),'features:',ds.feature_names,'shuffle seed:',args.shuffle_seed)
    configs=[('GCN-GRU spatial',ds.adj),('GCN-GRU shuffled graph',adj_shuffled),
             ('GRU identity adjacency',np.eye(len(ds.nodes),dtype=np.float32))]
    predictions={} # keys: (model, seed); arrays 271 by 80
    thresholds={}
    rows=[]; detail=[]
    for seed in args.seeds:
        for name,adj in configs:
            seed_all(seed)
            vm,model,_,_=run_gru(name,ds,adj,args.lookback,args.epochs,args.patience,
                    args.lr,args.batch_size,args.hidden,seed,dev,evaluate_test=False)
            threshold=float(vm['validation_f1_threshold'])
            prob=predict(model,xwin,torch.as_tensor(adj,device=dev),dev,args.batch_size)
            predictions[(name,seed)]=prob.astype(np.float32)
            thresholds[(name,seed)]=threshold
            for period,ii in index_sets.items():
                m=metric_scores(y_all[ii],prob[ii],threshold)
                rows.append({'period':period,'model':name,'seed':seed,**m})
                for r in grouped_scores(y_all[ii],prob[ii],dates_26[ii],
                                        [cat for _,cat in ds.nodes],threshold):
                    detail.append({'period':period,'model':name,'seed':seed,**r})
            print('COMPLETE',name,'seed',seed,'Jan-Jul AP',round(rows[-2]['ap'],3))
    # Two naive baselines, evaluated at thresholds selected on original validation anchors.
    for name,kind in [('Persistence previous 48h','recent48h'),
                      ('Seasonal naive previous-week 48h','previous_week_same_days')]:
        vscore=naive_score(ds.counts,ds.idx_val,ds.threshold_counts,kind)
        threshold=fit_score_threshold(ds.y[ds.idx_val],vscore)
        p=naive_score(counts_all,anchor_all,ds.threshold_counts,kind)
        for period,ii in index_sets.items():
            m=metric_scores(y_all[ii],p[ii],threshold,inclusive=False)
            rows.append({'period':period,'model':name,'seed':'deterministic',**m})
            for r in grouped_scores(y_all[ii],p[ii],dates_26[ii],
                                    [cat for _,cat in ds.nodes],threshold,inclusive=False):
                detail.append({'period':period,'model':name,'seed':'deterministic',**r})
        print('COMPLETE',name,'2025-val threshold',threshold)
    # Strong tabular baseline with EXACT SAME feature channels and 28-day history.
    # Other model hyperparameters fixed in script, not chosen based on 2026 outcomes.
    seed_all(42)
    train_windows=windows(ds.x,ds.idx_train,args.lookback).reshape(-1,args.lookback*ds.x.shape[-1])
    val_windows=windows(ds.x,ds.idx_val,args.lookback).reshape(-1,args.lookback*ds.x.shape[-1])
    xtrain=train_windows.reshape(-1,train_windows.shape[-1]); ytrain=ds.y[ds.idx_train].ravel().astype(int)
    xval=val_windows.reshape(-1,val_windows.shape[-1]); yval=ds.y[ds.idx_val].ravel().astype(int)
    # Balanced training weights, comparable to class-weighted neural models.
    prev=float(np.mean(ytrain)); weights=np.where(ytrain==1,0.5/max(prev,1e-6),0.5/max(1-prev,1e-6))
    gb=HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,max_leaf_nodes=15,
                                      l2_regularization=1.0,early_stopping=False,random_state=42)
    gb.fit(xtrain,ytrain,sample_weight=weights)
    vp=gb.predict_proba(xval)[:,1]
    from src.experiment import val_threshold
    threshold=float(val_threshold(yval,vp))
    pp=gb.predict_proba(xwin.reshape(-1,train_windows.shape[-1]))[:,1].reshape(y_all.shape)
    for period,ii in index_sets.items():
        m=metric_scores(y_all[ii],pp[ii],threshold)
        rows.append({'period':period,'model':'Pooled HistGradientBoosting (28-day inputs)','seed':'deterministic',**m})
        for r in grouped_scores(y_all[ii],pp[ii],dates_26[ii],
                                [cat for _,cat in ds.nodes],threshold):
            detail.append({'period':period,'model':'Pooled HistGradientBoosting (28-day inputs)',
                           'seed':'deterministic',**r})
    print('COMPLETE gradient boosting baseline')
    metrics_df=pd.DataFrame(rows)
    metrics_df.to_csv(out/'per_run_and_period_metrics.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'per_month_and_category_metrics.csv',index=False)
    year_over_year_counts(ds,dates_all,counts_all,out/'selected_nodes_year_over_year_counts.csv')
    stats=(metrics_df.groupby(['period','model'],dropna=False)
           .agg(n_runs=('f1','count'),mean_f1=('f1','mean'),sd_f1=('f1','std'),
                mean_ap=('ap','mean'),sd_ap=('ap','std'),
                mean_precision=('precision','mean'),mean_recall=('recall','mean'))
           .reset_index())
    stats.to_csv(out/'model_summary.csv',index=False)
    ci={}
    for period,ii in index_sets.items():
        arr={(name,s):predictions[(name,s)][ii] for name in NAMES for s in args.seeds}
        ci[period]=paired_bootstrap(y_all[ii],arr,args.seeds,thresholds,
                                   n_boot=args.bootstrap,block=args.block_days)
    manifest={'status':'SYNTHETIC_SMOKE_TEST_DO_NOT_REPORT' if args.synthetic_smoke else 'REVIEWER_EXPERIMENT_REAL_DATA',
              'source':ds.source,'past_sha256':sha256_file(args.past),
              'future_sha256':sha256_file(args.future),
              'windows':{w:{'anchors':len(ii),'cases':int(len(ii)*len(ds.nodes)),
                            'prevalence':float(y_all[ii].mean())} for w,ii in index_sets.items()},
              'model_seeds':args.seeds,'shuffled_graph_permutation_seed':args.shuffle_seed,
              'shuffled_graph_permutation_indices':perm.tolist(),
              'shuffled_graph_interpretation':'fixed adjacency node relabeling; preserves multiset of global weights, not node-specific degree',
              'fit_data':'2024-2025 training anchors; checkpoint and threshold selected only using 2025 validation',
              'no_training_on_2026':True,'window_jan_jul_is_post_hoc':True,
              '2026_jan_sep_has_known_complaint_category_break':True,
              'years_compared_for_seasonality':'same Jan-Jul calendar months of 2025 and 2026, selected frozen nodes',
              'bootstrap':ci,'environment':{'python':sys.version.split()[0],'platform':platform.platform(),
                                            'torch':torch.__version__,'numpy':np.__version__},
              'limitations':['Single fixed randomized topology; additional graph permutations strengthen controls.',
                             'Bootstrap CIs conditional on fitted seeds and selected geography, not city/generalization bounds.',
                             '28-day overlapping forecasts introduce strong dependence; resample whole dates together.',
                             'Category discontinuity not repaired; Jan-Jul was chosen after the issue was identified.']}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nMODEL SUMMARY\n',stats.to_string(index=False))
    for w,r in ci.items():
        print('\nBLOCK BOOTSTRAP:',w)
        for metric in ['delta_AP_graph_minus_identity','delta_F1_graph_minus_identity']:
            d=r[metric]
            print(' ',metric,'delta',round(d['point'],5),'95% percentile CI',
                  [round(v,5) for v in d['percentile_95_CI']])
    print('\nSAVED',out.resolve())

if __name__=='__main__':
    main()
