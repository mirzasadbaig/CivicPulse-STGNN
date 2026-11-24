#!/usr/bin/env python3
"""Controlled, validation-only graph-message-passing ablation.

Place next to original src/ folder inside CivicPulse_STGNN_NYC.
- Removes the neighbor-aggregate input feature from ALL models.
- Ensures spatial/hybrid models have identical X, labels and splits.
- Resets RNG before EACH neural model per seed.
- Evaluates validation only. Never opens test targets for scoring.
"""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path
import numpy as np
from src.data import load_data
from src.experiment import run_gru, baseline, seed_all, device_auto
import torch


def strip_graph_input(ds):
    """Remove graph-derived neighbor count from input features for a clean GCN ablation."""
    if 'neighbor count' not in ds.feature_names:
        raise RuntimeError('Expected neighbor count feature absent; review project source.')
    col = ds.feature_names.index('neighbor count')
    ds.x = np.delete(ds.x, col, axis=-1).copy()
    ds.feature_names = [f for i, f in enumerate(ds.feature_names) if i != col]
    return ds


def check_matching(spatial, hybrid):
    checks = {
        'node identities': spatial.nodes == hybrid.nodes,
        'days': np.array_equal(spatial.dates, hybrid.dates),
        'labels': np.array_equal(spatial.y, hybrid.y),
        'training split': np.array_equal(spatial.idx_train, hybrid.idx_train),
        'validation split': np.array_equal(spatial.idx_val, hybrid.idx_val),
        'thresholds': np.array_equal(spatial.threshold_counts, hybrid.threshold_counts),
        'feature names': spatial.feature_names == hybrid.feature_names,
        'feature values': np.allclose(spatial.x, hybrid.x, rtol=1e-6, atol=1e-6),
    }
    failed = [k for k, valid in checks.items() if not valid]
    if failed:
        raise AssertionError('Graph comparisons are not controlled: ' + ', '.join(failed))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/nyc_daily_2024_2025.csv')
    parser.add_argument('--output', default='results/fair_ablation_validation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7,17,27,37,42])
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lookback', type=int, default=28)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-zips', type=int, default=20)
    parser.add_argument('--max-nodes', type=int, default=80)
    parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    common=dict(lookback=args.lookback,max_zips=args.max_zips,max_nodes=args.max_nodes,quantile=.75)
    spatial=strip_graph_input(load_data(args.csv,graph='spatial',**common))
    hybrid=strip_graph_input(load_data(args.csv,graph='hybrid',**common))
    check_matching(spatial,hybrid)
    if spatial.source != 'NYC_311_PUBLIC_AGGREGATED':
        print('WARNING: SYNTHETIC SOFTWARE CHECK, NOT A REAL-DATA RESULT')
    print('Controlled data check passed. Features:',spatial.feature_names)
    print('Spatial edges:',int(np.count_nonzero(np.triu(spatial.raw_adj,1))))
    print('Hybrid edges:',int(np.count_nonzero(np.triu(hybrid.raw_adj,1))))
    print('Train / validation / test anchors:',len(spatial.idx_train),len(spatial.idx_val),len(spatial.idx_test))
    device=torch.device('cpu') if args.cpu else device_auto()
    rows=[]
    for seed in args.seeds:
        print('\n===== SEED',seed,'=====')
        seed_all(seed)
        logistic,_=baseline(spatial,evaluate_test=False)
        lm=logistic['validation']
        rows.append({'seed':seed,'model':'Logistic (no graph-derived features)', 'F1':lm['f1'],'PR_AUC':lm['average_precision_pr_auc']})
        for name,ds,adj in [
            ('GCN-GRU spatial',spatial,spatial.adj),
            ('GCN-GRU hybrid',hybrid,hybrid.adj),
            ('GRU identity adjacency',spatial,np.eye(len(spatial.nodes),dtype=np.float32))
        ]:
            # Identical seed resets for all model initializations and shuffling.
            seed_all(seed)
            m,_,_,_=run_gru(name,ds,adj,args.lookback,args.epochs,args.patience,args.lr,
                            args.batch_size,args.hidden,seed,device,evaluate_test=False)
            val=m['validation']
            rows.append({'seed':seed,'model':name,'F1':val['f1'],'PR_AUC':val['average_precision_pr_auc']})
            print('SUMMARY',seed,name,'F1',round(val['f1'],4),'AP',round(val['average_precision_pr_auc'],4))
    output_csv=out/'per_seed_validation.csv'
    with output_csv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','model','F1','PR_AUC']);w.writeheader();w.writerows(rows)
    stats={}
    for model in sorted(set(r['model'] for r in rows)):
        selected=[r for r in rows if r['model']==model]
        stats[model]={key:{'mean':statistics.mean(r[key] for r in selected),
                            'sd':statistics.stdev(r[key] for r in selected) if len(selected)>1 else None}
                      for key in ['F1','PR_AUC']}
    payload={'status':'VALIDATION_ONLY','source':spatial.source,'seeds':args.seeds,
             'removed_graph_input_feature':'neighbor count',
             'controlled_inputs_and_splits':True,
             'identical_seed_reset_per_model':True,
             'F1_validation_threshold_optimized_on_same_validation_set':True,
             'notes':'For graph comparison use validation AP as primary; do not treat these as new held-out test scores.',
             'statistics':stats,'rows':rows}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    print('\nWrote:',out/'results.json','and',output_csv)
    for m,results in stats.items():
        print(m, '| F1',round(results['F1']['mean'],4),'| AP',round(results['PR_AUC']['mean'],4))

if __name__=='__main__':main()
