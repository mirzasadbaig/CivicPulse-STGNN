#!/usr/bin/env python3
"""Recompute validation-selected thresholds and all 15 spatial-refit metric rows. No fitting."""
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'paper/evidence'; A=E/'original_archives'; OUT=E/'final_review'

def select(y,p):
    grid=np.linspace(.05,.95,91)
    return float(grid[np.argmax([f1_score(y.ravel(),p.ravel()>=t,zero_division=0) for t in grid])])

def run():
    with zipfile.ZipFile(A/'CivicPulse_Category_Threshold_Results.zip') as z, zipfile.ZipFile(A/'CivicPulse_Permutation_Results.zip') as prior:
        b='results/category_thresholds/'; old='results/benchmark_extension/'
        def arr(n):
            with np.load(io.BytesIO(z.read(b+n)),allow_pickle=False) as d: return dict(d)
        m=json.loads(z.read(b+'manifest.json')); args=m['args']; seeds=[7,17,27,37,42]
        assert args['seeds']==seeds and args['permutations']==0 and m['topologies']=={'spatial':list(range(80))}
        pm=json.loads(prior.read(old+'permutations_manifest.json'))
        for k in ['past_sha256','future_sha256']: assert m[k]==pm[k]
        lab=arr('labels.npz'); y,yv,dates,cats=[lab[k] for k in ['y','yv','dates','categories']]
        assert y.shape==(271,80) and yv.shape==(108,80) and set(np.unique(y))=={0,1} and set(np.unique(yv))=={0,1}
        assert np.array_equal(dates,pd.date_range('2026-01-01','2026-09-28').strftime('%Y-%m-%d').to_numpy())
        assert dict(zip(*np.unique(cats,return_counts=True)))=={'HEAT/HOT WATER':20,'Sewer':20,'Street Condition':20,'Water System':20}
        with np.load(io.BytesIO(prior.read(old+'gbm_predictions.npz')),allow_pickle=False) as d:
            np.testing.assert_array_equal(y,d['y']); np.testing.assert_array_equal(dates,d['dates'])
        rows=[]; cuts=[]; diffs=[]
        for seed in seeds:
            meta=json.loads(z.read(b+f'spatial_{seed}.json')); a=arr(f'spatial_{seed}.npz'); p,v=a['pred'],a['validation_pred']; assert p.shape==y.shape and v.shape==yv.shape
            for score in (p,v): assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
            t=select(yv,v); ts={c:select(yv[:,cats==c],v[:,cats==c]) for c in sorted(set(cats))}
            assert abs(t-meta['global_threshold'])<1e-12
            for c in ts: assert abs(ts[c]-meta['category_thresholds'][c])<1e-12
            cuts.append(dict(seed=seed,global_threshold=t,**ts)); tv=np.array([ts[c] for c in cats]); rr=[]
            with np.load(io.BytesIO(prior.read(old+f'spatial_{seed}.npz')),allow_pickle=False) as d: diffs.append(float(np.max(np.abs(p-d['pred']))))
            for period,yy,pp in [('2025 validation',yv,v),('Jan-Jul 2026',y[:210],p[:210]),('Jan-Sep 2026',y,p)]:
                truth=yy.ravel(); g=(pp>=t).ravel(); dec=(pp>=tv).ravel(); r=dict(topology='spatial',seed=seed,period=period,global_f1=f1_score(truth,g),ap=average_precision_score(truth,pp.ravel()),category_f1=f1_score(truth,dec),category_precision=precision_score(truth,dec,zero_division=0),category_recall=recall_score(truth,dec,zero_division=0));rr.append(r);rows.append(r)
            pd.testing.assert_frame_equal(pd.DataFrame(rr),pd.DataFrame(meta['rows']),check_exact=False,atol=1e-10,rtol=1e-10)
        df=pd.DataFrame(rows); pd.testing.assert_frame_equal(df,pd.read_csv(io.BytesIO(z.read(b+'runs.csv'))),check_exact=False,atol=1e-10,rtol=1e-10)
        cols=['global_f1','ap','category_f1','category_precision','category_recall']; means=df.groupby(['period','topology'])[cols].mean().reset_index(); pd.testing.assert_frame_equal(means,pd.read_csv(io.BytesIO(z.read(b+'topology_means.csv'))),check_exact=False,atol=1e-10,rtol=1e-10)
        summary=df.groupby(['period','topology'])[cols].agg(['mean','std']); summary.columns=['_'.join(c) for c in summary.columns]; summary=summary.reset_index(); df['delta_f1']=df.category_f1-df.global_f1
        delta=df.groupby('period').delta_f1.agg(['mean','std']).reset_index(); summary=summary.merge(delta.rename(columns={'mean':'delta_f1_mean','std':'delta_f1_std'}),on='period'); print(summary.to_string(index=False)); print('Maximum absolute 2026 score difference from prior spatial run:',max(diffs)); print('OK: all 15 metric rows and 25 validation-selected thresholds match.')
        return {'category_threshold_verified_runs.csv':df,'category_threshold_verified_summary.csv':summary,'category_threshold_selected_cutoffs.csv':pd.DataFrame(cuts)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    for n,df in run().items():
        if a.write: df.to_csv(OUT/n,index=False)
        else: pd.testing.assert_frame_equal(df,pd.read_csv(OUT/n),check_exact=False,rtol=1e-9,atol=1e-10)
