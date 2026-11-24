#!/usr/bin/env python3
"""Run from repository root; archived 2026 scores are not recalibrated."""
import argparse, io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from reviewer_experiments import (load_data,no_graph_feature,naive_score,fit_score_threshold,metric_scores,sha256_file,read_future_counts,build_external_features,read_2026_target)

def main():
    p=argparse.ArgumentParser();p.add_argument('--past',required=True);p.add_argument('--future',required=True);p.add_argument('--archive',default='paper/evidence/original_archives/CivicPulse_Permutation_Results.zip');p.add_argument('--output',default='results/final_review');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert sha256_file(a.past)=='5f98e176da8168a4b5dbc409c958b5c58322d835d43ad8f96cd289110d012fec'
    assert sha256_file(a.future)=='e442a3de8c0dd7652418816a28b0c118220b8921be910521cba163e8c25f8afe'
    ds=no_graph_feature(load_data(a.past,graph='spatial',lookback=28,max_zips=20,max_nodes=80,quantile=.75));yv=ds.y[ds.idx_val]
    rows=[]
    for name,kind in [('48-hour persistence','recent48h'),('Weekly seasonal naive','previous_week_same_days')]:
        v=naive_score(ds.counts,ds.idx_val,ds.threshold_counts,kind);t=fit_score_threshold(yv,v);rows.append(dict(model=name,**metric_scores(yv,v,t,inclusive=False)))
    pd.DataFrame(rows).to_csv(out/'naive_validation.csv',index=False)
    d,c,*_=read_future_counts(a.future,ds,False);dates,counts,x,err=build_external_features(ds,d,c);anchors,_,y=read_2026_target(ds,counts,dates);dates26=pd.DatetimeIndex(dates[anchors]);assert err<1e-5
    with zipfile.ZipFile(a.archive) as z:
        def get(name):return np.load(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('/'+name)))),allow_pickle=False)
        lab=get('permutation_labels.npz');np.testing.assert_array_equal(y,lab['y']);np.testing.assert_array_equal(dates26.strftime('%Y-%m-%d').to_numpy(dtype=str),lab['dates'])
        gbm=get('gbm_predictions.npz');np.testing.assert_array_equal(y,gbm['y']);np.testing.assert_array_equal(lab['dates'],gbm['dates'])
        probs=[('Spatial GCN-GRU',s,get(f'spatial_{s}.npz')['pred']) for s in [7,17,27,37,42]]+[('Corrected GBM',42,gbm['pred'])]
        rows=[]; bins=[]
        for period,ii in [('Jan-Jul 2026',np.arange(210)),('Jan-Sep 2026',np.arange(271))]:
            yy=y[ii].ravel(); base=float(yv.mean());rows.append(dict(period=period,model='Validation prevalence constant',seed=0,brier=np.mean((yy-base)**2),mean_score=base,prevalence=yy.mean()))
            for model,s,pp in probs:
                v=pp[ii].ravel();assert np.isfinite(v).all() and ((v>=0)&(v<=1)).all();rows.append(dict(period=period,model=model,seed=s,brier=brier_score_loss(yy,v),mean_score=v.mean(),prevalence=yy.mean()))
                for b in range(10):
                    ix=np.minimum((v*10).astype(int),9)==b
                    if ix.any():bins.append(dict(period=period,model=model,seed=s,bin=b,n=ix.sum(),mean_score=v[ix].mean(),event_rate=yy[ix].mean()))
        df=pd.DataFrame(rows);df.to_csv(out/'brier_per_seed.csv',index=False);df.groupby(['period','model']).agg(brier_mean=('brier','mean'),brier_sd=('brier','std'),mean_score=('mean_score','mean'),prevalence=('prevalence','mean')).reset_index().to_csv(out/'brier_summary.csv',index=False);pd.DataFrame(bins).to_csv(out/'reliability_bins.csv',index=False)
    print(pd.read_csv(out/'naive_validation.csv').to_string(index=False));print(pd.read_csv(out/'brier_summary.csv').to_string(index=False))
if __name__=='__main__':main()
