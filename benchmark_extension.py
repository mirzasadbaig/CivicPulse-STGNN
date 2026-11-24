#!/usr/bin/env python3
"""Exploratory extension after inspection of 2026; preserves historical outputs.
Run beside reviewer_experiments.py. All fitting/selection uses pre-2026 data.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from reviewer_experiments import (load_data, no_graph_feature, read_future_counts,
    build_external_features, read_2026_target, windows, shuffled_graph, run_gru,
    predict, seed_all, device_auto, metric_scores, naive_score, fit_score_threshold, sha256_file)
from src.experiment import val_threshold

def flatten(w):
    b,l,n,f=w.shape
    return w.transpose(0,2,1,3).reshape(b*n,l*f)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--past',required=True);p.add_argument('--future',required=True)
    p.add_argument('--output',default='results/benchmark_extension');p.add_argument('--mode',choices=['gbm','permutations'],required=True)
    p.add_argument('--cpu',action='store_true');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--permutations',type=int,default=20);p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27,37,42]);p.add_argument('--epochs',type=int,default=35)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    # Explicit layout regression test (distinct anchor, node, day and feature values).
    w=np.arange(2*3*4*2).reshape(2,3,4,2);flat=flatten(w)
    for b in range(2):
        for n in range(4):np.testing.assert_array_equal(flat[b*4+n],w[b,:,n,:].ravel())
    ds=no_graph_feature(load_data(a.past,graph='spatial',lookback=28,max_zips=20,max_nodes=80,quantile=.75))
    assert ds.source=='NYC_311_PUBLIC_AGGREGATED'
    d,c,*_=read_future_counts(a.future,ds,False)
    dates,counts,x,err=build_external_features(ds,d,c)
    anchors,_,y=read_2026_target(ds,counts,dates); dates26=pd.DatetimeIndex(dates[anchors]);wf=windows(x,anchors,28)
    periods={'Jan-Jul 2026':np.flatnonzero(dates26<=pd.Timestamp('2026-07-29')),'Jan-Sep 2026':np.arange(len(anchors))}
    assert [len(i) for i in periods.values()]==[210,271]
    manifest={'status':'EXPLORATORY_POST_HOC_2026','past_sha256':sha256_file(a.past),'future_sha256':sha256_file(a.future),'args':vars(a),'selection':'2025 validation only','torch':torch.__version__,'python':sys.version,'scaler_error':float(err)}
    mf=out/f'{a.mode}_manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Output directory contains a different run configuration; choose another --output')
    mf.write_text(json.dumps(manifest,indent=2))
    if a.mode=='gbm':
        tr=flatten(windows(ds.x,ds.idx_train,28));va=flatten(windows(ds.x,ds.idx_val,28));te=flatten(wf)
        yt=ds.y[ds.idx_train].ravel().astype(int);yv=ds.y[ds.idx_val].ravel().astype(int)
        assert len(tr)==len(yt) and len(va)==len(yv) and len(te)==y.size
        prev=yt.mean();balanced=np.where(yt==1,.5/prev,.5/(1-prev));best=None;grid=[]
        # Validation-only grid, includes exact original parameters with corrected layout.
        with threadpool_limits(limits=a.threads):
            for leaves in [15,31]:
                for weighted in [True,False]:
                    for iters in [100,200]:
                        params=dict(learning_rate=.05,max_iter=iters,max_leaf_nodes=leaves,l2_regularization=1.,early_stopping=False,random_state=42)
                        m=HistGradientBoostingClassifier(**params).fit(tr,yt,sample_weight=balanced if weighted else None)
                        vp=m.predict_proba(va)[:,1];ap=average_precision_score(yv,vp)
                        grid.append(dict(leaves=leaves,weighted=weighted,iters=iters,validation_ap=ap));print(grid[-1],flush=True)
                        if best is None or ap>best[0]:best=(ap,m,vp,grid[-1])
            ap,m,vp,cfg=best;t=val_threshold(yv,vp);pred=m.predict_proba(te)[:,1].reshape(y.shape)
        pd.DataFrame(grid).to_csv(out/'gbm_validation_grid.csv',index=False)
        rows=[dict(period='2025 validation',**metric_scores(yv,vp,t))]+[dict(period=k,**metric_scores(y[ii],pred[ii],t)) for k,ii in periods.items()]
        pd.DataFrame(rows).to_csv(out/'gbm_corrected_metrics.csv',index=False);np.savez_compressed(out/'gbm_predictions.npz',y=y,pred=pred,dates=dates26.astype(str).to_numpy(dtype=str))
        (out/'gbm_selected_config.json').write_text(json.dumps(cfg,indent=2));print(pd.DataFrame(rows).to_string(index=False))
    else:
        dev=torch.device('cpu') if a.cpu else device_auto();rng=np.random.default_rng(89123)
        perm_seeds=rng.choice(np.arange(100000,1000000),size=a.permutations,replace=False).tolist()
        configs=[('spatial',ds.adj,None)]+[(f'perm_{s}',*shuffled_graph(ds.raw_adj,s)) for s in perm_seeds]
        np.savez_compressed(out/'permutation_labels.npz',y=y,dates=dates26.astype(str).to_numpy(dtype=str))
        for name,adj,perm in configs:
            for seed in a.seeds:
                path=out/f'{name}_{seed}.json'
                if path.exists():continue
                seed_all(seed);vm,m,_,_=run_gru(name,ds,adj,28,a.epochs,8,.001,16,32,seed,dev,evaluate_test=False)
                t=float(vm['validation_f1_threshold']);pr=predict(m,wf,torch.as_tensor(adj,device=dev),dev,16)
                rows=[dict(period=k,**metric_scores(y[ii],pr[ii],t)) for k,ii in periods.items()]
                np.savez_compressed(out/f'{name}_{seed}.npz',pred=pr)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(topology=name,seed=seed,permutation=None if perm is None else perm.tolist(),validation_ap=vm['validation_best_AP'],rows=rows)));tmp.replace(path)
        rows=[]
        for name,_,_ in configs:
            for seed in a.seeds:
                r=json.loads((out/f'{name}_{seed}.json').read_text());rows.extend(dict(topology=name,seed=seed,**row) for row in r['rows'])
        df=pd.DataFrame(rows);df.to_csv(out/'permutation_runs.csv',index=False)
        summary=df.groupby(['period','topology'])[['ap','f1']].mean().reset_index();summary.to_csv(out/'permutation_topology_means.csv',index=False)
        ranks=[]
        for period,g in summary.groupby('period'):
            s=g[g.topology=='spatial'].iloc[0];null=g[g.topology!='spatial']
            for metric in ['ap','f1']:
                ranks.append(dict(period=period,metric=metric,spatial=s[metric],null_mean=null[metric].mean(),null_sd=null[metric].std(),null_min=null[metric].min(),null_max=null[metric].max(),n_topologies=len(null),fraction_null_at_least_spatial=float((null[metric]>=s[metric]).mean())))
        pd.DataFrame(ranks).to_csv(out/'permutation_rank_summary.csv',index=False);print(pd.DataFrame(ranks).to_string(index=False))
if __name__=='__main__':main()
