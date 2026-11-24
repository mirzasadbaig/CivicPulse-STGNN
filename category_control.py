#!/usr/bin/env python3
"""Exploratory joint ZIP relabeling and validation-only category thresholds.
Place beside reviewer_experiments.py; use a NEW output folder. No 2026 tuning.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from reviewer_experiments import (load_data,no_graph_feature,read_future_counts,build_external_features,read_2026_target,windows,run_gru,predict,seed_all,device_auto,metric_scores,normalize_raw,sha256_file)
from src.experiment import val_threshold
from sklearn.metrics import f1_score,precision_score,recall_score

def joint_zip_permutation(nodes,seed):
    zips=sorted({z for z,c in nodes});cats=sorted({c for z,c in nodes});lookup={node:i for i,node in enumerate(nodes)}
    assert len(lookup)==len(zips)*len(cats)==len(nodes)
    rng=np.random.default_rng(seed);mapping=dict(zip(zips,rng.permutation(zips)));perm=np.array([lookup[(mapping[z],c)] for z,c in nodes]);assert sorted(perm.tolist())==list(range(len(nodes)))
    for i,(z,c) in enumerate(nodes):assert nodes[perm[i]][1]==c
    # One shared ZIP mapping preserves co-location relationships across all categories.
    return perm

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--past',required=True);p.add_argument('--future',required=True);p.add_argument('--output',default='results/category_control');p.add_argument('--permutations',type=int,default=20);p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27,37,42]);p.add_argument('--threads',type=int,default=4);p.add_argument('--cpu',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args();torch.set_num_threads(a.threads)
    assert sha256_file(a.past)=='5f98e176da8168a4b5dbc409c958b5c58322d835d43ad8f96cd289110d012fec';assert sha256_file(a.future)=='e442a3de8c0dd7652418816a28b0c118220b8921be910521cba163e8c25f8afe'
    ds=no_graph_feature(load_data(a.past,graph='spatial',lookback=28,max_zips=20,max_nodes=80,quantile=.75));rng=np.random.default_rng(39521);seeds=rng.choice(np.arange(100000,1000000),size=a.permutations,replace=False);configs=[('spatial',ds.adj,np.arange(len(ds.nodes)))];seen=set()
    for s in seeds:
        perm=joint_zip_permutation(ds.nodes,int(s));key=tuple(perm);assert key not in seen and not np.array_equal(perm,np.arange(len(perm)));seen.add(key);raw=ds.raw_adj[np.ix_(perm,perm)];np.testing.assert_array_equal(np.sort(raw.ravel()),np.sort(ds.raw_adj.ravel()));configs.append((f'zipperm_{s}',normalize_raw(raw),perm))
    print(f'Checked {len(configs)-1} unique joint ZIP permutations; all preserve category labels and ZIP blocks.',flush=True)
    if a.check_only:return
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);dev=torch.device('cpu') if a.cpu else device_auto();manifest=dict(args=vars(a),past_sha256=sha256_file(a.past),future_sha256=sha256_file(a.future),device=str(dev),torch=str(torch.__version__),status='exploratory; validation-only selection',topologies={name:perm.tolist() for name,adj,perm in configs});mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Existing manifest differs; use new output folder')
    mf.write_text(json.dumps(manifest,indent=2));d,c,*_=read_future_counts(a.future,ds,False);dates,counts,x,err=build_external_features(ds,d,c);anchors,_,y=read_2026_target(ds,counts,dates);assert len(anchors)==271;wv=windows(ds.x,ds.idx_val,28);wf=windows(x,anchors,28);yv=ds.y[ds.idx_val];cats=np.array([c for z,c in ds.nodes]);np.savez_compressed(out/'labels.npz',y=y,yv=yv,dates=pd.DatetimeIndex(dates[anchors]).strftime('%Y-%m-%d').to_numpy(dtype=str),categories=cats)
    rows=[]
    for name,adj,perm in configs:
        for seed in a.seeds:
            f=out/f'{name}_{seed}.json';nf=f.with_suffix('.npz')
            if not (f.exists() and nf.exists()):
                seed_all(seed);vm,m,_,_=run_gru(name,ds,adj,28,35,8,.001,16,32,seed,dev,evaluate_test=False);ta=torch.as_tensor(adj,device=dev);vp=predict(m,wv,ta,dev,16);pred=predict(m,wf,ta,dev,16);t=float(vm['validation_f1_threshold']);thresholds={cat:float(val_threshold(yv[:,cats==cat],vp[:,cats==cat])) for cat in sorted(set(cats))};tv=np.array([thresholds[c] for c in cats]);rr=[]
                for period,yy,pp in [('2025 validation',yv,vp),('Jan-Jul 2026',y[:210],pred[:210]),('Jan-Sep 2026',y,pred)]:
                    base=metric_scores(yy,pp,t);dec=(pp>=tv).ravel();truth=yy.ravel();rr.append(dict(topology=name,seed=seed,period=period,global_f1=base['f1'],ap=base['ap'],category_f1=float(f1_score(truth,dec,zero_division=0)),category_precision=float(precision_score(truth,dec,zero_division=0)),category_recall=float(recall_score(truth,dec,zero_division=0))))
                np.savez_compressed(nf,pred=pred,validation_pred=vp);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(dict(global_threshold=t,category_thresholds=thresholds,rows=rr)));tmp.replace(f)
            rows.extend(json.loads(f.read_text())['rows']);print(f'Completed {name} seed {seed}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'runs.csv',index=False);means=df.groupby(['period','topology'])[['global_f1','ap','category_f1','category_precision','category_recall']].mean().reset_index();means.to_csv(out/'topology_means.csv',index=False);print(means.to_string(index=False))
if __name__=='__main__':main()
