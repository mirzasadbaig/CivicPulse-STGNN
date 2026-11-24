"""Run real-data experiments or clearly marked synthetic software checks.

Key safeguards: node selection and scaling trained before temporal holdout;
48-hour label embargo; thresholds selected exclusively using validation;
held-out test metrics calculated only once for each pre-registered run.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,precision_score,recall_score,
                             f1_score,confusion_matrix,roc_auc_score)
from .data import load_data,windows
from .model import GCNGRU
from .communities import describe_communities


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))


def device_auto():
    if torch.backends.mps.is_available():return torch.device('mps')
    return torch.device('cpu')


def predict(model,xd,adj,device,batch_size=16,get_emb=False):
    model.eval(); probs=[];embs=[]
    with torch.no_grad():
        for i in range(0,len(xd),batch_size):
            xb=torch.from_numpy(xd[i:i+batch_size]).to(device)
            output=model(xb,adj,return_embeddings=get_emb)
            if get_emb:
                z,h=output;embs.append(h.cpu().numpy())
            else:z=output
            probs.append(torch.sigmoid(z).cpu().numpy())
    p=np.concatenate(probs)
    if get_emb:return p,np.concatenate(embs)
    return p


def val_threshold(y,p):
    # One common threshold for all nodes; no per-node threshold test fitting.
    options=np.linspace(.05,.95,91)
    fs=[f1_score(y.ravel().astype(int),(p.ravel()>=t).astype(int),zero_division=0) for t in options]
    return float(options[np.argmax(fs)])


def score(y,p,threshold):
    y=np.asarray(y).ravel().astype(int);p=np.asarray(p).ravel()
    pred=(p>=threshold).astype(int)
    return {'n':len(y),'positives':int(y.sum()),'positive_rate':float(y.mean()),
            'threshold':float(threshold),'f1':float(f1_score(y,pred,zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'average_precision_pr_auc':float(average_precision_score(y,p)) if y.sum()>0 else None,
            'roc_auc':float(roc_auc_score(y,p)) if y.min()!=y.max() else None,
            'confusion_matrix_0_1':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def run_gru(name,ds,adj_np,lookback,epochs,patience,lr,batch_size,hidden,seed,device,evaluate_test=True):
    print(f'\n=== {name}: device={device} ===',flush=True)
    wt=windows(ds.x,ds.idx_train,lookback)
    wv=windows(ds.x,ds.idx_val,lookback)
    yy=ds.y[ds.idx_train]
    vy=ds.y[ds.idx_val]
    adj=torch.tensor(adj_np,device=device)
    model=GCNGRU(ds.x.shape[-1],hidden=hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    positives=float(yy.sum());negatives=float(yy.size-positives)
    pw=float(np.clip(negatives/max(positives,1),1,8))
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device))
    best=-1.0;best_ep=0;state=None;history=[]
    start=time.time()
    for epoch in range(1,epochs+1):
        model.train();order=np.random.permutation(len(wt));losses=[]
        for ids in np.array_split(order,max(1,int(np.ceil(len(order)/batch_size)))):
            xb=torch.from_numpy(wt[ids]).to(device)
            yb=torch.from_numpy(yy[ids]).to(device)
            opt.zero_grad(set_to_none=True)
            z=model(xb,adj);loss=criterion(z,yb);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.0)
            opt.step();losses.append(float(loss.item()))
        pv=predict(model,wv,adj,device,batch_size)
        ap=float(average_precision_score(vy.ravel(),pv.ravel())) if vy.sum()>0 else 0.0
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_ap':ap})
        print(f'{name} epoch {epoch:02d}: loss={np.mean(losses):.4f}, val_AP={ap:.4f}',flush=True)
        if ap>best+1e-4:
            best=ap;best_ep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch-best_ep>=patience:break
    if state is not None:model.load_state_dict(state)
    pv=predict(model,wv,adj,device,batch_size)
    threshold=val_threshold(vy,pv)
    ret={'validation_best_AP':best,'best_epoch':best_ep,'validation_f1_threshold':threshold,
         'elapsed_sec':round(time.time()-start,2),'history':history,
         'validation':score(vy,pv,threshold)}
    pt=None;emb=None
    if evaluate_test:
        xt=windows(ds.x,ds.idx_test,lookback)
        pt,emb=predict(model,xt,adj,device,batch_size,get_emb=True)
        ret['test']=score(ds.y[ds.idx_test],pt,threshold)
    return ret,model,pt,emb


def baseline(ds,evaluate_test=True):
    def tabular(idx):
        xx=ds.x[idx];avg=np.stack([ds.x[max(0,i-6):i+1].mean(axis=0) for i in idx])
        return np.concatenate([xx,avg],axis=-1).reshape(-1,2*ds.x.shape[-1])
    tr=tabular(ds.idx_train);val=tabular(ds.idx_val)
    ytr=ds.y[ds.idx_train].ravel().astype(int)
    if len(np.unique(ytr))<2:raise ValueError('Only one training class; adjust sampling/label')
    model=LogisticRegression(class_weight='balanced',max_iter=1500,random_state=0)
    model.fit(tr,ytr)
    pv=model.predict_proba(val)[:,1].reshape(ds.y[ds.idx_val].shape)
    threshold=val_threshold(ds.y[ds.idx_val],pv)
    ret={'validation_f1_threshold':threshold,'validation':score(ds.y[ds.idx_val],pv,threshold)}
    pt=None
    if evaluate_test:
        test=tabular(ds.idx_test)
        pt=model.predict_proba(test)[:,1].reshape(ds.y[ds.idx_test].shape)
        ret['test']=score(ds.y[ds.idx_test],pt,threshold)
    return ret,pt


def save_plot(out,results):
    names=list(results)
    f1=[results[k]['test']['f1'] for k in names]
    ap=[results[k]['test']['average_precision_pr_auc'] for k in names]
    fig,ax=plt.subplots(figsize=(9,4));pos=np.arange(len(names))
    ax.bar(pos-.18,f1,.34,label='Held-out F1')
    ax.bar(pos+.18,ap,.34,label='Held-out AP (PR-AUC)')
    ax.set_xticks(pos,names,rotation=15,ha='right');ax.set_ylim(0,1)
    ax.set_ylabel('Score');ax.set_title('Held-out test — report source and target definition')
    ax.legend();fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)


def main():
    a=argparse.ArgumentParser(description='CivicPulse: chronologically evaluated STGNN')
    a.add_argument('--csv',required=True)
    a.add_argument('--output',default='results/real_2024_2025')
    a.add_argument('--epochs',type=int,default=35)
    a.add_argument('--patience',type=int,default=8)
    a.add_argument('--lr',type=float,default=.001)
    a.add_argument('--hidden',type=int,default=32)
    a.add_argument('--batch-size',type=int,default=16)
    a.add_argument('--lookback',type=int,default=28)
    a.add_argument('--max-zips',type=int,default=20)
    a.add_argument('--max-nodes',type=int,default=80)
    a.add_argument('--quantile',type=float,default=.75)
    a.add_argument('--graph',choices=['spatial','hybrid'],default='spatial')
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--cpu',action='store_true')
    a.add_argument('--skip-ablation',action='store_true')
    a.add_argument('--validation-only',action='store_true',help='Use for hyperparameter selection without opening held-out test labels')
    args=a.parse_args()
    seed_all(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ds=load_data(args.csv,lookback=args.lookback,max_zips=args.max_zips,
                 max_nodes=args.max_nodes,quantile=args.quantile,graph=args.graph)
    print(f'SOURCE: {ds.source}; time range: {ds.dates[0].date()} .. {ds.dates[-1].date()}')
    print(f'Nodes={len(ds.nodes)}, days={len(ds.dates)}, train/val/test anchors='
          f'{len(ds.idx_train)}/{len(ds.idx_val)}/{len(ds.idx_test)}')
    print('Train / val / test event prevalence:',
          *[round(float(ds.y[idx].mean()),4) for idx in (ds.idx_train,ds.idx_val,ds.idx_test)])
    if ds.source!='NYC_311_PUBLIC_AGGREGATED':
        print('WARNING: THIS RUN IS SYNTHETIC SOFTWARE TEST, NOT SCIENTIFIC PERFORMANCE EVIDENCE.')
    for label,idx in [('train',ds.idx_train),('validation',ds.idx_val),('test',ds.idx_test)]:
        pos=int(ds.y[idx].sum()); total=int(ds.y[idx].size)
        if pos<5 or total-pos<5:
            raise ValueError(f'{label} contains {pos} positive and {total-pos} negative cases; '
                             'classification metrics are unstable. Download more months or revise '
                             'the target definition transparently before the final experiment.')
    device=torch.device('cpu') if args.cpu else device_auto()
    results={}
    results['Logistic regression'],bp=baseline(ds,evaluate_test=not args.validation_only)
    metric,model,proba,emb=run_gru('GCN-GRU',ds,ds.adj,args.lookback,args.epochs,
                                   args.patience,args.lr,args.batch_size,args.hidden,args.seed,device,
                                   evaluate_test=not args.validation_only)
    results['GCN-GRU']=metric
    if not args.skip_ablation:
        nodj=np.eye(len(ds.nodes),dtype=np.float32)
        ab,_,_,_=run_gru('GRU (no graph)',ds,nodj,args.lookback,args.epochs,
                         args.patience,args.lr,args.batch_size,args.hidden,args.seed,device,
                         evaluate_test=not args.validation_only)
        results['GRU (no graph)']=ab
    if args.validation_only:
        validation_run={'status':'VALIDATION_ONLY_NO_TEST_EXAMINED',
                        'source':ds.source,'graph':args.graph,'seed':args.seed,
                        'lookback_days':args.lookback,'hidden':args.hidden,
                        'lr':args.lr,'epochs_max':args.epochs,
                        'validation_results':results}
        (out/'validation_metrics.json').write_text(json.dumps(validation_run,indent=2))
        print('\nVALIDATION ONLY: held-out test not evaluated. Select hyperparameters using AP:')
        for name,r in results.items():
            print(f'{name:24} val_AP={r["validation"]["average_precision_pr_auc"]:.3f}, '
                  f'val_F1={r["validation"]["f1"]:.3f}')
        print(f'Wrote {out / "validation_metrics.json"}')
        return
    days=ds.dates[ds.idx_test]
    communities=describe_communities(ds.raw_adj,emb,days)
    json_data={'status':'MEASURED' if ds.source=='NYC_311_PUBLIC_AGGREGATED' else 'SYNTHETIC_SMOKE_TEST',
               'source':ds.source,'csv':args.csv,
               'target':'future 2-day complaint count strictly greater than per-node 75th percentile of TRAIN future 2-day counts',
               'target_quantile':args.quantile,
               'horizon_days':2,'lookback_days':args.lookback,
               'graph':args.graph,'seed':args.seed,'device':str(device),
               'dates':{'start':str(ds.dates[0].date()),'end':str(ds.dates[-1].date()),
                        'train_end':str(ds.dates[ds.idx_train[-1]].date()),
                        'validation_start':str(ds.dates[ds.idx_val[0]].date()),
                        'test_start':str(ds.dates[ds.idx_test[0]].date())},
               'nodes':[{'zip':z,'complaint_type':c,'high_demand_threshold':float(ds.threshold_counts[i])}
                        for i,(z,c) in enumerate(ds.nodes)],
               'results':results,
               'communities':{'snapshots':len(communities),
                              'mean_modularity':float(np.mean([r['modularity'] for r in communities])) if communities else None,
                              'mean_stability_ari':float(np.mean([r['stability_ari'] for r in communities if r['stability_ari'] is not None])) if len(communities)>1 else None}}
    (out/'metrics.json').write_text(json.dumps(json_data,indent=2))
    (out/'communities.json').write_text(json.dumps(communities,indent=2))
    # Community-level early-warning summaries for the last held-out snapshots.
    # These are descriptive post-processing, not extra supervised crisis labels.
    risk_rows=[]
    date_to_index={str(d.date()):i for i,d in enumerate(days)}
    for snap in communities:
        tid=date_to_index.get(snap['date'])
        if tid is None:continue
        for community_id,members in enumerate(snap['members']):
            avg_risk=float(proba[tid,members].mean())
            risk_rows.append({'date':snap['date'],'snapshot_community':community_id,
                              'n_nodes':len(members),'mean_high_demand_risk':avg_risk,
                              'member_nodes':'; '.join(f'{ds.nodes[j][0]}:{ds.nodes[j][1]}' for j in members)})
    pd.DataFrame(risk_rows).to_csv(out/'community_risk.csv',index=False)
    rows=[]
    for ti,day in enumerate(days):
        for ni,(z,c) in enumerate(ds.nodes):
            rows.append((str(day.date()),z,c,int(ds.y[ds.idx_test[ti],ni]),
                         float(ds.future_counts[ds.idx_test[ti],ni]),
                         float(ds.threshold_counts[ni]),float(proba[ti,ni]),float(bp[ti,ni])))
    pd.DataFrame(rows,columns=['anchor_date','zip','category','target_elevated_demand',
                               'future_48h_count','train_only_demand_threshold',
                               'gnn_probability','logistic_probability']).to_csv(out/'test_predictions.csv',index=False)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'input_features':ds.x.shape[-1],'nodes':ds.nodes,'source':ds.source},out/'model.pt')
    save_plot(out,results)
    print('\nFINAL TEST (not a tuning dataset):')
    for name,m in results.items():
        s=m['test'];print(f'{name:24} F1={s["f1"]:.3f}, AP={s["average_precision_pr_auc"]:.3f}, '
                        f'Precision={s["precision"]:.3f}, Recall={s["recall"]:.3f}')
    print(f'Wrote outputs to {out.resolve()}')

if __name__=='__main__':main()
