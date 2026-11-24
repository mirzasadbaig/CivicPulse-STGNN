"""Post-hoc diagnostics from saved predictions; no training or threshold changes.
Usage: python analyze_saved_predictions.py PERMUTATION_ZIP REVIEWER_ZIP OUTPUT_DIR
"""
import sys, io, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score, precision_recall_curve

p='results/benchmark_extension/'
with zipfile.ZipFile(sys.argv[1]) as z, zipfile.ZipFile(sys.argv[2]) as r:
    out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
    def read(n): return pd.read_csv(io.BytesIO(z.read(p+n)))
    def npz(n): return np.load(io.BytesIO(z.read(p+n)),allow_pickle=False)
    runs=read('permutation_runs.csv'); a=npz('permutation_labels.npz'); y=a['y']; dates=pd.DatetimeIndex(a['dates']); assert np.array_equal(y,npz('gbm_predictions.npz')['y'])
    assert runs.groupby(['topology','seed']).size().eq(2).all() and len(runs)==210
    err=0; perms={}; rows=[]
    for (top,seed),g in runs.groupby(['topology','seed']):
        j=json.loads(z.read(p+f'{top}_{seed}.json')); pred=npz(f'{top}_{seed}.npz')['pred']; assert pred.shape==y.shape and np.isfinite(pred).all()
        if top!='spatial':
            pp=tuple(j['permutation']); assert sorted(pp)==list(range(y.shape[1])); assert perms.setdefault(top,pp)==pp
        for row in g.itertuples():
            n=210 if row.period.startswith('Jan-Jul') else 271; yt=y[:n].ravel(); pr=pred[:n].ravel(); yp=pr>=row.threshold
            err=max(err,abs(average_precision_score(yt,pr)-row.ap),abs(f1_score(yt,yp)-row.f1),abs(precision_score(yt,yp)-row.precision),abs(recall_score(yt,yp)-row.recall))
        if top=='spatial':
            for month in range(1,8):
                ix=(dates.month==month)&(dates<=pd.Timestamp('2026-07-29')); yt=y[ix].ravel();pr=pred[ix].ravel();th=g.threshold.iloc[0];yp=pr>=th
                precision,recall,thresholds=precision_recall_curve(yt,pr); f=2*precision*recall/np.maximum(precision+recall,1e-12)
                rows.append(dict(month=month,seed=seed,prevalence=yt.mean(),ap=average_precision_score(yt,pr),frozen_f1=f1_score(yt,yp),frozen_recall=recall_score(yt,yp),alert_rate=yp.mean(),frozen_threshold=th,oracle_f1=f[:-1].max(),oracle_threshold=thresholds[f[:-1].argmax()]))
    assert len(set(perms.values()))==20 and err<1e-10
    m=pd.DataFrame(rows);m.to_csv(out/'monthly_threshold_diagnostics_per_seed.csv',index=False);m.drop(columns='seed').groupby('month').mean().to_csv(out/'monthly_threshold_diagnostics_mean.csv')
    yoy=pd.read_csv(io.BytesIO(r.read('selected_nodes_year_over_year_counts.csv'))); yoy=yoy[yoy.month<=7]; yoy.to_csv(out/'same_month_counts.csv',index=False)
    totals=yoy.groupby(['category','year']).complaints_in_selected_nodes.sum().unstack();totals['change_pct']=(totals[2026]/totals[2025]-1)*100;totals.to_csv(out/'jan_jul_yoy_totals.csv')
    means=runs.groupby(['period','topology'])[['ap','f1']].mean().reset_index(); means.to_csv(out/'verified_topology_means.csv',index=False)
    ranks=[]
    for period,g in means.groupby('period'):
        sp=g[g.topology=='spatial'].iloc[0];null=g[g.topology!='spatial']
        for metric in ['ap','f1']:
            k=int((null[metric]>=sp[metric]).sum());n=len(null)
            ranks.append(dict(period=period,metric=metric,spatial=sp[metric],random_mean=null[metric].mean(),random_sd=null[metric].std(),random_max=null[metric].max(),n_random=n,n_random_ge_spatial=k,rank=1+int((null[metric]>sp[metric]).sum()),plus_one_tail_fraction=(k+1)/(n+1)))
    pd.DataFrame(ranks).to_csv(out/'verified_topology_ranks.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(training_runs=105,unique_random_permutations=20,metric_rows=210,max_abs_metric_discrepancy=err,labels_match_gbm=True,scope='Saved-output checks, not independent retraining; all diagnostics post hoc.'),indent=2))
