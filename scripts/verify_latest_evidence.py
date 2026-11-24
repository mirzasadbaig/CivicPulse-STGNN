#!/usr/bin/env python3
"""Check archived metrics and dashboard coverage; no model fitting."""
from pathlib import Path
import io,json,re,subprocess,sys,tempfile,zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'paper'; A=P/'evidence/original_archives'
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable,str(P/'analyze_saved_predictions.py'),str(A/'CivicPulse_Permutation_Results.zip'),str(A/'CivicPulse_Reviewer_Experiment_Results.zip'),tmp],check=True)
    for n in ('verified_topology_means.csv','verified_topology_ranks.csv','monthly_threshold_diagnostics_mean.csv','monthly_threshold_diagnostics_per_seed.csv','same_month_counts.csv','jan_jul_yoy_totals.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(Path(tmp)/n),pd.read_csv(P/'evidence'/n),check_exact=False,rtol=1e-9,atol=1e-10)
    print('OK: 105 saved training runs, 20 unique relabelings, 210 metric rows and diagnostics checked.')
with zipfile.ZipFile(A/'CivicPulse_GBM_Corrected_Results.zip') as z:
    base='results/benchmark_extension/'
    with np.load(io.BytesIO(z.read(base+'gbm_predictions.npz')),allow_pickle=False) as p:
        y=p['y']; pred=p['pred']; dates=pd.DatetimeIndex(p['dates'])
    df=pd.read_csv(io.BytesIO(z.read(base+'gbm_corrected_metrics.csv')))
    pd.testing.assert_frame_equal(df,pd.read_csv(P/'evidence/gbm_corrected_metrics.csv'))
    for row in df[df.period.str.contains('2026')].itertuples():
        ix=dates<=pd.Timestamp('2026-07-29') if 'Jul' in row.period else np.ones(len(dates),dtype=bool)
        yt=y[ix].ravel(); pr=pred[ix].ravel(); yp=pr>=row.threshold
        assert len(yt)==row.n
        for k,v in [('ap',average_precision_score(yt,pr)),('f1',f1_score(yt,yp)),('precision',precision_score(yt,yp)),('recall',recall_score(yt,yp))]:
            assert abs(v-getattr(row,k))<1e-10,(k,row.period)
    print('OK: corrected GBM 2026 metrics match saved scores.')
s=(ROOT/'dashboard/index.html').read_text()
data,_=json.JSONDecoder().raw_decode(s.split('window.PULSE=',1)[1])
assert [len(data[k]) for k in ('nodes','daily','edges','forecast','snapshots')]==[80,731,276,108,21]
print('OK: dashboard coverage matches original exports. No browser interaction tested by this check.')
