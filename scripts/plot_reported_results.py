"""Draw the revised benchmark from saved evidence only."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parents[1];e=r/'paper/evidence'
h=pd.read_csv(e/'historical_model_summary.csv');g=pd.read_csv(e/'gbm_corrected_metrics.csv');t=pd.read_csv(e/'verified_topology_ranks.csv')
labels={'Persistence previous 48h':'48-hour persistence','GCN-GRU spatial':'Geographic GCN–GRU','GCN-GRU shuffled graph':'Initial shuffled GCN–GRU','GRU identity adjacency':'Identity GRU','Seasonal naive previous-week 48h':'Weekly seasonal naive'}
periods=['Jan-Jul 2026','Jan-Sep 2026'];names=list(labels.values())+['Corrected gradient boosting','Mean of 20 relabelings'];v=[]
for p in periods:
    a=h[h.period==p].set_index('model');v.append([a.loc[k,'mean_ap'] for k in labels]+[g[g.period==p].ap.iloc[0],t[(t.period==p)&(t.metric=='ap')].random_mean.iloc[0]])
fig,ax=plt.subplots(figsize=(10,4.7));y=np.arange(len(names));ax.barh(y-.18,v[0],.36,label='Jan–Jul (post-hoc)',color='#176f78');ax.barh(y+.18,v[1],.36,label='Jan–Sep (category break)',color='#9aafc4');ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,.7);ax.set_xlabel('Average precision (higher is better)');ax.legend(loc='lower right',frameon=False);ax.grid(axis='x',alpha=.2);ax.set_title('Recorded 2026 benchmark — corrected baseline');fig.tight_layout();fig.savefig(r/'assets/benchmark_ap.png',dpi=160);plt.close(fig)
