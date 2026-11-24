from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent; E=P/'evidence'; F=P/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
blue='#23577b'; teal='#187f7b'; orange='#b36a24'; gray='#697985'
def save(fig,name): fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.05); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.0,1.05)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
for i,(a,b) in enumerate([('NYC 311','Daily counts'),('80 nodes','20 ZIPs × 4 types'),('28-day history','10 features'),('Graph + GRU','Matched controls'),('48-hour target','AP and F1')]):
 x=.008+i*.2; ax.add_patch(FancyBboxPatch((x,.19),.176,.63,boxstyle='round,pad=.004,rounding_size=.025',fc=blue if i in (0,3) else '#e9f0f4',ec=blue,lw=.7)); c='white' if i in (0,3) else '#162c40'; ax.text(x+.088,.58,a,ha='center',va='center',color=c,fontsize=8.2,weight='bold'); ax.text(x+.088,.38,b,ha='center',va='center',color=c,fontsize=7.5)
 if i<4: ax.annotate('',xy=(x+.199,.505),xytext=(x+.178,.505),arrowprops=dict(arrowstyle='->',lw=.8,color=gray))
save(fig,'pipeline')
D=pd.read_csv(E/'verified_topology_means.csv'); print(D.columns,D.head()); R=pd.read_csv(E/'verified_topology_ranks.csv')
fig,axs=plt.subplots(2,1,figsize=(3.45,2.45),sharex=True)
for ax,period,base in zip(axs,['Jan-Jul 2026','Jan-Sep 2026'],[.612769978,.595001015]):
 d=D[D.period==period]; vals=d[d.topology!='spatial'].ap.to_numpy(); s=d[d.topology=='spatial'].ap.iloc[0]; ax.scatter(vals,.2+np.arange(len(vals))%4*.16,s=15,color=blue,alpha=.85);ax.axvline(s,color=teal,lw=1.8,label='Geographic');ax.axvline(base,color=orange,ls='--',lw=1.5,label='Persistence');ax.set_ylim(0,.9);ax.set_yticks([]);ax.set_title(period,loc='left',fontsize=9);ax.grid(axis='x',alpha=.2);ax.spines['left'].set_visible(False)
axs[-1].set_xlabel('Average precision'); axs[0].legend(loc='upper left',bbox_to_anchor=(0,1.62),ncol=2,fontsize=7,frameon=False); fig.subplots_adjust(hspace=.65); save(fig,'permutation')
M=pd.read_csv(E/'monthly_threshold_diagnostics_mean.csv'); fig,ax=plt.subplots(figsize=(3.45,2.25)); ax.plot(M.month,M.frozen_f1,'o-',color=blue,label='Frozen F1',ms=3);ax.plot(M.month,M.oracle_f1,'s--',color=orange,label='Retrospective best F1',ms=3);ax.plot(M.month,M.ap,'^-',color=teal,label='AP',ms=3);ax.set_xticks(range(1,8),['Jan','Feb','Mar','Apr','May','Jun','Jul']);ax.set_ylim(0,.9);ax.set_ylabel('Score');ax.grid(axis='y',alpha=.25);ax.legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2);save(fig,'monthly')
C=pd.read_csv(E/'official_categories_jul_sep_2026.csv'); print(C.columns,C.head())
fig,axs=plt.subplots(1,2,figsize=(3.45,1.85));
for ax,old,new in zip(axs,['Water System','Sewer'],['Water Maintenance','Sewer Maintenance']):
 vals=lambda lab: [C.loc[(C.month==m)&(C.complaint_type==lab),'count'].sum() for m in ['2026-07','2026-08','2026-09']];x=np.arange(3);ax.bar(x-.18,vals(old),.36,color=blue,label='Old label');ax.bar(x+.18,vals(new),.36,color=teal,label='Maintenance');ax.set_xticks(x,['Jul','Aug','Sep']);ax.set_title(old,fontsize=9);ax.tick_params(labelsize=7);ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Requests');fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=7);fig.tight_layout(w_pad=.9);save(fig,'category')
