"""Leakage-controlled graph/time-series feature creation for NYC 311 daily counts.

Definitions:
  Node = (ZIP, complaint category)
  Snapshot = one day
  Label = in next 2 days, count exceeds *training-only* 75th-percentile
          future-48h count for this node. Elevated demand, NOT verified crisis.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class Data:
    x: np.ndarray                  # [T,N,F]
    y: np.ndarray                  # [T,N]
    counts: np.ndarray             # [T,N]
    future_counts: np.ndarray      # [T,N]
    threshold_counts: np.ndarray   # [N]
    adj: np.ndarray                # [N,N]
    nodes: list[tuple[str,str]]
    dates: pd.DatetimeIndex
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    lookback: int
    source: str
    feature_names: list[str]
    raw_adj: np.ndarray
    zip_centroids: dict
    base_rate: float


def _km(lat1,lon1,lat2,lon2):
    from math import radians,sin,cos,asin,sqrt
    dlat=radians(lat2-lat1);dlon=radians(lon2-lon1)
    return 12742*asin(min(1,sqrt(sin(dlat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2)))


def load_data(csv: str, lookback=28, max_zips=20, max_nodes=80,
              train_fraction=.70, val_fraction=.15, quantile=.75, graph='spatial') -> Data:
    df=pd.read_csv(csv,dtype={'incident_zip':str})
    required={'day','incident_zip','complaint_type','n','lat','lon','source'}
    if not required.issubset(df):raise ValueError(f'Missing CSV columns: {required-set(df)}')
    source=set(df.source.dropna().unique())
    if len(source)!=1 or source.pop() not in {'NYC_311_PUBLIC_AGGREGATED','SMOKE_TEST_SYNTHETIC'}:
        raise ValueError('Unrecognized or mixed data provenance')
    source=df.source.iloc[0]
    df['day']=pd.to_datetime(df.day,errors='coerce').dt.normalize()
    df['incident_zip']=df.incident_zip.astype(str).str.strip().str.zfill(5)
    for k in ('n','lat','lon'):df[k]=pd.to_numeric(df[k],errors='coerce')
    df=df.dropna(subset=['day','incident_zip','complaint_type','n','lat','lon'])
    df=df[df.incident_zip.str.fullmatch(r'\d{5}') & (df.n>=0) & df.lat.between(40,42) & df.lon.between(-75,-72)]
    if df.empty: raise ValueError('No clean records; check CSV format and coordinates')
    dates=pd.date_range(df.day.min(),df.day.max(),freq='D')
    T=len(dates);horizon=2
    cut_train=int(train_fraction*T);cut_val=int((train_fraction+val_fraction)*T)
    if cut_train<lookback+40 or cut_val-cut_train<12 or T-cut_val<12:
        raise ValueError(f'Too few daily bins ({T}); use longer period or smaller lookback')
    # Only training period determines which ZIPs and nodes exist in the study.
    train_df=df[df.day<dates[cut_train]]
    zip_activity=train_df.groupby('incident_zip').n.sum().sort_values(ascending=False)
    zips=list(zip_activity.head(max_zips).index)
    cat_activity=train_df.groupby('complaint_type').n.sum().sort_values(ascending=False)
    cats=list(cat_activity.index)
    nodes=[(z,c) for z in zips for c in cats][:max_nodes]
    if len(nodes)<4:raise ValueError('Not enough nodes to model a graph')
    nodemap={node:i for i,node in enumerate(nodes)}
    counts=np.zeros((T,len(nodes)),dtype=np.float32)
    positions={day:i for i,day in enumerate(dates)}
    df['node']=[nodemap.get((z,c),-1) for z,c in zip(df.incident_zip,df.complaint_type)]
    for (day,ni),total in df[df.node>=0].groupby(['day','node']).n.sum().items():
        counts[positions[day],int(ni)]=float(total)
    # Centroids from training period only, weighted by number of complaints.
    zip_centroids={}
    for z,g in train_df[train_df.incident_zip.isin(zips)].groupby('incident_zip'):
        w=g.n.to_numpy(dtype=float);zip_centroids[z]=(float(np.average(g.lat,weights=w)),float(np.average(g.lon,weights=w)))
    raw=np.zeros((len(nodes),len(nodes)),dtype=np.float32)
    nearest={}
    for z in zips:
        if z not in zip_centroids:continue
        a,b=zip_centroids[z]
        other=sorted((_km(a,b,*zip_centroids[k]),k) for k in zips if k!=z and k in zip_centroids)
        nearest[z]={k for _,k in other[:3]}
    for i,(z,c) in enumerate(nodes):
        for j in range(i+1,len(nodes)):
            zz,cc=nodes[j]
            if z==zz and c!=cc:raw[i,j]=raw[j,i]=.25
            elif c==cc and (zz in nearest.get(z,set()) or z in nearest.get(zz,set())):
                d=_km(*zip_centroids[z],*zip_centroids[zz]);raw[i,j]=raw[j,i]=np.exp(-d/4.)
    if graph=='hybrid':
        # Historical correlation edges are computed ONLY on the training period.
        historical=counts[:cut_train].astype(float)
        cor=np.corrcoef(historical,rowvar=False)
        cor=np.nan_to_num(cor,nan=0,posinf=0,neginf=0)
        for i in range(len(nodes)):
            same_cat=[j for j in range(len(nodes)) if j!=i and nodes[j][1]==nodes[i][1]]
            for j in sorted(same_cat,key=lambda j:cor[i,j],reverse=True)[:2]:
                if cor[i,j]>.35:raw[i,j]=raw[j,i]=max(raw[i,j],float(.25*cor[i,j]))
    elif graph!='spatial':raise ValueError('graph must be spatial or hybrid')
    adj=raw+np.eye(len(nodes),dtype=np.float32)
    degree=np.maximum(adj.sum(1),1e-6)**-.5
    adj=(degree[:,None]*adj*degree[None,:]).astype(np.float32)
    # Features use only observations at or before each anchor day.
    neighbor_w=raw/np.maximum(raw.sum(1,keepdims=True),1e-7)
    feats=[]
    for t,d in enumerate(dates):
        curr=counts[t];past7=counts[max(0,t-6):t+1]
        past14=counts[max(0,t-13):t+1]
        prev7=counts[max(0,t-13):max(0,t-6)]
        same_weekday=counts[t-7] if t>=7 else np.zeros_like(curr)
        f=np.stack([np.log1p(curr),np.log1p(past7.mean(0)),np.log1p(past14.mean(0)),
                    np.log1p(prev7.mean(0)) if len(prev7) else np.zeros_like(curr),
                    np.clip(past7.mean(0)/(past14.mean(0)+1),0,5),
                    np.log1p(same_weekday),np.log1p(neighbor_w@curr),
                    np.full_like(curr,np.sin(2*np.pi*d.dayofweek/7)),
                    np.full_like(curr,np.cos(2*np.pi*d.dayofweek/7)),
                    np.full_like(curr,np.sin(2*np.pi*d.dayofyear/365.25)),
                    np.full_like(curr,np.cos(2*np.pi*d.dayofyear/365.25))],axis=1)
        feats.append(f)
    x=np.asarray(feats,dtype=np.float32)
    scaler=StandardScaler()
    scaler.fit(x[:cut_train].reshape(-1,x.shape[-1]))
    x=scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype(np.float32)
    x=np.clip(x,-8,8)
    future=np.zeros_like(counts)
    for t in range(T-horizon):future[t]=counts[t+1:t+1+horizon].sum(0)
    train=np.arange(lookback-1,cut_train-horizon,dtype=int)
    val=np.arange(cut_train,cut_val-horizon,dtype=int)
    test=np.arange(cut_val,T-horizon,dtype=int)
    if min(map(len,(train,val,test)))<8:raise ValueError('Too few anchors after horizon embargo')
    # Node-wise thresholds learned on train labels only. This defines an upper-quartile
    # *future 48h demand* event, NOT an emergency. Ties can affect prevalence.
    thresholds=np.quantile(future[train],quantile,axis=0).astype(np.float32)
    y=(future>thresholds[None,:]).astype(np.float32)
    rate=float(y[train].mean())
    if rate<.02 or rate>.65: print(f'Warning: unusual training positive rate {rate:.1%}; inspect target definition.')
    return Data(x,y,counts,future,thresholds,adj,nodes,dates,train,val,test,lookback,
                source,['log count','log 7d mean','log 14d mean','log earlier 7d mean','7d/14d ratio',
                        'same weekday count','neighbor count','weekday sin','weekday cos','year sin','year cos'],
                raw,zip_centroids,rate)


def windows(x,anchors,L):
    return np.stack([x[i-L+1:i+1] for i in anchors],axis=0)
