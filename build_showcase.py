"""Build a standalone portfolio observatory from research exports.
Optional --csv and --predictions expand real measured daily counts/risk; never synthesize observations.
"""
import argparse,json,html
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from importlib.util import spec_from_file_location,module_from_spec
ROOT=Path(__file__).resolve().parent

def aligned(snapshots,n):
    out=[];previous=None;stable=None
    for row in snapshots:
        labs=np.zeros(n,dtype=int)
        for i,m in enumerate(row['members']):
            for j in m: labs[j]=i
        if stable is None:stable=labs.copy()
        orig=sorted(set(stable));cur=sorted(set(labs));mat=np.array([[sum((stable==x)&(labs==y)) for y in cur] for x in orig]);ri,ci=linear_sum_assignment(-mat)
        mapid={cur[c]:orig[r] for r,c in zip(ri,ci)}
        ll=[int(mapid.get(v,v)) for v in labs]
        changes=[] if previous is None else [i for i,(a,b) in enumerate(zip(previous,ll)) if a!=b]
        out.append(dict(date=row['date'],labels=ll,changes=changes,modularity=round(row['modularity'],5),ari=row['stability_ari'],communities=row['n_communities']))
        previous=ll
    return out

def make(metrics,communities,out,csv=None,predictions=None):
    m=json.loads(Path(metrics).read_text()); c=json.loads(Path(communities).read_text()); nodes=[]
    for i,x in enumerate(m['nodes']):nodes.append(dict(id=i,zip=str(x['zip']),category=x['complaint_type'],threshold=x['high_demand_threshold']))
    snap=aligned(c,len(nodes))
    edges=[]
    for i,a in enumerate(nodes):
        for j in range(i+1,len(nodes)):
            if a['zip']==nodes[j]['zip']:edges.append(dict(source=i,target=j,weight=.25,type='Same ZIP'))
    scope='Verified same-ZIP edges only · geographic edges require original CSV'
    daily=[]; forecast={}
    if csv:
        from src.data import load_data
        data=load_data(str(csv),lookback=m['lookback_days'],max_zips=len(set(v['zip'] for v in nodes)),max_nodes=len(nodes),quantile=m['target_quantile'],graph=m['graph'])
        if data.nodes!=[(v['zip'],v['category']) for v in nodes]:raise ValueError('Node ordering mismatch: use the CSV from original run')
        if str(data.dates[0].date())!=m['dates']['start'] or str(data.dates[-1].date())!=m['dates']['end']:raise ValueError('Date mismatch')
        edges=[dict(source=i,target=j,weight=round(float(data.raw_adj[i,j]),4),type=('Same ZIP' if nodes[i]['zip']==nodes[j]['zip'] else 'Geographic')) for i in range(len(nodes)) for j in range(i+1,len(nodes)) if data.raw_adj[i,j]>0]
        scope='Full reconstructed graph using original NYC CSV · real daily counts'
        for k,date in enumerate(data.dates):daily.append({'date':str(date.date()),'counts':[int(v) for v in data.counts[k]]})
    if predictions:
        import csv as cs
        with open(predictions,newline='') as f:
            rd=cs.DictReader(f); keyed={(n['zip'],n['category']):n['id'] for n in nodes}
            for r in rd:
                date=r['anchor_date'][:10];idx=keyed.get((r['zip'].zfill(5),r['category']))
                if idx is None:continue
                forecast.setdefault(date,{})[str(idx)]={'prob':round(float(r['gnn_probability']),5),'actual':int(float(r['target_elevated_demand'])),'future':round(float(r['future_48h_count']),2)}
    val=m['results']['GCN-GRU']['test']; payload=dict(nodes=nodes,snapshots=snap,edges=edges,daily=daily,forecast=forecast,scope=scope,model={'f1':val['f1'],'ap':val['average_precision_pr_auc'],'precision':val['precision'],'recall':val['recall']},period=m['dates'],target=m['target'])
    markup=(ROOT/'dashboard'/'template.html').read_text()
    js=json.dumps(payload,ensure_ascii=True,separators=(',',':')).replace('</','<\\/')
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(markup.replace('/*PAYLOAD*/','window.PULSE='+js+';'))
    print('Saved',out,'days',len(daily),'snapshots',len(snap),'predictions',len(forecast),'edges',len(edges))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics',required=True);p.add_argument('--communities',required=True);p.add_argument('--csv');p.add_argument('--predictions');p.add_argument('--output',default='dashboard/index.html');x=p.parse_args();make(x.metrics,x.communities,x.output,x.csv,x.predictions)
