"""Synthetic smoke-test data ONLY. Never report its metrics as real-world evidence."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate(output='data/SMOKE_TEST_ONLY.csv',days=420,seed=7):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2024-01-01',periods=days)
    cats=['HEAT/HOT WATER','Water System','Sewer','Street Condition']
    zips=[f'112{i:02d}' for i in range(8)]
    records=[]
    for t,day in enumerate(dates):
        common=(1.4 if t%7 in (0,1) else 1.0)*(1+.4*np.sin(t/30))
        for iz,z in enumerate(zips):
            for ic,cat in enumerate(cats):
                base=2.5+ic*1.1+iz*.3
                lam=base*common*(1+0.6*np.sin((t-iz*2)/15))
                n=int(rng.poisson(max(.2,lam)))
                if n:
                    records.append({'day':str(day.date()),'incident_zip':z,'complaint_type':cat,
                                    'n':n,'lat':40.58+iz*.012,'lon':-74.02+iz*.009,
                                    'source':'SMOKE_TEST_SYNTHETIC'})
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(path,index=False)
    print(f'Created synthetic smoke-test ONLY: {path}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/SMOKE_TEST_ONLY.csv')
    a=p.parse_args();generate(a.output)
