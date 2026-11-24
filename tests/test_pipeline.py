from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.demo_data import generate
from src.data import load_data, windows
from src.experiment import val_threshold,score


def test_temporal_pipeline_no_future_in_features():
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'smoke.csv'
        generate(str(path),days=250)
        a=load_data(str(path),lookback=14,max_zips=5,max_nodes=20)
        assert a.x.shape[:2] == a.y.shape
        assert a.x.shape[1]==len(a.nodes)
        assert a.idx_train[-1]+2<a.idx_val[0]
        assert a.idx_val[-1]+2<a.idx_test[0]
        assert np.allclose(a.future_counts[a.idx_train[0]],a.counts[a.idx_train[0]+1:a.idx_train[0]+3].sum(0))
        assert a.source=='SMOKE_TEST_SYNTHETIC'
        assert windows(a.x,a.idx_train[:4],14).shape[1]==14
        assert len(a.idx_train)>0 and len(a.idx_val)>0 and len(a.idx_test)>0


def test_threshold_validation_only():
    y=np.array([[0,1,1,0]],dtype=int)
    p=np.array([[.1,.6,.8,.2]])
    t=val_threshold(y,p)
    assert 0<t<1
    out=score(y,p,t)
    assert out['f1']>=.66


def test_reject_wrong_source():
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'smoke.csv';generate(str(path),days=250)
        df=pd.read_csv(path,dtype={'incident_zip':str});df['source']='FAKE_NYC'
        df.to_csv(path,index=False)
        try:load_data(str(path))
        except ValueError as e:assert 'provenance' in str(e)
        else:raise AssertionError('Must reject unverified source strings')
