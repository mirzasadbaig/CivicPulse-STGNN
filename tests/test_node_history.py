import numpy as np
from benchmark_extension import flatten

def test_node_history_alignment():
    w=np.arange(2*3*4*2).reshape(2,3,4,2)
    rows=flatten(w)
    for b in range(2):
        for n in range(4):
            np.testing.assert_array_equal(rows[b*4+n],w[b,:,n,:].ravel())
