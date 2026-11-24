"""Descriptive community trajectories: meaningful but NOT validated crisis labels."""
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score


def describe_communities(raw_graph: np.ndarray,embeddings: np.ndarray,dates,
                         max_snapshots:int=21,seed=42):
    """Embeddings [time,node,hidden], drawn from held-out predictions.
    Louvain runs on *existing* graph edges weighted by latent cosine agreement.
    Returns communities, modularity and label-alignment (not ground-truth accuracy).
    """
    num=min(max_snapshots,len(embeddings));start=len(embeddings)-num
    results=[];previous=None
    for k in range(start,len(embeddings)):
        z=embeddings[k]
        n=len(z)
        norms=np.maximum(np.linalg.norm(z,axis=1),1e-6)
        sim=(z@z.T)/(norms[:,None]*norms[None,:])
        g=nx.Graph();g.add_nodes_from(range(n))
        for i,j in np.argwhere(np.triu(raw_graph,1)>0):
            weight=float(raw_graph[i,j]*(.5+.5*max(0,float(sim[i,j]))))
            if weight>0:g.add_edge(int(i),int(j),weight=weight)
        groups=nx.community.louvain_communities(g,weight='weight',seed=seed) if g.number_of_edges() else [{i} for i in range(n)]
        groups=[sorted(int(j) for j in c) for c in groups]
        label=np.zeros(n,dtype=int)
        for ci,group in enumerate(groups):label[group]=ci
        modularity=float(nx.community.modularity(g,[set(c) for c in groups],weight='weight')) if g.number_of_edges() else 0.
        stability=float(adjusted_rand_score(previous,label)) if previous is not None else None
        results.append({'date':str(dates[k].date()),'n_communities':len(groups),
                        'modularity':modularity,'stability_ari':stability,'members':groups})
        previous=label
    return results
