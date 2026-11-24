"""Small spatiotemporal neural network that runs on macOS CPU or MPS.
No PyTorch Geometric installation is necessary.
"""
import torch
from torch import nn

class GCNGRU(nn.Module):
    def __init__(self, features: int, hidden: int=32, dropout: float=.20):
        super().__init__()
        self.enc=nn.Linear(2*features,hidden)
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(hidden,1)
    def forward(self,x,adj,return_embeddings=False):
        # x: [batch, days, graph nodes, features]
        nx=torch.einsum('ij,btjf->btif',adj,x)
        h=torch.relu(self.enc(torch.cat([x,nx],dim=-1)))
        b,l,n,k=h.shape
        h=h.transpose(1,2).reshape(b*n,l,k)
        temporal,_=self.gru(h)
        embedding=temporal[:,-1,:].reshape(b,n,k)
        logits=self.out(self.dropout(embedding)).squeeze(-1)
        return (logits,embedding) if return_embeddings else logits
