"""Exact Stage151 numerical definitions; local orchestration imports removed.
Extracted functions are unchanged. See archive/original_scripts/stage151.py.
"""
import hashlib
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
prev=SimpleNamespace(hashlib=hashlib)

ARMS=('task_only','uni_mtl','conflict_mtl','binary_conflict','centered_only','conditional_direction','conditional_low','conditional_high','uni_shuffled',*(f'shuffle_{j}' for j in range(19)))

def projector(rank):
    def project(x,fit):
        z=(x-x[fit].mean(0))/x[fit].std(0).clip(.001)
        with threadpool_limits(limits=1):z=PCA(rank,svd_solver='randomized',random_state=123).fit(z[fit]).transform(z)
        return np.clip((z-z[fit].mean(0))/z[fit].std(0).clip(.001),-8,8)
    return project

def targets(u,y,v,ids):
    cells=defaultdict(list)
    for i,key in enumerate(zip(v,y)):cells[key].append(i)
    index=np.empty(len(y),int);counts=[];shuffles=[]
    for j,ix in enumerate(cells.values()):index[ix]=j;counts.append(len(ix))
    for rep in range(19):
        out=u.copy()
        for key,ix in cells.items():
            ix=np.array(ix);ix=ix[np.argsort(ids[ix])]
            token=f'151:{key[0]}:{float(key[1]).hex()}:{rep}'.encode()
            seed=int.from_bytes(prev.hashlib.sha256(token).digest()[:8],'little')
            out[ix]=u[np.random.default_rng(seed).permutation(ix)]
        shuffles.append(out)
    return np.stack([u]*8+[shuffles[0]]+shuffles),index,np.array(counts)

class Model(nn.Module):
    def __init__(self,rank,count,seed):
        super().__init__();torch.manual_seed(seed)
        a=[nn.Linear(rank,32) for _ in range(3)]
        def param(t):return nn.Parameter(t.unsqueeze(0).repeat(count,*([1]*t.ndim)))
        self.mw=param(torch.stack([v.weight.T for v in a]));self.mb=param(torch.stack([v.bias for v in a]))
        h,o=nn.Linear(96,47),nn.Linear(47,1)
        self.hw=param(h.weight.T);self.hb=param(h.bias);self.ow=param(o.weight.T);self.ob=param(o.bias)
        aux=[nn.Linear(32,1) for _ in range(3)];cl=nn.Linear(96,1)
        self.aw=param(torch.stack([v.weight[:,0:] .reshape(-1) for v in aux]));self.ab=param(torch.stack([v.bias[0] for v in aux]))
        self.cw=param(cl.weight.T);self.cb=param(cl.bias)
    def forward(self,x):
        h=F.gelu(torch.einsum('nmr,bmrh->bnmh',x,self.mw)+self.mb[:,None])
        flat=h.flatten(2)
        p=torch.bmm(F.gelu(torch.bmm(flat,self.hw)+self.hb[:,None]),self.ow).squeeze(-1)+self.ob
        u=(h*self.aw[:,None]).sum(-1)+self.ab[:,None]
        c=torch.bmm(flat,self.cw).squeeze(-1)+self.cb
        return p,u,c

def centered_loss(pred,truth,index,counts):
    sums=torch.zeros(pred.shape[0],len(counts),3,device=pred.device,dtype=pred.dtype)
    sums.index_add_(1,index,pred-truth)
    residual=(pred-truth)-(sums/counts[None,:,None])[:,index]
    valid=counts[index]>1
    return residual[:,valid].abs().mean((1,2)) if valid.any() else pred.sum((1,2))*0

def objective(p,u,logit,y,truth,c,index,counts):
    loss=(p-y).abs();task=loss.mean(1);uni=(u-truth).abs().mean((1,2));cond=centered_loss(u,truth,index,counts)
    result=task+.25*uni+.25*cond
    result[0]=task[0];result[1]=task[1]+.25*uni[1]
    result[2]=task[2]+.25*uni[2]+.5*loss[2,c].mean()
    result[3]=task[3]+.25*uni[3]+.25*F.binary_cross_entropy_with_logits(logit[3],c.float())
    result[4]=task[4]+.25*cond[4]
    result[6]=task[6]+.25*uni[6]+.0625*cond[6]
    result[7]=task[7]+.25*uni[7]+cond[7]
    result[8]=task[8]+.25*uni[8]
    return result
