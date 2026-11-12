"""Generic fixed-bank risk accounting. Sources, seeds, and null replicates are crossed."""
import numpy as np
from stage150_evaluate import ci,support_weights

def evaluate(d,names,null_indices,active,comparators,rng):
    y,c,v=[d[k] for k in ('y','c','video')];e=abs(d['prediction']-y);s,k,n=e.shape
    assert len(names)==k
    _,vi=np.unique(v,return_inverse=True);nv=vi.max()+1
    wb=rng.multinomial(nv,np.full(nv,1/nv),size=3000);ws=rng.multinomial(s,np.full(s,1/s),size=3000)/s
    nr=len(null_indices);wr=rng.multinomial(nr,np.full(nr,1/nr),size=3000)/nr
    mc,mn=support_weights(y,c,v);masks=dict(all=np.ones(n),C=c.astype(float),N=(~c).astype(float),N_outside=((~c)&(mn==0)).astype(float),matched_C=mc,matched_N=mn)
    points,boot,seeds,support={},{},{},{}
    for group,w in masks.items():
        counts=np.bincount(vi,weights=w,minlength=nv);den=wb@counts
        agg=np.stack([np.stack([np.bincount(vi,weights=a*w,minlength=nv) for a in row]) for row in e])
        nums=np.einsum('bs,skv,bv->bk',ws,agg,wb,optimize=True)
        sample=np.divide(nums,den[:,None],out=np.full_like(nums,np.nan),where=den[:,None]>0)
        nums_null=np.einsum('bs,br,srv,bv->b',ws,wr,agg[:,null_indices],wb,optimize=True)
        null=np.divide(nums_null,den,out=np.full(3000,np.nan),where=den>0)
        boot[group]=np.column_stack((sample,null));perseed=(e*w).sum(-1)/w.sum()
        seeds[group]=np.column_stack((perseed,perseed[:,null_indices].mean(1)));points[group]=seeds[group].mean(0)
        support[group]=dict(n=int((w>0).sum()),sources=int((counts>0).sum()),weight=float(w.sum()))
    all_names=(*names,'shuffled_primary');rows=[]
    for name in active:
        i=all_names.index(name)
        for ref in comparators:
            j=all_names.index(ref)
            if i==j:continue
            for group in masks:
                perseed=seeds[group][:,j]-seeds[group][:,i]
                rows.append(dict(method=name,reference=ref,group=group,mae=float(points[group][i]),**ci(points[group][j]-points[group][i],boot[group][:,j]-boot[group][:,i]),positive_seeds=int((perseed>0).sum()),seed_gains=perseed.tolist(),**support[group]))
            a=(points['matched_C'][j]-points['matched_C'][i])-(points['matched_N'][j]-points['matched_N'][i])
            samples=(boot['matched_C'][:,j]-boot['matched_C'][:,i])-(boot['matched_N'][:,j]-boot['matched_N'][:,i])
            rows.append(dict(method=name,reference=ref,group='A',**ci(a,samples)))
    return dict(rows=rows,support=support,mae={name:float(points['all'][i]) for i,name in enumerate(all_names)})
