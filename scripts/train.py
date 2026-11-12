"""Portable Stage151 bank runner on an authorized, already-prepared NPZ fold.

No raw dataset is distributed. This runner does not reconstruct the historical
pooling/PCA inputs: see data/README.md. It never accesses holdout during selection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.conditional_supervision import Model, ARMS, targets, objective

def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(a.data,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    x,y,u=d['x'],d['y'],d['u'];n=len(y)
    assert x.shape in [(n,24),(n,96)] and u.shape==(n,3)
    assert all(np.isfinite(v).all() for v in [x,y,u])
    fit,tune,hold=[d[k] for k in ['fit','tune','holdout']]
    assert all(v.dtype==bool and v.shape==(n,) and v.any() for v in [fit,tune,hold])
    assert np.all(fit.astype(int)+tune.astype(int)+hold.astype(int)==1)
    assert len(np.unique(d['id']))==n
    for left,right in [(fit,tune),(fit,hold),(tune,hold)]:
        assert set(d['source'][left]).isdisjoint(d['source'][right])
    assert d['conflict'].dtype==bool and d['conflict'].shape==(n,)
    assert d['conflict'][fit].any(), 'Conflict-weighted comparison needs nonempty FIT conflict group'
    spec=dict(input_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),seed=a.seed,
              core_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'src/conditional_supervision.py').read_bytes()).hexdigest(),
              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device=a.device,epochs=100)
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'result.json').exists():
        r=json.loads((a.out/'result.json').read_text());assert r['spec']==spec
        for name,digest in r['artifacts'].items():assert hashlib.sha256((a.out/name).read_bytes()).hexdigest()==digest
        print('Verified completed result; no retraining.');return
    lock=a.out/'worker.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        rank=x.shape[1]//3
        xt=torch.as_tensor(x.reshape(n,3,rank),dtype=torch.float32,device=a.device)
        yt=torch.as_tensor(y,dtype=torch.float32,device=a.device)
        truth,index,counts=targets(u[fit],y[fit],d['source'][fit],d['id'][fit])
        tt=torch.as_tensor(truth,dtype=torch.float32,device=a.device)
        ix=torch.as_tensor(index,device=a.device);ct=torch.as_tensor(counts,dtype=torch.float32,device=a.device)
        cf=torch.as_tensor(d['conflict'][fit],device=a.device)
        net=Model(rank,len(ARMS),a.seed).to(a.device)
        opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001,foreach=False)
        best=torch.full((len(ARMS),),float('inf'),device=a.device)
        selected=torch.zeros(len(ARMS),dtype=torch.long,device=a.device)
        best_state=copy.deepcopy(net.state_dict());start=0;latest=a.out/'latest.pt'
        if latest.exists():
            z=torch.load(latest,map_location=a.device,weights_only=True);assert z['spec']==spec
            net.load_state_dict(z['model']);opt.load_state_dict(z['optimizer'])
            best,selected,best_state,start=z['best'],z['selected_epochs'],z['best_state'],z['epoch']
        for ep in range(start+1,101):
            opt.zero_grad(set_to_none=True);pred,uni,logit=net(xt[fit])
            loss=objective(pred,uni,logit,yt[fit],tt,cf,ix,ct)
            assert torch.isfinite(loss).all();loss.sum().backward();opt.step()
            with torch.no_grad():
                pred,_,_=net(xt[tune]);score=(pred.clamp(-1,1)-yt[tune]).abs().mean(1)
                better=score<best;best[better]=score[better];selected[better]=ep
                for key,value in net.state_dict().items():best_state[key][better]=value[better]
            if ep%10==0:
                save(latest,dict(spec=spec,model=net.state_dict(),optimizer=opt.state_dict(),epoch=ep,
                     best=best,selected_epochs=selected,best_state=best_state))
                print(f'epoch {ep}/100',flush=True)
        net.load_state_dict(best_state)
        with torch.no_grad():pred=net(xt[hold])[0].clamp(-1,1).cpu().numpy()
        np.savez_compressed(a.out/'predictions.npz',prediction=pred)
        result=dict(spec=spec,arms=list(ARMS),selected_epochs=selected.tolist(),tune_mae=best.tolist(),
                    holdout_mae=np.abs(pred-y[hold]).mean(1).tolist(),
                    artifacts={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in ['latest.pt','predictions.npz']},
                    note='Single fold/seed only. Not a confidence interval or independent confirmation.')
        (a.out/'result.json').write_text(json.dumps(result,indent=2))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
