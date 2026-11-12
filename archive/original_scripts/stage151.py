"""Learn continuous within-cell modality differences rather than a binary conflict flag."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,tempfile,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import stage150 as prev
import stage124
base,g,portable=prev.base,prev.g,prev.portable
HERE=prev.HERE;OUT=HERE/'stage151';PROTOCOL=HERE/'STAGE151_PROTOCOL.md'
SEEDS=prev.SEEDS;CASES=prev.CASES
ARMS=('task_only','uni_mtl','conflict_mtl','binary_conflict','centered_only','conditional_direction','conditional_low','conditional_high','uni_shuffled',*(f'shuffle_{j}' for j in range(19)))

class Runtime(prev.Runtime):
    def __init__(self):
        super().__init__();self.path=OUT/'run_status.json';self.hold=g.GUARD/'cpu_holds/stage151.json';self.marker=g.GUARD/'workers/stage151.json'
    def start(self):
        assert not (g.GUARD/'cancel.flag').exists()
        base.cpu_ready(False,'Stage151 conditional directional supervision')
        g.save(self.hold,dict(active=True,pid=os.getpid(),authorization_id=self.config['authorization_id']))
        g.save(self.marker,dict(active=True,pid=os.getpid(),process_create_time=self.state['process_create_time'],controller_sha256=g.sha(__file__)))
        self.boundary(PROTOCOL);self.phase('training')

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

def fingerprint():return dict(code=g.sha(__file__),protocol=g.sha(PROTOCOL),prior=g.sha(prev.__file__),base=g.sha(base.__file__),portable=g.sha(portable.__file__),seeds=list(SEEDS),arms=list(ARMS))

def smoke():
    real=g.GUARD
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        try:
            g.GUARD=Path(tmp);config=dict(authorization_id='stage151-isolated');g.save(g.GUARD/'config.json',config)
            g.save(g.GUARD/'shutdown_request.json',dict(request_id='new151'));saved=g.GUARD/'checkpoint.json';g.save(saved,dict(saved=True))
            rt=object.__new__(Runtime);rt.config=config;rt.old_id='old';rt.started=time.monotonic();rt.path=g.GUARD/'status.json';rt.state=dict(pid=os.getpid(),process_create_time=g.process_time(os.getpid()))
            try:rt.boundary(saved);raise AssertionError('missing stop')
            except SystemExit as e:assert e.code==0
            assert g.read(g.GUARD/'shutdown_ack.json')['checkpoint_sha256']==g.sha(saved)
        finally:g.GUARD=real
    torch.manual_seed(151);pred=torch.randn(2,6,3);truth=torch.randn(2,6,3);index=torch.tensor([0,0,1,1,1,2]);counts=torch.tensor([2,3,1.])
    a=centered_loss(pred,truth,index,counts);shift=torch.randn(2,3,3)
    assert torch.allclose(a,centered_loss(pred+shift[:,index],truth,index,counts),atol=1e-6)
    # A singleton target cannot affect the conditional loss.
    changed=truth.clone();changed[:,-1]=100;assert torch.equal(a,centered_loss(pred,changed,index,counts))
    net=Model(8,len(ARMS),17);p,u,c=net(torch.randn(6,3,8));t=truth[:1].repeat(len(ARMS),1,1)
    out=objective(p,u,c,torch.randn(6),t,torch.tensor([1,0,1,0,0,0],dtype=torch.bool),index,counts);out.sum().backward();assert torch.isfinite(out).all()
    # Identical initialization across arms, independent parameters.
    assert torch.equal(p[0],p[1]) and torch.equal(net.mw[0],net.mw[1])
    g.save(OUT/'preflight.json',dict(spec=fingerprint(),guard_handshake=True,source_score_shift_invariant=True,singleton_exclusion=True,forward_backward=True,identical_initialization=True))
    print('Stage151 preflight passed.',flush=True)

def train(x,d,u,fit,tune,hold,seed,stem,spec,rt):
    rec=stem.with_suffix('.json');ckpt=stem.with_suffix('.pt');pp=stem.with_suffix('.npz')
    if rec.exists():
        r=g.read(rec);assert r['spec']==spec and r['checkpoint']==g.sha(ckpt) and r['predictions']==g.sha(pp)
        with np.load(pp,allow_pickle=False) as p:return p['prediction']
    rank=x.shape[1]//3
    xt=torch.as_tensor(x.reshape(len(x),3,rank),device='cuda',dtype=torch.float32);yt=torch.as_tensor(d['y'],device='cuda',dtype=torch.float32)
    truth,index,counts=targets(u[fit],d['y'][fit],d['video'][fit],d['id'][fit])
    tt=torch.as_tensor(truth,device='cuda',dtype=torch.float32);ix=torch.as_tensor(index,device='cuda');ct=torch.as_tensor(counts,device='cuda',dtype=torch.float32);cf=torch.as_tensor(d['c'][fit],device='cuda')
    net=Model(rank,len(ARMS),seed).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001,foreach=False)
    last=0;best=torch.full((len(ARMS),),float('inf'),device='cuda');epochs=torch.zeros(len(ARMS),device='cuda',dtype=torch.long);best_state=copy.deepcopy(net.state_dict())
    if ckpt.exists():
        r=torch.load(ckpt,map_location='cuda',weights_only=True);assert r['spec']==spec
        net.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);last=r['epoch'];best=r['best'];epochs=r['selected_epochs'];best_state=r['best_state']
    for ep in range(last+1,101):
        opt.zero_grad(set_to_none=True);p,up,cp=net(xt[fit]);value=objective(p,up,cp,yt[fit],tt,cf,ix,ct)
        assert torch.isfinite(value).all();value.sum().backward();opt.step()
        with torch.no_grad():
            p,_,_=net(xt[tune]);score=(p.clamp(-1,1)-yt[tune]).abs().mean(1);better=score<best;best[better]=score[better];epochs[better]=ep
            for key,tensor in net.state_dict().items():best_state[key][better]=tensor[better]
        if ep%10==0:
            base.torch_save(ckpt,dict(spec=spec,model=net.state_dict(),optimizer=opt.state_dict(),epoch=ep,best=best,selected_epochs=epochs,best_state=best_state));rt.boundary(ckpt)
    net.load_state_dict(best_state)
    with torch.no_grad():p,_,_=net(xt[hold]);pred=p.clamp(-1,1).cpu().numpy()
    base.npz(pp,dict(prediction=pred));g.save(rec,dict(spec=spec,checkpoint=g.sha(ckpt),predictions=g.sha(pp),selected_epochs=epochs.tolist(),tune_mae=best.tolist(),fits=len(ARMS),paired_fit_examples=int((counts[index]>1).sum()),fit_examples=int(fit.sum()),changed_unimodal_vectors=[int((row!=truth[0]).any(1).sum()) for row in truth[9:]]))
    rt.state['completed_new_fits']+=len(ARMS);rt.boundary(rec);return pred

def run(rt):
    spec=fingerprint();assert g.read(OUT/'preflight.json')['spec']==spec
    _,_,bundle,_,_,_=base.definitions();all_u=stage124.ratings();seal={}
    for rank in (8,32):
        portable.fit_project=projector(rank);load=portable.sources(bundle)
        for case in CASES:
            parts=[]
            for fold in range(3):
                x,d,y,fit,tune,hold=load(case,fold);u=np.array([all_u[case][i] for i in d['id']])
                assert all(set(d['video'][a]).isdisjoint(set(d['video'][b])) for a,b in ((fit,tune),(fit,hold),(tune,hold)))
                cell=dict(spec=spec,rank=rank,case=case,fold=fold,x=prev.digest_array(x),fit_u=prev.digest_array(u[fit]),fit_y=prev.digest_array(y[fit]),ids=prev.digest_array(d['id']))
                predictions=[]
                for seed in SEEDS:
                    stem=OUT/f'r{rank}_{case}_{fold}_{seed}'
                    predictions.append(train(x,d,u,fit,tune,hold,seed,stem,{**cell,'seed':seed},rt));seal[stem.with_suffix('.json').name]=g.sha(stem.with_suffix('.json'))
                parts.append(dict(**{k:d[k][hold] for k in ('id','y','c','video')},prediction=np.stack(predictions)))
                print(f'Stage151 rank={rank} {case} fold={fold} sealed; fits={rt.state["completed_new_fits"]}',flush=True)
            base.npz(OUT/f'predictions_r{rank}_{case}.npz',{k:np.concatenate([a[k] for a in parts],axis=-1) for k in parts[0]})
    g.save(OUT/'fit_seal.json',dict(spec=spec,receipts=seal,predictions={p.name:g.sha(p) for p in OUT.glob('predictions_*.npz')},all_fits=90*len(ARMS),scope='adaptive retrospective development'))
    rt.state.update(status='training_complete',active_training=False);rt.boundary(OUT/'fit_seal.json');print('Stage151 complete; predictions sealed before scoring.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True);g.BASE=OUT;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.smoke:smoke();return
    if (OUT/'fit_seal.json').exists():print('Completed seal; no retraining.');return
    for marker in (g.GUARD/'workers').glob('*.json'):
        r=g.read(marker)
        if r.get('active'):
            t=g.process_time(r['pid']);assert t is None or abs(t-r['process_create_time'])>.01,marker
    rt=Runtime();lock=OUT/'worker.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:rt.start();run(rt)
    except Exception as e:
        rt.state.update(status='failed',error=repr(e));g.save(OUT/f'failure_{os.getpid()}.json',rt.state);g.save(rt.path,rt.state);raise
    finally:rt.finish();lock.unlink()

if __name__=='__main__':main()
