"""Source-cross-fitted matched excess-risk correction; exploratory, no test selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, copy, hashlib, json, time, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import stage118 as base
import stage123 as portable

g=base.g
HERE=Path(__file__).resolve().parent
OUT=HERE/'stage150'
PROTOCOL=HERE/'STAGE150_PROTOCOL.md'
SEEDS=(17,43,89,131,197)
CASES=('v1','v2','UniC')
ARMS=('residual','conflict','matched','mean_regret','positive_regret',
      'matched_positive_low','matched_positive','matched_positive_high',
      'matched_distill','source_regret',*(f'shuffle_{j}' for j in range(19)))

def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()

class Runtime(g.SafeRuntime):
    def __init__(self):
        super().__init__()
        self.path=OUT/'run_status.json'
        self.hold=g.GUARD/'cpu_holds/stage150.json'
        self.marker=g.GUARD/'workers/stage150.json'
    def start(self):
        assert not (g.GUARD/'cancel.flag').exists()
        base.cpu_ready(False,'Stage150 source-cross-fitted research writes')
        g.save(self.hold,dict(active=True,pid=os.getpid(),authorization_id=self.config['authorization_id']))
        g.save(self.marker,dict(active=True,pid=os.getpid(),process_create_time=self.state['process_create_time'],controller_sha256=g.sha(__file__)))
        self.boundary(PROTOCOL)
        self.phase('training')

def folds(video,ids):
    names,counts=np.unique(video,return_counts=True)
    totals=np.zeros(3,int); mapping={}
    for j in sorted(range(len(names)),key=lambda j:(-counts[j],str(names[j]))):
        k=int(totals.argmin()); mapping[names[j]]=k; totals[k]+=counts[j]
    return np.array([mapping[v] for v in video])

def shuffle_flags(c,y,v,ids,rep):
    out=c.copy(); groups=defaultdict(list)
    for j,key in enumerate(zip(v,y)):
        groups[key].append(j)
    for (source,score),ii in groups.items():
        ii=np.array(ii); ii=ii[np.argsort(ids[ii])]
        token=f'150:{source}:{float(score).hex()}:{rep}'.encode()
        seed=int.from_bytes(hashlib.sha256(token).digest()[:8],'little')
        out[ii]=np.random.default_rng(seed).permutation(c[ii])
        assert out[ii].sum()==c[ii].sum()
    return out

def weights(c,y,v):
    a=np.zeros(len(c)); groups=defaultdict(list)
    for i,key in enumerate(zip(v,y)):
        groups[key].append(i)
    for ii in groups.values():
        ii=np.array(ii); yes=ii[c[ii]]; no=ii[~c[ii]]
        if len(yes) and len(no):
            a[yes]=len(no)/(len(yes)+len(no))
    a/=max(a.sum(),1e-12)
    n=(~c).astype(float); n/=max(n.sum(),1)
    q=c.astype(float); q/=max(q.sum(),1)
    return a,n,q

class Residual(base.Readouts):
    def __init__(self,width,count,seed):
        super().__init__(width,count,seed)
        nn.init.zeros_(self.w2); nn.init.zeros_(self.b2)
    def forward(self,x):
        return .25*torch.tanh(super().forward(x))

def reference(x,y,fit,tune,hold,video,ids,seed,stem,spec,rt):
    receipt=stem.with_suffix('.json'); predpath=stem.with_suffix('.npz'); ckpt=stem.with_suffix('.pt')
    if receipt.exists():
        r=g.read(receipt); assert r['spec']==spec and r['predictions']==g.sha(predpath) and r['checkpoint']==g.sha(ckpt)
        with np.load(predpath,allow_pickle=False) as a: return a['b']
    inner=folds(video[fit],ids[fit]); masks=np.stack([inner!=k for k in range(3)])
    assert all(set(video[fit][m]).isdisjoint(set(video[fit][~m])) for m in masks)
    net=base.Readouts(x.shape[1],3,seed).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001,foreach=False)
    xt=torch.as_tensor(x,dtype=torch.float32,device='cuda'); yt=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    mt=torch.as_tensor(masks,dtype=torch.float32,device='cuda')
    last=0
    if ckpt.exists():
        z=torch.load(ckpt,map_location='cuda',weights_only=True); assert z['spec']==spec
        net.load_state_dict(z['model']); opt.load_state_dict(z['optimizer']);last=z['epoch']
    for ep in range(last+1,101):
        opt.zero_grad(set_to_none=True)
        loss=(net(xt[fit])-yt[fit]).abs()
        ((loss*mt).sum(1)/mt.sum(1)).sum().backward(); opt.step()
        if ep%10==0:
            base.torch_save(ckpt,dict(spec=spec,model=net.state_dict(),optimizer=opt.state_dict(),epoch=ep))
            rt.boundary(ckpt)
    with torch.no_grad(): p=net(xt).clamp(-1,1).cpu().numpy()
    b=p.mean(0); ix=np.flatnonzero(fit); b[ix]=p[inner,ix]
    base.npz(predpath,dict(b=b,inner=inner,all_reference=p))
    g.save(receipt,dict(spec=spec,checkpoint=g.sha(ckpt),predictions=g.sha(predpath),source_folds={str(k):sorted(set(video[fit][inner==k].tolist())) for k in range(3)},fits=3))
    rt.state['completed_new_fits']+=3;rt.boundary(receipt)
    return b

def losses(pred,y,b,a,n,q,source_masks):
    error=(pred-y).abs(); ref=(b-y).abs(); delta=error-ref
    ordinary=error.mean(1)
    incentive=(error*a).sum(1)
    pos=(F.relu(delta)*n).sum(1)
    mean=F.relu((delta*n).sum(1))
    distill=((pred-b).square()*n).sum(1)
    value=ordinary+.5*incentive+2*pos
    value[0]=ordinary[0]
    value[1]=ordinary[1]+.5*(error[1]*q[1]).sum()
    value[2]=ordinary[2]+.5*incentive[2]
    value[3]=ordinary[3]+.5*incentive[3]+2*mean[3]
    value[4]=ordinary[4]+.5*(error[4]*q[4]).sum()+2*pos[4]
    value[5]=ordinary[5]+.5*incentive[5]+.5*pos[5]
    value[7]=ordinary[7]+.5*incentive[7]+8*pos[7]
    value[8]=ordinary[8]+.5*incentive[8]+2*distill[8]
    value[9]=ordinary[9]+.5*incentive[9]+2*F.relu(source_masks@delta[9]).mean()
    return value

def correction(x,y,b,fit,tune,hold,c,v,ids,seed,stem,spec,rt):
    receipt=stem.with_suffix('.json');predpath=stem.with_suffix('.npz');ckpt=stem.with_suffix('.pt')
    if receipt.exists():
        r=g.read(receipt); assert r['spec']==spec and r['checkpoint']==g.sha(ckpt) and r['predictions']==g.sha(predpath)
        with np.load(predpath,allow_pickle=False) as p: return p['prediction']
    cf,yf,vf,idf=c[fit],y[fit],v[fit],ids[fit]
    ws=[weights(cf,yf,vf) for _ in range(10)]
    for j in range(19): ws.append(weights(shuffle_flags(cf,yf,vf,idf,j),yf,vf))
    a,n,q=[torch.as_tensor(np.stack([w[k] for w in ws]),device='cuda',dtype=torch.float32) for k in range(3)]
    source_masks=np.stack([(~cf)&(vf==s) for s in np.unique(vf) if ((~cf)&(vf==s)).any()]).astype(float)
    source_masks/=source_masks.sum(1)[:,None]
    st=torch.as_tensor(source_masks,device='cuda',dtype=torch.float32)
    z=np.column_stack((x,b))
    xt=torch.as_tensor(z,dtype=torch.float32,device='cuda'); yt=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    bt=torch.as_tensor(b,dtype=torch.float32,device='cuda')
    net=Residual(z.shape[1],len(ARMS),seed).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001,foreach=False)
    last=0; best=float('inf');best_epoch=0;best_state=None; history=[]
    if ckpt.exists():
        r=torch.load(ckpt,map_location='cuda',weights_only=True);assert r['spec']==spec
        net.load_state_dict(r['model']);opt.load_state_dict(r['optimizer'])
        last,best,best_epoch,best_state,history=[r[k] for k in ('epoch','best','best_epoch','best_state','history')]
    for ep in range(last+1,101):
        opt.zero_grad(set_to_none=True)
        p=(bt[fit]+net(xt[fit])).clamp(-1,1)
        objective=losses(p,yt[fit],bt[fit],a,n,q,st)
        assert torch.isfinite(objective).all()
        objective.sum().backward();opt.step()
        with torch.no_grad():
            tp=(bt[tune]+net(xt[tune])).clamp(-1,1)
            metric=float((tp[0]-yt[tune]).abs().mean())
        history.append(metric)
        if metric<best:
            best=metric;best_epoch=ep;best_state={k:t.detach().clone() for k,t in net.state_dict().items()}
        if ep%10==0:
            base.torch_save(ckpt,dict(spec=spec,model=net.state_dict(),optimizer=opt.state_dict(),epoch=ep,best=best,best_epoch=best_epoch,best_state=best_state,history=history))
            rt.boundary(ckpt)
    net.load_state_dict(best_state)
    with torch.no_grad(): prediction=(bt[hold]+net(xt[hold])).clamp(-1,1).cpu().numpy()
    base.npz(predpath,dict(prediction=prediction))
    g.save(receipt,dict(spec=spec,checkpoint=g.sha(ckpt),predictions=g.sha(predpath),selected_epoch=best_epoch,tune_ordinary_residual_mae=best,fits=len(ARMS),matched_fit_conflict=int((a[0].cpu().numpy()>0).sum()),flags_changed=[int((shuffle_flags(cf,yf,vf,idf,j)!=cf).sum()) for j in range(19)]))
    rt.state['completed_new_fits']+=len(ARMS);rt.boundary(receipt)
    return prediction

def fingerprint():
    return dict(code=g.sha(__file__),protocol=g.sha(PROTOCOL),base=g.sha(base.__file__),portable=g.sha(portable.__file__),seeds=list(SEEDS),arms=list(ARMS))

def smoke():
    real=g.GUARD
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        try:
            g.GUARD=Path(tmp);config=dict(authorization_id='stage150-isolated')
            g.save(g.GUARD/'config.json',config);g.save(g.GUARD/'shutdown_request.json',dict(request_id='new150'))
            saved=g.GUARD/'checkpoint.json';g.save(saved,dict(committed=True))
            rt=object.__new__(Runtime);rt.config=config;rt.old_id='old';rt.started=time.monotonic();rt.path=g.GUARD/'status.json'
            rt.state=dict(pid=os.getpid(),process_create_time=g.process_time(os.getpid()))
            try: rt.boundary(saved);raise AssertionError('stop missing')
            except SystemExit as e: assert e.code==0
            ack=g.read(g.GUARD/'shutdown_ack.json');assert ack['checkpoint_sha256']==g.sha(saved) and ack['request_id']=='new150'
        finally: g.GUARD=real
    c=np.array([1,0,1,0,0],bool);y=np.array([0,0,1,1,2.]);v=np.array(['a']*5);ids=np.array(list('abcde'))
    a,n,q=weights(c,y,v);assert np.isclose(a.sum(),1) and np.isclose(n.sum(),1) and a[-1]==0
    for j in range(19):
        cf=shuffle_flags(c,y,v,ids,j)
        assert all(cf[y==s].sum()==c[y==s].sum() for s in np.unique(y))
    # Positive individual regret cannot cancel a hurt item against a helped one.
    delta=torch.tensor([.2,-.2]);assert F.relu(delta.mean())==0 and F.relu(delta).mean()>.09
    torch.manual_seed(150);x=torch.randn(7,25);net=Residual(25,29,17)
    assert torch.equal(net(x),torch.zeros(29,7))
    # Safe checkpoint continuation: optimizer replay must exactly match.
    opt=torch.optim.AdamW(net.parameters(),lr=.003,foreach=False)
    target=torch.randn(7)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);(net(x)-target).square().mean(1).sum().backward();opt.step()
    saved=copy.deepcopy((net.state_dict(),opt.state_dict()))
    opt.zero_grad(set_to_none=True);(net(x)-target).square().mean(1).sum().backward();opt.step();expected=net(x).detach().clone()
    other=Residual(25,29,17);other.load_state_dict(saved[0]);opt2=torch.optim.AdamW(other.parameters(),lr=.003,foreach=False);opt2.load_state_dict(saved[1])
    opt2.zero_grad(set_to_none=True);(other(x)-target).square().mean(1).sum().backward();opt2.step()
    assert torch.equal(expected,other(x)) and torch.max(other(x).abs())<=.25
    assert torch.cuda.is_available()
    g.save(OUT/'preflight.json',dict(spec=fingerprint(),guard_handshake=True,optimizer_resume_exact=True,zero_initial_correction=True,noncancelling_regret=True,shuffle_preserves_cells=True,device=torch.cuda.get_device_name(0)))
    print('Stage150 preflight passed: guard, resume, objective, shuffle, zero correction.',flush=True)

def run(rt):
    spec=fingerprint();assert g.read(OUT/'preflight.json')['spec']==spec
    _,_,bundle,_,_,_=base.definitions();load=portable.sources(bundle)
    seal={};packs={}
    for case in CASES:
        parts=[]
        for fold in range(3):
            x,d,y,fit,tune,hold=load(case,fold)
            assert all(set(d['video'][a]).isdisjoint(set(d['video'][b])) for a,b in ((fit,tune),(fit,hold),(tune,hold)))
            cell_spec=dict(spec=spec,case=case,fold=fold,x=digest_array(x),y=digest_array(y),id=digest_array(d['id']),masks=[digest_array(m) for m in (fit,tune,hold)])
            preds=[];refs=[]
            for seed in SEEDS:
                stem=OUT/f'{case}_{fold}_{seed}'
                refstem=OUT/f'{case}_{fold}_{seed}_reference'
                this_spec={**cell_spec,'seed':seed}
                b=reference(x,y,fit,tune,hold,d['video'],d['id'],seed,refstem,this_spec,rt)
                prediction=correction(x,y,b,fit,tune,hold,d['c'],d['video'],d['id'],seed,stem,this_spec,rt)
                refs.append(b[hold]);preds.append(prediction)
                for p in (stem.with_suffix('.json'),refstem.with_suffix('.json')):seal[p.name]=g.sha(p)
                print(f'Stage150 sealed {case} fold={fold} seed={seed}; new fits={rt.state["completed_new_fits"]}',flush=True)
            parts.append(dict(**{k:d[k][hold] for k in ('id','y','c','video')},prediction=np.stack(preds),reference=np.stack(refs)))
        packs[case]={k:np.concatenate([a[k] for a in parts],axis=-1) for k in parts[0]}
        base.npz(OUT/f'predictions_{case}.npz',packs[case])
    g.save(OUT/'fit_seal.json',dict(spec=spec,receipts=seal,predictions={p.name:g.sha(p) for p in OUT.glob('predictions_*.npz')},all_fits=45*(3+len(ARMS)),scope='retrospective development; no holdout-based candidate selection'))
    rt.state.update(status='training_complete',active_training=False);rt.boundary(OUT/'fit_seal.json')
    print('All predictions sealed. No holdout metrics computed by training controller.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);g.BASE=OUT
    if args.smoke:smoke();return
    if (OUT/'fit_seal.json').exists():print('Completed seal exists; no retraining.');return
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
