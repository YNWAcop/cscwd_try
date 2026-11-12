"""Add two initialization seeds to the unchanged original audit without duplicate fits."""
import os,argparse,tempfile,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
import stage150 as prev
base,g,portable=prev.base,prev.g,prev.portable
HERE=prev.HERE;OUT=HERE/'stage152';PROTOCOL=HERE/'STAGE152_PROTOCOL.md'

class Runtime(prev.Runtime):
    def __init__(self):
        super().__init__();self.path=OUT/'run_status.json';self.hold=g.GUARD/'cpu_holds/stage152.json';self.marker=g.GUARD/'workers/stage152.json'
    def needs_boundary(self):
        request=g.read(g.GUARD/'shutdown_request.json') if (g.GUARD/'shutdown_request.json').exists() else None
        return g.pending_request(request,self.old_id) or (g.GUARD/'cancel.flag').exists() or (OUT/'stop.flag').exists() or g.read(g.GUARD/'config.json')!=self.config or time.monotonic()-self.started>3590
    def start(self):
        assert not (g.GUARD/'cancel.flag').exists();base.cpu_ready(False,'Stage152 original audit seed extension')
        g.save(self.hold,dict(active=True,pid=os.getpid(),authorization_id=self.config['authorization_id']))
        g.save(self.marker,dict(active=True,pid=os.getpid(),process_create_time=self.state['process_create_time'],controller_sha256=g.sha(__file__)))
        self.boundary(PROTOCOL);self.phase('training')

def spec():return dict(code=g.sha(__file__),protocol=g.sha(PROTOCOL),base=g.sha(base.__file__),portable=g.sha(portable.__file__))

def smoke():
    real=g.GUARD
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        try:
            g.GUARD=Path(tmp);config=dict(authorization_id='stage152-isolated');g.save(g.GUARD/'config.json',config)
            g.save(g.GUARD/'shutdown_request.json',dict(request_id='new152'));saved=g.GUARD/'checkpoint.json';g.save(saved,dict(saved=True))
            rt=object.__new__(Runtime);rt.config=config;rt.old_id='old';rt.started=time.monotonic();rt.path=g.GUARD/'status.json';rt.state=dict(pid=os.getpid(),process_create_time=g.process_time(os.getpid()))
            assert rt.needs_boundary()
            try:rt.boundary(saved);raise AssertionError('missing stop')
            except SystemExit as e:assert e.code==0
            assert g.read(g.GUARD/'shutdown_ack.json')['checkpoint_sha256']==g.sha(saved)
        finally:g.GUARD=real
    g.save(OUT/'preflight.json',dict(spec=spec(),guard_handshake=True));print('Stage152 guard preflight passed.',flush=True)

def run(rt):
    current=spec();assert g.read(OUT/'preflight.json')['spec']==current
    native,group,permute,adversary=base.cohort_inputs();_,_,bundle,_,_,_=base.definitions();common=portable.sources(bundle)
    seal={};parents={}
    for interface,parent in [('native','stage118'),('common','stage123')]:
        oldseal=g.read(HERE/parent/'fit_seal.json');oldresult=g.read(HERE/parent/'result.json')
        assert g.sha(HERE/parent/'fit_seal.json')==oldresult['fit_seal']
        parts_by_case={}
        for case in prev.CASES:
            parts=[]
            for fold in range(3):
                if interface=='native':x,d,fit,tune,hold,eps=native(case,fold)
                else:x,d,_,fit,tune,hold=common(case,fold);eps=.05
                groups,support=base.assignments(d,fit,fold,group,permute);preds=[]
                for seed in prev.SEEDS:
                    if seed in base.SEEDS:
                        rec=HERE/parent/f'{case}_{fold}_{seed}.json';assert oldseal[rec.name]==g.sha(rec)
                        r=g.read(rec);p=rec.with_suffix('.npz');assert r['predictions']==g.sha(p) and r['checkpoint']==g.sha(rec.with_suffix('.pt'))
                        with np.load(p,allow_pickle=False) as a:pred=a['prediction']
                        parents[str(rec)]=g.sha(rec)
                    else:
                        stem=OUT/f'{interface}_{case}_{fold}_{seed}';rec=stem.with_suffix('.json');p=stem.with_suffix('.npz')
                        cell=dict(spec=current,interface=interface,case=case,fold=fold,seed=seed,x=prev.digest_array(x),epsilon=eps)
                        if rec.exists():
                            r=g.read(rec);assert r['spec']==cell and r['predictions']==g.sha(p) and r['checkpoint']==g.sha(stem.with_suffix('.pt'))
                            with np.load(p,allow_pickle=False) as a:pred=a['prediction']
                        else:
                            pred,ep,val=base.train_bank(x,d,fit,tune,hold,eps,groups,adversary,seed,'cuda',stem,cell,rt)
                            base.npz(p,dict(prediction=pred));g.save(rec,dict(spec=cell,predictions=g.sha(p),checkpoint=g.sha(stem.with_suffix('.pt')),selected_epoch=ep,uniform_tune_mae=val,fit_support=support));rt.state['completed_new_fits']+=22
                        seal[rec.name]=g.sha(rec);rt.boundary(rec)
                    preds.append(pred)
                parts.append(dict(**{k:d[k][hold] for k in ('id','y','c','video')},prediction=np.stack(preds)))
                print(f'Stage152 {interface} {case} fold={fold} sealed; new fits={rt.state["completed_new_fits"]}',flush=True)
            base.npz(OUT/f'predictions_{interface}_{case}.npz',{k:np.concatenate([a[k] for a in parts],axis=-1) for k in parts[0]})
    g.save(OUT/'fit_seal.json',dict(spec=current,receipts=seal,reused=parents,predictions={p.name:g.sha(p) for p in OUT.glob('predictions_*.npz')},new_fits=792,reused_fits=1188))
    rt.state.update(status='training_complete',active_training=False);rt.boundary(OUT/'fit_seal.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();OUT.mkdir(exist_ok=True);g.BASE=OUT;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.smoke:smoke();return
    if (OUT/'fit_seal.json').exists():print('Complete; no retraining.');return
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
