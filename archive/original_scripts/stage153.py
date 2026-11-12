"""Two-way projection of relative modality errors, reusing sealed simpler baselines."""
import os,sys
from pathlib import Path
import numpy as np
import torch
import stage151 as engine
import stage151_recover as recovery
g=engine.g;HERE=engine.HERE;OUT=HERE/'stage153';PROTOCOL=HERE/'STAGE153_PROTOCOL.md'
ARMS=('modality_only','double_centered','double_low','double_high',*(f'shuffle_{j}' for j in range(19)))
original_targets=engine.targets
original_center=engine.centered_loss
original_train=engine.train

def targets(u,y,v,ids):
    truth,index,counts=original_targets(u,y,v,ids)
    return np.concatenate((truth[:4],truth[9:]),axis=0),index,counts

def double_center(pred,truth,index,counts):
    e=pred-truth;e=e-e.mean(-1,keepdim=True)
    return original_center(e,torch.zeros_like(e),index,counts)

def objective(p,u,logit,y,truth,c,index,counts):
    task=(p-y).abs().mean(1);uni=(u-truth).abs().mean((1,2));dual=double_center(u,truth,index,counts)
    loss=task+.25*uni+.25*dual
    e=u-truth
    loss[0]=task[0]+.25*uni[0]+.25*(e[0]-e[0].mean(-1,keepdim=True)).abs().mean()
    loss[2]=task[2]+.25*uni[2]+.0625*dual[2]
    loss[3]=task[3]+.25*uni[3]+dual[3]
    return loss

def fingerprint():
    return dict(code=g.sha(__file__),protocol=g.sha(PROTOCOL),engine=g.sha(engine.__file__),recovery=g.sha(recovery.__file__),portable=g.sha(engine.portable.__file__),seeds=list(engine.SEEDS),arms=list(ARMS),parent=g.sha(HERE/'stage151/fit_seal.json'))

class Runtime(engine.Runtime):
    def __init__(self):
        super().__init__();self.path=OUT/'run_status.json';self.hold=g.GUARD/'cpu_holds/stage153.json';self.marker=g.GUARD/'workers/stage153.json'

def train(x,d,u,fit,tune,hold,seed,stem,spec,rt):
    prediction=original_train(x,d,u,fit,tune,hold,seed,stem,spec,rt)
    # The reusable engine's reporting slice starts after nine arms; this bank has four.
    path=stem.with_suffix('.json');r=g.read(path)
    if r.get('null_metadata_start')!=4:
        truth,_,_=targets(u[fit],d['y'][fit],d['video'][fit],d['id'][fit])
        r['changed_unimodal_vectors']=[int((row!=truth[0]).any(1).sum()) for row in truth[4:]]
        r['null_metadata_start']=4;g.save(path,r);rt.boundary(path)
    return prediction

def setup():
    engine.OUT=OUT;engine.PROTOCOL=PROTOCOL;engine.ARMS=ARMS;engine.targets=targets;engine.objective=objective;engine.fingerprint=fingerprint;engine.Runtime=Runtime;engine.train=train
    engine.portable.sources=recovery.recovered_loader

if __name__=='__main__':
    setup()
    if '--smoke' in sys.argv:
        # Retain engine's saved-checkpoint/new-request handshake and add both nullspace checks.
        engine.main()
        torch.manual_seed(153);a=torch.randn(2,6,3);b=torch.randn(2,6,3);ix=torch.tensor([0,0,1,1,1,2]);n=torch.tensor([2,3,1.])
        d=double_center(a,b,ix,n)
        sample_shift=torch.randn(2,6,1);cell_shift=torch.randn(2,3,3)
        assert torch.allclose(d,double_center(a+sample_shift+cell_shift[:,ix],b,ix,n),atol=1e-6)
        receipt=g.read(OUT/'preflight.json');receipt.update(sample_common_valence_invariant=True,cell_modality_bias_invariant=True);g.save(OUT/'preflight.json',receipt)
        print('Stage153 both projection invariances passed.',flush=True)
    else:
        receipt=g.read(OUT/'preflight.json');assert receipt['sample_common_valence_invariant'] and receipt['cell_modality_bias_invariant']
        engine.main()
