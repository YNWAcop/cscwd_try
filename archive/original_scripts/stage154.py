"""Complete assignment controls for the frozen conditional-supervision candidate."""
import os,sys,hashlib
from collections import defaultdict
import numpy as np
import torch
import stage151 as engine
import stage151_recover as recovery

g=engine.g;HERE=engine.HERE;OUT=HERE/'stage154';PROTOCOL=HERE/'STAGE154_PROTOCOL.md'
MODES=('global','source_only','score_only')
ARMS=tuple(f'{mode}_{j}' for mode in MODES for j in range(19))
original_targets=engine.targets
original_train=engine.train

def groups(mode,y,v):
    result=defaultdict(list)
    keys={'global':np.zeros(len(y),int),'source_only':v,'score_only':y}[mode]
    for i,key in enumerate(keys):result[key].append(i)
    return result

def targets(u,y,v,ids):
    _,index,counts=original_targets(u,y,v,ids)
    output=[]
    for mode in MODES:
        cells=groups(mode,y,v)
        for rep in range(19):
            shuffled=u.copy()
            for key,ix in cells.items():
                ix=np.array(ix);ix=ix[np.argsort(ids[ix])]
                token=f'154:{mode}:{key}:{rep}'.encode()
                seed=int.from_bytes(hashlib.sha256(token).digest()[:8],'little')
                shuffled[ix]=u[np.random.default_rng(seed).permutation(ix)]
            output.append(shuffled)
    return np.stack(output),index,counts

def objective(p,u,logit,y,truth,c,index,counts):
    return (p-y).abs().mean(1)+.25*(u-truth).abs().mean((1,2))+.25*engine.centered_loss(u,truth,index,counts)

def fingerprint():
    return dict(code=g.sha(__file__),protocol=g.sha(PROTOCOL),engine=g.sha(engine.__file__),recovery=g.sha(recovery.__file__),portable=g.sha(engine.portable.__file__),seeds=list(engine.SEEDS),arms=list(ARMS),parent=g.sha(HERE/'stage151/fit_seal.json'))

class Runtime(engine.Runtime):
    def __init__(self):
        super().__init__();self.path=OUT/'run_status.json';self.hold=g.GUARD/'cpu_holds/stage154.json';self.marker=g.GUARD/'workers/stage154.json'
    def start(self):
        assert not (g.GUARD/'cancel.flag').exists()
        engine.base.cpu_ready(False,'Stage154 candidate four-shuffle audit')
        g.save(self.hold,dict(active=True,pid=os.getpid(),authorization_id=self.config['authorization_id']))
        g.save(self.marker,dict(active=True,pid=os.getpid(),process_create_time=self.state['process_create_time'],controller_sha256=g.sha(__file__)))
        self.boundary(PROTOCOL);self.phase('training')

def train(x,d,u,fit,tune,hold,seed,stem,spec,rt):
    prediction=original_train(x,d,u,fit,tune,hold,seed,stem,spec,rt)
    path=stem.with_suffix('.json');r=g.read(path)
    if r.get('null_metadata_start')!=0:
        truth,_,_=targets(u[fit],d['y'][fit],d['video'][fit],d['id'][fit])
        r['changed_unimodal_vectors']=[int((a!=u[fit]).any(1).sum()) for a in truth]
        r['null_metadata_start']=0;r['schemes']=list(MODES);g.save(path,r);rt.boundary(path)
    return prediction

def setup():
    engine.OUT=OUT;engine.PROTOCOL=PROTOCOL;engine.ARMS=ARMS;engine.targets=targets;engine.objective=objective;engine.fingerprint=fingerprint;engine.Runtime=Runtime;engine.train=train
    engine.portable.sources=recovery.recovered_loader

if __name__=='__main__':
    setup()
    if '--smoke' in sys.argv:
        engine.main()
        u=np.arange(36).reshape(12,3)/36;y=np.tile([-.5,0,.5],4);v=np.repeat(['a','b','c'],4);ids=np.array([f'id{i:02d}' for i in range(12)])
        truth,index,counts=targets(u,y,v,ids)
        assert truth.shape==(57,12,3)
        for j,mode in enumerate(MODES):
            for rep in range(19):
                for ix in groups(mode,y,v).values():
                    assert sorted(map(tuple,u[ix]))==sorted(map(tuple,truth[j*19+rep,ix]))
        receipt=g.read(OUT/'preflight.json');receipt.update(joint_vector_permutation=True,preserved_mode_multisets=True,three_null_schemes=True);g.save(OUT/'preflight.json',receipt)
        print('Stage154 permutation invariants and saved-checkpoint handshake passed.',flush=True)
    else:
        assert g.read(OUT/'preflight.json')['preserved_mode_multisets']
        engine.main()
