"""Assertion-only recovery: original Stage123 loader hardcoded its output width to 24."""
import inspect,sys
import numpy as np
import stage151 as m

def recovered_loader(bundle):
    text=inspect.getsource(original)
    old='assert z.shape==(len(d["y"]),24) and np.isfinite(z).all()'
    assert text.count(old)==1
    new='assert z.shape[0]==len(d["y"]) and z.shape[1] in (24,96) and np.isfinite(z).all()'
    scope=dict(m.portable.__dict__)
    exec(compile(text.replace(old,new),__file__,'exec'),scope)
    return scope['sources'](bundle)

original=m.portable.sources
if __name__=='__main__':
    m.OUT.mkdir(exist_ok=True)
    receipt=m.OUT/'recovery_preflight.json'
    if '--verify' in sys.argv:
        _,_,bundle,_,_,_=m.base.definitions()
        m.portable.fit_project=m.projector(8);a=original(bundle);b=recovered_loader(bundle)
        for case in m.CASES:
            xa,da,*_=a(case,0);xb,db,*_=b(case,0)
            np.testing.assert_array_equal(xa,xb)
            for k in da:np.testing.assert_array_equal(da[k],db[k])
        m.portable.fit_project=m.projector(32);wide=recovered_loader(bundle)
        for case in m.CASES:assert wide(case,0)[0].shape[1]==96
        m.g.save(receipt,dict(recovery_code=m.g.sha(__file__),original_code=m.g.sha(m.__file__),portable_code=m.g.sha(m.portable.__file__),rank8_exact=True,rank32_shape=True,change='Only loader width assertion, 24 -> allowed {24,96}. No training/prediction/selection code changed. Complete rank8 receipts reused.',guard_preflight=m.g.sha(m.OUT/'preflight.json')))
        print('Stage151 recovery verified: rank8 inputs exact, rank32 inputs 96-dimensional.',flush=True)
    else:
        r=m.g.read(receipt);assert r['recovery_code']==m.g.sha(__file__) and r['original_code']==m.g.sha(m.__file__) and r['portable_code']==m.g.sha(m.portable.__file__)
        m.portable.sources=recovered_loader
        m.main()
