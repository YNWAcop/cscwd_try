import ast, unittest
from pathlib import Path
import numpy as np
import torch
from src import conditional_supervision as core

class CoreTests(unittest.TestCase):
    def test_exact_historical_definitions(self):
        r=Path(__file__).resolve().parents[1]
        def definitions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        old=definitions(r/'archive/original_scripts/stage151.py');new=definitions(r/'src/conditional_supervision.py')
        for name in ['Model','targets','objective','centered_loss','projector']:self.assertEqual(old[name],new[name])
    def test_conditional_invariance_singletons_and_gradients(self):
        torch.manual_seed(151);p=torch.randn(2,6,3,requires_grad=True);u=torch.randn(2,6,3)
        ix=torch.tensor([0,0,1,1,1,2]);counts=torch.tensor([2.,3.,1.])
        loss=core.centered_loss(p,u,ix,counts);shift=torch.randn(2,3,3)
        torch.testing.assert_close(loss,core.centered_loss(p+shift[:,ix],u,ix,counts))
        u[:,-1]=100;torch.testing.assert_close(loss,core.centered_loss(p,u,ix,counts))
        loss.sum().backward();self.assertTrue(torch.equal(p.grad[:,-1],torch.zeros_like(p.grad[:,-1])))
    def test_paired_initialization_and_backward(self):
        model=core.Model(8,len(core.ARMS),17);p,u,c=model(torch.randn(6,3,8))
        torch.testing.assert_close(p[0],p[1],rtol=0,atol=0)
        y=torch.zeros(6);truth=torch.zeros_like(u);ix=torch.tensor([0,0,1,1,1,2]);counts=torch.tensor([2.,3.,1.])
        loss=core.objective(p,u,c,y,truth,torch.tensor([True,False,True,False,False,False]),ix,counts)
        loss.sum().backward();self.assertTrue(torch.isfinite(loss).all())

if __name__=='__main__':unittest.main()
