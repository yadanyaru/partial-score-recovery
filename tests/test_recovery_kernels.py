"""Small deterministic checks of the exported recovery implementation."""
from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import numpy as np

HAVE_GPU_DEPENDENCIES=all(importlib.util.find_spec(x) for x in ['torch','transformers'])

class PredictiveGeometry(unittest.TestCase):
    def test_equal_marginals_different_predictive_risk(self):
        H=np.array([[1.,-1.],[-1.,1.]])
        self.assertAlmostEqual(np.trace(H@np.eye(2))/2,1.)
        self.assertAlmostEqual(np.trace(H@np.ones((2,2)))/2,0.)
        Hp=np.ones((2,2))
        self.assertAlmostEqual(np.trace(Hp@np.ones((2,2)))/2,2.)

    @unittest.skipUnless(HAVE_GPU_DEPENDENCIES,'Install requirements-gpu.txt for implementation checks; no GPU needed here')
    def test_shared_quadratic_matches_exact_covariance(self):
        import torch
        from sample_precision_ablation import shared_quadratic
        f=torch.tensor([0.,.1,.1,.7,1.],dtype=torch.float64)
        a=torch.tensor([[1.,-2.,3.,.4,7.],[-4.,1.,.1,2.,-3.]],dtype=torch.float64)
        covariance=torch.minimum(f[:,None],f[None,:])-f[:,None]*f[None,:]
        expected=torch.einsum('bi,ij,bj->b',a,covariance,a)
        actual=shared_quadratic(a,f)
        torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-13)

    @unittest.skipUnless(HAVE_GPU_DEPENDENCIES,'Install requirements-gpu.txt for implementation checks; no GPU needed here')
    def test_public_decoder_recovers_noiseless_prediction(self):
        import torch
        from risk_selected_precision_pretrained import public_decoder
        gen=torch.Generator().manual_seed(123)
        W=torch.randn(30,3,dtype=torch.float64,generator=gen)
        b=torch.randn(30,dtype=torch.float64,generator=gen)
        h=torch.randn(3,dtype=torch.float64,generator=gen)
        ids=torch.arange(8);D,_,_=public_decoder(W,ids)
        lp=torch.log_softmax(W@h+b,0)
        recovered=D@(lp[ids]-b[ids])
        torch.testing.assert_close(recovered,h,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(torch.softmax(W@recovered+b,0),lp.exp(),atol=1e-12,rtol=1e-12)

    @unittest.skipUnless(HAVE_GPU_DEPENDENCIES,'Install requirements-gpu.txt for implementation checks; no GPU needed here')
    def test_sample_selection_uses_quadratics(self):
        import torch
        from review_precision_utility import choose
        from sample_precision_ablation import shared_quadratic
        W=torch.tensor([[1.,0.],[0.,1.],[-1.,.5]],dtype=torch.float64)
        D=torch.tensor([[1.,.2],[.3,1.]],dtype=torch.float64)
        f=torch.tensor([.2,.7],dtype=torch.float64);err=torch.tensor([-.2,.3],dtype=torch.float64)
        pairs=torch.tensor([[0,1],[0,2],[1,2]])
        p=torch.ones(3,dtype=torch.float64)/3
        idx,scores=choose(p,W,D,f,err,pairs)
        a=(W[pairs[:,0]]-W[pairs[:,1]])@D
        expected=torch.stack([a.square()@(f*(1-f)),shared_quadratic(a,f),(a@err).square()],1).mean(0)
        torch.testing.assert_close(scores,expected)
        self.assertEqual(idx,int(expected.argmin()))

if __name__=='__main__':unittest.main()
