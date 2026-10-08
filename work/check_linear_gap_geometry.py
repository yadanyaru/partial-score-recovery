import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json,pathlib
def bases(p,number):
 x=np.arange(p)[:,None];b=np.arange(p)[None,:]
 return [np.eye(p,dtype=complex)]+[np.exp(2j*np.pi*((a*x*x+b*x)%p)/p)/np.sqrt(p) for a in range(number-1)]
def realrows(U):
 return np.r_[np.c_[U.real.T,U.imag.T],np.c_[-U.imag.T,U.real.T]]
diagnostics=[]
for p in (7,13,17):
 frames=bases(p,min(13,p+1));maxerror=0.
 for i,U in enumerate(frames):
  assert np.max(abs(U.conj().T@U-np.eye(p)))<1e-12
  for V in frames[:i]:maxerror=max(maxerror,float(np.max(abs(abs(U.conj().T@V)-1/np.sqrt(p)))))
 head=np.concatenate([realrows(U) for U in frames[:6]])
 gram=head@head.T;np.fill_diagonal(gram,0)
 assert abs(gram).max()<=1/np.sqrt(p)+1e-12
 rank=np.linalg.matrix_rank(head[1:]-head[0]);assert rank==2*p
 diagnostics.append({'p':p,'six_basis_rank':int(rank),'coherence':float(abs(gram).max()),'MUB_absolute_error':maxerror})
r=260;p=67;r0=2*p;V=6*(r+1)
raw=np.concatenate([realrows(U) for U in bases(p,13)])[:V]
head=np.zeros((V,r));head[:,:r0]=raw
rng=np.random.default_rng(8881);radius=1/(64*np.sqrt(p))
# Small normalized perturbation: merely numerical geometry, not genericity proof.
noise=rng.normal(size=head.shape);noise*=radius/np.linalg.norm(noise,axis=1)[:,None]
head+=noise;head/=np.linalg.norm(head,axis=1)[:,None]
gram=head@head.T;np.fill_diagonal(gram,0)
rank=np.linalg.matrix_rank(head[1:]-head[0]);assert rank==r
assert abs(gram).max()<2/np.sqrt(p)
result={'scope':'CPU numerical geometry only; no exact frequency-genericity or rational-head certificate',
        'prime_frame_checks':diagnostics,
        'all_rank_example':{'r':r,'p':p,'r0':r0,'available_rows':13*r0,'selected_V':V,
                            'perturbed_affine_rank':int(rank),'coherence':float(abs(gram).max()),'bound':2/np.sqrt(p)}}
pathlib.Path('outputs/linear_gap_geometry_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
