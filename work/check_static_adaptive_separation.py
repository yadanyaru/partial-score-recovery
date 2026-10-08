"""CPU-only diagnostics: exposed random head and explicit two-frequency phase witness."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json,pathlib
from scipy.special import logsumexp
rng=np.random.default_rng(737)
r=512;V=6*(r+1);W=rng.normal(size=(V,r));W/=np.linalg.norm(W,axis=1)[:,None]
eta=0.
for begin in range(0,V,128):
 gram=W[begin:begin+128]@W.T
 for local in range(len(gram)):gram[local,begin+local]=0.
 eta=max(eta,float(abs(gram).max()))
assert eta<1/3
a,b=V-2,V-1;n=W[a]+W[b];d=W[a]-W[b]
endpoint=1+W[a]@W[b];othermax=float(np.max((W[:-2]@n)))
assert endpoint-othermax>0
g=1/r;delta=g/(16*eta);t=delta*(1-W[a]@W[b])
JS=np.log(2)-(-((1+np.tanh(t))/2)*np.log((1+np.tanh(t))/2)-((1-np.tanh(t))/2)*np.log((1-np.tanh(t))/2))
assert JS>=g*g/(4608*eta*eta)
# This diagnostic head intentionally uses an explicitly exposed edge rather than
# the global coherence hypothesis, and only two observed coordinates.
a=np.array([.5,0,np.sqrt(.75)]);b=np.array([-.5,0,np.sqrt(.75)])
u=np.array([.05,np.sqrt(1-.05**2-.4**2),-.4])
v=np.array([-.07,np.sqrt(1-.07**2-.7**2),-.7])
head=np.stack([a,b,u,v]);n=a+b;d=a-b;g=.02;delta=g/(16*.95)
c=1+a@b;intercept=np.log(2*np.cosh(delta*(1-a@b)))
alpha=float(u@n-c)
found=None
for integer in range(-100000,-1):
 T=(integer*g+intercept)/alpha
 if T<100:continue
 logs=[]
 for sign in (-1,1):
  z=head@(T*n+sign*delta*d);logs.append(z-logsumexp(z))
 codes=[np.rint(x[2:]/g).astype(np.int64) for x in logs]
 if np.array_equal(codes[0],codes[1]):
  margins=[float(np.min(g/2-abs(x[2:]-codes[0]*g))) for x in logs]
  if min(margins)>g/8:
   found={'T':T,'selected_codes':codes[0].tolist(),'strict_margin':min(margins)};break
assert found is not None
result={'scope':'CPU diagnostics, not rational-independence proof or all-subset certificate',
        'random_head':{'r':r,'V':V,'max_pair_coherence':eta,'sample_endpoint_gap':endpoint-othermax,
                       'g':1/r,'binary_JS':JS,'certified_analytic_pair_floor':(1/r)**2/(4608*eta*eta)},
        'two_selected_phase_witness':found}
pathlib.Path('outputs/static_adaptive_separation_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
