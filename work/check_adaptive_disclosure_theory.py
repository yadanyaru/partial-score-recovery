"""CPU counterexample searches / diagnostics for audited exact-arithmetic theorem."""
import json,pathlib,hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eigvalsh
rng=np.random.default_rng(20261007)
rows=[]
for r in (1,2,3):
 for iteration in range(20):
  V=3*r+4;W=rng.normal(size=(V,r));bias=rng.normal(size=V)
  B=W[1:r+1]-W[0];alpha=np.linalg.solve(B.T,(W-W[0]).T).T
  tau=float(abs(alpha).max());g1=.01/(r*tau);g2=.002
  h=rng.normal(size=r);logits=W@h+bias;ell=logits-logsumexp(logits)
  first=np.rint(ell[:r+1]/g1)*g1
  h0=np.linalg.solve(B,first[1:]-first[0]-bias[1:r+1]+bias[0])
  p0=np.exp(W@h0+bias-logsumexp(W@h0+bias));mu=p0@W
  F=(W-mu).T@(p0[:,None]*(W-mu))
  S=rng.choice(V,r+2,replace=False);weights=rng.uniform(.1,1,size=r+2);weights/=weights.sum()
  WS=W[S];muS=weights@WS;X=WS-muS;CS=X.T@(weights[:,None]*X)
  c=float(eigvalsh(CS,F)[0]);assert c>0
  observed=np.rint(ell[S]/g2)*g2;response=observed-bias[S]
  hLS=np.linalg.solve(CS,X.T@(weights*response))
  # Projection into the first-message parallelotope using coordinates z=B(h-h0).
  Binv=np.linalg.inv(B)
  def objective(z):
   v=h0+Binv@z-hLS;return .5*v@CS@v
  def jac(z):return Binv.T@CS@(h0+Binv@z-hLS)
  fit=minimize(objective,np.clip(B@(hLS-h0),-g1,g1),jac=jac,
               bounds=[(-g1,g1)]*r,method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-13,'maxiter':1000})
  hp=h0+Binv@fit.x;lp=W@hp+bias;lp-=logsumexp(lp)
  p=np.exp(ell);q=np.exp(lp)
  losses=[float(p@(ell-lp)),float(q@(lp-ell))]
  bound=np.exp(2*r*tau*g1)*g2*g2/(8*c)
  e=observed-ell[S];raw=hLS-h
  assert raw@CS@raw<=g2*g2/4+1e-12
  assert max(losses)<=bound+1e-10
  # Numerically test envelope at every vertex of D, not as a global proof.
  envelope=[]
  import itertools
  for s in itertools.product((-1,1),repeat=r):
   hv=h0+Binv@(g1*np.array(s));pv=np.exp(W@hv+bias-logsumexp(W@hv+bias));m=pv@W
   FV=(W-m).T@(pv[:,None]*(W-m));envelope.append(float(eigvalsh(FV,F)[-1]))
  assert max(envelope)<=np.exp(2*r*tau*g1)+1e-10
  rows.append({'r':r,'case':iteration,'c':c,'bound':bound,'KL':losses,'vertex_curvature_ratio':max(envelope)})
# Common logit row shifts leave p unchanged but break a regression without intercept.
W=np.array([[10.],[11.],[12.]]);h=np.array([.3]);ell=(W@h).ravel();ell-=logsumexp(ell)
naive=np.linalg.lstsq(W,ell,rcond=None)[0]
assert abs(float(naive[0])-.3)>.1
result={'scope':'60 CPU diagnostic/counterexample attempts; no GPU/pretrained run; exact proof audited separately',
        'rows':rows,'no_intercept_counterexample':{'true_h':.3,'naive_h':float(naive[0]),'W':W.tolist()},
        'check_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
pathlib.Path('outputs/adaptive_disclosure_theory_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'cases':len(rows),'largest_KL_to_bound':max(max(x['KL'])/x['bound'] for x in rows),'no_intercept_counterexample':result['no_intercept_counterexample']}))
