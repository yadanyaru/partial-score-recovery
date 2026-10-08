"""Frozen zero/common/boundary-phase deterministic rounding witnesses."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json,math,hashlib,argparse
from scipy.special import logsumexp
from explicit_known_grid import head
ROOT=__import__("runtime_paths").ROOT
P=ROOT/'outputs/COMMON_PHASE_SHARED_PROTOCOL.json';O=ROOT/'outputs/common_phase_shared_witness.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(a,g):return a-g*math.floor(a/g+.5)
def freeze():
    P.write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),head_source_sha256=sha(Path(__file__).with_name('explicit_known_grid.py')),ranks=[64,1024,1536],grid_exponents=[1,4],pairs=3,phases=['zero','g/7','-0.4g','endpoint_boundary'],seed=202610095,witnesses=['static_complement','adaptive_full'],frozen_before_metrics=True),indent=2),encoding='utf-8')
def run():
    pro=json.loads(P.read_text());assert pro['source_sha256']==sha(Path(__file__));assert pro['head_source_sha256']==sha(Path(__file__).with_name('explicit_known_grid.py'))
    rows=[]
    for r in pro['ranks']:
        W,D,Q=head(r);V=len(W);eta=3/Q;rng=np.random.default_rng(pro['seed']+r)
        for exponent in pro['grid_exponents']:
            g=2.**(-math.ceil(math.log2(r))-exponent);period=D*D*g;T0=4*(math.log(1e7*(V-2)/g)+g/8);M=math.ceil((T0+g/2)/period)
            for _ in range(pro['pairs']):
                a,b=map(int,rng.choice(V,2,replace=False));N=(W@(W[a]+W[b])).astype(float)/(D*D);Z=(W@(W[a]-W[b])).astype(float)/(D*D);Ad=1-float(W[a]@W[b])/(D*D)
                for label,phi in [('zero',0.),('g/7',g/7),('-0.4g',-.4*g),('endpoint_boundary',canonical(-math.log(2)-g/2,g))]:
                    for kind in pro['witnesses']:
                        delta=g/(16*eta) if kind=='static_complement' else g/100
                        margin=abs(canonical(-math.log(2)-phi,g))
                        s=0. if kind=='static_complement' or margin>=g/8 else g/4
                        reference=delta if kind=='static_complement' else s
                        t=canonical(-math.log(2)-math.log(math.cosh(Ad*reference))-phi-g/2,g);T=M*period+t
                        code=[];ps=[];distance=[];fractions=[]
                        for sign in (-1,1):
                            logits=T*N+(s+sign*delta)*Z;lp=logits-logsumexp(logits);scaled=(lp-phi)/g
                            code.append(np.floor(scaled).astype(np.int64));ps.append(np.exp(lp));frac=scaled-np.floor(scaled);fractions.append(frac);distance.append(np.minimum(frac,1-frac))
                        mask=np.ones(V,dtype=bool)
                        if kind=='static_complement':mask[[a,b]]=False
                        collision=bool(np.array_equal(code[0][mask],code[1][mask]));assert collision
                        intervals=sorted((min(a,b),max(a,b)) for a,b in zip(fractions[0][mask],fractions[1][mask]));total=0.;left=right=None
                        for lo,hi in intervals:
                            if left is None:left,right=lo,hi
                            elif lo<=right:right=max(right,hi)
                            else:total+=right-left;left,right=lo,hi
                        agreement=1-total-(right-left if left is not None else 0);assert agreement>=.08
                        m=(ps[0]+ps[1])/2;js=sum(float(np.sum(p[p>0]*np.log(p[p>0]/m[p>0]))) for p in ps)/2
                        rows.append(dict(rank=r,mesh=g,pair=[a,b],phase_type=label,phase=phi,kind=kind,t=t,tangent_center=s,T=T,collision=collision,common_component_mass=agreement,minimum_boundary_margin_mesh=min(float(x[mask].min()) for x in distance),JS=js))
    O.write_text(json.dumps(dict(protocol_sha256=sha(P),cases=len(rows),collisions=sum(x['collision'] for x in rows),minimum_margin=min(x['minimum_boundary_margin_mesh'] for x in rows),rows=rows),indent=2),encoding='utf-8');print(json.dumps(dict(cases=len(rows),collisions=sum(x['collision'] for x in rows))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze() if a.freeze else run()
