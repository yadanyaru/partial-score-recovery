"""Prespecified exact-kernel coupling witnesses; no reconstruction tuning."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,math,hashlib,argparse
import numpy as np
from scipy.special import logsumexp
from explicit_known_grid import head
ROOT=__import__("runtime_paths").ROOT
P=ROOT/'outputs/EXPLICIT_SHARED_COIN_PROTOCOL.json';O=ROOT/'outputs/explicit_shared_coin_witness.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    P.write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),head_source_sha256=sha(Path(__file__).with_name('explicit_known_grid.py')),ranks=[64,1024,1536],grid_exponents=[1,4],pairs_per_rank_grid=3,seed=202610081,witnesses=['static_complement','adaptive_full_vector'],quantity='Probability that a common-U coupling produces identical observed code vectors',frozen_before_metrics=True),indent=2))
def union_length(intervals):
    total=0.;left=right=None
    for a,b in sorted(intervals):
        if left is None:left,right=a,b
        elif a<=right:right=max(right,b)
        else:total+=right-left;left,right=a,b
    return total+(right-left if left is not None else 0)
def run():
    protocol=json.loads(P.read_text());assert protocol['source_sha256']==sha(Path(__file__));assert protocol['head_source_sha256']==sha(Path(__file__).with_name('explicit_known_grid.py'))
    rows=[]
    for r in protocol['ranks']:
        W,D,Q=head(r);V=len(W);eta=3/Q;rng=np.random.default_rng(protocol['seed']+r)
        for exponent in protocol['grid_exponents']:
            d=2.**(-math.ceil(math.log2(r))-exponent);a=d/128*round(-128*math.log(2)/d);phi=a-d/2
            T0=4*(math.log(1e7*(V-2)/d)+d/8);period=D*D*d;T=period*math.ceil(T0/period)
            for _ in range(protocol['pairs_per_rank_grid']):
                i,j=map(int,rng.choice(V,2,replace=False));N=(W@(W[i]+W[j])).astype(float)/(D*D);Z=(W@(W[i]-W[j])).astype(float)/(D*D)
                for kind in protocol['witnesses']:
                    delta=d/(16*eta) if kind=='static_complement' else d/100
                    mask=np.ones(V,dtype=bool)
                    if kind=='static_complement':mask[[i,j]]=False
                    lows=[];fractions=[];ps=[]
                    for sign in (-1,1):
                        logits=T*N+sign*delta*Z;lp=logits-logsumexp(logits);ps.append(np.exp(lp));scaled=(lp[mask]-phi)/d;lo=np.floor(scaled).astype(np.int64);lows.append(lo);fractions.append(scaled-lo)
                    assert np.array_equal(lows[0],lows[1])
                    intervals=[(min(a,b),max(a,b)) for a,b in zip(*fractions)]
                    agreement=1-union_length(intervals)
                    assert agreement>=.5
                    mean=(ps[0]+ps[1])/2;js=sum(float(np.sum(p[p>0]*np.log(p[p>0]/mean[p>0]))) for p in ps)/2
                    rows.append(dict(rank=r,mesh=d,pair=[i,j],kind=kind,observed_count=int(mask.sum()),common_component_mass_lower_bound=agreement,JS=js,common_component_weighted_JS=agreement*js,fraction_min=min(float(f.min()) for f in fractions),fraction_max=max(float(f.max()) for f in fractions)))
    O.write_text(json.dumps(dict(protocol_sha256=sha(P),cases=len(rows),minimum_common_mass=min(x['common_component_mass_lower_bound'] for x in rows),rows=rows),indent=2));print(json.dumps(dict(cases=len(rows),minimum_common_mass=min(x['common_component_mass_lower_bound'] for x in rows))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze() if a.freeze else run()
