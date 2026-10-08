"""Prespecified distinct-vocabulary finite-field collision witnesses.

The source and protocol are frozen before metrics. This extends vocabulary by
new incoherent character rows; no token is copied. All geometry uses integer
numerators, and all prescribed witnesses are retained.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
from scipy.special import logsumexp

ROOT=__import__("runtime_paths").ROOT
PROTOCOL = ROOT / 'outputs/review_low_disclosure_protocol.json'
OUT = ROOT / 'outputs/review_low_disclosure_results.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def finite_field(Q):
    poly = {32: 0b100101}[Q]
    def multiply(a, b):
        z = 0
        while b:
            if b & 1:
                z ^= a
            b >>= 1
            a <<= 1
            if a & Q:
                a ^= poly
        return z
    mult = np.asarray([[multiply(a,b) for b in range(Q)] for a in range(Q)], dtype=np.int16)
    trace = []
    for a in range(Q):
        z, t = 0, a
        for _ in range(Q.bit_length()-1):
            z ^= t
            t = int(mult[t,t])
        assert z in (0,1)
        trace.append(z)
    return mult, np.asarray(trace, dtype=np.int16)

def head(r, V):
    r0 = 1
    while 4*r0 <= r:
        r0 *= 4
    Q, D = math.isqrt(r0), 5*math.isqrt(r0)
    assert 6*(r+1) <= V <= (Q+1)*r0
    mult, trace = finite_field(Q)
    W = np.zeros((V,r), dtype=np.int16)
    W[np.arange(r0), np.arange(r0)] = D
    xy = mult.copy()
    # Construct one full basis through broadcast field arithmetic.
    uv_y = mult[:,None,None,:]  # v,1,1,y
    u_x = mult[:,None,:,None]   # u,1,x,1
    cursor = r0
    for a in range(Q):
        phase = mult[a,xy][None,None,:,:] ^ u_x ^ uv_y.transpose(1,0,2,3)
        block = (5*(1-2*trace[phase])).reshape(r0,r0)
        take = min(r0, V-cursor)
        W[cursor:cursor+take,:r0] = block[:take]
        cursor += take
        if cursor == V:
            break
    assert cursor == V
    # Row r0 is uniform and remains untouched, proving affine faithfulness.
    for j in range(r0,r):
        i = r0+1+j-r0
        value = int(W[i,0])//5
        W[i,0], W[i,j] = 4*value, 3*value
    norms = np.einsum('ij,ij->i', W, W, dtype=np.int64)
    assert np.all(norms == D*D)
    # Equal hash count proves the concrete bank contains no repeated row.
    row_hashes = {hashlib.sha256(row.tobytes()).digest() for row in W}
    assert len(row_hashes) == V
    return W,D,Q

def canonical(a,g):
    return a-g*math.floor(a/g+.5)

def freeze():
    spec = dict(source_sha256=sha(Path(__file__)), ranks=[1024,1536],
                vocabulary_per_rank='unique ascending {6*(r+1),16*r0,(Q+1)*r0}',
                grid_exponents=[1,4], pairs=3, phases=['zero','g/7','-0.4g','endpoint_boundary'],
                witnesses=['static_complement','adaptive_full'], seed=2026100816,
                K='5*(r+1)', frozen_before_metrics=True,
                checks=['integer row norms','distinct row hashes','sampled pair coherence',
                        'complement/full nearest code equality','JS lower bound','boundary margin'])
    PROTOCOL.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print(str(PROTOCOL))

def run():
    spec = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert spec['source_sha256'] == sha(Path(__file__))
    results, geometry = [], []
    start = time.perf_counter()
    for r in spec['ranks']:
        r0 = 1
        while 4*r0 <= r:
            r0 *= 4
        Q = math.isqrt(r0)
        for V in sorted({6*(r+1),16*r0,(Q+1)*r0}):
            begin = time.perf_counter()
            W,D,Q = head(r,V)
            K,eta = 5*(r+1),3/Q
            pi = (V-K)*(V-K-1)/(V*(V-1))
            rng = np.random.default_rng(spec['seed']+r+V)
            pair_bank = [list(map(int,rng.choice(V,2,replace=False))) for _ in range(spec['pairs'])]
            geometry.append(dict(rank=r,V=V,K=K,K_over_V=K/V,pi=pi,
                                 denominator=D,eta=eta,distinct_rows=V,
                                 unit_norm_integer=True,head_sha256=hashlib.sha256(W.tobytes()).hexdigest(),
                                 construction_seconds=time.perf_counter()-begin))
            for exponent in spec['grid_exponents']:
                g = 2.**(-math.ceil(math.log2(r))-exponent)
                period = D*D*g
                T0 = 4*(math.log(1e7*(V-2)/g)+g/8)
                M = math.ceil((T0+g/2)/period)
                for a,b in pair_bank:
                    N = W@(W[a].astype(np.int64)+W[b].astype(np.int64))
                    Z = W@(W[a].astype(np.int64)-W[b].astype(np.int64))
                    N,Z = N.astype(float)/(D*D),Z.astype(float)/(D*D)
                    inner = float(W[a].astype(np.int64)@W[b].astype(np.int64))/(D*D)
                    assert abs(inner) <= eta
                    Ad = 1-inner
                    for label,phi in [('zero',0.),('g/7',g/7),('-0.4g',-.4*g),
                                      ('endpoint_boundary',canonical(-math.log(2)-g/2,g))]:
                        for kind in spec['witnesses']:
                            delta = g/(16*eta) if kind=='static_complement' else g/100
                            margin = abs(canonical(-math.log(2)-phi-g/2,g))
                            s = 0. if kind=='static_complement' or margin>=g/8 else g/4
                            reference = delta if kind=='static_complement' else s
                            t = canonical(-math.log(2)-math.log(math.cosh(Ad*reference))-phi,g)
                            T = M*period+t
                            codes, ps, distances = [], [], []
                            for sign in (-1,1):
                                logits = T*N+(s+sign*delta)*Z
                                lp = logits-logsumexp(logits)
                                scaled = (lp-phi)/g
                                nearest = np.floor(scaled+.5)
                                codes.append(nearest.astype(np.int64))
                                ps.append(np.exp(lp))
                                distances.append(.5-np.abs(scaled-nearest))
                            mask = np.ones(V,dtype=bool)
                            if kind=='static_complement':
                                mask[[a,b]] = False
                            collision = bool(np.array_equal(codes[0][mask],codes[1][mask]))
                            assert collision
                            mean = (ps[0]+ps[1])/2
                            js = sum(float(np.sum(p[p>0]*np.log(p[p>0]/mean[p>0]))) for p in ps)/2
                            js_lower = 9*delta*delta/512
                            assert js >= js_lower
                            results.append(dict(rank=r,V=V,K=K,K_over_V=K/V,pi=pi,g=g,
                                                pair=[a,b],pair_inner=inner,phase_type=label,phase=phi,
                                                kind=kind,collision=collision,JS=js,JS_lower=js_lower,
                                                JS_over_lower=js/js_lower,
                                                static_risk_lower=pi*9*g*g/(131072*eta*eta),
                                                static_lower_over_rg2=pi*9/(131072*eta*eta*r),
                                                minimum_boundary_margin_mesh=min(float(x[mask].min()) for x in distances),
                                                T=T,t=t,tangent_center=s,radius=2*(M*period+g/2)+2*g/(16*eta)))
            print(json.dumps(dict(rank=r,V=V,K_over_V=K/V,completed=len(results))),flush=True)
    doc = dict(protocol_sha256=sha(PROTOCOL),source_sha256=sha(Path(__file__)),
               seconds=time.perf_counter()-start,cases=len(results),
               collisions=sum(x['collision'] for x in results),
               minimum_margin=min(x['minimum_boundary_margin_mesh'] for x in results),
               minimum_JS_over_lower=min(x['JS_over_lower'] for x in results),
               geometry=geometry,rows=results)
    OUT.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in doc.items() if k not in ['rows','geometry']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    freeze() if parser.parse_args().freeze else run()
