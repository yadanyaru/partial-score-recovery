"""Explicit rational character heads and deterministic grid witnesses; no fitting."""
from pathlib import Path
import numpy as np
import json, math, hashlib
ROOT=__import__("runtime_paths").ROOT
def head(r):
    r0=4**int(math.log(r,4)); Q=math.isqrt(r0)
    poly={8:0b1011,32:0b100101}[Q]
    def mul(a,b):
        z=0
        while b:
            if b&1:z^=a
            b>>=1;a<<=1
            if a&Q:a^=poly
        return z
    def tr(a):
        z=0;t=a
        for _ in range(Q.bit_length()-1):z^=t;t=mul(t,t)
        assert z in (0,1)
        return z
    V=6*(r+1); D=5*Q
    rows=[np.eye(1,r,k=j,dtype=np.int64)[0]*D for j in range(r0)]
    for a in range(Q):
        for u in range(Q):
            for v in range(Q):
                row=np.zeros(r,dtype=np.int64)
                row[:r0]=[5*(1-2*tr(mul(a,mul(x,y))^mul(u,x)^mul(v,y))) for x in range(Q) for y in range(Q)]
                rows.append(row)
                if len(rows)==V:break
            if len(rows)==V:break
        if len(rows)==V:break
    assert len(rows)==V
    for j in range(r0,r):
        i=r0+1+j-r0; value=int(rows[i][0]);assert value in (-5,5)
        rows[i][0]=4*(value//5);rows[i][j]=3*(value//5)
    W=np.asarray(rows);assert np.all(np.sum(W*W,axis=1)==D*D)
    return W,D,Q
def run():
    result=[]
    for r in (64,1024,1536):
        W,D,Q=head(r);V=len(W);eta=3/Q
        # Structural rank follows from unlifted standard rows, the uniform row,
        # and one nonzero new coordinate per lifted row; no numerical rank test.
        rng=np.random.default_rng(20261007+r)
        for g in (2.**(-math.ceil(math.log2(r))-1),2.**(-math.ceil(math.log2(r))-4)):
            phase=g/128*round(-128*math.log(2)/g)
            T0=4*(math.log(1e7*(V-2)/g)+g/8)
            period=D*D*g;T=period*math.ceil(T0/period)
            for _ in range(3):
                a,b=map(int,rng.choice(V,2,replace=False))
                n=W[a]+W[b];v=W[a]-W[b]
                N=(W@n).astype(float)/(D*D);Z=(W@v).astype(float)/(D*D)
                for label,delta in [('static',g/(16*eta)),('all_scores',g/100)]:
                    codes=[]; probs=[]
                    for sign in (-1,1):
                        logits=T*N+sign*delta*Z;m=float(logits.max())
                        ell=logits-m-math.log(float(np.exp(logits-m).sum()))
                        codes.append(np.floor((ell-phase)/g+.5).astype(np.int64));probs.append(np.exp(ell))
                    mask=np.ones(V,dtype=bool)
                    if label=='static':mask[[a,b]]=False
                    collision=bool(np.array_equal(codes[0][mask],codes[1][mask]))
                    mean=(probs[0]+probs[1])/2
                    js=sum(float(np.sum(p[p>0]*np.log(p[p>0]/mean[p>0]))) for p in probs)/2
                    result.append(dict(r=r,V=V,g=g,pair=[a,b],kind=label,collision=collision,JS=js,delta=delta,T=T,radius=2*T+2*g/(16*eta),denominator=D,head_sha256=hashlib.sha256(W.tobytes()).hexdigest()))
                    assert collision
    out=ROOT/'outputs'/'explicit_known_grid_results.json'
    out.write_text(json.dumps(dict(protocol='Prespecified ranks, dyadic grids, three uniform pairs per rank/grid, both witness types; full exact integer unit rows.',cases=result),indent=2))
    print(json.dumps(dict(cases=len(result),collisions=sum(x['collision'] for x in result),output=str(out))))
if __name__=='__main__':run()
