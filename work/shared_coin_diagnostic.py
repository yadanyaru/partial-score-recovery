"""Exact shared-coin expectation and correlation-sensitive local loss diagnostic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from scipy.special import logsumexp
ROOT=__import__("runtime_paths").ROOT
P=ROOT/'outputs/SHARED_COIN_PROTOCOL.json';O=ROOT/'outputs/shared_coin_results.json'
BASE=ROOT/'outputs/STOCHASTIC_PRECISION_DIAGNOSTIC_PROTOCOL.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def freeze():
    save(P,dict(source_sha256=sha(Path(__file__)),base_sha256=sha(BASE),cases='All 144 existing CPU cases, no selection',independent_draws=4096,seed=202610080,laws=['independent','shared'],phase=0,shared_expectation='Exact integration over sorted fractional breakpoints; no Monte Carlo',decoder='Same public intercept OLS; no kernel-aware optimization',hypothesis='Error covariance predicts local KL, without assuming shared coins always worse',frozen_before_metrics=True))
def loss(lp,W,bias,decoded):
    lq=decoded@W.T+bias;lq-=logsumexp(lq,axis=1,keepdims=True)
    f=np.sum(np.exp(lp)*(lp-lq),axis=1);b=np.sum(np.exp(lq)*(lq-lp),axis=1)
    return f,b,np.maximum(f,b)
def run():
    pro=json.loads(P.read_text());assert sha(Path(__file__))==pro['source_sha256'];assert sha(BASE)==pro['base_sha256']
    base=json.loads(BASE.read_text());heads={h['id']:h for h in base['heads']};designs={d['head_id']:d for d in base['designs']};rows=[]
    for case in base['cases']:
        head=heads[case['head_id']];des=designs[case['head_id']]
        W=np.array(head['W']);bias=np.array(head['bias']);h=np.array(case['h']);ids=np.array(des['ids']);D=np.array(des['decoder'])[:-1]
        lp=W@h+bias;lp-=logsumexp(lp);d=case['g2']/2;x=lp[ids];k=np.floor(x/d).astype(np.int64);f=x/d-k;m=len(ids)
        prob=np.exp(lp);mu=prob@W;F=(W-mu).T@(prob[:,None]*(W-mu));H=D.T@F@D
        covs={'independent':np.diag(f*(1-f)),'shared':np.minimum(f[:,None],f[None,:])-f[:,None]*f[None,:]}
        breaks=np.unique(np.r_[0.,f,1.]);weights=np.diff(breaks);u=(breaks[1:]+breaks[:-1])/2
        shared=k+(u[:,None]<f)
        rng=np.random.default_rng(pro['seed']+case['case_id']);indep=k+(rng.random((pro['independent_draws'],m))<f)
        laws={}
        for name,codes,wt in [('shared',shared,weights),('independent',indep,np.full(len(indep),1/len(indep)))]:
            decoded=(d*codes-bias[ids])@D.T
            fw,rv,mx=loss(lp,W,bias,decoded)
            prediction=float(.5*d*d*np.sum(H*covs[name]))
            actual=float(wt@fw)
            laws[name]=dict(mean_forward=actual,mean_reverse=float(wt@rv),mean_maximum=float(wt@mx),quadratic_forward=prediction,relative_quadratic_error=abs(prediction-actual)/max(actual,1e-20),evaluations=len(codes),exact_expectation=name=='shared',MC_standard_error_forward=float(np.std(fw,ddof=1)/np.sqrt(len(fw))) if name=='independent' else None)
        rows.append(dict(case_id=case['case_id'],head_id=case['head_id'],rank=head['rank'],mesh=d,charged_scores=m,laws=laws,shared_to_independent_forward= laws['shared']['mean_forward']/laws['independent']['mean_forward']))
    summary=dict(completed=len(rows),shared_larger_forward=sum(x['laws']['shared']['mean_forward']>x['laws']['independent']['mean_forward'] for x in rows),median_shared_to_independent_forward=float(np.median([x['shared_to_independent_forward'] for x in rows])))
    for name in ('shared','independent'):
        summary[name]=dict(median_forward=float(np.median([x['laws'][name]['mean_forward'] for x in rows])),median_relative_quadratic_error=float(np.median([x['laws'][name]['relative_quadratic_error'] for x in rows])),maximum_relative_quadratic_error=max(x['laws'][name]['relative_quadratic_error'] for x in rows))
    save(O,dict(protocol_sha256=sha(P),summary=summary,rows=rows));print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze() if a.freeze else run()
