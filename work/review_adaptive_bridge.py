"""Matched-budget pretrained adaptive disclosure, fixed before final metrics.

Only observer-visible pilot messages enter adaptive design. Native teacher
states are used by the simulator and for post-reconstruction diagnostics.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import qr,eigh,eigvalsh
from review_box_qp import project_box
from frozen_models import REVISIONS

ROOT=__import__("runtime_paths").ROOT
PROTOCOL=ROOT/'outputs/REVIEW_ADAPTIVE_PROTOCOL.json'
OUT=ROOT/'outputs/review_adaptive'
ARMS=('static_uniform_5n','static_zero_leverage_5n','static_doptimal_5n',
      'static_doptimal_pilot_4n','adaptive_fisher_pilot_4n')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8'); tmp.replace(path)
def sync(): torch.cuda.synchronize()
def timed(fn):
    sync(); t=time.perf_counter(); value=fn(); sync(); return value,time.perf_counter()-t

def freeze():
    models=[]
    for name,revision in REVISIONS.items():
        path=ROOT/'outputs/review_native_states'/f"{name.replace('/','__')}.json"
        meta=json.loads(path.read_text(encoding='utf8'))['metadata']
        models.append(dict(model=name,revision=revision,state_file=str(path),state_sha256=sha(path),
                           head_cache=meta['head_cache'],head_cache_sha256=meta['head_cache_sha256']))
    save(PROTOCOL,dict(source_sha256=sha(__file__),projection_source_sha256=sha(Path(__file__).with_name('review_box_qp.py')),frozen_before_metrics=True,models=models,
         paragraphs=list(range(64)),grids=[.0005,.02,1.0],arms=list(ARMS),seed=2026100817,
         budget='5(r+1) returned scalar requests, repeated IDs charged',
         pilot='n=r+1 seeded public 4n-pool pivoted QR affine rows; nearest rounding with same g as fine stage',
         static_full='5n fine draws without observing pilot; weighted free-intercept LS',
         pilot_arms='n common pilot scores plus4n fine draws; weighted LS followed by pilot-box metric projection',
         pilot_box='|B(h-h0)|<=g: every true teacher is feasible under nearest rounded contrast scores',
         zero_leverage='full-vocabulary h=0 augmented Fisher leverage, target softmax(b)',
         adaptive='full-vocabulary h0 augmented Fisher leverage, target softmax(W h0+b); h0 uses only pilot responses',
         fisher_regularization='1e-10*trace(F)/r identity for proposal computation only; regression uses unregularized Gram',
         static_doptimal='fixed 8n public candidate pool;20 multiplicative D-optimal weight updates; draws from resulting distribution; unweighted LS',
         selection='with replacement common uniforms per paragraph/arm budget, duplicate scalars fully charged',
         projection_options=dict(tolerance=1e-9,maxiter=10000),
         projection_solver='Lawson-Hanson nonnegative quadratic dual active-set; diagonal-normalized constraints; z/g unit box; primal feasibility and dual multipliers recorded',
         smoke='first existing confirmation context or new context0, output separate; excluded from final evidence',
         timing='three repeat designs for first context; three repeated decoding calls first context each arm; CUDA synchronized',
         diagnostics='actual unregularized Fisher, selected covariance generalized eigenvalues, pilot logit oscillation, condition numbers, absolute both KL, tail mass and top-token fidelity',
         diagnostic_implementation='cache Fisher eigen-whitening once per paragraph/anchor; CPU SciPy symmetric eigenvalue diagnostics; no effect on designs, messages, decoder or losses',
         storage='fineID and importanceweight arrays stored losslessly in compressed per-cell NPZ with SHA256; JSON holds references and numerical summaries',
         theorem_status='numerical public proposals; no BSS or maximum-determinant certificate asserted',
         failure_policy='all576 prescribed cells retained, failed arm recorded explicitly; no outcome-dependent redesign'))
    print(json.dumps(dict(frozen=True,planned_cells=576,protocol_sha256=sha(PROTOCOL))))

def geometry(W,bias,h):
    lp=torch.log_softmax(W@h+bias,0); p=lp.exp(); mu=p@W
    F=torch.zeros((W.shape[1],W.shape[1]),device=W.device,dtype=W.dtype)
    for begin in range(0,len(W),4096):
        X=W[begin:begin+4096]-mu; F.addmm_(X.T,p[begin:begin+4096,None]*X)
    return lp,p,mu,(F+F.T)/2

def leverage(W,p,mu,F):
    ridge=1e-10*torch.trace(F)/F.shape[0]
    L=torch.linalg.cholesky(F+ridge*torch.eye(F.shape[0],device=W.device,dtype=W.dtype))
    values=[]
    for begin in range(0,len(W),4096):
        X=W[begin:begin+4096]-mu
        Z=torch.linalg.solve_triangular(L,X.T,upper=False)
        values.append(p[begin:begin+4096]*(1+Z.square().sum(0)))
    proposal=torch.cat(values); proposal/=proposal.sum()
    return proposal,dict(ridge=float(ridge),fisher_trace=float(torch.trace(F)))

def doptimal(W,pool,iterations):
    X=W[pool]; center=X.mean(0); Z=X-center
    G=Z.T@Z/len(Z); L=torch.linalg.cholesky(G)
    Y=torch.linalg.solve_triangular(L,Z.T,upper=False).T
    A=torch.cat((Y,torch.ones((len(Y),1),device=W.device,dtype=W.dtype)),1)
    weights=torch.full((len(A),),1/len(A),device=W.device,dtype=W.dtype)
    for _ in range(iterations):
        M=A.T@(weights[:,None]*A); C=torch.linalg.cholesky(M)
        Z1=torch.linalg.solve_triangular(C,A.T,upper=False)
        lv=Z1.square().sum(0)
        weights*=lv/A.shape[1]; weights/=weights.sum()
    C=torch.linalg.cholesky(A.T@(weights[:,None]*A))
    lv=torch.linalg.solve_triangular(C,A.T,upper=False).square().sum(0)
    return weights,dict(iterations=iterations,candidate_count=len(pool),max_candidate_augmented_leverage=float(lv.max()),
                        rank_plus_one=A.shape[1],scope='candidate-pool approximate D-design; numerical heuristic')

def draw(proposal,target,count,uniforms):
    cdf=torch.cumsum(proposal,0); cdf[-1]=1
    ids=torch.searchsorted(cdf,uniforms[:count]).clamp_max(len(proposal)-1)
    raw=target[ids]/proposal[ids]/count; weights=raw/raw.sum()
    return ids,weights

def regress(W,bias,ids,weights,received):
    X=W[ids]; mu=weights@X; C=X-mu
    G=C.T@(weights[:,None]*C); G=(G+G.T)/2
    L=torch.linalg.cholesky(G)
    D=torch.cholesky_solve(C.T*weights[None,:],L)
    h=D@(received-bias[ids])
    return h,G,D

def project(h,G,h0,B,Binv,g,options):
    zls=B@(h-h0)
    if float(zls.abs().max())<=g:
        return h,dict(active=False,success=True,iterations=0,active_constraints=0,box_violation=0.,
                      stationarity_infinity_norm=0.,relative_stationarity=0.,absolute_duality_gap=0.,relative_duality_gap=0.)
    C=B@torch.linalg.solve(G,B.T); C=(C+C.T)/2
    zn,meta=project_box(zls.cpu().numpy(),C.cpu().numpy(),g,**options)
    z=torch.tensor(zn,device=h.device,dtype=h.dtype)
    return h0+Binv@z,meta

def metrics(lp,lq,gold):
    p,q=lp.exp(),lq.exp(); tail=torch.ones(len(p),device=p.device,dtype=torch.bool)
    tail[torch.topk(p,20).indices]=False
    return dict(forward=float(p@(lp-lq)),reverse=float(q@(lq-lp)),
                maximum=float(torch.maximum(p@(lp-lq),q@(lq-lp))),
                total_variation=float(.5*(p-q).abs().sum()),
                tail_mass_absolute_error=float((q[tail].sum()-p[tail].sum()).abs()),
                top1_teacher=int(p.argmax()),top1_decoded=int(q.argmax()),
                top1_agreement=int(p.argmax()==q.argmax()),
                first_token_gold=gold,gold_logprob_teacher=float(lp[gold]),gold_logprob_decoded=float(lq[gold]))

def relative_frame(F):
    e,U=eigh(F.cpu().numpy(),check_finite=False); ridge=1e-12*float(e[-1]); vals=np.maximum(e,ridge)
    root=U*vals[None,:]**-.5
    return root,dict(fisher_min_eigenvalue=float(e[0]),fisher_max_eigenvalue=float(e[-1]),
                fisher_condition=float(e[-1]/e[0]) if float(e[0])>0 else None,
                comparison_eigenvalue_floor=ridge)

def relative_geometry(frame,G):
    root,meta=frame;ratio=root.T@G@root;ev=eigvalsh((ratio+ratio.T)/2,check_finite=False)
    return dict(meta,relative_min=float(ev[0]),relative_max=float(ev[-1]))

def run(plan,protocol,smoke=False):
    slug=plan['model'].replace('/','__'); path=OUT/(f'{slug}_smoke.json' if smoke else f'{slug}.json')
    previous=json.loads(path.read_text(encoding='utf8')) if path.exists() else None
    if previous and previous['metadata'].get('completed'):raise FileExistsError(path)
    raw=json.loads(Path(plan['state_file']).read_text(encoding='utf8'))
    assert sha(plan['state_file'])==plan['state_sha256']
    head=np.load(plan['head_cache']); Wcpu=np.asarray(head['W'],dtype=np.float64)
    W=torch.tensor(Wcpu,device='cuda',dtype=torch.float64)
    bias=torch.tensor(np.asarray(head['bias'],dtype=np.float64),device='cuda')
    V,r=W.shape; n=r+1; rng=np.random.default_rng(protocol['seed'])
    pool=rng.choice(V,min(8*n,V),replace=False)
    # Public pivoted QR; no teacher labels, hidden states or responses supplied.
    qrpool=pool[:4*n]; ref=int(qrpool[0]); C=(Wcpu[qrpool]-Wcpu[ref]).T
    _,_,piv=qr(C,pivoting=True,mode='economic'); pilot=np.r_[ref,qrpool[piv[:r]]]
    assert len(np.unique(pilot))==n
    pilotids=torch.tensor(pilot,device='cuda'); B=W[pilotids[1:]]-W[pilotids[0]]; Binv=torch.linalg.inv(B)
    zlp,zp,zmu,zF=geometry(W,bias,torch.zeros(r,device='cuda',dtype=torch.float64))
    (zq,zmeta),zseconds=timed(lambda:leverage(W,zp,zmu,zF))
    (dq,dm),dseconds=timed(lambda:doptimal(W,torch.tensor(pool,device='cuda'),20))
    denseq=torch.zeros(V,device='cuda',dtype=torch.float64); denseq[torch.tensor(pool,device='cuda')]=dq
    uniform=torch.full((V,),1/V,device='cuda',dtype=torch.float64)
    result=dict(metadata=dict(model=plan['model'],revision=plan['revision'],protocol_sha256=sha(PROTOCOL),
                source_sha256=sha(__file__),rank=r,V=V,score_budget=5*n,K_over_V=5*n/V,
                completed=False,smoke=smoke,pilot_ids=pilot.tolist(),pilot_condition=float(torch.linalg.cond(B)),
                zero_leverage=zmeta,doptimal=dm,static_design_seconds=dict(zero_leverage=zseconds,doptimal=dseconds),
                timing_repeats=[],static_setup_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                device=torch.cuda.get_device_name(0)),rows=[],failures=[])
    prior_elapsed=0.
    if previous:
        result=previous;meta=result['metadata'];prior_elapsed=meta['elapsed_seconds']
        meta.setdefault('resume_history',[]).append(dict(source_sha256=meta['source_sha256'],
             protocol_sha256=meta['protocol_sha256'],preserved_cells=len(result['rows']),
             elapsed_seconds=prior_elapsed,reason='diagnostic-only eigen caching/CPU amendment; existing outcomes reused'))
        meta.update(source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL))
        for row in result['rows']:row.setdefault('diagnostic_version','GPU_v1')
    done={(row['paragraph'],row['grid']) for row in result['rows']}
    save(path,result); torch.cuda.reset_peak_memory_stats()
    paragraphs=[0] if smoke else protocol['paragraphs']; start=time.perf_counter()
    for paragraph in paragraphs:
        if all((paragraph,g) in done for g in protocol['grids']):continue
        state=raw['rows'][paragraph]; h=torch.tensor(state['hidden_state'],device='cuda',dtype=torch.float64)
        lp,p,mu,F=geometry(W,bias,h); gold=int(state['gold_ids'][0]);Fframe=relative_frame(F)
        for g in protocol['grids']:
            if (paragraph,g) in done:continue
            cell=dict(paragraph=paragraph,grid=g,status='complete',arms={},diagnostic_version='CPU_cached_v2');packet={}
            try:
                recv=g*torch.round(lp[pilotids]/g)
                corrected=recv-bias[pilotids]; h0=Binv@(corrected[1:]-corrected[0])
                (anchor_lp,ap,amu,AF),fseconds=timed(lambda:geometry(W,bias,h0))
                (aq,ameta),aseconds=timed(lambda:leverage(W,ap,amu,AF))
                AFframe=relative_frame(AF)
                actualshift=W@(h0-h)
                cell.update(pilot_forward=float(p@(lp-anchor_lp)),pilot_reverse=float(ap@(anchor_lp-lp)),
                  pilot_hidden_l2_error=float(torch.linalg.vector_norm(h0-h)),pilot_logit_oscillation=float(actualshift.max()-actualshift.min()),
                  pilot_box_truth_violation=max(0,float((B@(h-h0)).abs().max())-g),
                  adaptive_design_seconds=aseconds+fseconds,adaptive_leverage_seconds=aseconds,
                  adaptive_Fisher_seconds=fseconds,adaptive_geometry=ameta)
                generator=torch.Generator(device='cuda').manual_seed(protocol['seed']+paragraph)
                u=torch.rand(5*n,generator=generator,device='cuda',dtype=torch.float64)
                for arm in ARMS:
                    try:
                        withpilot='pilot' in arm; count=4*n if withpilot else 5*n
                        if arm.startswith('adaptive'): proposal,target=aq,ap
                        elif 'zero_leverage' in arm: proposal,target=zq,zp
                        elif 'doptimal' in arm: proposal,target=denseq,denseq
                        else: proposal,target=uniform,uniform
                        ids,weights=draw(proposal,target,count,u); received=g*torch.round(lp[ids]/g)
                        def decode():
                            hh,GG,DD=regress(W,bias,ids,weights,received)
                            pp=dict(active=False,success=True,iterations=0,box_violation=0.)
                            if withpilot: hh,pp=project(hh,GG,h0,B,Binv,g,protocol['projection_options'])
                            return hh,GG,DD,pp
                        (estimate,G,D,pr),seconds=timed(decode)
                        lq=torch.log_softmax(W@estimate+bias,0)
                        dh=estimate-h; shift=W@dh
                        diagnostic_begin=time.perf_counter();Gcpu=G.cpu().numpy();Geig=eigvalsh(Gcpu,check_finite=False)
                        anchor_cov=relative_geometry(AFframe,Gcpu);teacher_cov=relative_geometry(Fframe,Gcpu)
                        diagnostic_seconds=time.perf_counter()-diagnostic_begin
                        out=dict(metrics=metrics(lp,lq,gold),fine_draw_count=count,pilot_count=n if withpilot else 0,
                          scalar_request_count=count+(n if withpilot else 0),unique_fine_ids=int(torch.unique(ids).numel()),
                          minimum_weight=float(weights.min()),maximum_weight=float(weights.max()),condition=float(Geig[-1]/Geig[0]),
                          relative_anchor_covariance=anchor_cov,relative_teacher_covariance=teacher_cov,diagnostic_seconds=diagnostic_seconds,
                          decoded_quadratic_forward=float(.5*dh@F@dh),decoded_logit_oscillation=float(shift.max()-shift.min()),
                          projection=pr,decode_seconds=seconds)
                        if not pr['success']: out['status']='projection_iteration_limit'
                        else: out['status']='complete'
                        packet[f'{arm}_ids']=ids.cpu().numpy().astype(np.int32)
                        packet[f'{arm}_weights']=weights.cpu().numpy()
                        cell['arms'][arm]=out
                        if paragraph==0:
                            repeats=[]
                            for _ in range(3): _,t=timed(decode); repeats.append(t)
                            result['metadata']['timing_repeats'].append(dict(grid=g,operation='decode',arm=arm,seconds=repeats))
                    except Exception:
                        cell['arms'][arm]=dict(status='failed',traceback=traceback.format_exc())
                        result['failures'].append(dict(paragraph=paragraph,grid=g,arm=arm,traceback=traceback.format_exc()))
                if paragraph==0:
                    times=[]
                    for _ in range(3): _,t=timed(lambda:leverage(W,ap,amu,AF)); times.append(t)
                    result['metadata']['timing_repeats'].append(dict(grid=g,operation='adaptive_leverage_excluding_Fisher',seconds=times))
            except Exception:
                cell.update(status='failed',traceback=traceback.format_exc());result['failures'].append(cell.copy())
            if packet:
                packetpath=OUT/'packets'/f'{slug}_p{paragraph}_g{g}.npz';packetpath.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(packetpath,**packet)
                cell.update(packet_file=str(packetpath.relative_to(ROOT)),packet_sha256=sha(packetpath))
            result['rows'].append(cell); result['metadata'].update(elapsed_seconds=prior_elapsed+time.perf_counter()-start,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            save(path,result)
            print(json.dumps(dict(model=slug,paragraph=paragraph,grid=g,arms=len(cell['arms']),elapsed=result['metadata']['elapsed_seconds'])),flush=True)
    result['metadata']['completed']=True; save(path,result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--model');args=parser.parse_args()
    if args.freeze: return freeze()
    protocol=json.loads(PROTOCOL.read_text(encoding='utf8'));assert sha(__file__)==protocol['source_sha256']
    assert sha(Path(__file__).with_name('review_box_qp.py'))==protocol['projection_source_sha256']
    plan=next(x for x in protocol['models'] if x['model']==args.model)
    run(plan,protocol,args.smoke)
if __name__=='__main__':main()
