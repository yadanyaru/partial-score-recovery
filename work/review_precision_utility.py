"""Frozen broad-mesh, absolute-risk and LAMBADA utility evaluation of precision selection."""
import os
os.environ.setdefault('HF_HOME', str(__import__('runtime_paths').HF_CACHE))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse, gc, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
import torch
from frozen_models import REVISIONS
from risk_selected_precision_pretrained import public_decoder
from sample_precision_ablation import shared_quadratic

ROOT=__import__("runtime_paths").ROOT
OUT=ROOT/'outputs/review_precision'
PROTOCOL=ROOT/'outputs/REVIEW_PRECISION_UTILITY_PROTOCOL.json'
CACHE=__import__("runtime_paths").HEAD_CACHE
LAWS=['independent','shared','nearest']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sync():torch.cuda.synchronize()
def freeze():
    assert not PROTOCOL.exists(),'Preserve frozen experiment'
    original=json.loads((ROOT/'outputs/RISK_SELECTED_PRECISION_PROTOCOL.json').read_text())
    models=[]
    for name,revision in REVISIONS.items():
        states=ROOT/'outputs/review_native_states'/f"{name.replace('/','__')}.json"
        old=next(x for x in original['models'] if x['model']==name)
        models.append(dict(model=name,revision=revision,state_file=states.relative_to(ROOT).as_posix(),
            state_sha256=sha(states),selected_ids=old['selected_ids'],W_sha256=old['W_sha256'],bias_sha256=old['bias_sha256']))
    save(PROTOCOL,dict(source_sha256=sha(__file__),decoder_source_sha256=sha(ROOT/'work/risk_selected_precision_pretrained.py'),
        pair_source_sha256=sha(ROOT/'work/sample_precision_ablation.py'),models=models,
        corpus_sha256=sha(ROOT/'work/review_lambada_corpus.json'),paragraphs=list(range(128)),
        grids=[.002,.02,.2,1.,4.],mesh='d=g/2',pairs=512,rounding_draws=128,seed=2026100817,
        laws=LAWS,selection='min sampled second moment before fresh rounding coins; ties by law order',
        receiver='fixed previous public head-only leverage IDs; intercept-aware LS; unweighted',
        teacher='float64 public head on native FP32 final states; all answer-prefix states',
        task='LAMBADA answer-token sequence: expected exact answer-token match and teacher-forced answer NLL',
        task_probability='response coins independent across answer-prefix states; exact-match expectation is product of conditional next-token correctness probabilities',
        metrics=['forward_KL','reverse_KL','maximum_KL','TV','teacher_top20_complement_mass_absolute_error',
                 'gold_token_NLL','gold_token_accuracy','teacher_top1_agreement'],
        uncertainty='128 rounding draws per stochastic candidate; paired passage bootstrap, all128 passages',
        payload='same chosen IDs, lattice and public fixed-width score fields for all kernels; actual repeats charged',
        timing='CUDA synchronized wall-clock selection/encoding; profile first8 contexts atg=.02,5 repetitions; dense first2,3repetitions',
        frozen_before_metrics=True,failure_policy='all passages/meshes/prefixes retained; no tuning on evaluation outcomes'))
    print(json.dumps(dict(frozen=True,passages=128,grids=5,models=3,pairs=512,draws=128)))

def choose(p,W,decoder,f,error,pairs):
    a=(W[pairs[:,0]]-W[pairs[:,1]])@decoder
    estimates=torch.stack([a.square()@(f*(1-f)),shared_quadratic(a,f),(a@error).square()],1).mean(0)
    return int(torch.argmin(estimates)),estimates

def measure(fn,reps):
    fn();sync();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    times=[]
    for _ in range(reps):
        sync();t=time.perf_counter();fn();sync();times.append(1000*(time.perf_counter()-t))
    return dict(milliseconds=times,median_ms=float(np.median(times)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),baseline_allocated_bytes=int(base),
        additional_peak_bytes=int(torch.cuda.max_memory_allocated()-base))

def evaluate(errors,teacher_lp,p,W,decoder,gold,top20,base_delta):
    # Differences in logits avoid cancellation from the large common score intercept.
    values={k:[] for k in ['forward','reverse','maximum','TV','tail_mass_error','gold_NLL','gold_accuracy','teacher_top1_agreement']}
    teacher_top1=int(torch.argmax(p));teacher_topmass=p[top20].sum()
    for start in range(0,len(errors),16):
        dz=(errors[start:start+16]@decoder.T+base_delta)@W.T
        lq=torch.log_softmax(teacher_lp[None,:]+dz,dim=1);q=lq.exp()
        forward=(p*(teacher_lp-lq)).sum(1);reverse=(q*(lq-teacher_lp)).sum(1)
        prediction=torch.argmax(lq,dim=1)
        batch=dict(forward=forward,reverse=reverse,maximum=torch.maximum(forward,reverse),
            TV=.5*(q-p).abs().sum(1),tail_mass_error=(q[:,top20].sum(1)-teacher_topmass).abs(),
            gold_NLL=-lq[:,gold],gold_accuracy=(prediction==gold).double(),
            teacher_top1_agreement=(prediction==teacher_top1).double())
        for k,v in batch.items():values[k].append(v)
    result={}
    for k,chunks in values.items():
        v=torch.cat(chunks);result[k]=float(v.mean());result[k+'_MC_SE']=float(v.std(unbiased=True)/len(v)**.5) if len(v)>1 else 0.
    result['draws']=len(errors)
    return result

def run_one(plan,protocol):
    name=plan['model'];slug=name.replace('/','__');dest=OUT/f'{slug}.json'
    if dest.exists():
        old=json.loads(dest.read_text());assert old['metadata']['completed'];return
    source=ROOT/plan['state_file'];assert sha(source)==plan['state_sha256']
    states=json.loads(source.read_text());cache=Path(states['metadata']['head_cache']);assert sha(cache)==states['metadata']['head_cache_sha256']
    z=np.load(cache);wc=z['W'].astype(np.float64);bc=z['bias'].astype(np.float64)
    assert hashlib.sha256(wc.tobytes()).hexdigest()==plan['W_sha256'];assert hashlib.sha256(bc.tobytes()).hexdigest()==plan['bias_sha256']
    W=torch.tensor(wc,device='cuda');bias=torch.tensor(bc,device='cuda');del wc,bc,z;gc.collect()
    ids=torch.tensor(plan['selected_ids'],device='cuda');decoder,_,design=public_decoder(W,ids)
    result=dict(metadata=dict(model=name,revision=plan['revision'],protocol_sha256=sha(PROTOCOL),
        source_sha256=sha(__file__),completed=False,rank=W.shape[1],V=len(W),selected_count=len(ids),
        disclosure_fraction=len(ids)/len(W),design=design,hardware=torch.cuda.get_device_name(),
        torch_version=torch.__version__,precision='float64',chosen_ids=plan['selected_ids']),rows=[],failures=[],timings=[])
    replay={};started=time.perf_counter()
    for item in states['rows']:
        j=item['paragraph']
        for step,gold in enumerate(item['gold_ids']):
            try:
                h=torch.tensor(item['answer_hidden_states'][step],device='cuda',dtype=torch.float64)
                lp=torch.log_softmax(W@h+bias,0);p=lp.exp();selected=lp[ids]
                base_delta=(selected-bias[ids])@decoder.T-h
                top20=torch.topk(p,20).indices
                seed=protocol['seed']+sum(name.encode())*100000+j*100+step
                gen=torch.Generator(device='cuda');gen.manual_seed(seed)
                pairs=torch.multinomial(p,2*protocol['pairs'],replacement=True,generator=gen).reshape(-1,2)
                # Response randomness uses a separate generator; same uniforms compare meshes.
                rg=torch.Generator(device='cuda');rg.manual_seed(seed+100000000)
                independent_u=torch.rand((protocol['rounding_draws'],len(ids)),device='cuda',dtype=torch.float64,generator=rg)
                shared_u=torch.rand((protocol['rounding_draws'],1),device='cuda',dtype=torch.float64,generator=rg)
                replay[f'pairs_{j}_{step}']=pairs.cpu().numpy()
                replay[f'shared_uniform_{j}_{step}']=shared_u.cpu().numpy()
                a=(W[pairs[:,0]]-W[pairs[:,1]])@decoder
                for gi,g in enumerate(protocol['grids']):
                    d=g/2;scaled=selected/d;floor=torch.floor(scaled);f=scaled-floor;en=torch.round(scaled)-scaled
                    Q=torch.stack([a.square()@(f*(1-f)),shared_quadratic(a,f),(a@en).square()],1).mean(0)*d*d/4
                    choice=int(torch.argmin(Q));policy=LAWS[choice]
                    up=independent_u<f
                    replay[f'independent_up_{j}_{step}_{gi}']=np.packbits(up.cpu().numpy(),axis=1)
                    errors={'nearest':en[None,:]*d,'independent':(up.double()-f)*d,'shared':((shared_u<f).double()-f)*d}
                    laws={law:evaluate(errors[law],lp,p,W,decoder,gold,top20,base_delta) for law in LAWS}
                    laws['selected']=dict(laws[policy],selected_kernel=policy)
                    radius=states['metadata']['public_state_radius'];extent=2*radius*float(torch.linalg.vector_norm(W,dim=1).max())+float(bias.max()-bias.min())+math.log(len(W))
                    bits=math.ceil(math.log2(2*math.ceil(extent/d+1)+1))+math.ceil(math.log2(len(W)))
                    result['rows'].append(dict(paragraph=j,source_row=item['source_row'],answer_step=step,gold_id=gold,
                        grid=g,mesh=d,status='completed',laws=laws,estimated_Q=Q.cpu().tolist(),selected_kernel=policy,
                        seed=seed,exact_score_decoder_residual_norm=float(torch.linalg.vector_norm(base_delta)),
                        teacher_gold_NLL=float(-lp[gold]),teacher_correct=int(torch.argmax(lp))==gold,
                        teacher_entropy=float(-(p*lp).sum()),packet_bits=len(ids)*bits,charged_scores=len(ids)))
                    if step==0 and j<8 and g==.02:
                        def near():return torch.round(selected/d)*d
                        def sampled():
                            pair=torch.multinomial(p,2*protocol['pairs'],replacement=True).reshape(-1,2)
                            selected_index,_=choose(p,W,decoder,f,en,pair)
                            if selected_index==2:return torch.round(selected/d)*d
                            u=torch.rand((1 if selected_index==1 else len(ids)),device='cuda',dtype=torch.float64)
                            return d*(floor+(u<f))
                        timing=dict(paragraph=j,grid=g,nearest_encode=measure(near,5),sample512_select_encode=measure(sampled,5))
                        if j<2:
                            def dense():
                                mu=p@W;centered=W-mu;F=centered.T@(p[:,None]*centered);H=decoder.T@F@decoder
                                C=torch.minimum(f[:,None],f[None,:])-f[:,None]*f[None,:]
                                return torch.stack([(torch.diag(H)*f*(1-f)).sum(),(H*C).sum(),en@H@en])
                            timing['dense_select']=measure(dense,3)
                        result['timings'].append(timing)
                    del errors,laws
                del a,independent_u,shared_u,pairs
            except Exception:
                result['failures'].append(dict(paragraph=j,answer_step=step,traceback=traceback.format_exc()))
        if (j+1)%8==0:
            save(dest,result);print(json.dumps(dict(model=name,passages=j+1,rows=len(result['rows']),failures=len(result['failures']),elapsed=time.perf_counter()-started)),flush=True)
    replaypath=CACHE/f'{slug}_precision_replay.npz';np.savez_compressed(replaypath,**replay)
    result['metadata'].update(completed=len(result['failures'])==0 and len(result['rows'])==5*sum(len(x['gold_ids']) for x in states['rows']),
        elapsed=time.perf_counter()-started,replay_file=str(replaypath),replay_sha256=sha(replaypath),
        prescribed_rows=5*sum(len(x['gold_ids']) for x in states['rows']))
    save(dest,result);print(json.dumps(dict(model=name,completed=result['metadata']['completed'],rows=len(result['rows']))),flush=True)
    del W,bias,decoder,result,replay;gc.collect();torch.cuda.empty_cache()

def main():
    args=argparse.ArgumentParser();args.add_argument('--freeze',action='store_true');args.add_argument('--run',action='store_true');x=args.parse_args()
    if x.freeze:freeze();return
    spec=json.loads(PROTOCOL.read_text());assert spec['source_sha256']==sha(__file__)
    assert spec['decoder_source_sha256']==sha(ROOT/'work/risk_selected_precision_pretrained.py')
    assert spec['pair_source_sha256']==sha(ROOT/'work/sample_precision_ablation.py')
    torch.set_num_threads(4)
    with torch.inference_mode():
        for plan in spec['models']:run_one(plan,spec)

if __name__=='__main__':main()
