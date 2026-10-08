"""Frozen teacher-pair implementation ablation, reusing all confirmation cells."""
import os
os.environ.setdefault('HF_HOME', str(__import__('runtime_paths').HF_CACHE))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse, gc, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from risk_selected_precision_pretrained import public_decoder

ROOT=__import__("runtime_paths").ROOT
OUT=ROOT/'outputs'; PROTOCOL=OUT/'SAMPLED_PRECISION_PROTOCOL.json'
LAWS=['independent_unbiased_stochastic','known_shared_coin_exact','nearest']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def shared_quadratic(a,f):
    """Exact E_U[(a dot (1{U<f}-f))^2], sorted fractions, batched rows."""
    sorted_f,order=torch.sort(f)
    b=a[:,order]
    prefix=torch.cat([torch.zeros((len(a),1),device=a.device,dtype=a.dtype),b.cumsum(1)],1)
    tail=b.sum(1)[:,None]-prefix
    breaks=torch.cat([torch.zeros(1,device=a.device,dtype=a.dtype),sorted_f,torch.ones(1,device=a.device,dtype=a.dtype)])
    widths=breaks[1:]-breaks[:-1]
    return ((tail-(a@f)[:,None]).square()*widths).sum(1)
def freeze():
    assert not PROTOCOL.exists(),'Preserve frozen protocol'
    old=json.loads((OUT/'RISK_SELECTED_PRECISION_PROTOCOL.json').read_text())
    records=[]
    for plan in old['models']:
        source=OUT/'risk_selected_precision'/f"{plan['model'].replace('/','__')}.json"
        records.append(dict(plan=plan,source=source.relative_to(ROOT).as_posix(),sha256=sha(source)))
    save(PROTOCOL,dict(source_sha256=sha(__file__),decoder_source_sha256=sha(ROOT/'work'/'risk_selected_precision_pretrained.py'),models=records,sample_counts=[32,128,512],provider_replicates=8,seed=2026100801,laws=LAWS,scope='implementation ablation on existing 96 confirmation cells; no new heldout confirmation',selection='empirical pairwise quadratic risk only; never measured KL',coins='provider pairs independent of later rounding coins; conditional risk taken from preserved original candidate ledger',all_cells_retained=True))
    print('Frozen 96 cells x 3 pair budgets x 8 provider replicates.')
def run():
    protocol=json.loads(PROTOCOL.read_text());assert sha(__file__)==protocol['source_sha256']
    assert sha(ROOT/'work'/'risk_selected_precision_pretrained.py')==protocol['decoder_source_sha256']
    torch.set_num_threads(4)
    for mi,item in enumerate(protocol['models']):
        plan=item['plan'];name=plan['model'];start=time.perf_counter()
        destination=OUT/'sampled_precision'/f"{name.replace('/','__')}.json"
        assert not destination.exists(),f'Preserve existing ablation: {destination}'
        raw=json.loads((ROOT/item['source']).read_text());assert sha(ROOT/item['source'])==item['sha256']
        result=dict(metadata=dict(model=name,protocol_sha256=sha(PROTOCOL),completed=False),rows=[],failures=[])
        try:
            model=AutoModelForCausalLM.from_pretrained(name,revision=plan['revision'],torch_dtype=torch.float32,local_files_only=True)
            layer=model.get_output_embeddings()
            wc=layer.weight.detach().cpu().double().numpy().copy()
            bc=np.zeros(len(wc)) if layer.bias is None else layer.bias.detach().cpu().double().numpy().copy()
            assert hashlib.sha256(wc.tobytes()).hexdigest()==plan['W_sha256']
            assert hashlib.sha256(bc.tobytes()).hexdigest()==plan['bias_sha256']
            del layer,model;gc.collect()
            W=torch.tensor(wc,device='cuda',dtype=torch.float64);bias=torch.tensor(bc,device='cuda',dtype=torch.float64)
            ids=torch.tensor(plan['selected_ids'],device='cuda',dtype=torch.long)
            decoder,_,design=public_decoder(W,ids)
            result['metadata']['design']=design
            for paragraph in range(16):
                try:
                    teacher=torch.tensor(raw['metadata']['hidden_states'][str(paragraph)],device='cuda',dtype=torch.float64)
                    lp=torch.log_softmax(W@teacher+bias,dim=0);p=lp.exp()
                    maxn=max(protocol['sample_counts']);reps=protocol['provider_replicates']
                    gen=torch.Generator(device='cuda');gen.manual_seed(protocol['seed']+mi*1000+paragraph)
                    pair_ids=torch.multinomial(p,2*maxn*reps,replacement=True,generator=gen).reshape(reps,maxn,2)
                    aa=(W[pair_ids[:,:,0]]-W[pair_ids[:,:,1]]).reshape(-1,W.shape[1])@decoder
                    # Save sampled token IDs so the finite ablation can be replayed without RNG implementation dependence.
                    selected=lp[ids]
                    for g in [.002,.02]:
                        cell=next(x for x in raw['rows'] if x['paragraph']==paragraph and x['grid']==g)
                        spacing=g/2;f=selected/spacing-torch.floor(selected/spacing)
                        e=torch.round(selected/spacing)-selected/spacing
                        quadratics=torch.stack([(aa.square()@(f*(1-f))),shared_quadratic(aa,f),(aa@e).square()],dim=1)*spacing**2/4
                        quadratics=quadratics.reshape(reps,maxn,3)
                        for n in protocol['sample_counts']:
                            estimates=quadratics[:,:n].mean(1)
                            choices=torch.argmin(estimates,dim=1).cpu().tolist()
                            for rep,j in enumerate(choices):
                                law=LAWS[j]
                                result['rows'].append(dict(paragraph=paragraph,grid=g,pairs=n,provider_seed_index=rep,status='completed',chosen=law,estimated_Q=estimates[rep].cpu().tolist(),dense_Q=[cell['laws'][law]['quadratic_forward'] for law in LAWS],dense_chosen=cell['laws']['covariance_selected']['selected_kernel'],selected_forward=cell['laws'][law]['mean_forward'],nearest_forward=cell['laws']['nearest']['mean_forward'],dense_selected_forward=cell['laws']['covariance_selected']['mean_forward'],selected_maximum=cell['laws'][law]['mean_maximum']))
                    result.setdefault('sampled_pair_ids',{})[str(paragraph)]=pair_ids.cpu().tolist()
                    save(destination,result)
                    print(json.dumps(dict(model=name,paragraph=paragraph,elapsed=time.perf_counter()-start)),flush=True)
                except Exception:
                    result['failures'].append(dict(paragraph=paragraph,traceback=traceback.format_exc()))
            del W,bias,decoder,aa;gc.collect();torch.cuda.empty_cache()
        except Exception:
            result['failures'].append(dict(stage='model',traceback=traceback.format_exc()))
        done={(x['paragraph'],x['grid'],x['pairs'],x['provider_seed_index']) for x in result['rows']}
        for para in range(16):
            for g in [.002,.02]:
                for n in protocol['sample_counts']:
                    for rep in range(protocol['provider_replicates']):
                        if (para,g,n,rep) not in done: result['rows'].append(dict(paragraph=para,grid=g,pairs=n,provider_seed_index=rep,status='failed'))
        result['metadata'].update(completed=all(x['status']=='completed' for x in result['rows']),elapsed=time.perf_counter()-start)
        save(destination,result)
        print(json.dumps(dict(model=name,complete=result['metadata']['completed'],rows=len(result['rows']))),flush=True)
if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--freeze',action='store_true');args.add_argument('--run',action='store_true');a=args.parse_args()
    if a.freeze:freeze()
    elif a.run:run()
