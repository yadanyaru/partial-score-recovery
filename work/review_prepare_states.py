"""Freeze native LAMBADA answer-prefix states; preserve model/head identities."""
import os
os.environ.setdefault('HF_HOME', str(__import__('runtime_paths').HF_CACHE))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from frozen_models import REVISIONS

ROOT=__import__("runtime_paths").ROOT
CACHE=__import__("runtime_paths").HEAD_CACHE
OUT=ROOT/'outputs/review_native_states'
CORPUS=ROOT/'work/review_lambada_corpus.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,allow_nan=False),encoding='utf-8');tmp.replace(p)

def main():
    torch.set_num_threads(4);CACHE.mkdir(parents=True,exist_ok=True)
    corpus=json.loads(CORPUS.read_text(encoding='utf-8'))
    protocol=ROOT/'outputs/REVIEW_STATE_PROTOCOL.json'
    spec=dict(source_sha256=sha(__file__),corpus_sha256=sha(CORPUS),models=REVISIONS,
              paragraphs=list(range(128)),max_context_tokens=256,
              states='native FP32 final-normalized states at every teacher-forced answer prefix',
              teacher='float64 public affine head; original model parameters unchanged',
              tokenization='full text tokenization must extend context tokenization exactly',frozen_before_metrics=True)
    if protocol.exists():assert json.loads(protocol.read_text())==spec
    else:save(protocol,spec)
    for name,rev in REVISIONS.items():
        slug=name.replace('/','__');dest=OUT/f'{slug}.json'
        if dest.exists():
            old=json.loads(dest.read_text());assert old['metadata']['completed'];continue
        started=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(name,revision=rev,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
        tok=AutoTokenizer.from_pretrained(name,revision=rev,local_files_only=True)
        layer=model.get_output_embeddings();w=layer.weight.detach().cpu().numpy().copy()
        bias=np.zeros(len(w),dtype=np.float32) if layer.bias is None else layer.bias.detach().cpu().numpy().copy()
        cachepath=CACHE/f'{slug}_head.npz';np.savez(cachepath,W=w,bias=bias)
        oldplan=next(x for x in json.loads((ROOT/'outputs/RISK_SELECTED_PRECISION_PROTOCOL.json').read_text())['models'] if x['model']==name)
        assert hashlib.sha256(w.astype(np.float64).tobytes()).hexdigest()==oldplan['W_sha256']
        assert hashlib.sha256(bias.astype(np.float64).tobytes()).hexdigest()==oldplan['bias_sha256']
        norm=model.gpt_neox.final_layer_norm if name.startswith('EleutherAI') else model.transformer.ln_f if name.startswith('openai-community') else model.model.norm
        nw=norm.weight.detach().double();nb=torch.zeros_like(nw) if getattr(norm,'bias',None) is None else norm.bias.detach().double()
        radius=2*(w.shape[1]**.5*float(nw.abs().max())+float(torch.linalg.vector_norm(nb)))+1
        rows=[]
        with torch.inference_mode():
            for j,item in enumerate(corpus):
                context=tok.encode(item['context'],add_special_tokens=False)
                full=tok.encode(item['text'].rstrip(),add_special_tokens=False)
                assert full[:len(context)]==context,(name,j,'context tokenization boundary')
                gold=full[len(context):];assert gold
                kept=context[-256:]
                ids=torch.tensor([kept+gold],device='cuda')
                native=model(ids,output_hidden_states=True,use_cache=False)
                answer_states=native.hidden_states[-1][0,len(kept)-1:len(kept)+len(gold)-1].detach().cpu().double()
                assert len(answer_states)==len(gold)
                rows.append(dict(paragraph=j,source_row=item['source_row'],hidden_state=answer_states[0].tolist(),
                    answer_hidden_states=answer_states.tolist(),gold_ids=gold,token_count=len(kept),
                    original_context_tokens=len(context),answer=item['answer']))
                del native,ids
                if (j+1)%32==0:print(json.dumps(dict(stage='states',model=name,passages=j+1,elapsed=time.perf_counter()-started)),flush=True)
        result=dict(metadata=dict(model=name,revision=rev,source_sha256=sha(__file__),protocol_sha256=sha(protocol),
            corpus_sha256=sha(CORPUS),completed=True,head_cache=str(cachepath),head_cache_sha256=sha(cachepath),
            W_sha256=oldplan['W_sha256'],bias_sha256=oldplan['bias_sha256'],rank=w.shape[1],V=len(w),public_state_radius=radius),rows=rows)
        save(dest,result)
        print(json.dumps(dict(model=name,passages=len(rows),answer_states=sum(len(x['gold_ids']) for x in rows),elapsed=time.perf_counter()-started)),flush=True)
        del model,layer,norm,tok,w,bias,nw,nb;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
