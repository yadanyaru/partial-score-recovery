"""Frozen three-model response-law study with reused public IDs and intercept LS."""
import os
os.environ.setdefault('HF_HOME', str(__import__('runtime_paths').HF_CACHE))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from frozen_models import REVISIONS

ROOT=__import__("runtime_paths").ROOT
PROTOCOL = ROOT/'outputs'/'PRETRAINED_STOCHASTIC_PRECISION_PROTOCOL.json'
OUT = ROOT/'outputs'/'pretrained_stochastic_precision'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False)
    for attempt in range(12):
        temporary = path.with_suffix(f'.tmp{attempt}')
        temporary.write_text(text)
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.1*(attempt+1), 1.))

def freeze():
    models = []
    for name, revision in REVISIONS.items():
        previous_path = ROOT/'outputs'/'pretrained_disclosure_formal'/f"{name.replace('/', '__')}.json"
        previous = json.loads(previous_path.read_text())
        design = previous['designs']['public_leverage__9041']
        ids = design['public_ids']
        models.append(dict(model=name, revision=revision, selected_ids=ids,
                           prior_design_file_sha256=sha(previous_path),
                           W_sha256=previous['metadata']['W_sha256'],
                           bias_sha256=previous['metadata']['bias_sha256']))
    save(PROTOCOL, dict(source_sha256=sha(__file__), corpus_sha256=sha(ROOT/'work'/'corpus.json'),
                        frozen_before_new_response_metrics=True,
                        models=models, paragraphs=list(range(8,64)), max_tokens=64,
                        grids=[.002,.02], mesh='g/2', repetitions=32, seed=202610071,
                        designs='reuse head-only public leverage seed9041; d+101 IDs; no new outcome-based selection',
                        decoder='unweighted intercept-aware augmented LS; no teacher state argument',
                        response_laws=['nearest_same_mesh','independent_unbiased_stochastic','coherent_biased_lattice'],
                        coherent_law='public maximum-leverage row versus public mean-nearest row; ceil/floor according to public contrast coefficients',
                        numerical_scope='float64 frozen public head on native FP32 last states; numerical diagnostics only',
                        failure_policy='all336 prescribed model/paragraph/grid cells retained',
                        claims_excluded=['ordinary rounding superiority','API prevalence of coherent bias',
                                         'native API certification','adaptive algorithm improvement','heldout text generalization']))
    print(json.dumps(dict(frozen=True, models=len(models), prescribed_cases=336)))

def public_decoder(W, ids):
    chosen = W[ids]
    center = chosen.mean(dim=0)
    centered = chosen-center
    gram = centered.T@centered
    L = torch.linalg.cholesky(gram)
    decoder = torch.cholesky_solve(centered.T, L)
    values = []
    for begin in range(0,len(W),4096):
        white = torch.linalg.solve_triangular(L,(W[begin:begin+4096]-center).T,upper=False)
        values.append(white.square().sum(dim=0)+1/len(ids))
    leverage = torch.cat(values)
    high = int(torch.argmax(leverage))
    low = int(torch.argmin((W-center).square().sum(dim=1)))
    signs = torch.sign((W[high]-W[low])@decoder)
    return decoder, signs, dict(coefficient_norm_max_numeric=float(leverage.max()),
                                coherent_pair=[high,low], selected_count=len(ids),
                                minimum_cholesky_diagonal=float(L.diag().min()),
                                scope='public float64 coefficient proposal, not an enclosed certificate')

def run_one(plan, protocol):
    name = plan['model']
    path = OUT/f"{name.replace('/','__')}.json"
    if path.exists():
        raise FileExistsError(f'Preserve existing result before rerun: {path}')
    result = dict(metadata=dict(model=name, revision=plan['revision'], source_sha256=sha(__file__),
                                protocol_sha256=sha(PROTOCOL), completed=False), rows=[], failures=[])
    corpus = json.loads((ROOT/'work'/'corpus.json').read_text(encoding='utf-8'))
    states = {}
    start = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(name,revision=plan['revision'],
                torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
        tokenizer = AutoTokenizer.from_pretrained(name,revision=plan['revision'],local_files_only=True)
        layer = model.get_output_embeddings()
        Wcpu = layer.weight.detach().cpu().double().numpy().copy()
        bcpu = np.zeros(len(Wcpu)) if layer.bias is None else layer.bias.detach().cpu().double().numpy().copy()
        assert hashlib.sha256(Wcpu.tobytes()).hexdigest() == plan['W_sha256']
        assert hashlib.sha256(bcpu.tobytes()).hexdigest() == plan['bias_sha256']
        if name.startswith('EleutherAI'):
            norm = model.gpt_neox.final_layer_norm
        elif name.startswith('openai-community'):
            norm = model.transformer.ln_f
        else:
            norm = model.model.norm
        norm_weight = norm.weight.detach().cpu().double()
        norm_bias = torch.zeros_like(norm_weight) if getattr(norm,'bias',None) is None else norm.bias.detach().cpu().double()
        radius = 2*(math.sqrt(Wcpu.shape[1])*float(norm_weight.abs().max())+float(torch.linalg.vector_norm(norm_bias)))+1
        for paragraph in protocol['paragraphs']:
            try:
                encoded = tokenizer(corpus[paragraph]['text'],return_tensors='pt',truncation=True,
                                    max_length=protocol['max_tokens']).to('cuda')
                with torch.inference_mode():
                    native = model(**encoded,output_hidden_states=True)
                states[paragraph] = native.hidden_states[-1][0,-1].detach().cpu().double()
                del native, encoded
            except Exception:
                result['failures'].append(dict(stage='state',paragraph=paragraph,traceback=traceback.format_exc()))
        del norm, layer, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        W = torch.tensor(Wcpu,device='cuda',dtype=torch.float64)
        bias = torch.tensor(bcpu,device='cuda',dtype=torch.float64)
        ids = torch.tensor(plan['selected_ids'],device='cuda',dtype=torch.long)
        decoder, signs, design = public_decoder(W,ids)
        extent = 2*radius*float(torch.linalg.vector_norm(W,dim=1).max())+float(bias.max()-bias.min())+math.log(len(W))
        result['metadata'].update(design=design, public_state_radius_proposal=radius,
                                   radius_scope='conservative public final-normalization envelope; float32 diagnostic',
                                   teacher='recomputed float64 head on native FP32 states',
                                   V=len(W), rank=W.shape[1], selected_ids=plan['selected_ids'])
        for paragraph in protocol['paragraphs']:
            for g in protocol['grids']:
                record = dict(paragraph=paragraph,grid=g,status='failed')
                try:
                    teacher = states[paragraph].to('cuda')
                    assert float(torch.linalg.vector_norm(teacher)) <= radius
                    lp = torch.log_softmax(W@teacher+bias,dim=0)
                    p = lp.exp()
                    selected, spacing = lp[ids], g/2
                    floor = torch.floor(selected/spacing)
                    fraction = selected/spacing-floor
                    generator = torch.Generator(device='cuda')
                    generator.manual_seed(protocol['seed']+paragraph+int(g*1000000))
                    stochastic = floor+(torch.rand((protocol['repetitions'],len(ids)),generator=generator,
                                                  device='cuda',dtype=torch.float64)<fraction).double()
                    codes = dict(nearest_same_mesh=torch.round(selected/spacing)[None,:],
                                 independent_unbiased_stochastic=stochastic,
                                 coherent_biased_lattice=(floor+(signs>0).double())[None,:])
                    laws = {}
                    for law, code in codes.items():
                        received = code*spacing
                        error = received-selected
                        assert float(error.abs().max()) <= spacing+1e-10
                        estimated = (received-bias[ids])@decoder.T
                        lq = torch.log_softmax(estimated@W.T+bias,dim=1)
                        forward = (p*(lp-lq)).sum(dim=1)
                        reverse = (lq.exp()*(lq-lp)).sum(dim=1)
                        maximum = torch.maximum(forward,reverse)
                        laws[law] = dict(mean_forward=float(forward.mean()),mean_reverse=float(reverse.mean()),
                                         mean_maximum=float(maximum.mean()), maximum_loss=float(maximum.max()),
                                         repetitions=len(code), received_codes=code.long().cpu().tolist(),
                                         max_score_error=float(error.abs().max()))
                    J = math.ceil(extent/spacing+1)
                    bits = math.ceil(math.log2(2*J+1))+math.ceil(math.log2(len(W)))
                    record.update(status='completed',laws=laws,mesh=spacing,
                                  same_fixed_slot_payload=len(ids)*bits,
                                  payload_scope='same public common range, IDs and score fields; common framing excluded',
                                  numeric_expected_max_bound=5*design['coefficient_norm_max_numeric']*spacing**2/8)
                except Exception:
                    record['traceback'] = traceback.format_exc()
                result['rows'].append(record)
            if (paragraph-7)%8 == 0:
                save(path,result)
                print(json.dumps(dict(model=name,paragraphs_done=paragraph-7,elapsed=time.perf_counter()-start)),flush=True)
        del W,bias,decoder
    except Exception:
        result['failures'].append(dict(stage='model',traceback=traceback.format_exc()))
    expected = {(p,g) for p in protocol['paragraphs'] for g in protocol['grids']}
    seen = {(x['paragraph'],x['grid']) for x in result['rows']}
    for p,g in sorted(expected-seen):
        result['rows'].append(dict(paragraph=p,grid=g,status='failed',failure='model stage prevented execution; retained'))
    result['metadata'].update(completed=all(x['status']=='completed' for x in result['rows']),
                               prescribed_cases=len(expected),elapsed=time.perf_counter()-start)
    save(path,result)
    print(json.dumps(dict(model=name,completed=result['metadata']['completed'],cases=len(result['rows']),
                           failures=sum(x['status']!='completed' for x in result['rows']))),flush=True)
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        torch.set_num_threads(4)
        protocol = json.loads(PROTOCOL.read_text())
        assert protocol['source_sha256'] == sha(__file__)
        assert protocol['corpus_sha256'] == sha(ROOT/'work'/'corpus.json')
        for plan in protocol['models']:
            run_one(plan,protocol)
