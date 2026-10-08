"""Pre-specified corpus diagnostics, not a native-API or numerical certificate.

No work is performed on import. Audit/freeze this source before launching.
"""
import os
os.environ.setdefault('HF_HOME', str(__import__('runtime_paths').HF_CACHE))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import argparse, gc, hashlib, json, math, pathlib, time, traceback
import numpy as np
import torch
from scipy.linalg import qr
from transformers import AutoModelForCausalLM, AutoTokenizer
from frozen_models import REVISIONS

ROOT=__import__("runtime_paths").ROOT
SEEDS = (9041, 9101, 9301)
GRIDS = (.002, .02)
METHODS = ('public_qr', 'public_spanner', 'random_square',
           'random_oversampled', 'public_leverage')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def save(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def public_reference(Wcpu, seed):
    rng = np.random.default_rng(seed)
    V, d = Wcpu.shape
    pool = rng.choice(V, min(4*d+1, V), replace=False)
    P = Wcpu[pool]
    ref = int(pool[np.argmin(np.linalg.norm(P-P.mean(axis=0), axis=1))])
    _, _, pivot = qr((P-Wcpu[ref]).T, pivoting=True, mode='economic')
    basis = pool[pivot[:d]]
    if len(basis) != d or ref in basis:
        raise ValueError('public QR initialization lacks nonreference full basis')
    return ref, pool, basis


def leverage_distribution(W, ref, chunk=4096):
    # Exact algebra C^T C, C_y=W_y-W_ref; computed float64, diagnostic only.
    V, d = W.shape
    sums = W.sum(dim=0)
    r = W[ref]
    G = W.T@W - torch.outer(sums, r) - torch.outer(r, sums) + V*torch.outer(r, r)
    G = (G+G.T)/2
    L = torch.linalg.cholesky(G)
    values = np.empty(V, dtype=np.float64)
    for begin in range(0, V, chunk):
        C = W[begin:begin+chunk]-r
        whitened = torch.linalg.solve_triangular(L, C.T, upper=False)
        values[begin:begin+chunk] = whitened.square().sum(dim=0).cpu().numpy()
    values[ref] = 0.
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('invalid numerical leverage distribution')
    return values/values.sum(), {
        'scheme': 'without replacement, probability proportional to numerical row-contrast leverage',
        'Gram_method': 'float64 analytic centered Gram, Cholesky, chunked triangular solves; no exact leverage certificate',
        'leverage_sum_diagnostic': float(values.sum()),
    }


def spanner(W, ref, initial, max_swaps=2000):
    V, d = W.shape
    basis = initial.copy()
    C = W-W[ref]
    J = torch.linalg.inv(C[torch.as_tensor(basis, device=W.device)])
    A = C@J
    ledger = []
    for iteration in range(max_swaps+1):
        # Chunked absolute maxima avoid a second full-vocabulary matrix.
        value, flat = -1., None
        for begin in range(0, V, 4096):
            maximum, local = torch.max(A[begin:begin+4096].abs().reshape(-1), 0)
            if float(maximum) > value:
                value, flat = float(maximum), begin*d+int(local)
        if value <= 1.05:
            break
        if iteration == max_swaps:
            raise RuntimeError(f'maxvol limit {max_swaps}; max coefficient {value}')
        row, col = divmod(int(flat), d)
        old = int(basis[col])
        alpha = A[row].clone()
        pivot = alpha[col].clone()
        if not torch.isfinite(pivot) or abs(float(pivot)) <= 1.:
            raise RuntimeError('invalid volume-increasing swap pivot')
        delta = alpha.clone(); delta[col] -= 1
        old_column = A[:, col].clone()
        J = J-J[:, col, None]*delta[None, :]/pivot
        A.addmm_(old_column[:, None], delta[None, :], beta=1., alpha=-1./float(pivot))
        del old_column
        basis[col] = row
        ledger.append([row, col, old, value])
        if (iteration+1) % 25 == 0:
            del A
            J = torch.linalg.inv(C[torch.as_tensor(basis, device=W.device)])
            A = C@J
    del A
    J = torch.linalg.inv(C[torch.as_tensor(basis, device=W.device)])
    recomputed = C@J
    final = float(recomputed.abs_().max())
    del recomputed
    if final > 1.05+1e-9:
        raise RuntimeError(f'recomputed final max coefficient {final} exceeds threshold')
    return basis, {'swaps': len(ledger), 'swap_ledger': ledger,
                   'max_abs_alpha_diagnostic': final,
                   'certificate': 'none; numerical volume swaps, ideal exact-alpha theorem separate'}


def decoder_system(W, ids):
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicated disclosed IDs')
    ref = int(ids[0])
    D = W[torch.as_tensor(ids[1:], device=W.device)]-W[ref]
    # CPU SVD is explicit and supplies one algorithm for square and tall cases.
    U, singular, Vh = np.linalg.svd(D.cpu().numpy(), full_matrices=False)
    d = W.shape[1]
    rank = int(np.count_nonzero(singular > np.finfo(float).eps*max(D.shape)*singular[0]))
    if rank != d:
        raise ValueError(f'numerical contrast rank {rank}, expected {d}')
    inverse = (Vh.T/singular)@U.T
    meta = {'public_ids': ids.tolist(), 'score_count': len(ids),
            'decoder': 'public float64 SVD pseudoinverse of unweighted row contrasts; factorized reconstruction from disclosed ratios only',
            'oversampling_weighting': 'none; leverage sampling is a heuristic unweighted least-squares baseline',
            'numerical_rank': rank, 'condition': float(singular[0]/singular[-1]),
            'sigma_min': float(singular[-1]), 'ideal_coefficient_certificate': None}
    return ids, inverse, meta


def planned_rows(paragraphs):
    for paragraph in paragraphs:
        for seed in SEEDS:
            for method in METHODS:
                for grid in GRIDS:
                    yield paragraph, seed, method, grid
        for grid in GRIDS:
            yield paragraph, None, 'passive_top_dplus1', grid


def fail_row(paragraph, seed, method, grid, reason):
    return {'paragraph': paragraph, 'design_seed': seed, 'method': method,
            'grid': grid, 'status': 'failed', 'failure': reason}


def metrics(lp, qlp, tail):
    p, q = lp.exp(), qlp.exp()
    pm, qm = float(p[tail].sum()), float(q[tail].sum())
    return {'forward_kl': float((p*(lp-qlp)).sum()),
            'reverse_kl': float((q*(qlp-lp)).sum()),
            'teacher_top20_complement_mass': pm, 'decoded_complement_mass': qm,
            'tail_mass_signed_error': qm-pm, 'tail_mass_absolute_error': abs(qm-pm)}


def run_model(model_name, args):
    revision = REVISIONS[model_name]
    corpus = json.loads((ROOT/'work/corpus.json').read_text(encoding='utf-8'))
    paragraphs = list(range(8, 64))[:args.paragraph_limit]
    if len(corpus) < 64:
        raise ValueError('corpus shorter than pre-specified 64 paragraphs')
    slug = model_name.replace('/', '__')
    outpath = ROOT/'outputs/pretrained_disclosure_formal'/f'{slug}.json'
    if outpath.exists():
        raise FileExistsError(f'output already exists; preserve it before a separately authorized rerun: {outpath}')
    result = {'metadata': {
        'model': model_name, 'revision': revision, 'paragraphs': paragraphs,
        'design_seeds': list(SEEDS), 'grids': list(GRIDS), 'max_length': 64,
        'public_methods': list(METHODS), 'public_variants': 15, 'passive_variants': 1,
        'planned_rows': len(paragraphs)*16*2,
        'scope': 'pre-specified corpus evaluation; all values approximate numerical diagnostics, not numerical certificates or an unseen test set',
        'oracle': 'float64 public head applied to native frozen-model float32 final hidden state; distinct from native output logits',
        'native_comparison': 'double log-softmax of native float32 logits versus recomputed double public head',
        'rounding': 'torch.round(logprob/grid) ties-to-even, float64 numerical grid; no clipping',
        'teacher_event': 'complement of teacher top20 IDs; teacher IDs used only to define evaluation event, not public selection or decoder',
        'selection_scope': 'public W only; passive top IDs depend on paragraph oracle',
        'code_sha256': sha(__file__), 'frozen_models_sha256': sha(ROOT/'work/frozen_models.py'),
        'corpus_sha256': sha(ROOT/'work/corpus.json'),
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'smoke_subset': len(paragraphs) != 56, 'completed': False,
        'certificate': 'none; ideal exact-alpha uniform theorem not a guarantee for implemented approximate decoder'},
        'designs': {}, 'rows': [], 'failures': []}
    save(result, outpath)
    device = torch.device(args.device)
    states = {}
    start = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                  torch_dtype=torch.float32, local_files_only=True).to(device).eval()
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
        layer = model.get_output_embeddings()
        Wcpu = layer.weight.detach().cpu().double().numpy().copy()
        bcpu = np.zeros(len(Wcpu)) if layer.bias is None else layer.bias.detach().cpu().double().numpy().copy()
        result['metadata'].update(V=Wcpu.shape[0], d=Wcpu.shape[1],
              W_sha256=hashlib.sha256(Wcpu.tobytes()).hexdigest(),
              bias_sha256=hashlib.sha256(bcpu.tobytes()).hexdigest(),
              model_weight_dtype=str(layer.weight.dtype), device=str(device))
        for paragraph in paragraphs:
            try:
                enc = tok(corpus[paragraph]['text'], return_tensors='pt', truncation=True,
                          max_length=64).to(device)
                with torch.inference_mode():
                    native = model(**enc, output_hidden_states=True)
                states[paragraph] = (native.hidden_states[-1][0, -1].detach().cpu().double(),
                                     native.logits[0, -1].detach().cpu().float(),
                                     int(enc['input_ids'].shape[1]))
                del native, enc
            except Exception:
                result['failures'].append({'stage': 'native_state', 'paragraph': paragraph,
                                           'traceback': traceback.format_exc()})
            save(result, outpath)
        # Release the body before allocating full-vocabulary float64 contrasts.
        del layer, model, tok
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        result['metadata']['native_state_stage_seconds'] = time.perf_counter()-start
        W = torch.as_tensor(Wcpu, device=device, dtype=torch.float64)
        bias = torch.as_tensor(bcpu, device=device, dtype=torch.float64)
        V, d = W.shape
        result['metadata']['id_bits_per_score'] = math.ceil(math.log2(V))
        systems = {}
        for seed in SEEDS:
            ref = pool = initial = None
            sync(device); reference_begin = time.perf_counter()
            try:
                ref, pool, initial = public_reference(Wcpu, seed)
            except Exception:
                reason = traceback.format_exc()
                result['failures'].append({'stage': 'public_reference', 'seed': seed, 'traceback': reason})
            reference_seconds = time.perf_counter()-reference_begin
            result['metadata'].setdefault('shared_reference_QR_setup_seconds', {})[str(seed)] = reference_seconds
            leverage = leverage_meta = None
            for method in METHODS:
                key = f'{method}__{seed}'
                sync(device); begin = time.perf_counter()
                try:
                    if ref is None: raise RuntimeError('public reference initialization failed')
                    rng = np.random.default_rng(seed)
                    candidates = np.delete(np.arange(V), ref)
                    details = {'reference_id': ref, 'seed': seed,
                               'QR_initial_pool_ids': pool.tolist(),
                               'public_reference_rule': 'nearest row to seeded public pool mean'}
                    if method == 'public_qr': chosen = initial
                    elif method == 'public_spanner':
                        chosen, extra = spanner(W, ref, initial); details.update(extra)
                    elif method == 'random_square': chosen = rng.choice(candidates, d, replace=False)
                    elif method == 'random_oversampled': chosen = rng.choice(candidates, d+100, replace=False)
                    else:
                        leverage, leverage_meta = leverage_distribution(W, ref)
                        chosen = rng.choice(V, d+100, replace=False, p=leverage)
                        details.update(leverage_meta)
                    ids, inverse, meta = decoder_system(W, np.r_[ref, chosen])
                    sync(device)
                    meta.update(details, status='ready', public_design_seconds=time.perf_counter()-begin,
                                public_design_seconds_including_shared_reference_setup=time.perf_counter()-begin+reference_seconds,
                                reference_QR_setup_shared_across_methods=True)
                    systems[key] = (ids, inverse)
                    result['designs'][key] = meta
                except Exception:
                    reason = traceback.format_exc()
                    result['designs'][key] = {'status': 'failed', 'failure': reason,
                                             'public_design_seconds': time.perf_counter()-begin}
                    result['failures'].append({'stage': 'public_design', 'key': key, 'traceback': reason})
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
                save(result, outpath)
        for paragraph in paragraphs:
            if paragraph not in states:
                result['rows'].extend(fail_row(p, s, m, g, 'native-state stage failed')
                                     for p, s, m, g in planned_rows([paragraph]))
                save(result, outpath); continue
            hcpu, native_logits, ntokens = states.pop(paragraph)
            try:
                lp = torch.log_softmax(W@hcpu.to(device)+bias, dim=0)
                native_lp = torch.log_softmax(native_logits.double().to(device), dim=0)
                native_discrepancy = float((lp-native_lp).abs().max())
                top20 = torch.topk(lp, 20).indices
                tail = torch.ones(V, dtype=torch.bool, device=device); tail[top20] = False
            except Exception:
                reason = traceback.format_exc()
                result['rows'].extend(fail_row(p, s, m, g, reason) for p, s, m, g in planned_rows([paragraph]))
                save(result, outpath); continue
            active = dict(systems)
            passive_key = 'passive_top_dplus1'
            passive_meta = None
            try:
                sync(device); begin = time.perf_counter()
                pids = torch.argsort(lp, descending=True)[:d+1].cpu().numpy()
                ids, inverse, passive_meta = decoder_system(W, pids)
                sync(device)
                passive_meta.update(status='ready', paragraph=paragraph,
                                    passive_design_seconds=time.perf_counter()-begin)
                active[passive_key] = (ids, inverse)
                result['designs'][f'{passive_key}__{paragraph}'] = passive_meta
            except Exception:
                result['designs'][f'{passive_key}__{paragraph}'] = {'status': 'failed', 'failure': traceback.format_exc()}
            variants = [(s, m, f'{m}__{s}') for s in SEEDS for m in METHODS]+[(None, passive_key, passive_key)]
            for seed, method, key in variants:
                if key not in active:
                    result['rows'].extend(fail_row(paragraph, seed, method, g, 'design failed; see retained design record') for g in GRIDS)
                    continue
                ids, inverse_cpu = active[key]
                try:
                    indices = torch.as_tensor(ids, device=device)
                    inverse = torch.as_tensor(inverse_cpu, device=device)
                    ell = lp[indices]
                    disclosed_bias = bias[indices]
                    def decode(scores):
                        rhs = scores[1:]-scores[0]-disclosed_bias[1:]+disclosed_bias[0]
                        reconstructed = inverse@rhs
                        return torch.log_softmax(W@reconstructed+bias-bias[indices[0]], dim=0)
                    clean = decode(ell)
                    clean_errors = {'clean_forward_kl': metrics(lp, clean, tail)['forward_kl'],
                                    'clean_reverse_kl': metrics(lp, clean, tail)['reverse_kl'],
                                    'clean_logprob_max_error': float((lp-clean).abs().max())}
                    for grid in GRIDS:
                        codes = torch.round(ell/grid)
                        if not torch.isfinite(codes).all(): raise ValueError('nonfinite score codes')
                        integers = [int(x) for x in codes.cpu().tolist()]
                        rounded = codes*grid
                        qlp = decode(rounded)
                        row = {'paragraph': paragraph, 'source_row': corpus[paragraph]['source_row'],
                               'domain': corpus[paragraph]['domain'], 'input_tokens': ntokens,
                               'design_seed': seed, 'method': method, 'grid': grid, 'status': 'ok',
                               'score_count': len(ids), 'id_bits': len(ids)*math.ceil(math.log2(V)),
                               'signed_integer_score_bits': sum(1+abs(x).bit_length() for x in integers),
                               'score_bit_encoding': 'raw variable-width sum is a diagnostic without lengths; fixed-width payload below is decodable given width header; excludes framing/known grid',
                               'fixed_width_score_magnitude_bits': max(abs(x).bit_length() for x in integers),
                               'fixed_width_score_payload_bits': len(ids)*(1+max(abs(x).bit_length() for x in integers)),
                               'score_codes': integers, 'score_code_min': min(integers), 'score_code_max': max(integers),
                               'score_logprob_min': float(ell.min()), 'score_logprob_max': float(ell.max()),
                               'clipping': None, 'actual_max_rounding_error': float((rounded-ell).abs().max()),
                               'native_fp32_logprob_discrepancy': native_discrepancy,
                               'decoder_uses_teacher_hidden': False,
                               'selection_ids_record': f'{passive_key}__{paragraph}' if seed is None else key,
                               **metrics(lp, qlp, tail), **clean_errors}
                        if not all(math.isfinite(row[x]) for x in ('forward_kl','reverse_kl','tail_mass_absolute_error')):
                            raise ValueError('nonfinite decoding metrics')
                        row['total_scalar_message_bits_excluding_framing'] = row['id_bits']+row['signed_integer_score_bits']
                        row['fixed_width_total_payload_bits_excluding_header'] = row['id_bits']+row['fixed_width_score_payload_bits']
                        result['rows'].append(row)
                except Exception:
                    reason = traceback.format_exc()
                    existing = {(r['paragraph'],r['design_seed'],r['method'],r['grid']) for r in result['rows']}
                    result['rows'].extend(fail_row(paragraph, seed, method, g, reason) for g in GRIDS
                                         if (paragraph,seed,method,g) not in existing)
                    result['failures'].append({'stage': 'decode', 'paragraph': paragraph, 'key': key, 'traceback': reason})
                save(result, outpath)
            print(f'{model_name}: paragraph {paragraph} complete, rows={len(result["rows"])}', flush=True)
        result['metadata']['completed'] = True
    except Exception:
        result['failures'].append({'stage': 'model', 'traceback': traceback.format_exc()})
        existing = {(r['paragraph'],r['design_seed'],r['method'],r['grid']) for r in result['rows']}
        for p, s, m, g in planned_rows(paragraphs):
            if (p,s,m,g) not in existing:
                result['rows'].append(fail_row(p,s,m,g,'model-level failure; see traceback'))
    finally:
        result['metadata']['total_seconds'] = time.perf_counter()-start
        result['metadata']['observed_rows'] = len(result['rows'])
        result['metadata']['successful_rows'] = sum(r['status']=='ok' for r in result['rows'])
        save(result, outpath)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(REVISIONS), help='omit to run all three sequentially')
    parser.add_argument('--paragraph-limit', type=int, default=56, help='smoke subset of pre-specified 8..63; no protocol-changing offsets')
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    args = parser.parse_args()
    if not 1 <= args.paragraph_limit <= 56:
        parser.error('--paragraph-limit must be 1..56')
    torch.set_num_threads(4)
    for model in ([args.model] if args.model else list(REVISIONS)):
        run_model(model, args)


if __name__ == '__main__':
    main()
