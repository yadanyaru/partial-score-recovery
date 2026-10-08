"""Frozen CPU response-law diagnostic; approximate linear decoding, not certificates."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT=__import__("runtime_paths").ROOT
P = ROOT/'outputs'/'STOCHASTIC_PRECISION_DIAGNOSTIC_PROTOCOL.json'
R = ROOT/'outputs'/'stochastic_precision_diagnostic.json'
OLD = ROOT/'outputs'/'adaptive_disclosure_synthetic_protocol.json'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def freeze():
    previous = json.loads(OLD.read_text())
    designs = []
    for index, head in enumerate(previous['heads']):
        W = np.array(head['W'])
        z = np.column_stack([W, np.ones(len(W))])
        n = z.shape[1]
        leverage = np.sum(z*np.linalg.solve(z.T@z, z.T).T, axis=1)
        rng = np.random.default_rng(20261007+index)
        ids = rng.choice(len(W), size=4*n, replace=False, p=leverage/leverage.sum())
        decoder = np.linalg.solve(z[ids].T@z[ids], z[ids].T)
        A = z@decoder
        C = float(np.max(np.sum(A*A, axis=1)))
        contrast_l1 = np.sum(np.abs(A[:, None, :]-A[None, :, :]), axis=2)
        i, j = np.unravel_index(np.argmax(contrast_l1), contrast_l1.shape)
        signs = np.sign(A[i]-A[j])
        designs.append(dict(head_id=head['id'], ids=ids.tolist(), decoder=decoder.tolist(),
                            coefficient_squared_norm_max=C,
                            adversarial_pair=[int(i), int(j)], signs=signs.tolist(),
                            charged_scores=4*n,
                            design_scope='public augmented-row leverage, fixed seed, without replacement; not BSS'))
    save(P, dict(source_sha256=sha(Path(__file__)), reused_protocol_sha256=sha(OLD),
                 frozen_before_metrics=True, heads=previous['heads'], cases=previous['cases'],
                 designs=designs, repetitions=1024, response_seed=202610070,
                 mesh='d=g2/2',
                 laws=['nearest_same_mesh', 'independent_unbiased_stochastic', 'coherent_biased_lattice'],
                 primary_question='Does independent centering control expected predictive loss while matched correlated reports can increase it?',
                 not_claimed=['ordinary-context stochastic superiority', 'pretrained improvement',
                              'minimax lower verified numerically', 'rigorous floating-point certificate'],
                 teacher_use='simulation/evaluation only; designs and decoder are public'))
    print(json.dumps(dict(frozen=True, cases=len(previous['cases']), designs=len(designs))))

def metrics(lp, decoded, W, bias):
    lq = decoded@W.T+bias
    lq -= logsumexp(lq, axis=1, keepdims=True)
    p = np.exp(lp)
    forward = np.sum(p*(lp-lq), axis=1)
    reverse = np.sum(np.exp(lq)*(lq-lp), axis=1)
    return forward, reverse, np.maximum(forward, reverse)

def run():
    protocol = json.loads(P.read_text())
    assert protocol['source_sha256'] == sha(Path(__file__))
    assert protocol['reused_protocol_sha256'] == sha(OLD)
    heads = {x['id']:x for x in protocol['heads']}
    designs = {x['head_id']:x for x in protocol['designs']}
    rows = []
    for case in protocol['cases']:
        record = dict(case_id=case['case_id'], head_id=case['head_id'], status='failed')
        try:
            head, design = heads[case['head_id']], designs[case['head_id']]
            W, bias, teacher = np.array(head['W']), np.array(head['bias']), np.array(case['h'])
            ids, decoder = np.array(design['ids']), np.array(design['decoder'])
            lp = W@teacher+bias
            lp -= logsumexp(lp)
            d = case['g2']/2
            selected = lp[ids]
            lower_code = np.floor(selected/d).astype(np.int64)
            fraction = selected/d-lower_code
            repeated = protocol['repetitions']
            rng = np.random.default_rng(protocol['response_seed']+case['case_id'])
            stochastic_code = lower_code+(rng.random((repeated, len(ids))) < fraction)
            nearest_code = np.rint(selected/d).astype(np.int64)[None, :]
            coherent_code = (lower_code+(np.array(design['signs']) > 0))[None, :]
            law_results = {}
            for name, codes in [('nearest_same_mesh', nearest_code),
                                ('independent_unbiased_stochastic', stochastic_code),
                                ('coherent_biased_lattice', coherent_code)]:
                received = d*codes
                fitted = (received-bias[ids])@decoder.T
                f, b, maximum = metrics(lp, fitted[:, :-1], W, bias)
                limit = 5*design['coefficient_squared_norm_max']*d*d/8
                law_results[name] = dict(repetitions=len(codes), mean_forward=float(np.mean(f)),
                                        mean_reverse=float(np.mean(b)), mean_maximum=float(np.mean(maximum)),
                                        maximum_loss=float(np.max(maximum)),
                                        MC_standard_error_maximum=float(np.std(maximum, ddof=1)/np.sqrt(len(codes))) if len(codes)>1 else None,
                                        mean_error_per_coordinate=(received-selected).mean(axis=0).tolist(),
                                        maximum_absolute_error=float(np.max(np.abs(received-selected))),
                                        received_codes=codes.tolist())
                assert law_results[name]['maximum_absolute_error'] <= d+1e-12
            radius = 4.
            score_extent = 2*radius*np.linalg.norm(W, axis=1).max()+bias.max()-bias.min()+np.log(len(W))
            code_extent = int(np.ceil(score_extent/d+1))
            score_bits = int(np.ceil(np.log2(2*code_extent+1)))
            id_bits = int(np.ceil(np.log2(len(W))))
            record.update(status='completed', rank=head['rank'], state_norm=case['state_norm'],
                          mesh=d, laws=law_results,
                          public_expected_max_bound_numeric_proposal=limit,
                          bound_scope='conditional expectation bound using numerical coefficient norm; not a certificate',
                          equal_fixed_slot_payload=len(ids)*(score_bits+id_bits),
                          score_bits=score_bits, id_bits=id_bits, charged_scores=len(ids),
                          payload_scope='same common public radius/range/mesh; excludes common framing',
                          empirical_mean_below_bound=law_results['independent_unbiased_stochastic']['mean_maximum'] <= limit)
        except Exception as error:
            record['failure'] = f'{type(error).__name__}: {error}'
        rows.append(record)
    valid = [x for x in rows if x['status']=='completed']
    summary = {}
    for law in protocol['laws']:
        summary[law] = dict(median_case_mean_forward=float(np.median([x['laws'][law]['mean_forward'] for x in valid])),
                            median_case_mean_reverse=float(np.median([x['laws'][law]['mean_reverse'] for x in valid])),
                            median_case_mean_maximum=float(np.median([x['laws'][law]['mean_maximum'] for x in valid])))
    summary['stochastic_lower_mean_than_coherent'] = sum(x['laws']['independent_unbiased_stochastic']['mean_maximum'] < x['laws']['coherent_biased_lattice']['mean_maximum'] for x in valid)
    summary['stochastic_lower_mean_than_nearest'] = sum(x['laws']['independent_unbiased_stochastic']['mean_maximum'] < x['laws']['nearest_same_mesh']['mean_maximum'] for x in valid)
    summary['numeric_expectation_bound_exceedances'] = sum(not x['empirical_mean_below_bound'] for x in valid)
    save(R, dict(protocol_sha256=sha(P), source_sha256=sha(Path(__file__)), prescribed_cases=len(protocol['cases']),
                 completed=len(valid), retained_failures=len(rows)-len(valid), summary=summary, rows=rows))
    print(json.dumps(dict(completed=len(valid), failures=len(rows)-len(valid), summary=summary)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run()
