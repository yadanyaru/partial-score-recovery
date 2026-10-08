from pathlib import Path
import hashlib
import json
import numpy as np

root=__import__("runtime_paths").ROOT
protocol_path=root/'outputs'/'PRETRAINED_STOCHASTIC_PRECISION_PROTOCOL.json'
protocol=json.loads(protocol_path.read_text())
groups=[]
models=[]
for plan in protocol['models']:
    path=root/'outputs'/'pretrained_stochastic_precision'/f"{plan['model'].replace('/','__')}.json"
    result=json.loads(path.read_text())
    rows=result['rows']
    expected={(p,g) for p in protocol['paragraphs'] for g in protocol['grids']}
    assert len(rows)==112 and {(x['paragraph'],x['grid']) for x in rows}==expected
    assert result['metadata']['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert result['metadata']['source_sha256']==protocol['source_sha256']
    assert all(x['status']=='completed' for x in rows)
    models.append(dict(model=plan['model'],file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       coefficient_norm_max_numeric=result['metadata']['design']['coefficient_norm_max_numeric'],
                       successful=112,prescribed=112))
    for grid in protocol['grids']:
        part=[x for x in rows if x['grid']==grid]
        for law in protocol['response_laws']:
            statistics={}
            for metric in ('mean_forward','mean_reverse','mean_maximum'):
                values=np.array([x['laws'][law][metric] for x in part])
                statistics[metric]=dict(mean=float(values.mean()),median=float(np.median(values)),maximum=float(values.max()))
            groups.append(dict(model=plan['model'],grid=grid,law=law,successful=len(part),prescribed=56,
                                statistics=statistics,score_count=result['metadata']['design']['selected_count'],
                                fixed_payload_bits=part[0]['same_fixed_slot_payload']))
        a='independent_unbiased_stochastic'
        for comparator in ('nearest_same_mesh','coherent_biased_lattice'):
            count=sum(x['laws'][a]['mean_maximum']<x['laws'][comparator]['mean_maximum'] for x in part)
            print(json.dumps(dict(model=plan['model'],grid=grid,comparator=comparator,
                                  stochastic_lower_case_mean=count,denominator=56)))
        assert len({x['same_fixed_slot_payload'] for x in part})==1
        models[-1].setdefault('comparisons',[]).append(dict(grid=grid,
            stoch_lower_than_nearest=sum(x['laws'][a]['mean_maximum']<x['laws']['nearest_same_mesh']['mean_maximum'] for x in part),
            stoch_lower_than_coherent=sum(x['laws'][a]['mean_maximum']<x['laws']['coherent_biased_lattice']['mean_maximum'] for x in part),
            empirical_mean_bound_exceedances=sum(x['laws'][a]['mean_maximum']>x['numeric_expected_max_bound'] for x in part)))
summary=dict(status='complete336prescribed numerical cases',models=models,groups=groups,
              primary_estimator='mean of32stochastic reconstructions within each fixed paragraph; then descriptive paragraph summaries',
              limitations=['reused pre-specified corpus; not independent articles or unseen-text holdout',
                           'synthetic response laws on native states; not existing-API prevalence',
                           'ordinary nearest precision has smaller pointwise error range',
                           'public leverage subset and numerical inverse, not exact D-optimal/BSS',
                           'numeric coefficient bound not enclosed certificate'])
dest=root/'outputs'/'PRETRAINED_STOCHASTIC_PRECISION_SUMMARY.json'
dest.write_text(json.dumps(summary,indent=2))
lines=['# Matched response laws on three pretrained heads','',
       'All336prescribed cells completed. Each stochastic cell averages32reconstructions. These are descriptive numerical diagnostics, not native-API guarantees or held-out generalization.','',
       '| Model | g | Response law | Mean forward KL | Mean reverse KL | Scores | Payload bits |',
       '|---|---:|---|---:|---:|---:|---:|']
for group in groups:
    lines.append(f"| {group['model']} | {group['grid']} | {group['law']} | {group['statistics']['mean_forward']['mean']:.8g} | {group['statistics']['mean_reverse']['mean']:.8g} | {group['score_count']} | {group['fixed_payload_bits']} |")
lines+=['','## Matched case comparisons','',json.dumps(models,indent=2),
        '', 'The coherent response is an allowed constructed calibration error. Its comparison does not establish that real APIs exhibit that error. Stochastic rounding is not claimed to improve ordinary nearest rounding. All laws share IDs, decoder, mesh and a common fixed-format payload.']
(root/'outputs'/'PRETRAINED_STOCHASTIC_PRECISION_REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(completed=336,groups=len(groups),models=models)))
