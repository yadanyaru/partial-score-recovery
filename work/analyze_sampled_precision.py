from pathlib import Path
import json,hashlib,numpy as np
ROOT=__import__("runtime_paths").ROOT;OUT=ROOT/'outputs'
protocol=json.loads((OUT/'SAMPLED_PRECISION_PROTOCOL.json').read_text());allrows=[];models=[]
for item in protocol['models']:
    path=OUT/'sampled_precision'/f"{item['plan']['model'].replace('/','__')}.json"
    raw=json.loads(path.read_text());assert raw['metadata']['completed'] and len(raw['rows'])==768
    assert not raw['failures'] and all(x['status']=='completed' for x in raw['rows'])
    groups=[]
    for g in [.002,.02]:
        for n in protocol['sample_counts']:
            rows=[x for x in raw['rows'] if x['grid']==g and x['pairs']==n]
            groups.append(dict(grid=g,pairs=n,count=len(rows),mean_ratio_to_nearest=sum(x['selected_forward'] for x in rows)/sum(x['nearest_forward'] for x in rows),mean_ratio_to_dense=sum(x['selected_forward'] for x in rows)/sum(x['dense_selected_forward'] for x in rows),dense_choice_agreement=sum(x['chosen']==x['dense_chosen'] for x in rows)/len(rows),higher_than_nearest=sum(x['selected_forward']>x['nearest_forward'] for x in rows),lower_than_nearest=sum(x['selected_forward']<x['nearest_forward'] for x in rows),equal_to_nearest=sum(x['selected_forward']==x['nearest_forward'] for x in rows)))
    models.append(dict(model=raw['metadata']['model'],groups=groups,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));allrows+=raw['rows']
pooled=[]
for n in protocol['sample_counts']:
    rows=[x for x in allrows if x['pairs']==n]
    pooled.append(dict(pairs=n,replicated_decisions=len(rows),dense_choice_agreement=sum(x['chosen']==x['dense_chosen'] for x in rows)/len(rows),higher_than_nearest=sum(x['selected_forward']>x['nearest_forward'] for x in rows),mean_ratio_to_nearest=sum(x['selected_forward'] for x in rows)/sum(x['nearest_forward'] for x in rows)))
report=dict(completed_decisions=len(allrows),distinct_confirmation_cells=96,provider_seed_replicates=8,scope=protocol['scope'],models=models,pooled=pooled)
(OUT/'SAMPLED_PRECISION_SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# Teacher-pair selection implementation ablation','','All 2304 decisions retained: 96 existing cells, three prespecified budgets, eight provider sampling seeds. Conditional rounding losses come from the preserved confirmation ledger, not new rounding simulations. This is not an independent heldout confirmation. Replicates and nested budgets are dependent; do not count these as 2304 independent data points.','','| Model | g | Pairs | Group mean / nearest | Agreement | Higher /128 |','|---|---:|---:|---:|---:|---:|']
for m in models:
    for g in m['groups']:lines.append(f"| {m['model']} | {g['grid']} | {g['pairs']} | {g['mean_ratio_to_nearest']:.6f} | {g['dense_choice_agreement']:.6f} | {g['higher_than_nearest']}/128 |")
lines+=['','Sampling uses the teacher distribution inside the provider; the receiver and transmitted packet are unchanged. No dense Fisher matrix or risk matrix is formed. For each fixed teacher and kernel, the pairwise quadratic estimate is unbiased; minimization of these estimates is not unbiased selection. Small-budget errors and increases are retained. The one independent candidate conditional loss remains an inherited 256-draw estimate. No latency comparison is inferred from total script execution time.']
(OUT/'SAMPLED_PRECISION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(report['pooled']))
