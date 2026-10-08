from pathlib import Path
import json,hashlib,numpy as np
root=__import__("runtime_paths").ROOT
protocol=json.loads((root/'outputs/PRETRAINED_SHARED_COIN_PROTOCOL.json').read_text())
models=[];allrows=[]
for plan in protocol['models']:
    path=root/'outputs/pretrained_shared_coin'/f"{plan['model'].replace('/','__')}.json"
    raw=json.loads(path.read_text());rows=[x for x in raw['rows'] if x['status']=='completed']
    assert len(raw['rows'])==112 and len(rows)==112 and raw['metadata']['completed']
    groups=[]
    for g in protocol['grids']:
        subset=[x for x in rows if x['grid']==g];stats={}
        for law in protocol['response_laws']:
            stats[law]={k:float(np.median([x['laws'][law][k] for x in subset])) for k in ['mean_forward','mean_reverse','mean_maximum','relative_quadratic_error']}
            stats[law]['maximum_relative_quadratic_error']=max(x['laws'][law]['relative_quadratic_error'] for x in subset)
        ratios=[x['laws']['known_shared_coin_exact']['mean_forward']/x['laws']['independent_unbiased_stochastic']['mean_forward'] for x in subset]
        groups.append(dict(grid=g,count=len(subset),laws=stats,median_shared_to_independent_forward=float(np.median(ratios)),shared_larger_forward=sum(t>1 for t in ratios)))
    models.append(dict(model=plan['model'],groups=groups,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    allrows.extend(dict(model=plan['model'],**x) for x in rows)
summary=dict(completed=len(allrows),models=models,shared_larger_forward=sum(x['laws']['known_shared_coin_exact']['mean_forward']>x['laws']['independent_unbiased_stochastic']['mean_forward'] for x in allrows),median_shared_relative_quadratic_error=float(np.median([x['laws']['known_shared_coin_exact']['relative_quadratic_error'] for x in allrows])))
(root/'outputs/PRETRAINED_SHARED_COIN_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# Known shared-coin kernel on pretrained heads','',f"All {len(allrows)} prescribed cases completed. Scalar rounding marginals, IDs, decoder, mesh and fixed-slot payload match. Shared expectation is exact interval integration; independent expectation uses 256 draws. All native contexts reuse the earlier frozen corpus, not an unseen text holdout.",'','| Model | g | Shared/independent median forward ratio | Shared covariance median relative error | Shared larger cases |','|---|---:|---:|---:|---:|']
for item in models:
    for group in item['groups']:lines.append(f"| {item['model']} | {group['grid']} | {group['median_shared_to_independent_forward']:.6g} | {group['laws']['known_shared_coin_exact']['relative_quadratic_error']:.6g} | {group['shared_larger_forward']}/56 |")
lines.extend(['','These compare prescribed linear reconstructions. They do not establish optimal-decoder dominance, a prevalence claim about API noise, or worst-case lower bounds on pretrained heads. Full minimax proofs concern explicit constructed heads.'])
(root/'outputs/PRETRAINED_SHARED_COIN_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary))
