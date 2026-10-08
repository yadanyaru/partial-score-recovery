from pathlib import Path
import json,hashlib,numpy as np
ROOT=__import__("runtime_paths").ROOT;OUT=ROOT/'outputs'
protocol=json.loads((OUT/'RISK_SELECTED_PRECISION_PROTOCOL.json').read_text())
models=[];allrows=[]
for plan in protocol['models']:
    path=OUT/'risk_selected_precision'/f"{plan['model'].replace('/','__')}.json"
    raw=json.loads(path.read_text());assert raw['metadata']['completed'] and len(raw['rows'])==32
    assert all(x['status']=='completed' for x in raw['rows']);groups=[]
    for g in protocol['grids']:
        rows=[x for x in raw['rows'] if x['grid']==g];ratios={};laws={}
        for law in protocol['response_laws']:
            laws[law]={k:float(np.median([x['laws'][law][k] for x in rows])) for k in ['mean_forward','mean_reverse','mean_maximum','relative_quadratic_error']}
        for baseline in ['nearest','independent_unbiased_stochastic','known_shared_coin_exact']:
            ratios[baseline]=dict(median_casewise_forward_ratio=float(np.median([x['laws']['covariance_selected']['mean_forward']/x['laws'][baseline]['mean_forward'] for x in rows])),lower_forward=sum(x['laws']['covariance_selected']['mean_forward']<x['laws'][baseline]['mean_forward'] for x in rows),mean_forward_ratio=float(np.mean([x['laws']['covariance_selected']['mean_forward'] for x in rows])/np.mean([x['laws'][baseline]['mean_forward'] for x in rows])),higher_forward=sum(x['laws']['covariance_selected']['mean_forward']>x['laws'][baseline]['mean_forward'] for x in rows))
        groups.append(dict(grid=g,count=len(rows),laws=laws,ratios=ratios,shared_selected=sum(x['laws']['covariance_selected']['selected_kernel']=='known_shared_coin_exact' for x in rows)))
    models.append(dict(model=plan['model'],groups=groups,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elapsed=raw['metadata']['elapsed']))
    allrows.extend(dict(model=plan['model'],**x) for x in raw['rows'])
summary=dict(completed=len(allrows),models=models,shared_selected=sum(x['laws']['covariance_selected']['selected_kernel']=='known_shared_coin_exact' for x in allrows))
for base in ['nearest','independent_unbiased_stochastic','known_shared_coin_exact']:
    summary[base]=dict(higher_forward=sum(x['laws']['covariance_selected']['mean_forward']>x['laws'][base]['mean_forward'] for x in allrows),lower_forward=sum(x['laws']['covariance_selected']['mean_forward']<x['laws'][base]['mean_forward'] for x in allrows),median_casewise_forward_ratio=float(np.median([x['laws']['covariance_selected']['mean_forward']/x['laws'][base]['mean_forward'] for x in allrows])))
summary['median_shared_covariance_relative_error']=float(np.median([x['laws']['known_shared_coin_exact']['relative_quadratic_error'] for x in allrows]))
(OUT/'RISK_SELECTED_PRECISION_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# Fresh validation of provider-selected precision','',f"All {len(allrows)} prespecified cases completed, no failures. Shared kernel selected in {summary['shared_selected']} cases.",'','| Model | g | Selected / independent | Selected / nearest | Shared selected |','|---|---:|---:|---:|---:|']
for item in models:
    for group in item['groups']:lines.append(f"| {item['model']} | {group['grid']} | {group['ratios']['independent_unbiased_stochastic']['median_casewise_forward_ratio']:.4f} | {group['ratios']['nearest']['median_casewise_forward_ratio']:.4f} | {group['shared_selected']}/16 |")
lines+=['','Selection uses second-moment predictions, never measured KL. The experiment evaluates all three candidates and applies this deterministic provider rule in law; drawing candidate samples during evaluation before computing the selection does not change its distribution because selection depends only on the fixed teacher. The decoder receives neither teacher probabilities nor a selection flag. Exact shared integration; 256 independent draws. Reported ratios compare conditional mean forward loss casewise. Provider matrix computation is offline and not a speed-up claim. Different validation paragraphs are not a guarantee of independence between Wikipedia topics.', '',json.dumps({k:v for k,v in summary.items() if k!='models'},indent=2)]
(OUT/'RISK_SELECTED_PRECISION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='models'}))
