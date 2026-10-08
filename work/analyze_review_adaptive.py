"""Descriptive comparison of every frozen pretrained adaptive bridge cell."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=__import__("runtime_paths").ROOT;OUT=ROOT/'outputs/review_adaptive'
ARMS=('static_uniform_5n','static_zero_leverage_5n','static_doptimal_5n','static_doptimal_pilot_4n','adaptive_fisher_pilot_4n')
def analyze():
    allrows=[];groups=[];numerical=[];raws=[]
    for path in sorted(OUT.glob('*.json')):
        if 'smoke' in path.name.lower() or 'PROTOCOL' in path.name:continue
        obj=json.loads(path.read_text(encoding='utf8'))
        if 'rows' not in obj or not obj.get('metadata',{}).get('completed'):continue
        raws.append(obj); model=obj['metadata']['model']
        for grid in sorted({x['grid'] for x in obj['rows']}):
            rows=[x for x in obj['rows'] if x['grid']==grid]
            complete=[x for x in rows if all(a in x['arms'] and x['arms'][a].get('status')=='complete' for a in ARMS)]
            for x in complete:
                record=dict(model=model,paragraph=x['paragraph'],grid=grid,pilot_logit_oscillation=x['pilot_logit_oscillation'])
                for a in ARMS:
                    z=x['arms'][a];record[a]=z['metrics']['forward'];numerical.append(z)
                allrows.append(record)
            if not complete:continue
            risks=np.array([[x['arms'][a]['metrics']['forward'] for a in ARMS] for x in complete]);rng=np.random.default_rng(2026100843)
            boot=rng.integers(0,len(risks),(10000,len(risks)));row=dict(model=model,grid=grid,planned=len(rows),complete=len(complete),arms={})
            beststatic=risks[:,:3].min(1);adaptive=risks[:,-1]
            for i,a in enumerate(ARMS):
                vals=[x['arms'][a] for x in complete]
                ratio=risks[:,i].mean()/risks[:,1].mean(); bs=risks[boot,i].mean(1)/risks[boot,1].mean(1)
                row['arms'][a]=dict(mean_forward=float(risks[:,i].mean()),median_forward=float(np.median(risks[:,i])),
                     mean_reverse=float(np.mean([v['metrics']['reverse'] for v in vals])),
                     mean_tail_absolute_error=float(np.mean([v['metrics']['tail_mass_absolute_error'] for v in vals])),
                     top1_agreements=sum(v['metrics']['top1_agreement'] for v in vals),ratio_to_zero_leverage=ratio,
                     ratio_bootstrap95=list(np.quantile(bs,[.025,.975]).astype(float)),
                     median_condition=float(np.median([v['condition'] for v in vals])),
                     median_quadratic_relative_error=float(np.median([abs(v['decoded_quadratic_forward']-v['metrics']['forward'])/max(v['metrics']['forward'],1e-30) for v in vals])),
                     max_quadratic_relative_error=float(np.max([abs(v['decoded_quadratic_forward']-v['metrics']['forward'])/max(v['metrics']['forward'],1e-30) for v in vals])),
                     median_quadratic_reverse_relative_error=float(np.median([abs(v['decoded_quadratic_forward']-v['metrics']['reverse'])/max(v['metrics']['reverse'],1e-30) for v in vals])),
                     max_quadratic_reverse_relative_error=float(np.max([abs(v['decoded_quadratic_forward']-v['metrics']['reverse'])/max(v['metrics']['reverse'],1e-30) for v in vals])),
                     quadratic_to_mean_forward=float(np.mean([v['decoded_quadratic_forward'] for v in vals])/risks[:,i].mean()),
                     mean_decode_seconds=float(np.mean([v['decode_seconds'] for v in vals])),
                     Fisher_floor_active_count=sum(v['relative_anchor_covariance']['fisher_min_eigenvalue']<v['relative_anchor_covariance']['comparison_eigenvalue_floor'] for v in vals),
                     median_relative_anchor_min=float(np.median([v['relative_anchor_covariance']['relative_min'] for v in vals])))
            row.update(adaptive_lower_than_uniform=int((adaptive<risks[:,0]).sum()),adaptive_lower_than_zero_leverage=int((adaptive<risks[:,1]).sum()),
                adaptive_lower_than_doptimal=int((adaptive<risks[:,2]).sum()),adaptive_lower_than_static_oracle=int((adaptive<beststatic).sum()),
                adaptive_to_best_fixed_static_mean=float(adaptive.mean()/risks[:,:3].mean(0).min()),
                best_fixed_static_arm=ARMS[int(np.argmin(risks[:,:3].mean(0)))],
                adaptive_lower_than_same_pilot_control=int((adaptive<risks[:,3]).sum()),
                adaptive_to_same_pilot_control_mean=float(adaptive.mean()/risks[:,3].mean()),
                median_covariance_coverage_gain_over_zero_leverage=float(np.median([x['arms'][ARMS[-1]]['relative_anchor_covariance']['relative_min']/x['arms'][ARMS[1]]['relative_anchor_covariance']['relative_min'] for x in complete])),
                adaptive_to_casewise_static_oracle=float(adaptive.mean()/beststatic.mean()),
                median_pilot_logit_oscillation=float(np.median([x['pilot_logit_oscillation'] for x in complete])),
                max_pilot_box_truth_violation=max(x['pilot_box_truth_violation'] for x in complete),
                median_adaptive_design_seconds=float(np.median([x['adaptive_design_seconds'] for x in complete])))
            groups.append(row)
    projected=[x['projection'] for x in numerical if x['projection'].get('active')]
    result=dict(completed_models=len(raws),complete_cells=len(allrows),groups=groups,
       projection=dict(active_count=len(projected),failed=sum(not x['success'] for x in projected),
         max_primal_violation=max((x['box_violation'] for x in projected),default=0.),
         max_relative_stationarity=max((x['relative_stationarity'] for x in projected),default=0.),
         max_relative_duality_gap=max((x['relative_duality_gap'] for x in projected),default=0.),
         max_complementarity=max((x.get('complementarity_infinity_norm',0.) for x in projected),default=0.),
         max_active_constraints=max((x['active_constraints'] for x in projected),default=0)),
       resource=[dict(model=x['metadata']['model'],elapsed_seconds=x['metadata']['elapsed_seconds'],
         peak_cuda_allocated_bytes=x['metadata']['peak_cuda_allocated_bytes'],
         static_setup_peak_cuda_allocated_bytes=x['metadata']['static_setup_peak_cuda_allocated_bytes'],
         static_design_seconds=x['metadata']['static_design_seconds']) for x in raws],
       interpretation='Native LAMBADA held states, chosen-coordinate disclosure. Every arm uses5(r+1) returned scalar requests. Repeated IDs charged. Approximate public designs and numerical projection; no BSS certificate. Bootstrap resamples the64paragraphs within each model/grid group; groups share paragraphs and are dependent.')
    if allrows:
        for key,baseline in [('zero_leverage','static_zero_leverage_5n'),('uniform','static_uniform_5n')]:
            x=np.log([v['pilot_logit_oscillation']+1e-20 for v in allrows]);y=np.log([v['adaptive_fisher_pilot_4n']/v[baseline] for v in allrows])
            rho,p=spearmanr(x,y);result.setdefault('descriptive_geometry_associations',{})[key]=dict(spearman_rho=float(rho),pvalue_omitted='shared contexts and grids')
        with open(OUT/'case_comparisons.csv','w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    names={'EleutherAI/pythia-70m':'P70M','openai-community/gpt2':'GPT-2','Qwen/Qwen2.5-0.5B':'Q0.5B'}
    table=[r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{3pt}',
      r'\begin{tabular}{llrrrr}',r'\toprule',r'Model & $g$ & A/U & A/Z & A/D & A/P\\',r'\midrule']
    for group in groups:
        aa=group['arms'][ARMS[-1]]['mean_forward']
        values=[aa/group['arms'][a]['mean_forward'] for a in ARMS[:-1]]
        table.append(names[group['model']]+' & '+str(group['grid'])+' & '+' & '.join(f'{v:.3f}' for v in values)+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}',
      r'\caption{Adaptive/static mean forward KL on 64 frozen native LAMBADA contexts per group. U: uniform; Z: zero-anchor leverage; D: approximate public D-optimal design, each using $5(r+1)$ fine requests. P: the same D-design with a shared $r+1$ pilot and $4(r+1)$ fine requests. Adaptive A uses the same pilot/fine allocation. Every request, including repeated IDs, is charged. Here $g$ is the actual nearest-grid spacing; values below one favor A.}\label{tab:nativeadaptive}',r'\end{table}'])
    (OUT/'native_adaptive_main_table.tex').write_text('\n'.join(table)+'\n',encoding='utf8')
    print(json.dumps(dict(models=len(raws),cells=len(allrows),groups=len(groups),projection=result['projection'])))
if __name__=='__main__':analyze()
