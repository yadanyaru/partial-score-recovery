"""Passage-level absolute risk, task utility and measured cost; all frozen cases."""
from pathlib import Path
import csv,json
import numpy as np

ROOT=__import__("runtime_paths").ROOT;OUT=ROOT/'outputs'
def main():
    groups=[];passages=[];costs=[]
    rng=np.random.default_rng(2026100821)
    bootstrap=rng.integers(0,128,size=(10000,128))
    for file in sorted((OUT/'review_precision').glob('*.json')):
        raw=json.loads(file.read_text());meta=raw['metadata'];assert meta['completed'] and not raw['failures']
        name=meta['model'];rows=raw['rows'];assert len(rows)==meta['prescribed_rows']
        for g in sorted({r['grid'] for r in rows}):
            local=[]
            for j in range(128):
                seq=sorted((r for r in rows if r['paragraph']==j and r['grid']==g),key=lambda r:r['answer_step'])
                assert [r['answer_step'] for r in seq]==list(range(len(seq)))
                item=dict(model=name,grid=g,paragraph=j,source_row=seq[0]['source_row'],
                    teacher_word_NLL=sum(r['teacher_gold_NLL'] for r in seq),
                    teacher_word_accuracy=float(all(r['teacher_correct'] for r in seq)),
                    first_selected_kernel=seq[0]['selected_kernel'],answer_tokens=len(seq))
                for law in ['nearest','independent','shared','selected']:
                    first=seq[0]['laws'][law]
                    for k in ['forward','reverse','maximum','TV','tail_mass_error','gold_NLL','gold_accuracy','teacher_top1_agreement']:
                        item[f'{law}_{k}']=first[k]
                    item[f'{law}_word_NLL']=sum(r['laws'][law]['gold_NLL'] for r in seq)
                    item[f'{law}_word_accuracy']=float(np.prod([r['laws'][law]['gold_accuracy'] for r in seq]))
                    item[f'{law}_forward_MC_SE']=first['forward_MC_SE']
                passages.append(item);local.append(item)
            arr=lambda key:np.array([x[key] for x in local])
            near=arr('nearest_forward');selected=arr('selected_forward')
            ratios=selected[bootstrap].mean(1)/near[bootstrap].mean(1)
            def difference_interval(key):
                delta=arr('selected_'+key)-arr('nearest_'+key)
                return dict(mean=float(delta.mean()),interval95=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist())
            group=dict(model=name,grid=g,mesh=g/2,passages=128,score_count=meta['selected_count'],
                disclosure_fraction=meta['disclosure_fraction'],selected_to_nearest_forward=float(selected.mean()/near.mean()),
                ratio_interval95=np.quantile(ratios,[.025,.975]).tolist(),
                absolute_forward_reduction=float((near-selected).mean()),
                selected_lower_cases=int((selected<near).sum()),selected_equal_cases=int((selected==near).sum()),
                selected_higher_cases=int((selected>near).sum()),
                choices={law:sum(x['first_selected_kernel']==law for x in local) for law in ['nearest','shared','independent']},
                teacher_word_NLL=float(arr('teacher_word_NLL').mean()),teacher_word_accuracy=float(arr('teacher_word_accuracy').mean()),
                word_NLL_difference=difference_interval('word_NLL'),word_accuracy_difference=difference_interval('word_accuracy'),
                tail_mass_error_difference=difference_interval('tail_mass_error'))
            for law in ['nearest','independent','shared','selected']:
                group[law]={k:float(arr(f'{law}_{k}').mean()) for k in ['forward','reverse','maximum','TV','tail_mass_error',
                    'gold_NLL','gold_accuracy','teacher_top1_agreement','word_NLL','word_accuracy']}
                group[law]['group_forward_MC_SE']=float(np.linalg.norm(arr(f'{law}_forward_MC_SE'))/128)
            groups.append(group)
        timings=raw['timings'];cost=dict(model=name,rank=meta['rank'],V=meta['V'],score_count=meta['selected_count'],
            precision=meta['precision'],hardware=meta['hardware'],elapsed=meta['elapsed'])
        for operation in ['nearest_encode','sample512_select_encode','dense_select']:
            selected=[x[operation] for x in timings if operation in x]
            values=[v for x in selected for v in x['milliseconds']]
            cost[operation]=dict(contexts=len(selected),repetitions=len(values),median_ms=float(np.median(values)),
                interval_ms=np.quantile(values,[.25,.75]).tolist(),max_additional_peak_bytes=max(x['additional_peak_bytes'] for x in selected),
                max_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in selected))
        cost['dense_to_pair_time_ratio']=cost['dense_select']['median_ms']/cost['sample512_select_encode']['median_ms']
        costs.append(cost)
    assert len(groups)==15 and len(passages)==1920
    report=dict(groups=groups,costs=costs,bootstrap=dict(unit='passage',replicates=10000,seed=2026100821),
        scope='128 uniformly sampled LAMBADA passages; paired within this sample; stochastic kernels128draws, policy512pairs',
        word_task='expected exact match of all answer tokens with independent per-prefix packets; answer NLLteacher-forced')
    (OUT/'REVIEW_PRECISION_UTILITY_SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUT/'REVIEW_PRECISION_PASSAGE_RESULTS.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(passages[0]));w.writeheader();w.writerows(passages)
    lines=['# New LAMBADA precision and task results','',
        'All128 passages per model at all five meshes. Absolute KL is in nats. Task exact match refers to the complete answer-token sequence.','',
        '| Model | g | Nearest KL | Selected KL | Ratio [95%] | Tail nearest/selected | Word accuracy nearest/selected | Word NLL change |',
        '|---|---:|---:|---:|---|---|---|---:|']
    for x in groups:
        n=x['nearest'];s=x['selected'];ci=x['ratio_interval95']
        lines.append(f"| {x['model']} | {x['grid']:g} | {n['forward']:.5g} | {s['forward']:.5g} | {x['selected_to_nearest_forward']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | {n['tail_mass_error']:.4g}/{s['tail_mass_error']:.4g} | {n['word_accuracy']:.4f}/{s['word_accuracy']:.4f} | {x['word_NLL_difference']['mean']:+.5g} |")
    lines+=['','## Cost (synchronized GPU measurements)','',
        '| Model | Nearest ms | Pair512 ms | Dense ms | Dense/pair | Pair/dense additional peak MiB |','|---|---:|---:|---:|---:|---:|']
    for x in costs:
        a=x['sample512_select_encode'];d=x['dense_select']
        lines.append(f"| {x['model']} | {x['nearest_encode']['median_ms']:.3f} | {a['median_ms']:.3f} | {d['median_ms']:.3f} | {x['dense_to_pair_time_ratio']:.1f} | {a['max_additional_peak_bytes']/2**20:.2f}/{d['max_additional_peak_bytes']/2**20:.2f} |")
    (OUT/'REVIEW_PRECISION_UTILITY_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
