"""Strict completed-ledger analysis; failures retained, descriptive paragraph bootstrap."""
import pathlib,json,hashlib,numpy as np
ROOT=__import__("runtime_paths").ROOT
protocol=json.loads((ROOT/'outputs/PRETRAINED_DISCLOSURE_FORMAL_PROTOCOL.json').read_text())
repair=json.loads((ROOT/'outputs/PRETRAINED_DISCLOSURE_IO_REPAIR_ADDENDUM.json').read_text())
summary={'protocol_sha256':hashlib.sha256((ROOT/'outputs/PRETRAINED_DISCLOSURE_FORMAL_PROTOCOL.json').read_bytes()).hexdigest(),'io_repair_addendum_sha256':hashlib.sha256((ROOT/'outputs/PRETRAINED_DISCLOSURE_IO_REPAIR_ADDENDUM.json').read_bytes()).hexdigest(),'scope':'approximate float64-head diagnostics, three frozen models; paragraph bootstrap descriptive because shared source articles; no native API risk certificate','models':{}}
rng=np.random.default_rng(20261007)
for model in protocol['models']:
 path=ROOT/'outputs/pretrained_disclosure_formal'/f"{model.replace('/','__')}.json"
 data=json.loads(path.read_text());meta=data['metadata'];rows=data['rows']
 assert meta['completed'] and len(rows)==1792,(model,meta['completed'],len(rows))
 allowed={protocol['source_hashes']['work/pretrained_disclosure_formal.py']}
 if model=='EleutherAI/pythia-70m': allowed.add(repair['repaired_driver_sha256'])
 assert meta['code_sha256'] in allowed
 assert meta['corpus_sha256']==protocol['source_hashes']['work/corpus.json']
 assert meta['frozen_models_sha256']==protocol['source_hashes']['work/frozen_models.py']
 expected={(p,s,m,g) for p in range(8,64) for s in protocol['seeds'] for m in protocol['methods'][:-1] for g in protocol['grids']}
 expected|={(p,None,'passive_top_dplus1',g) for p in range(8,64) for g in protocol['grids']}
 actual=[(r['paragraph'],r['design_seed'],r['method'],r['grid']) for r in rows]
 assert len(set(actual))==len(actual) and set(actual)==expected
 out={'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'executed_driver_sha256':meta['code_sha256'],'execution_retry_events':meta.get('io_replace_retry_events',[]),'completed_rows':len(rows),'failed_rows':sum(r['status']!='ok' for r in rows),'V':meta['V'],'d':meta['d'],'groups':[],'paired_comparisons':[]}
 for method in protocol['methods']:
  for grid in protocol['grids']:
   group=[r for r in rows if r['method']==method and r['grid']==grid];ok=[r for r in group if r['status']=='ok']
   entry={'method':method,'grid':grid,'prescribed':len(group),'successful':len(ok),'failed':len(group)-len(ok),'conditional_on_success':len(ok)!=len(group)}
   for metric in ['forward_kl','reverse_kl','tail_mass_absolute_error','clean_logprob_max_error','native_fp32_logprob_discrepancy','fixed_width_score_payload_bits','fixed_width_total_payload_bits_excluding_header','score_count']:
    if ok:
     values=[r[metric] for r in ok];entry[metric]={'mean':float(np.mean(values)),'median':float(np.median(values)),'max':float(max(values))}
   out['groups'].append(entry)
 for grid in protocol['grids']:
  for comparator in ['public_qr','random_square','random_oversampled','public_leverage','passive_top_dplus1']:
   paired=[]
   for p in range(8,64):
    aa=[r for r in rows if r['paragraph']==p and r['method']=='public_spanner' and r['grid']==grid];bb=[r for r in rows if r['paragraph']==p and r['method']==comparator and r['grid']==grid]
    if all(r['status']=='ok' for r in aa+bb):
     paired.append((p,np.mean([r['forward_kl'] for r in aa]),np.mean([r['forward_kl'] for r in bb])))
   entry={'grid':grid,'comparator':comparator,'paired_paragraphs':len(paired),'missing_paragraphs':56-len(paired),'different_score_count':comparator in ['random_oversampled','public_leverage'],'scope':'descriptive only; design seeds averaged within paragraph, paragraph clusters may share articles'}
   if paired:
    delta=np.array([a-b for _,a,b in paired]);samples=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
    entry.update(mean_forward_kl_difference=float(delta.mean()),descriptive_bootstrap_95=[float(x) for x in np.quantile(samples,[.025,.975])],paragraphs_spanner_lower=int(sum(a<b for _,a,b in paired)))
   out['paired_comparisons'].append(entry)
 summary['models'][model]=out
(ROOT/'outputs/PRETRAINED_DISCLOSURE_FORMAL_SUMMARY.json').write_text(json.dumps(summary,indent=2))
lines=['# Frozen three-model selected-score evaluation','','All 5,376 prescribed cells are retained. Results are approximate public-head diagnostics, not native-API or uniform numerical risk certificates. Public rules are classical baselines. Score counts and fixed-width payload bits are reported separately; framing/width headers are excluded. Failed cases are counted and successful-only summaries labeled.','','| Model | Method | Grid | Success / prescribed | Mean forward KL | Mean reverse KL | Mean score count | Mean fixed-width ID+score payload bits |','|---|---|---:|---:|---:|---:|---:|---:|']
for model,datum in summary['models'].items():
 for e in datum['groups']:
  def value(k):return f"{e[k]['mean']:.6g}" if k in e else 'NA'
  lines.append(f"| {model} | {e['method']} | {e['grid']} | {e['successful']} / {e['prescribed']} | {value('forward_kl')} | {value('reverse_kl')} | {value('score_count')} | {value('fixed_width_total_payload_bits_excluding_header')} |")
lines+=['','Paired bootstrap results in the JSON are descriptive paragraph resampling summaries. Paragraphs are not certified independent articles. The dataset is pre-specified, not an unseen-text holdout. No method-superiority conclusion is hardcoded.']
(ROOT/'outputs/PRETRAINED_DISCLOSURE_FORMAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('completed analysis',sum(x['completed_rows'] for x in summary['models'].values()))
