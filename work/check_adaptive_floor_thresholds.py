import numpy as np,json,pathlib
rng=np.random.default_rng(792)
records=[]
for g in (.0001,.001,.01,.1,.5,1.):
 for trial in range(30):
  phases=rng.uniform(-g,g,size=2);boundaries=[1/3,2/3]
  for coordinate in (0,1):
   lo=int(np.floor((np.log(1/3)-phases[coordinate])/g-.5))-2
   hi=int(np.ceil((np.log(2/3)-phases[coordinate])/g-.5))+2
   thresholds=np.exp(phases[coordinate]+(np.arange(lo,hi+1)+.5)*g)
   if coordinate:thresholds=1-thresholds
   boundaries.extend(float(x) for x in thresholds if 1/3<x<2/3)
  boundaries=np.sort(boundaries);gaps=np.diff(boundaries);j=int(np.argmax(gaps));L=float(gaps[j]);p0=(boundaries[j]+boundaries[j+1])/2;e=g/1000
  ps=[p0-e/2,p0+e/2];codes=[];minimum_margin=np.inf
  for p in ps:
   ell=np.array([np.log(p),np.log1p(-p)])
   code=np.floor((ell-phases)/g+.5).astype(np.int64);codes.append(code)
   margin=g/2-abs(ell-(phases+code*g));minimum_margin=min(minimum_margin,float(margin.min()))
  assert L>=g/15-1e-13 and np.array_equal(codes[0],codes[1])
  assert minimum_margin>=g/40-1e-13
  records.append({'g':g,'L_over_g':L/g,'endpoint_score_margin_over_g':minimum_margin/g})
result={'scope':'180 CPU endpoint threshold diagnostics only; no real-head phase-independence or GPU test',
        'checks':len(records),'min_interval_over_g':min(x['L_over_g'] for x in records),
        'min_margin_over_g':min(x['endpoint_score_margin_over_g'] for x in records)}
pathlib.Path('outputs/adaptive_floor_threshold_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
