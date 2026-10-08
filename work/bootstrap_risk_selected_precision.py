from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).resolve().parent.parent/'outputs'
seed=202610096; results=[]
for mi,path in enumerate(sorted((OUT/'risk_selected_precision').glob('*.json'))):
 raw=json.loads(path.read_text()); rng=np.random.default_rng(seed+mi)
 for g in [0.002,0.02]:
  rows=[x for x in raw['rows'] if x['grid']==g]
  a=np.array([x['laws']['covariance_selected']['mean_forward'] for x in rows])
  b=np.array([x['laws']['nearest']['mean_forward'] for x in rows])
  indices=rng.integers(0,len(rows),size=(10000,len(rows)))
  ci=np.quantile(a[indices].mean(axis=1)/b[indices].mean(axis=1),[.025,.975])
  results.append(dict(model=raw['metadata']['model'],grid=g,mean_forward_ratio=float(a.mean()/b.mean()),paired_paragraph_bootstrap_95_interval=ci.tolist(),scope='within-ledger paragraph resampling; not an across-dataset or topic-independence guarantee'))
report=dict(seed=seed,repetitions=10000,unit='paired paragraph per model/grid, no cross-grid pooling',results=results)
(OUT/'RISK_SELECTED_PRECISION_BOOTSTRAP.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
