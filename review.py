"""Anonymous review entry point. Run `python review.py --help`."""
from pathlib import Path
import argparse, gzip, hashlib, json, os, shutil, subprocess, sys

REPO=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'work'))
from runtime_paths import CACHE,HF_CACHE

def digest(data):return hashlib.sha256(data).hexdigest()
def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def workspace(name):
    root=REPO/'runs'/name
    (root/'outputs/figures').mkdir(parents=True,exist_ok=True)
    (root/'work').mkdir(exist_ok=True)
    for p in (REPO/'work').glob('*.py'):shutil.copyfile(p,root/'work'/p.name)
    return root

def run(root,script,*args):
    env=os.environ.copy()
    env.update(PREDREC_ROOT=str(root),PREDREC_CACHE=str(CACHE),HF_HOME=str(HF_CACHE),
               PYTHONIOENCODING='utf-8',PYTHONUTF8='1',MPLBACKEND='Agg')
    subprocess.run([sys.executable,str(root/'work'/script),*args],cwd=root,env=env,check=True)

def materialize(name='reference'):
    root=workspace(name)
    manifest=json.loads((REPO/'config/reference_manifest.json').read_text())
    for item in manifest:
        source=REPO/item['path'];data=source.read_bytes()
        if source.suffix=='.gz':data=gzip.decompress(data)
        assert digest(data)==item['sha256'],item['path']
        dest=root/item['target'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert dest.read_bytes()==data,f'Existing workspace differs: {dest.name}; use another workspace'
        else:dest.write_bytes(data)
    return root

def prepare_data(root,download=True):
    """Reconstruct all four corpora using public source rows and exact text hashes."""
    from urllib.request import urlopen
    import io
    import pyarrow.parquet as pq
    selections=json.loads((REPO/'config/data_selections.json').read_text())
    data_cache=CACHE/'datasets';data_cache.mkdir(parents=True,exist_ok=True)
    tables={}
    for name,spec in selections.items():
        domain=spec['domain']
        if domain not in tables:
            if domain=='lambada_openai_test':
                url='https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl'
                local=data_cache/'lambada_test.jsonl'
            else:
                split='validation' if domain.endswith('validation') else 'test'
                url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
                local=data_cache/f'wikitext_{split}.parquet'
            if not local.exists():
                if not download:raise FileNotFoundError('Dataset cache missing; run prepare-data with network access')
                with urlopen(url,timeout=120) as response:local.write_bytes(response.read())
            raw=local.read_bytes()
            if domain=='lambada_openai_test':
                assert digest(raw)=='4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226'
                tables[domain]=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
            else:tables[domain]=pq.read_table(io.BytesIO(raw)).to_pylist()
        records=[]
        for i,(idx,expected) in enumerate(zip(spec['rows'],spec['text_sha256'])):
            text=tables[domain][idx]['text'];assert digest(text.encode('utf-8'))==expected,(name,i)
            if domain=='lambada_openai_test':
                import re
                match=re.search(r'\S+\s*$',text);answer=match.group().rstrip()
                prefix=text[:match.start()];context=prefix.rstrip();separator=prefix[len(context):]
                values=dict(sample_index=i,domain=domain,task='final_word_cloze',language='en',
                            source_row=idx,source_row_indexing='zero_based',text=text,
                            context=context,answer=answer,continuation=separator+answer)
            else:values=dict(domain=domain,text=text,source_row=idx)
            records.append({key:values[key] for key in spec['fields']})
        encoded=json.dumps(records,ensure_ascii=False,indent=2).encode('utf-8')
        choices=[encoded,encoded+b'\n']
        data=next((b for b in choices if digest(b)==spec['sha256']),None)
        assert data is not None,f'Corpus serialization mismatch: {name}'
        path=root/'work'/f'{name}.json'
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
    print('Reconstructed four corpora; every selected text and corpus checksum matches.')

def cpu_checks(name):
    root=workspace(name)
    run(root,'check_review_instance_theory.py')
    run(root,'check_adaptive_disclosure_theory.py')
    run(root,'check_static_adaptive_separation.py')
    run(root,'check_adaptive_floor_thresholds.py')
    print('CPU mathematical checks completed. Results are in runs/'+name+'/outputs.')

def analysis(name):
    root=materialize(name)
    for script in ['analyze_review_precision.py','analyze_pretrained_disclosure_formal.py',
                   'analyze_stochastic_precision.py','analyze_pretrained_shared_coin.py',
                   'analyze_risk_selected_precision.py','analyze_sampled_precision.py',
                   'analyze_review_adaptive.py']:
        run(root,script)
    write_json(root/'analysis_complete.json',dict(status='completed',inputs='included reference outcomes'))

def download_models():
    os.environ['HF_HOME']=str(HF_CACHE);os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import snapshot_download
    from frozen_models import REVISIONS
    for model,revision in REVISIONS.items():
        snapshot_download(repo_id=model,revision=revision)
    print('Downloaded all three pinned model revisions.')

def require_gpu():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('This stage needs CUDA; CPU checks and reference analyses do not.')

def stage(name,which):
    require_gpu() if which not in {'low-disclosure','topk'} else None
    root=workspace(name)
    # Pinned public designs are inputs; original outcomes are never silently
    # reused as outputs in a fresh computational stage.
    ref=materialize('_inputs')
    seed='RISK_SELECTED_PRECISION_PROTOCOL.json'
    target=root/'outputs'/seed
    if not target.exists():shutil.copyfile(ref/'outputs'/seed,target)
    if which=='low-disclosure':
        protocol=root/'outputs/review_low_disclosure_protocol.json'
        if not protocol.exists():run(root,'review_low_disclosure.py','--freeze')
        run(root,'review_low_disclosure.py');return
    if which in {'states','precision','shared','confirmation','sampling','baseline','response'}:
        prepare_data(root)
    if which=='states':run(root,'review_prepare_states.py')
    elif which=='precision':
        if not (root/'outputs/REVIEW_STATE_PROTOCOL.json').exists():run(root,'review_prepare_states.py')
        if not (root/'outputs/REVIEW_PRECISION_UTILITY_PROTOCOL.json').exists():run(root,'review_precision_utility.py','--freeze')
        run(root,'review_precision_utility.py','--run')
        run(root,'analyze_review_precision.py')
    elif which in {'topk','adaptive'}:
        if not (root/'outputs/REVIEW_STATE_PROTOCOL.json').exists():
            raise RuntimeError('Run the states stage in this workspace first.')
        script='review_topk_interface.py' if which=='topk' else 'review_adaptive_bridge.py'
        proto='REVIEW_TOPK_PROTOCOL.json' if which=='topk' else 'REVIEW_ADAPTIVE_PROTOCOL.json'
        if not (root/'outputs'/proto).exists():run(root,script,'--freeze')
        if which=='adaptive':
            from frozen_models import REVISIONS
            for model in REVISIONS:run(root,script,'--model',model)
            run(root,'analyze_review_adaptive.py')
        else:run(root,script)
    elif which=='baseline':
        shutil.copyfile(ref/'outputs/PRETRAINED_DISCLOSURE_FORMAL_PROTOCOL.json',root/'outputs/PRETRAINED_DISCLOSURE_FORMAL_PROTOCOL.json')
        shutil.copyfile(ref/'outputs/PRETRAINED_DISCLOSURE_IO_REPAIR_ADDENDUM.json',root/'outputs/PRETRAINED_DISCLOSURE_IO_REPAIR_ADDENDUM.json')
        run(root,'pretrained_disclosure_formal.py')
    elif which in {'shared','confirmation','sampling','response'}:
        if which!='sampling':
            baseline=root/'outputs/pretrained_disclosure_formal'
            if not baseline.exists():shutil.copytree(ref/'outputs/pretrained_disclosure_formal',baseline)
        script,proto={
            'shared':('pretrained_shared_coin.py','PRETRAINED_SHARED_COIN_PROTOCOL.json'),
            'confirmation':('risk_selected_precision_pretrained.py','RISK_SELECTED_PRECISION_PROTOCOL.json'),
            'sampling':('sample_precision_ablation.py','SAMPLED_PRECISION_PROTOCOL.json'),
            'response':('pretrained_stochastic_precision.py','PRETRAINED_STOCHASTIC_PRECISION_PROTOCOL.json')}[which]
        if which=='sampling' and not (root/'outputs/risk_selected_precision').exists():
            raise RuntimeError('Run confirmation in this workspace first.')
        # The seed protocol carries reusable IDs; confirmation must recreate its
        # source/corpus provenance before executing.
        if which=='confirmation' and not (root/'outputs/risk_selected_precision').exists():
            target.unlink()
        if not (root/'outputs'/proto).exists():run(root,script,'--freeze')
        run(root,script,'--run')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','smoke','materialize','analyze',
                                     'prepare-data','download-models','run'])
    p.add_argument('--workspace',default=None,help='Subdirectory below runs/ (fresh name for independent reruns)')
    p.add_argument('--stage',choices=['low-disclosure','states','precision','topk','adaptive','baseline',
                                     'response','shared','confirmation','sampling'])
    a=p.parse_args();name=a.workspace or ('reference' if a.command in {'materialize','analyze'} else 'rerun')
    if name in {'.','..'} or '/' in name or '\\' in name:p.error('Workspace must be a single directory name')
    if a.command=='verify':
        from tools.verify_release import verify
        verify(REPO)
    elif a.command=='smoke':cpu_checks(a.workspace or 'cpu-smoke')
    elif a.command=='materialize':materialize(name)
    elif a.command=='analyze':analysis(name)
    elif a.command=='prepare-data':prepare_data(workspace(name))
    elif a.command=='download-models':download_models()
    elif a.command=='run':
        if not a.stage:p.error('run requires --stage')
        stage(name,a.stage)

if __name__=='__main__':main()
