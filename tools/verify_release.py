"""Validate anonymous release contents without network access or model weights."""
from pathlib import Path
import ast,gzip,hashlib,json,re

def verify(root):
    manifest=json.loads((root/'config/reference_manifest.json').read_text())
    for item in manifest:
        path=root/item['path'];data=path.read_bytes()
        if path.suffix=='.gz':data=gzip.decompress(data)
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
    patterns={
        'absolute Windows path':re.compile(r'(?i)\b[A-Z]:[/\\]'),
        'user home path':re.compile(r'/(?:Users|home)/[^/\s]+/'),
        'credential':re.compile(r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}\b'),
        'private SSH key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    }
    findings=[];files=[]
    for path in root.rglob('*'):
        rel=path.relative_to(root)
        if any(x in {'runs','cache','.git','__pycache__','.venv'} for x in rel.parts):continue
        if not path.is_file():continue
        files.append(path)
        assert path.stat().st_size<99*1024**2,rel.as_posix()
        if path.suffix in {'.npz','.png','.pdf'}:continue
        data=path.read_bytes()
        if path.suffix=='.gz':data=gzip.decompress(data)
        text=data.decode('utf-8')
        if path.suffix=='.py':ast.parse(text)
        for label,pattern in patterns.items():
            if pattern.search(text):findings.append((rel.as_posix(),label))
    assert not findings,findings
    report=dict(status='passed',reference_files=len(manifest),release_files=len(files),
                python_sources_parse=True,absolute_personal_paths_found=0,
                credential_patterns_found=0,git_metadata_excluded=True,
                max_file_bytes=max(p.stat().st_size for p in files),
                scope='Repository content; hosting account and future commit metadata must be anonymized separately.')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
