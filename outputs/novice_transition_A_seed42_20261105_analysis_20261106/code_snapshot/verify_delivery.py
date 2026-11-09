"""Read-only verification of the current in-place output manifest."""
from pathlib import Path
import argparse,json,hashlib
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
root=args.root;manifest=json.loads((root/'artifact_manifest.json').read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
for row in manifest['files']:
    p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
assert json.loads((root/'validation.json').read_text(encoding='utf8'))['complete'] is True
print(json.dumps({'passed':True,'verified_files':len(manifest['files']),'root':str(root)},ensure_ascii=False))
