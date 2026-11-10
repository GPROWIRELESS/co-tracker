"""Validate local report links and freeze a hash inventory; no source mutations."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse
import argparse
import hashlib
import json

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        for k in ('src','href'):
            if k in a:self.paths.append(a[k])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args();root=args.root.resolve()
    page=root/'report.html';assert page.is_file()
    h=Links();h.feed(page.read_text(encoding='utf-8'))
    errors=[];checked=0
    for link in h.paths:
        parsed=urlparse(link)
        if parsed.scheme or parsed.netloc:continue
        if not parsed.path:
            if parsed.fragment and parsed.fragment not in h.ids:errors.append(link)
        else:
            checked+=1
            if not (root/unquote(parsed.path)).exists():errors.append(link)
    assert not errors,errors
    v=json.loads((root/'validation/independent_validation.json').read_text(encoding='utf-8'))
    assert v['passed']
    fm=json.loads((root/'figure_manifest.json').read_text(encoding='utf-8'))
    assert fm['n_figure_pages']==24
    for rel,expected in fm['input_sha256'].items():assert sha(root/rel)==expected
    for fig in fm['figures']:
        for rel in fig['files']:assert (root/rel).is_file() and (root/rel).stat().st_size>1000
    prov=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    unchanged={path:sha(Path(path))==expected for path,expected in prov['input_sha256'].items()}
    assert all(unchanged.values())
    delivery=dict(passed=True,local_html_links_checked=checked,broken_links=[],figure_pages=24,
                  png_count=len(list((root/'figures').glob('*.png'))),pdf_count=len(list((root/'figures').glob('*.pdf'))),
                  independent_statistics_passed=True,source_and_historical_inputs_unchanged=unchanged,
                  figure_inputs_match_statistics=True)
    (root/'validation/delivery_validation.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[]
    for f in sorted(root.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or f.name=='artifact_manifest.json':continue
        files.append(dict(path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
    (root/'artifact_manifest.json').write_text(json.dumps(dict(files=files,file_count=len(files)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,files=len(files),total_bytes=sum(r['bytes'] for r in files),links_checked=checked)))

if __name__=='__main__':main()
