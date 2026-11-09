"""Extract only real-observation algorithms from the pinned v3 snapshot."""
from pathlib import Path
import ast
import hashlib
import json

BASE = Path(r'D:\GitHub_Project\co-tracker\outputs\target_locking_analysis\20261105T063658759189Z_v3_full\code_snapshot')
OUT = Path(__file__).resolve().parents[1] / 'code'
OUT.mkdir(parents=True, exist_ok=True)
provenance = []

def selected(file, names):
    source = (BASE/file).read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    result = []
    for name in names:
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        text = ''.join(lines[node.lineno-1:node.end_lineno])
        result.append(text)
        provenance.append(dict(source_file=file, name=name, line_start=node.lineno,
                               line_end=node.end_lineno, sha256=hashlib.sha256(text.encode()).hexdigest()))
    return '\n\n'.join(result) + '\n'

geometry_names = ['_size','_radii','_points','_vec','target_center','_geometry','classify',
                  '_scanlines','_interval','_intersection','_count','areas','times',
                  '_blocks','smooth_gaze','temporal']
header = '''"""Unchanged v3 real-gaze geometry, native exposure, smoothing and temporal algorithms.

Only observation functions are retained. Target and PSM centers use original
continuous pixels; gaze accepts the closed image boundary. Pixel areas are
counted at (column+0.5,row+0.5). No probability or prediction code is included.
"""
import numpy as np

NAMES = ("other", "target_only", "psm_only", "overlap")
HZ = 41.0
MAX_GAP_S = 0.100
MIN_DWELL_S = 0.120

'''
(OUT/'tl_observed_geometry.py').write_text(header+selected('tl_geometry.py', geometry_names),encoding='utf-8')

header = '''"""Target Locking v3 real-observation measurement entry points."""
import numpy as np
import tl_observed_geometry as geo

STAGES = {"A": ("reach ring", "grab ring", "transfer ring", "place ring"),
          "C": ("reach string", "wrap around psm", "pull through and cinch", "threading string through tower")}
SCENARIOS = {'main': (50,50,None,False), 'r25': (25,25,None,False),
             'r100': (100,100,None,False), 'gripper100': (50,100,None,False),
             'smooth': (50,50,None,True)}

'''
body = selected('tl_common.py',['in_image','stage_masks','ratio','summarize'])
prepare = selected('analyze_target_locking.py',['prepare'])
prepare = prepare.replace('def prepare(rec,args):','def prepare_arrays(d):')
prepare = prepare.replace("    d=inputs.read_record(rec,args.data_root,args.repo,gaze_allowed_codes=None,audit_cache_dir=args.cache_dir)\n",'')
prepare = prepare.replace("STAGES[rec['task']]", "STAGES[d['task']]")
(OUT/'tl_observed_core.py').write_text(header+body+'\n'+prepare+'\n'+selected('analyze_target_locking.py',['support']),encoding='utf-8')

bootstrap = selected('tl_stats.py',['percentiles','corr','ParticipantBootstrap'])
tree = ast.parse(bootstrap)
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'bss')
lines = bootstrap.splitlines(keepends=True)
del lines[node.lineno-1:node.end_lineno]
bootstrap = ''.join(lines)
(OUT/'tl_observed_bootstrap.py').write_text('''"""Unchanged v3 participant bootstrap; irrelevant evaluation method removed."""
import numpy as np
from scipy.stats import rankdata

'''+bootstrap,encoding='utf-8')

header = '''"""Unchanged v3 equal-recording then equal-participant summaries.

content_id MUST be the unique viewing-record key, participant the observer.
Source-video identifiers are separate provenance fields.
"""
import numpy as np
import pandas as pd
from tl_observed_bootstrap import ParticipantBootstrap

GROUP = ["scope", "scenario", "task", "stage", "source"]
PERSON = GROUP + ["participant"]
OCC = ["L_T", "L_P", "L_O", "L_N", "L_target", "L_gripper", "D", "C",
       "C_U", "C_excess", "E_target", "E_gripper", "B", "B_excess"]
AREA = ["area_T", "area_P", "area_O", "area_N"]
TIME = ["direct_rate", "via_rate", "dwell_T_ms", "dwell_P_ms", "dwell_O_ms"]

'''
(OUT/'tl_observed_statistics.py').write_text(header+selected('tl_stats.py',['present','group_items','participant_summary']),encoding='utf-8')
record = dict(reference_snapshot=str(BASE),
              source_files_sha256={f:hashlib.sha256((BASE/f).read_bytes()).hexdigest() for f in ['tl_geometry.py','tl_common.py','tl_stats.py','analyze_target_locking.py']},
              retained_functions=provenance,
              changes=['prepare accepts already-loaded data instead of invoking the active loader',
                       'prepare task comes from data dictionary instead of the recording dictionary',
                       'geometry import renamed to the extracted observed-only module',
                       'ParticipantBootstrap.bss removed; all retained methods unchanged'])
(OUT/'core_extraction_provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Extracted:', ', '.join(p.name for p in OUT.glob('tl_observed_*.py')))
