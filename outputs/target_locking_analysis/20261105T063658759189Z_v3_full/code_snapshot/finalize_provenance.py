"""Finalize provenance after the analysis exits, before independent validation."""
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
import numpy
import pandas
import scipy

work = Path(__file__).resolve().parent
repo = Path('D:/GitHub_Project/co-tracker')
run = repo / 'outputs/target_locking_analysis/v3 full'
cfg_path = run / 'run_config.json'
cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
assert cfg['status'] == 'complete', cfg['status']
assert cfg['recordings_ok'] == 84 and cfg['predictions_ok'] == 50 and cfg['issues'] == 0

validator = repo / 'tools/validate_target_locking_v2.py'
shutil.copy2(validator, run / 'code_snapshot' / validator.name)
digest = hashlib.sha256(validator.read_bytes()).hexdigest()
cfg['code_sha256'][validator.name] = digest
cfg['validation_code_sha256'] = digest
test = repo / 'tests/test_target_locking_v3.py'
shutil.copy2(test, run / 'code_snapshot' / test.name)
cfg['regression_test_sha256'] = hashlib.sha256(test.read_bytes()).hexdigest()
cfg['auxiliary_code_sha256'] = {}
for name in ('build_change_summary.py','finalize_provenance.py','final_output_check.py'):
    source = work/name
    shutil.copy2(source,run/'code_snapshot'/name)
    cfg['auxiliary_code_sha256'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
cfg['validation_code_note'] = 'Independent v3 validator installed after measurement start; measurement sources unchanged.'
cfg['execution_environment'] = dict(python=sys.version, executable=sys.executable,
    platform=platform.platform(), numpy=numpy.__version__, pandas=pandas.__version__,
    scipy=scipy.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__)
cfg['run_command_arguments'] = ['--repo', str(repo), '--output-dir', str(run),
    '--reference-run', cfg['reference_run'], '--cache-from', cfg['reference_run'],
    '--no-supersede', '--nboot', '10000']
cfg['provenance_finalized_utc'] = datetime.now(timezone.utc).isoformat()
for name, expected in cfg['code_sha256'].items():
    assert hashlib.sha256((run/'code_snapshot'/name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((repo/'tools'/name).read_bytes()).hexdigest() == expected, name
cfg_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ('input_metadata_validation.json','regression_tests.log','full_run.log'):
    shutil.copy2(work/name,run/name)
print('Provenance finalized; all code hashes verified.')
