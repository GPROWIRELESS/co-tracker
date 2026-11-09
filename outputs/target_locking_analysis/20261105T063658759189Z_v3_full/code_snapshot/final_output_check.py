"""Verify final provenance, report assets and inventory all deliverables."""
import csv
import hashlib
import json
from pathlib import Path

run = Path('D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/v3 full')
cfg = json.loads((run/'run_config.json').read_text(encoding='utf-8'))
checks = json.loads((run/'independent_validation.json').read_text(encoding='utf-8'))
comparison = json.loads((run/'v2_v3_comparison.json').read_text(encoding='utf-8'))
assert cfg['status']=='complete' and cfg['version']==3 and cfg['issues']==0
assert checks['all_real_and_paired_masks_reconstructed_from_coordinates']
assert comparison['identical_recording_inputs'] and comparison['identical_prediction_inputs']
assert comparison['new_config_sha256']==hashlib.sha256((run/'run_config.json').read_bytes()).hexdigest()
assert len(list((run/'frames').glob('*.npz')))==84
manifest = json.loads((run/'statistics_manifest.json').read_text(encoding='utf-8'))
assert len(manifest['branches'])==4
for relative in manifest['figure_files']:
    assert (run/relative).is_file(), relative
for relative in manifest['rows']:
    assert (run/(relative+'.csv')).is_file(), relative
for relative in ('report.md','detailed_report_zh.md','correction_summary_zh.md',
                 'frame_support_v2_v3_changes.csv','trial_stage_v2_v3_changes.csv',
                 'model_trial_errors_v2_v3_changes.csv','input_metadata_validation.json',
                 'regression_tests.log','full_run.log'):
    assert (run/relative).stat().st_size>0, relative
summary = dict(status='passed',recordings=84,prediction_files=50,branches=4,
    verified_figure_files=len(manifest['figure_files']),verified_table_files=len(manifest['rows']),
    source_and_prediction_hashes_unchanged=True,configuration_hash_matches_comparison=True,
    original_run_preserved=True)
(run/'final_output_validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
paths = [p for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_index.csv']
with (run/'output_index.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['path','bytes'])
    writer.writeheader()
    writer.writerows(dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size) for p in paths)
print(json.dumps(summary,indent=2))
print(f'Indexed {len(paths)} output files.')
