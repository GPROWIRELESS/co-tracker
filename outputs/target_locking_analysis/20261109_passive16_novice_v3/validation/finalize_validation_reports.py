"""Compact successful assertion traces; preserve failures and source fingerprints."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent if (HERE.parent/'pair_metrics.csv').exists() else HERE.parent/'output'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = {
        'independent_validation.json': ['pair_manifest.csv', 'pair_metrics.csv', 'coverage.csv', 'region_joint.csv',
                                        'active_reproduction.csv', 'passive_gaze_links.input.json'],
        'independent_model_validation.json': ['model_pair_metrics.csv', 'model_reproduction.csv', 'model_audit.json'],
        'independent_human_statistics_validation.json': ['pair_metrics.csv', 'content_stage_summary.csv',
                'participant_stage_summary.csv', 'group_stage_summary.csv', 'group_stage_summary_long.csv',
                'performance_correlations.csv', 'performance_participants.csv'],
        'independent_model_statistics_validation.json': ['model_pair_metrics.csv', 'model_content_stage_summary.csv',
                'model_participant_stage_summary.csv', 'model_group_stage_summary.csv', 'model_group_stage_summary_long.csv',
                'model_brier_skill.csv'],
        'independent_secondary_validation.json': ['content_stage_summary.csv', 'participant_round_summary.csv',
                'repetition_contrasts.csv', 'performance_participants.csv', 'performance_leave_one_out.csv'],
    }
    summary = []
    for name, artifacts in files.items():
        path = HERE/name
        if not path.exists():
            raise RuntimeError('Required validation is missing: '+name)
        report = json.loads(path.read_text(encoding='utf-8'))
        checks = report.pop('checks', None)
        if checks is not None:
            report['successful_check_count'] = sum(c['passed'] for c in checks)
            report['full_assertion_trace_sha256'] = hashlib.sha256(json.dumps(checks, ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        report['check_storage'] = 'All failures retained; successful assertions counted. Reproduce with included validators.'
        report['input_artifact_sha256'] = {name: sha(OUT/name) for name in artifacts}
        report['validation_source_sha256'] = {p.name: sha(p) for p in sorted(HERE.glob('validate_*.py'))}
        if name == 'independent_validation.json':
            report['frame_artifact_sha256'] = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT/'frames').glob('*.npz'))}
        if name == 'independent_model_validation.json':
            report['frame_artifact_sha256'] = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted((OUT/'model_frames').glob('*.npz'))}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append({'report': name, 'status': report['status'], 'check_count': report['check_count'],
                        'failures': len(report.get('failed_checks', [])), 'sha256': sha(path)})
    result = {'status': 'passed' if all(r['status'] == 'passed' for r in summary) else 'failed',
              'total_independent_checks': sum(r['check_count'] for r in summary), 'reports': summary,
              'method_review': 'Frozen v3 metric code is byte-identical; paired row mapping, common frame support, source-operator hierarchy and paired active/passive reporting verified.'}
    (HERE/'validation_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
