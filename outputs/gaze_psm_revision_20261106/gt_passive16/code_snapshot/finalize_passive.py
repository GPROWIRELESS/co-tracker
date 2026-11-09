"""Package audited passive results without altering computed numerical values."""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes

import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent
OUT = WORK.parent / 'outputs/gaze_psm_revision_20261106/gt_passive16'
BASELINE = Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106\gt84')
PYTHON = Path(r'C:\ProgramData\anaconda3\envs\sam3\python.exe')
sys.path.insert(0, str(OUT / 'code_snapshot'))
from run_gt84_revision import clean, csvout, save, sha

STAGES = {'A': ['reach ring', 'grab ring', 'transfer ring', 'place ring', 'drop ring'],
          'C': ['reach string', 'wrap around psm', 'threading string through tower', 'pull through and cinch']}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(mode_rows(csv.DictReader(stream)))


def read_json(path):
    return normalize_modes(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def finite(value):
    try:
        return np.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def enrich_csv(path, identities):
    data = read_csv(path)
    for row in data:
        identity = identities.get(row.get('content_id'))
        if identity:
            for field in ('recording_id', 'observer_user', 'observer_skill', 'source_content_id', 'source_user', 'src_skill', 'seq', 'task_number'):
                row.setdefault(field, identity[field])
    csvout(path, data)


def main():
    manifest = read_json(OUT / 'run_manifest.json')
    validation = read_json(OUT / 'validation.json')
    independent = read_json(WORK / 'validation/passive_independent_validation.json')
    regression = read_json(WORK / 'validation/gt84_representative_regression.json')
    assert independent['passed'] and regression['passed']
    identities = {record['recording_id']: record for record in manifest['records']}
    records = list(identities.values())
    inventory = read_json(WORK / 'inputs/input_record_inventory.json')
    source_audit = read_json(WORK / 'input_audit/input_ledger.json')
    source_audit = {row['record_id']: row for row in source_audit}
    constructor = read_json(WORK / 'input_audit/constructor_provenance.json')
    assert constructor['independent_comparison_matches_all16']
    user_confirmations = {
        'coordinate_system': '视频画面归一化坐标 1280*1024',
        'row_mapping': '直接按行数对应就可以',
        'row_rule': 'NPY row i, passive video frame i, source video frame i, and DLC row i; zero-based local indices',
        'clock': 'precise video_ts matched by original Frame_Number, as frozen GT84',
        'raw_VID_FRAME_role': 'audit only, not a remapping or exclusion criterion by explicit user instruction',
    }
    for row in inventory:
        audit = source_audit[row['recording_id']]
        row.update(all_exclusion_reasons=audit.get('exclusion_reasons', []),
                   gaze_array_rows=audit.get('passive_npy_rows'),
                   source_exists=audit.get('source_npy_exists'),
                   source_stage_available=audit.get('stage_exists'),
                   source_dlc_available=audit.get('dlc_exists'))
        if row['included']:
            row.update(mapping_status='user_authorised_row_correspondence_with_equal_counts_and_frame_keys',
                       mapping_authorisation=user_confirmations['row_mapping'],
                       pixel_mapping_evidence=user_confirmations['coordinate_system'])
    csvout(OUT / 'input_record_inventory.csv', inventory)
    csvout(OUT / 'analysis_record_inventory.csv', [row for row in inventory if row['included']])
    save(OUT / 'input_record_inventory.json', inventory)
    save(OUT / 'user_input_confirmations.json', user_confirmations)

    for folder, source in [('input_audit', WORK / 'input_audit'), ('input_mapping', WORK / 'inputs/frame_mapping'),
                           ('reproduction_inputs', WORK / 'inputs'), ('baseline_audit', WORK / 'baseline_audit'),
                           ('report_code', WORK / 'report_code')]:
        shutil.copytree(source, OUT / folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    validation_code = OUT / 'validation_code'
    validation_code.mkdir(exist_ok=True)
    for path in (WORK / 'validation').glob('*.py'):
        shutil.copy2(path, validation_code / path.name)
    for path in (WORK / 'validation').glob('*.json'):
        shutil.copy2(path, OUT / path.name)
    shutil.copy2(WORK / 'baseline_fingerprint_before.json', OUT / 'baseline_fingerprint_before.json')
    shutil.copy2(Path(__file__), OUT / 'code_snapshot/finalize_passive.py')
    shutil.copy2(WORK / 'passive_run.log', OUT / 'numerical_run.log')

    # Retain the unmasked original NPY coordinates and their exact historical raw-row provenance.
    for variant in manifest['variants']:
        for recording_id in identities:
            path = OUT / 'point_provenance' / variant / f'{recording_id}.npz'
            with np.load(path, allow_pickle=False) as stream:
                provenance = {key: stream[key] for key in stream.files}
            with np.load(WORK / 'inputs/cache' / f'{recording_id}.npz', allow_pickle=False) as stream:
                for field in ('native_gaze_norm_original', 'native_gaze_px'):
                    provenance['input__' + field] = stream[field]
            historical = WORK / 'input_audit/native_to_raw_mapping' / f'{recording_id}.npz'
            with np.load(historical, allow_pickle=False) as stream:
                for field in stream.files:
                    provenance['historical_input__' + field] = stream[field]
            np.savez_compressed(path, **provenance)

    for filename in ('side_selection.csv', 'spatial_records.csv', 'matched_mode_coverage.csv', 'unpaired_donor_coverage.csv', 'unpaired_precompute_timing.csv'):
        enrich_csv(OUT / filename, identities)
    pairings = read_csv(OUT / 'unpaired_pairings.csv')
    for row in pairings:
        for prefix, key in [('target', 'target'), ('donor', 'donor')]:
            identity = identities.get(row[key])
            row[prefix + '_observer_user'] = identity['observer_user'] if identity else ''
            row[prefix + '_source_content_id'] = identity['source_content_id'] if identity else ''
        row['shared_source_stimulus'] = bool(row['donor'] and row['target_source_content_id'] == row['donor_source_content_id'])
    csvout(OUT / 'unpaired_pairings.csv', pairings)

    all_curves = []
    with (OUT / 'record_curves.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            curve = normalize_modes(json.loads(line))
            curve['r'] = np.asarray(curve['r'], dtype=float)
            all_curves.append(curve)
    summaries = read_csv(OUT / 'group_summary.csv')
    lookup = {tuple(row[key] for key in ('variant', 'task', 'side', 'signal', 'stage', 'mode')): row for row in summaries}
    coverage = []
    for variant, config in manifest['variants'].items():
        for task, stages in STAGES.items():
            for stage in stages:
                annotated = [record for record in records if record['task'] == task and stage in source_audit[record['recording_id']].get('stage_values', {})]
                for side in ('l', 'r'):
                    for signal in config.get('signals', ('position', 'velocity')):
                        for mode in ('PSM in stage', 'Both in stage'):
                            key = (variant, task, side, signal, stage, mode)
                            group = lookup.get(key, {})
                            curves = [curve for curve in all_curves if tuple(curve[field] for field in ('variant', 'task', 'side', 'signal', 'stage', 'mode')) == key]
                            complete = [curve for curve in curves if np.isfinite(curve['r']).all()]
                            observers = {curve['user'] for curve in complete}
                            status = 'missing_stage_annotation' if not annotated else 'supported' if len(observers) >= 3 else 'insufficient_complete_observers'
                            coverage.append(dict(zip(('variant', 'task', 'side', 'signal', 'stage', 'mode'), key),
                                                 stage_annotated_records=len(annotated), computed_record_curves=len(curves),
                                                 complete_records=len(complete), complete_observers=len(observers),
                                                 status=status, peak_r=group.get('peak_r') or None, peak_lag_s=group.get('peak_lag_s') or None))
    csvout(OUT / 'stage_coverage.csv', coverage)

    native_discrepancies = []
    for row in constructor['records']:
        native_discrepancies.append(dict(recording_id=row['record_id'], rows=row['rows'],
            min_frame_difference=row['selected_raw_vid_frame_minus_native_row_min'],
            median_frame_difference=row['selected_raw_vid_frame_minus_native_row_median'],
            max_frame_difference=row['selected_raw_vid_frame_minus_native_row_max'],
            different_frame_rows=row['selected_raw_video_frame_differs_native_row_count'],
            historical_clamped_rows=row['historical_coordinate_clamped_rows'],
            historical_nearest_dt_ms_max=row['historical_native_clock_nearest_dt_ms_max'],
            analysis_rule='user authorised original NPY row i to video and DLC row i',
            used_for_exclusion=False, current_npy_reproduced_exactly=row['all_fields_reproduced_exactly']))
    csvout(OUT / 'native_row_frame_discrepancies.csv', native_discrepancies)

    shared_all = defaultdict(set)
    shared_included = defaultdict(set)
    for row in inventory:
        shared_all[row['source_content_id']].add(row['observer_user'])
        if row['included']:
            shared_included[row['source_content_id']].add(row['observer_user'])
    primary = [row for row in summaries if row['variant'] == 'primary']
    record_summary = read_csv(OUT / 'record_summary.csv')
    reference = read_csv(OUT / 'unpaired_reference.csv')
    main_ref = [row for row in reference if row['reference_scope'] == 'elapsed_time_model_consistent' and finite(row['reference_tail_fraction'])]
    audit = dict(
        population_counts=dict(watchings=len(inventory), observers=len({row['observer_user'] for row in inventory}), sources=len(shared_all),
                               task_counts=dict(Counter(row['task'] for row in inventory))),
        inclusion_counts=dict(records=len(records), observers=len({row['user'] for row in records}), sources=len(shared_included),
                              task_counts=dict(Counter(row['task'] for row in records))),
        exclusions=dict(Counter(row['exclusion_reason'] for row in inventory if row['exclusion_reason'])),
        task_scope=dict(primary=['A', 'C'], novice_only=dict(Counter(row['task'] for row in inventory if row['observer_skill'] == row['src_skill'] == 'novice')),
                        User2_1_1='Task B independently resolved by score/assignment/decoded video; active source still missing'),
        video_intersection=dict(all_manifest_watches=sum(row['gt84_source_intersection'] for row in inventory),
                                all_manifest_sources=len({row['source_content_id'] for row in inventory if row['gt84_source_intersection']}),
                                admitted_watches=len(records), admitted_sources=len(shared_included),
                                selection='novice observer and source, Task A/C, available required inputs; no automatic intersection restriction'),
        shared_stimulus=dict(full_manifest_sources_with_multiple_observers=sum(len(observers) > 1 for observers in shared_all.values()),
                             included_sources_with_multiple_observers=sum(len(observers) > 1 for observers in shared_included.values()),
                             actual_shared_source_reference_pairings=sum(row['shared_source_stimulus'] for row in pairings),
                             donor_rule='same task, different observer; identical source content remains eligible'),
        mapping_summary=dict(npy_video_equal_count_records=122, analysed_rows_verified=16, coordinate_confirmation=user_confirmations['coordinate_system'],
                             row_confirmation=user_confirmations['row_mapping'], user_authorised_row_mapping=True,
                             raw_frame_difference_min=min(row['min_frame_difference'] for row in native_discrepancies),
                             raw_frame_difference_max=max(row['max_frame_difference'] for row in native_discrepancies),
                             clock='canonical precise source acquisition video_ts, identical mapped passive video_ts',
                             decoded_frame_checks=112, historical_constructor_exact_records=16),
        mapping_warning=dict(unresolved=False, user_authorised_row_mapping=True,
                             detail='按用户指定，NPY 第 i 行与视频第 i 帧及 DLC 第 i 行对应。122 条 NPY 与各自 passive 视频行数一致，16 条分析输入的原始帧号、精确时间轴和 DLC 行索引也一致。原始 VID_FRAME 差异及历史最近邻构造保留在审计中，按用户指示不改变对应关系。lag 的秒数沿用精确源视频采集时钟。'),
        method_validation=dict(numerical_consistency=independent['passed'], user_authorised_input_mapping=True,
                               representative_GT84=regression['passed'], independent_primary_max_error=independent['direct_primary']['maximum_absolute_correlation_error']),
        findings=dict(primary_group_cells_with_computed_stage=len(primary), predefined_primary_cells=72,
                      supported_primary_groups=sum(finite(row['peak_r']) for row in primary),
                      primary_interpretations=dict(Counter(row['lag_interpretation'] for row in primary if finite(row['peak_r']))),
                      primary_record_curves=sum(row['variant'] == 'primary' for row in record_summary),
                      primary_identifiable_record_curves=sum(row['variant'] == 'primary' and row['lag_identifiable'] == 'True' for row in record_summary),
                      main_reference_supported_groups=len(main_ref), main_reference_valid_min=min(int(row['n_valid']) for row in main_ref),
                      main_reference_valid_max=max(int(row['n_valid']) for row in main_ref),
                      reference_tail_interpretation='peak correlation height descriptive reference, not a test of positive lag or formal FDR'),
    )
    save(OUT / 'passive_metadata_audit.json', audit)

    # Freeze every baseline artifact again, not only code, and retain exact executed sources.
    before = read_json(WORK / 'baseline_fingerprint_before.json')
    changes = []
    for row in before['files']:
        path = BASELINE / row['path']
        if not path.is_file() or sha(path) != row['sha256']:
            changes.append(row['path'])
    after = dict(baseline=str(BASELINE), files_checked=len(before['files']), unchanged=not changes, changed_files=changes)
    save(OUT / 'baseline_fingerprint_after.json', after)
    assert not changes
    frozen_hashes = read_json(BASELINE / 'run_manifest.json')['code_sha256']
    kernels = {name: sha(OUT / 'code_snapshot' / name) == digest for name, digest in frozen_hashes.items() if name != 'run_gt84_revision.py'}
    assert all(kernels.values())
    manifest.setdefault('executed_code_sha256', manifest['code_sha256'].copy())
    manifest.update(user_input_confirmations=user_confirmations, source_content_and_observer_mapping='content_id retained as unique recording_id for frozen schema, user always observer_user',
                    frozen_module_matches=kernels, baseline_requested_tools_exists=False,
                    actual_numerical_tool_source=str(BASELINE / 'code_snapshot'),
                    input_mapping_rule='explicit user-authorised row correspondence, independently checked frame keys and equal counts',
                    input_mapping_necessary_difference='original VID_FRAME differs from historical NPY associations; user explicitly chose row correspondence',
                    random_seed_rules=read_json(WORK / 'baseline_audit/baseline_audit.json')['randomness'])
    command = [str(PYTHON), str(WORK / 'code/run_gt84_revision.py'), '--source', str(WORK / 'inputs'), '--output', str(OUT), '--workers', '4', '--bootstrap', '5000', '--null-count', '5000', '--no-report']
    manifest['actual_command_argv'] = command
    manifest['report_command_argv'] = [str(PYTHON), str(OUT / 'report_code/report_passive_revision.py'), str(OUT)]
    manifest['additional_provenance_postprocessing'] = 'finalize_passive.py appends unmasked NPY/raw-row provenance and identity columns, preserving all numerical curve/statistic values'
    manifest['software'].update(matplotlib=importlib.metadata.version('matplotlib'), opencv=cv2_version(), python_executable=str(PYTHON))
    manifest['code_sha256'] = {path.name: sha(path) for path in (OUT / 'code_snapshot').glob('*.py')}
    save(OUT / 'run_manifest.json', manifest)
    save(OUT / 'execution_commands.json', dict(numerical_argv=command, finalization_argv=[str(PYTHON), str(Path(__file__))],
                                              report_argv=manifest['report_command_argv'], numerical_seed_rules=manifest['random_seed_rules']))
    (OUT / 'runtime_packages.txt').write_text(subprocess.check_output([str(PYTHON), '-m', 'pip', 'freeze'], text=True), encoding='utf-8')

    validation.update(independent_passive_validation=independent, representative_GT84_regression=regression,
                      baseline_unchanged=not changes, baseline_files_checked=len(before['files']),
                      numerical_modules_identical=kernels, complete_group_bootstrap_validation=read_json(OUT / 'group_bootstrap_full_validation.json'),
                      input_mapping=dict(status='user_authorised_row_correspondence', confirmations=user_confirmations,
                                         exact_frame_clock_dlc_joins=True, native_rows_equal_video_frames_all122=True,
                                         raw_VID_FRAME_differences_retained=True, raw_VID_FRAME_used_for_remapping=False),
                      actual_physical_playback_delay_claim=False)
    evidence_manifest = read_json(OUT / 'input_manifest.json')
    extra = read_json(WORK / 'input_audit/raw_and_task_evidence_hashes.json')
    extra_rows = extra if isinstance(extra, list) else [dict(path=path, sha256=digest, role='raw_eye_or_task_mapping_evidence') for path, digest in extra.items()]
    seen = {row['path'] for row in evidence_manifest}
    for row in extra_rows:
        if row['path'] not in seen:
            evidence_manifest.append(row)
            seen.add(row['path'])
    changed_inputs = [row['path'] for row in evidence_manifest if sha(row['path']) != row['sha256']]
    validation['input_unchanged'] = not changed_inputs
    validation['changed_inputs'] = changed_inputs
    assert not changed_inputs
    save(OUT / 'input_manifest.json', evidence_manifest)
    save(OUT / 'validation.json', validation)
    write_methods(manifest, independent, kernels)
    print(json.dumps(dict(output=str(OUT), records=len(records), observers=8, source_videos=16, numerical_validation=independent['passed'],
                          baseline_unchanged=not changes, input_unchanged=not changed_inputs), ensure_ascii=False))


def cv2_version():
    import cv2
    return cv2.__version__


def write_methods(manifest, independent, kernels):
    entries = [
        ('冻结版本与入口', 'gt84-fixed-v2.1-npy-20261106 的实际代码与 manifest', '仅输入、身份、路径、历史不适用文件和报告适配', 'run_gt84_revision.py; baseline_audit.json', '通过', 'outputs/tools 不存在，使用冻结 code_snapshot'),
        ('分析单位', 'active recording，按 user 汇总和重抽样', '每次 passive 观看唯一 recording_id，user=observer_user', 'run_gt84_revision.record_job; records.json', '通过', '身份改为真实观看者'),
        ('样本与任务', 'A/C，预定 A5+C4 阶段', 'novice–novice A7+C9，共16观看、8观看者、16源视频', 'input_record_inventory.csv; stage_coverage.csv', '通过', 'B/D不套用A/C阶段，User2_1_1查实为B'),
        ('画面与坐标', '归一化 gaze 乘对应源图像尺寸', '用户确认视频归一化1280×1024，左右DLC均在同一源图像', 'user_input_confirmations.json; decoded_evidence.json', '通过', '显示坐标定义由采集者确认'),
        ('帧与行对应', '原始 Frame_Number 对齐精确video_ts，DLC按保留视频局部行号', '按用户指定row i对应video frame i和DLC row i，且逐项帧号核对一致', 'input_mapping/*.npz; npy_video_frame_count_audit.csv', '按用户指定通过', '原VID_FRAME差异保留，用户明确采用按行对应'),
        ('时间轴', '精确video_ts原始帧键，非NPY近似时间或名义FPS', '精确passive/source video_ts逐值一致，按源采集时钟秒数分析', 'prepare_passive_inputs.timeline; input_mapping', '通过', '不把结果称为已校准的实际播放生理时延'),
        ('gaze有效性', '两个坐标有限且均在闭区间[0,1]，valid/Status仅审计', '从passive NPY读取同规则，保留原生valid和历史clamp来源', 'prepare_passive_inputs.py; point_provenance', '通过', '历史NPY已有clamp/最近邻填充，本轮直接使用所给NPY'),
        ('PSM有效性', '保留全部有限二维点，likelihood/图外/跳变只标记', '质量规则删除数0，不加权', 'gaze_psm_preprocess_revision.prepare_signals', '通过', '无'),
        ('网格与插值', '41Hz，最大间隙0.1秒，不外推，稳定排序重复时间取首', '原样复用冻结interpolate_valid与网格规则', 'gaze_psm_preprocess_revision.interpolate_valid', '通过', '无'),
        ('Position滤波', '有限连续段内3点居中boxcar，原边缘/短段规则', '冻结filter_valid逐字相同', 'gaze_psm_preprocess_revision.filter_valid', '通过', '无'),
        ('Velocity滤波', '未经position平滑的输入独立SG5二阶后差分×41，有符号二维速度', '原样复用，速度中点赋stage/occurrence', 'prepare_signals; _velocity_labels', '通过', '无'),
        ('配对与lag方向', 'gaze(t−τ)对PSM(t)，主±2秒165点', '原样复用，正τ表示gaze轨迹对应较早', 'gaze_psm_reliability_v2.occurrence_statistics', '通过', '无'),
        ('阶段与occurrence', 'PSM in stage只限制PSM，Both in stage两端同一occurrence', 'source真实连续occurrence按原帧映射，within跨界0', 'occurrence_statistics; independent validation', '通过', '缺失drop ring保留覆盖状态'),
        ('二维相关估计', '每lag每occurrence按有效配对中心化，汇总协方差及平方和后归一化', '冻结FFT充分统计量与聚合，独立显式配对核复核全部440primary', 'occurrence_statistics; aggregate_statistics', '通过', f"独立最大误差{independent['direct_primary']['maximum_absolute_correlation_error']:.3g}"),
        ('支持阈值', 'occurrence至少2对，总至少82对，双方二维RMS≥1，分母非退化', '实际min_overlap=2秒，min_rms=1，未使用底层其他默认值', 'run_gt84_revision.BASE; _from_totals', '通过', '无'),
        ('峰与可靠性', 'signed maximum，原并列处理，0.01/0.05/0.10近峰与0.05/0.10/0.20方向容差', '原样复用全部峰形/LOO/半段/边界及稳定性诊断', 'peak_features; reliability', '通过', '无'),
        ('8个variant', 'primary、position1/5、velocitybox3/SG7/SG11、trim_edges、window4', '配置与冻结manifest逐值一致，window4为329lag', 'run_manifest.variants', '通过', '无'),
        ('组聚合', '全lag完整录制固定，录制在user内等权Fisher-z，user间等权，裁剪±(1−1e−7)，至少3人', 'user为观看者，所有384主/选择侧组独立核对', 'gaze_psm_statistics_v2.group_curves', '通过', '无'),
        ('组bootstrap', '5000次整user有放回，SHA256派生seed42', '逐lag范围、边界频率及全部variant峰分布保存并复核', 'group_curves; group_bootstrap_full_validation.json', '通过', '补存全部variant分布，数值与既有表相同'),
        ('occurrence bootstrap', 'primary500/敏感性200，非循环相邻块长度2，precision0.4秒，登记/贡献至少4', '全部2420条录制条件核对', 'reliability; block_weights', '通过', '无'),
        ('GT选择侧', 'primary逐录制stage/signal/mode选左右signed peak，平局左，保留完整赢家曲线', '原样运行gt_selected_side', 'run_gt84_revision.selected_curves', '通过', '无'),
        ('共同mode支持', '两种mode共同全lag完整录制重算', '原样运行matched_mode_*', 'run_gt84_revision.matched_modes', '通过', '无'),
        ('阶段差值', '共同user，同步5000bootstrap，B−A，至少3人，保留边界', '观看者共同支持，72个可用差值分布独立复核', 'gt_supplement.supplement', '通过', '无'),
        ('聚合支持对照', '相同信号固定器械，对照available/complete记录支持', '原样运行', 'gt_supplement.supplement; gaze_psm_group.aggregate', '通过', '无'),
        ('空间分析', 'primary position完整录制，零lag及组峰lag，距离中位数与25/50/100px，录制/观看者等权', '原样运行，Both in stage跨occurrence配对为0', 'spatial_and_boundaries; independent validation', '通过', '无'),
        ('非配对参考', '同task不同user，先等概率user再录制，elapsed不伸缩不循环不外推', '不同观看者，source相同仍可选，固定5000尝试失败不补抽', 'gt_reference.donor_choices; run_reference', '通过', '纳入16源视频互异，实际同source donor为0'),
        ('duration子集与BH', '子集重算观察值，seed31415，A5+C4 family含缺失，尾部(1+计数)/(1+有效次数)', '两套参考、128条件与16个BH family独立重建', 'gt_reference._bh_descriptive; run_reference', '通过', '无'),
        ('legacy回归', 'Active历史CSV对照文件存在', '两个同名文件明确不适用，无passive历史基线', 'run_gt84_revision.regression', '不适用', '未读取Active历史结果作为passive回归'),
        ('报告图表', '13种图，每种PNG/PDF', '同样13图，观看者支持、全预定阶段、缺失/边界标记', 'report_code/report_passive_revision.py', '待最终渲染核对', '报告文字按passive实际数值重写'),
    ]
    rows = [dict(zip(('方法项', 'GT84规则', 'passive实现', '代码依据', '验证结果', '必要差异'), entry)) for entry in entries]
    csvout(OUT / 'methods_consistency.csv', rows)
    text = '# GT84 与 passive 方法一致性核对\n\n'
    text += '数值核采用冻结版本 gt84-fixed-v2.1-npy-20261106。输入按用户最终指定的逐行对应规则处理，保留原始 VID_FRAME 差异和历史构造审计。\n\n'
    text += '| 方法项 | GT84规则 | passive实现 | 代码依据 | 验证结果 | 必要差异 |\n|---|---|---|---|---|---|\n'
    text += '\n'.join('| ' + ' | '.join(str(value).replace('|', '/') for value in entry) + ' |' for entry in entries) + '\n'
    (OUT / 'methods_consistency.md').write_text(text, encoding='utf-8')
    save(OUT / 'methods_consistency.json', dict(baseline_version=manifest['baseline_version'], numerical_kernels_byte_identical=kernels,
                                              method_rows=rows, user_authorised_row_mapping=True))


if __name__ == '__main__':
    main()
