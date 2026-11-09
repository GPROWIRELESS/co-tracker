"""Complete the passive report's inventory and verify every required artifact."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent
OUT = WORK.parent / 'outputs/gaze_psm_revision_20261106/gt_passive16'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    baseline = load(WORK / 'baseline_audit/baseline_audit.json')
    figures = [row['stem'] for row in baseline['figures']]
    assert len(set(figures)) == 13
    figures_checked = []
    for stem in figures:
        png = OUT / 'figures' / (stem + '.png')
        pdf = OUT / 'figures' / (stem + '.pdf')
        with Image.open(png) as image:
            image.verify()
        raw = pdf.read_bytes()
        assert raw.startswith(b'%PDF-') and b'%%EOF' in raw[-1024:], pdf
        figures_checked.append(dict(stem=stem, png_bytes=png.stat().st_size, pdf_bytes=pdf.stat().st_size))
    report = (OUT / 'report.html').read_text(encoding='utf-8')
    urls = re.findall(r'(?:href|src)=[\"\']([^\"\']+)', report)
    missing = [url for url in urls if not url.startswith(('http:', 'https:', '#', 'data:')) and not (OUT / url.split('#')[0]).exists()]
    assert not missing, missing
    assert 'class="mapping-note"' in report and 'class="mapping-warning"' not in report
    with (OUT / 'stage_coverage.csv').open(encoding='utf-8-sig', newline='') as stream:
        coverage = list(csv.DictReader(stream))
    primary = [row for row in coverage if row['variant'] == 'primary']
    assert len(primary) == 72
    assert Counter(row['status'] for row in primary) == {'supported': 56, 'missing_stage_annotation': 8, 'insufficient_complete_observers': 8}
    expected_top = {row['path'] for row in baseline['top_level_output_files']}
    assert all((OUT / name).is_file() for name in expected_top), [name for name in expected_top if not (OUT / name).is_file()]
    for variant in load(OUT / 'run_manifest.json')['variants']:
        for folder in ('point_provenance', 'bootstrap_peaks', 'occurrence_statistics'):
            assert len(list((OUT / folder / variant).glob('*.npz'))) == 16, (folder, variant)
    fingerprints = load(OUT / 'input_manifest.json')
    changed = [row['path'] for row in fingerprints if sha(row['path']) != row['sha256']]
    assert not changed, changed
    assert load(OUT / 'baseline_fingerprint_after.json')['unchanged']
    independent = load(OUT / 'passive_independent_validation.json')
    assert independent['passed']
    assert load(OUT / 'group_bootstrap_full_validation.json')['passed']

    status_text = '通过，13组PNG/PDF和离线链接已核对'
    methods = OUT / 'methods_consistency.csv'
    with methods.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['方法项'] == '报告图表':
            row['验证结果'] = status_text
        if row['方法项'] == '峰与可靠性':
            row['passive实现'] += '；positive_peak 表示峰高r>0'
    with methods.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    method_json = load(OUT / 'methods_consistency.json')
    method_json['method_rows'] = rows
    save(OUT / 'methods_consistency.json', method_json)
    for name in ('methods_consistency.md', 'report.html'):
        path = OUT / name
        path.write_text(path.read_text(encoding='utf-8').replace('待最终渲染核对', status_text), encoding='utf-8')

    summary = '''

## 本次数据与结果

输入清单共122次观看，11名观看者、44个源视频。User2_1_1 经任务评分表和解码视频核实为B后，完整任务分布为A35、C38、B35、D14。
排除任一端为intermediate的98次观看，剩24次novice–novice观看，其中A7、C9、B8。主分析纳入A/C共16次观看、8名观看者、16个源视频。
所有122条NPY的行数与各自passive视频帧数一致。16条分析输入的NPY、passive/source视频、DLC行数和原始帧键也一致，按用户指定逐行对应。

全部8个variant、固定左右侧、两种signal和两种mode均完成。共2420条逐录制条件曲线，其中primary为440条，全部具有完整lag支持。预定72个主组单元中56个支持充分。A的drop ring缺少阶段标注，C的threading string through tower不足3名完整支持观看者，相关单元保留缺失。

Position的28个支持充分组峰高介于−0.0620和0.2532，velocity的28组介于0.0091和0.0598。峰高采用有符号最大相关。56个支持充分主组中23个受到搜索窗限制、31个方向不确定、2个重抽样峰lag范围位于负侧。440个逐录制主条件中仅3个通过原lag_identifiable规则，显示峰位定位总体不稳定。

空间指标单独描述。零lag支持充分条件下，观看者分层平均的录制中位距离为135.9–419.1px，≤50px比例为0.34%–12.29%。峰高、峰lag与空间接近度分别保留。positive_peak仅表示峰值相关r>0。正lag表示gaze轨迹对应较早，百分位重采样范围作为探索性描述。

两套非配对参考均按每组5000次尝试运行。主参考56个观察支持充分条件的有效参考数为114–5000，其余失败次数及原因完整导出。尾部比例比较参考峰高与观察峰高，BH为预定义family内的描述性调整。当前纳入的16个源视频互异，实际同源donor配对为0，抽样规则仍允许不同观看者观看同一源视频。

全部440条primary由独立显式配对核重新计算，有限掩码一致，最大绝对相关误差为8.1324×10⁻¹⁴。GT84代表性A/C回归72条曲线最大误差为2.2204×10⁻¹⁶，原生NPY到冻结gaze网格误差为0像素。组聚合、bootstrap、阶段差值、非配对抽样及尾部比例均通过独立复核。
269个实际输入及补充证据文件指纹一致，GT84的2846个文件指纹一致。指定outputs/tools目录不存在，实际采用GT84冻结code_snapshot中的数值核。两个legacy回归文件明确记录无passive历史基线。

## 结果与复现文件

- [方法一致性表](methods_consistency.csv) 和 [可读版本](methods_consistency.md)
- [全部观看输入清单](input_record_inventory.csv) 和 [纳入记录清单](analysis_record_inventory.csv)
- [预定阶段完整覆盖](stage_coverage.csv)
- [组曲线与峰](group_summary.csv) 和 [逐lag结果](group_curves.csv)
- [逐录制可靠性](record_summary.csv) 和 [完整曲线](record_curves.jsonl)
- [零lag和组峰位空间指标](spatial_summary.csv)
- [非配对参考](unpaired_reference.csv) 和 [参考方法](unpaired_reference_method.json)
- [全部variant组峰位分布索引](group_bootstrap_full_index.json) 和 [5000次完整分布](group_bootstrap_full_peak_lags.npz)
- [NPY行数与视频帧数](npy_video_frame_count_audit.csv)
- [逐行原始VID_FRAME差异审计](native_row_frame_discrepancies.csv) 和 [用户对应规则](user_input_confirmations.json)
- [独立数值验证](passive_independent_validation.json) 和 [GT84代表性回归](gt84_representative_regression.json)

point_provenance保存8个variant的逐点来源、有效掩码、滤波边缘、插值标记、原始NPY坐标及历史原始眼动行映射。input_mapping保存原始帧键、局部行号、精确时间和阶段occurrence。input_audit保留历史构造代码、任务证据、解码画面对比和VID_FRAME差异。

冻结分析表的content_id作为唯一录制键，与recording_id同值。source_content_id保存源视频身份，user和observer_user均为真实观看者。source_user单独保存操作者。

code_snapshot、report_code和validation_code保存本次代码。reproduction_inputs保存实际输入缓存与配置。execution_commands.json记录实际运行命令和seed派生规则，runtime_packages.txt保存运行依赖版本。重新计算时请为--output指定新的空目录。
'''
    with (OUT / 'README.md').open('a', encoding='utf-8') as stream:
        stream.write(summary)
    save(OUT / 'report_code/manifest.json', dict(role='offline passive report and13 figures from recomputed passive results',
                                                script='report_passive_revision.py', sha256=sha(OUT / 'report_code/report_passive_revision.py'),
                                                postprocessing='complete_delivery.py appends numerical summary and artifact verification to README'))
    shutil.copy2(Path(__file__), OUT / 'code_snapshot/complete_delivery.py')
    manifest = load(OUT / 'run_manifest.json')
    manifest['code_sha256'] = {path.name: sha(path) for path in (OUT / 'code_snapshot').glob('*.py')}
    manifest['report_code_sha256'] = {'report_passive_revision.py': sha(OUT / 'report_code/report_passive_revision.py')}
    manifest['executed_analysis_branches'] = ['all8variants_fixed_left_right', 'gt_selected_side_primary', 'matched_modes',
        'spatial_zero_and_group_peak', 'stage_contrasts', 'available_complete_support_comparison',
        'observer_group_bootstrap', 'occurrence_bootstrap', 'elapsed_time_unpaired_reference', 'duration_eligible_subset_reference']
    save(OUT / 'run_manifest.json', manifest)
    validation = load(OUT / 'validation.json')
    validation['artifact_validation'] = dict(passed=True, figures=figures_checked, png_count=13, pdf_count=13,
                                           checked_local_links=len(urls), missing_local_links=missing,
                                           required_baseline_top_level_files_present=True, stage_coverage_cells=len(coverage),
                                           primary_stage_coverage_cells=72, input_files_verified=len(fingerprints))
    save(OUT / 'validation.json', validation)
    save(OUT / 'artifact_validation.json', validation['artifact_validation'])
    files = [dict(path=path.relative_to(OUT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'file_manifest.json']
    save(OUT / 'file_manifest.json', dict(files=files, files_listed=len(files), excludes_self=True))
    print(json.dumps(dict(passed=True, files=len(files) + 1, figures=13, local_links=len(urls), immutable_inputs=len(fingerprints)), ensure_ascii=False))


if __name__ == '__main__':
    main()
