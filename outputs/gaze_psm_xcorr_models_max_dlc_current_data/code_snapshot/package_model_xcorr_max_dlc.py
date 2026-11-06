"""Verify, document and package the completed max-DLC analysis."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict, Counter
import numpy as np
from pypdf import PdfReader, PdfWriter

MODELS = ['all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot']
SOURCES = ['gt', *MODELS]
FIELDS = ('content_id', 'task', 'side', 'signal', 'stage', 'mode')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def merge(paths, output):
    writer = PdfWriter()
    index = []
    for title, path in paths:
        page = len(writer.pages) + 1
        writer.append(str(path), outline_item=title)
        index.append(dict(start_page=page, title=title, pdf=str(path.relative_to(output.parent.parent))))
    with output.open('wb') as f:
        writer.write(f)
    return len(writer.pages), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output
    validation = json.loads((root/'validation.json').read_text(encoding='utf-8'))
    assert validation['all_assertions_passed'] and validation['source_files_unchanged']
    assert validation['shuffles'] == 100 and validation['max_dlc_gt_regression']['curves'] == 152
    data = json.loads((root/'comparison_results.json').read_text(encoding='utf-8'))
    key = lambda r: tuple(r[k] for k in FIELDS)
    refkeys = {key(r) for r in data['models']['gt']['record_curves']}
    for name, result in data['models'].items():
        assert refkeys == {key(r) for r in result['record_curves']}
        assert all(r['side'] == 'max' and r['mode'] in {'psm_stage', 'within_stage'} for r in result['record_curves'])
    predictions = read_csv(root/'prediction_manifest.csv')
    for row in predictions:
        assert sha(Path(row['prediction_path'])) == row['prediction_sha256'], row['prediction_path']
    assert len(predictions) == 50
    null_coverage = []
    for name in SOURCES:
        for group in data['models'][name]['group_curves']:
            null_coverage.append(dict(source=name, **{k: group[k] for k in ('task', 'side', 'signal', 'stage', 'mode')},
                                      status=group['status'], n_stable_users=group['n_stable_users'],
                                      n_null_replicates=group['n_null_replicates'], null_peak_p=group['null_peak_p']))
    write_csv(root/'unpaired_reference_coverage.csv', null_coverage)
    primary = [r for r in data['paired_records'] if r['signal'] == 'position' and r['mode'] == 'psm_stage']
    common = set.intersection(*[{key(r) for r in primary if r['model'] == m and r['eligible']} for m in MODELS])
    same = {k for k in common if all(r['same_selected_side'] for r in primary if key(r) == k)}
    summary = []
    for cohort, keys in [('common5', common), ('common5_same_gt_side', same)]:
        for task in ('A', 'C'):
            for model in MODELS:
                rows = [r for r in primary if r['task'] == task and r['model'] == model and key(r) in keys]
                people = defaultdict(list)
                for r in rows:
                    people[r['user']].append(r)
                summary.append(dict(cohort=cohort, task=task, model=model, n_record_stage_pairs=len(rows),
                                    n_participants=len(people),
                                    participant_equal_mean_absolute_max_lag_difference_ms=1000*float(np.mean([np.mean([r['abs_lag_error_s'] for r in rr]) for rr in people.values()])) if rows else None,
                                    median_absolute_max_lag_difference_ms=1000*float(np.median([r['abs_lag_error_s'] for r in rows])) if rows else None,
                                    selected_side_agreement_rate=float(np.mean([r['same_selected_side'] for r in rows])) if rows else None))
    write_csv(root/'delivery_summary.csv', summary)
    H = lambda task, model: next(r for r in summary if r['cohort'] == 'common5' and r['task'] == task and r['model'] == model)
    source_figures = [(f'{name}: {path.stem}', path) for name in SOURCES for path in sorted((root/name/'figures').glob('*.pdf'))]
    assert len(source_figures) == 48, len(source_figures)
    (root/'figures').mkdir(exist_ok=True)
    source_pages, source_index = merge(source_figures, root/'figures'/'legacy_source_report.pdf')
    write_csv(root/'figures'/'legacy_report_page_index.csv', source_index)
    comparison = root/'figures'/'comparison_report.pdf'
    assert comparison.exists(), comparison
    complete_pages, complete_index = merge([('Paired model/GT comparisons', comparison),
                                            ('Original max-DLC source figures', root/'figures'/'legacy_source_report.pdf')],
                                           root/'figures'/'complete_model_xcorr_report.pdf')
    write_csv(root/'figures'/'complete_report_sections.csv', complete_index)
    assert len(PdfReader(root/'figures'/'complete_model_xcorr_report.pdf').pages) == complete_pages
    counts = {task: len({key(r) for r in primary if r['task'] == task and key(r) in common}) for task in ('A', 'C')}
    same_counts = {task: len({key(r) for r in primary if r['task'] == task and key(r) in same}) for task in ('A', 'C')}
    stage_counts = {task: dict(Counter(r['stage'] for r in primary if r['task'] == task and r['model'] == MODELS[0] and key(r) in common)) for task in ('A', 'C')}
    null_counts = {name: dict(Counter(r['n_null_replicates'] for r in null_coverage if r['source'] == name)) for name in SOURCES}
    text = f'''# 模型 gaze 与真实 gaze：严格 max-DLC 方法的 A/C 测试集分析

本目录是用户指定 `outputs/gaze_psm_xcorr_max_dlc` 方法后的正式重算版本。此前 `gaze_psm_xcorr_models` 使用 SAM mask 和不同主分析选侧规则，其数值不作为本版本结论。

范围为 Task A、C 各5个测试视频，合计10个视频、7名参与者、五类模型50份 prediction。未进行 score correlation。

## 结果入口

- `figures/complete_model_xcorr_report.pdf`：完整图集，{complete_pages}页。
- `figures/comparison_report.pdf`：模型/GT 直接比较图，建议先看共同五模型集合的图。
- `figures/legacy_source_report.pdf`：六个 gaze 来源按原 max-DLC 绘图函数生成的48页曲线与峰值 lag 图，包含 bootstrap 与随机对照。
- `paired_record_summary.csv`：{len(data['paired_records'])}条 recording×stage×signal×mode×model 配对结果，含选侧、峰值、lag、差值、质量标志和排除原因。
- `paired_group_summary.csv`：双方共同完整 lag 支持参与者上的组曲线峰差和配对 bootstrap 区间。
- `paired_summary.csv`、`paired_overview.csv`：分阶段及跨阶段描述性汇总。
- `common5/position_psm_stage_summary.csv`：五模型共同可用主分析集合；同时提供所有模型都与 GT 选同一侧的子集。
- `delivery_summary.csv`：共同集合的绝对 max-lag 差异（毫秒），先在参与者内平均，再对参与者等权。
- `side_selection.csv`：每个来源独立选侧及与 GT 的一致性。
- `unpaired_reference_coverage.csv`、`unpaired_pairings.csv`：随机对照的实际有效次数及100次抽样记录。
- `phase_validity.csv`、`stage_inventory.csv`、`prediction_manifest.csv`、`alignment_audit.csv`、`validation.json`：质量、阶段、输入与核验记录。

## 严格沿用的计算规则

1. PSM 使用 `gaze_psm_xcorr_dlc/cache` 中独立的左/右 DLC x/y，cache version 5；不按 likelihood 设阈值，jump 阈值仅用于 QC。GT 使用缓存的高频原始 gaze（Status=2）。
2. 直接调用原 DLC pipeline、core、group 和 max-DLC 的数值函数。41 Hz，lag ±2秒，最大插值间隙0.1秒，80 ms居中平滑；保留二维 pooled covariance/variance 的归一化 xcorr，以及位置/速度 RMS 阈值。
3. 仅保留 `psm_stage`、`within_stage`，不纳入 whole-task。所有非空 stage 沿用原代码。重复 occurrence 分别中心化后池化充分统计，不首尾拼接；stage 最少池化重叠2秒。
4. `psm_stage` 以 PSM 时刻为阶段锚，允许 gaze 来自相邻阶段；`within_stage` 要求两者位于同一连续 occurrence。正 lag 表示 gaze 领先 PSM。
5. 对每个 recording×signal×stage×mode，**GT 和每个模型独立比较左右曲线的最大有符号 r，选择较高一侧并保留整条曲线**。不是绝对值，不是逐 lag 取最大，不固定在 GT 侧。相同峰选左，仅单侧有效则选该侧，双侧无效保留缺失标记。
6. 每个参与者内录制等权 Fisher-z 平均，再对参与者等权；组峰从组曲线寻找，要求完整 lag 支持及至少3名参与者。每个来源保留1,000次参与者 bootstrap。
7. 保留100次同任务、不同参与者的 unpaired reference，仅在本批10个测试视频内抽样，seed=42且保留旧记录顺序。每个对照先重新计算左右曲线、重新选侧，再按原层级聚合。40种确定性的 target/donor 对预计算后按100次原抽样组合，结果等价。有效对照次数可少于100，按实际值报告。

本批每个来源的36组中，26组有100个有效对照、4组有20个、2组有45个、4组为0。C 的 `pull through and cinch` 为20个；`threading string through tower` 的 within-stage 为45个；A 的 `drop ring` 因支持不足为0。

## 模型与 GT 配对的补充规则

- 模型采用原概率图期望坐标，乘视频 width/height，再按同一时间轴插值和平滑。帧号和相对纳秒时间与10份原缓存完全对应。
- 五模型与 GT 使用共同有效 gaze 支持；本次没有额外删去 GT 网格点，真实 gaze 结果仍与旧 DLC/max-DLC 对应测试记录一致。
- `eligible` 用于配对描述：正峰、非边界峰、完整 lag 支持以及有效组状态，不使用任何 score 条件。原始不合格结果仍保存，原 source 聚合不被该配对筛选改写。
- 两来源的组比较重新限制到共同完整 lag 参与者，同步进行1,000次配对 bootstrap；区间未经多重比较校正。组峰差不是逐视频平均绝对误差。
- 独立选侧可能选中不同器械。因此主表 lag 差是**最大相关侧统计量的差异**，不能一律解释为同一 PSM 的时间误差。选侧一致率单独报告；共同且都同 GT 侧的子集作为敏感性分析，不改变主分析选侧规则。

## 共同集合的描述性结果

position / psm_stage 的五模型共同可用集合：A={counts['A']}、C={counts['C']}个 recording×stage 条件。所有模型均与 GT 选同侧的子集：A={same_counts['A']}、C={same_counts['C']}。这些是重复阶段条件，不能当作独立参与者数量。

共同主分析不含 `drop ring` 或 `pull through and cinch`，不能将这些比较概括为全部 A/C 阶段。逐阶段数量：A `{stage_counts['A']}`；C `{stage_counts['C']}`。

| 模型 | A：参与者等权平均绝对 max-lag 差异 (ms) | C：参与者等权平均绝对 max-lag 差异 (ms) |
|---|---:|---:|
'''
    for model in MODELS:
        av, cv = H('A', model)['participant_equal_mean_absolute_max_lag_difference_ms'], H('C', model)['participant_equal_mean_absolute_max_lag_difference_ms']
        text += f'| {model} | {av:.1f} | {cv:.1f} |\n'
    text += f'''
以上是同一可用集合的描述性差异，不是显著性排名。高 xcorr 不保证 gaze 空间位置相同，相关选侧不能识别实际操作器械，lag 也不能单独证明预测能力或因果关系。

## 验证与复现

- 旧 DLC 左右 GT 344条曲线与旧 max-DLC 选侧 GT 152条曲线分别回归核验；最大数值差为 `{validation['max_dlc_gt_regression']['max_abs_difference']:.3g}`。选侧和缺失支持完全一致。
- 原 max 选侧前四项单元测试通过（4/4）；已知5格 PSM 延迟正确恢复正 lag。
- 50份 prediction 的输入哈希在运行后再次核验；已校验的原代码、旧结果、配置及缓存未改变。
- 缓存保留在 `cache/<source>`；左右中间曲线在 `left_right_intermediate`，用于审计，不属于主 max 输出。原文件结构在 `gt/` 及五个模型子目录中保留 `record_* / novice_* / group_* / bootstrap_* / unpaired_peaks / contrasts / results / config / figures`。
- `code_snapshot/` 保存本次使用的原源码及执行脚本。未重新解码原始视频；复现从已有 DLC 缓存开始。

依次执行（需要 numpy、scipy、matplotlib、pypdf）：

```text
python code_snapshot/analyze_model_xcorr_max_dlc.py --output <new_output> --workers 4
python code_snapshot/plot_model_xcorr_max_dlc.py --output <new_output>
python code_snapshot/package_model_xcorr_max_dlc.py --output <new_output>
```

原数据和模型路径见脚本常量；迁移机器时更新这些路径。`config.json` 保留实际计算时的路径，交付目录见 `delivery_metadata.json`。
'''
    (root/'README.md').write_text(text, encoding='utf-8')
    shutil.copy2(__file__, root/'code_snapshot'/Path(__file__).name)
    plot_script = Path(__file__).with_name('plot_model_xcorr_max_dlc.py')
    shutil.copy2(plot_script, root/'code_snapshot'/plot_script.name)
    validation.update(original_side_selection_tests_passed=4, all_prediction_hashes_rechecked=True,
                      all_sources_have_identical_condition_keys=True, common5_primary_counts=counts,
                      common5_same_gt_side_primary_counts=same_counts, complete_report_pages=complete_pages,
                      common5_primary_stage_counts=stage_counts, effective_null_replicate_counts=null_counts,
                      original_source_figures=48)
    (root/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    delivery = dict(delivery_root=r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_models_max_dlc',
                    calculation_root=str(root), method_reference=str(Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_max_dlc')),
                    supersedes_for_this_request='gaze_psm_xcorr_models (mask-based version)',
                    report='figures/complete_model_xcorr_report.pdf', report_pages=complete_pages)
    (root/'delivery_metadata.json').write_text(json.dumps(delivery, indent=2), encoding='utf-8')
    manifest = [dict(relative_path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'artifact_manifest.csv']
    write_csv(root/'artifact_manifest.csv', manifest)
    print(json.dumps(dict(common5=counts, common5_same_gt_side=same_counts, report_pages=complete_pages,
                          files=len(manifest)+1, delivery_summary=summary), indent=2))


if __name__ == '__main__':
    main()
