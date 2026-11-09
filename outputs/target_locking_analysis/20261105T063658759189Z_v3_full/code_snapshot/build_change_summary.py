"""Build a Chinese correction overview from verified v2/v3 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


TASKS = {"A": "Sea Spikes", "C": "Knot-tying"}
METRICS = ("L_target", "L_gripper", "C")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def table(headers, rows):
    def escape(value):
        return str(value).replace("|", "／").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(escape, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(map(escape, row)) + " |" for row in rows),
    ])


def primary(path):
    data = pd.read_csv(path)
    return data[(data.analysis_branch == "probability_mc") &
                (data.scenario == "main") & (data.stage == "whole") &
                (data.valid_s > 0)].copy()


def participant_means(data, metric):
    """Match analysis: equal trial weights within participant, then equal people."""
    values = data.loc[np.isfinite(data[metric]), ["participant", metric]]
    return values.groupby("participant")[metric].mean()


def estimate(data, metric):
    means = participant_means(data, metric)
    return (float(means.mean()) if len(means) else np.nan), len(means)


def number(value, digits=3):
    return f"{value:.{digits}f}" if np.isfinite(value) else "NA"


def comparison_cell(old, new, metric, scale=100, digits=3):
    before, n_before = estimate(old, metric)
    after, n_after = estimate(new, metric)
    delta = (after - before) * scale
    change = f"{delta:+.{digits}f}" if np.isfinite(delta) else "NA"
    return (f"{number(before * scale, digits)} → {number(after * scale, digits)}"
            f"（Δ {change}；n={n_before}→{n_after}）")


def select(data, task, scope, source):
    return data[(data.task == task) & (data.scope == scope) & (data.source == source)]


def ranking(data, task, metric, models):
    values = []
    for model in models:
        score, n = estimate(select(data, task, "paired_test", model), metric)
        if np.isfinite(score):
            values.append((score, model))
    groups = []
    for score, model in sorted(values):
        if groups and score == groups[-1][0]:
            groups[-1][1].append(model)
        else:
            groups.append((score, [model]))
    signature = tuple(tuple(sorted(names)) for _, names in groups)
    return " < ".join(" = ".join(names) for _, names in groups) or "不可估", signature


def relative_link(path, label=None):
    return f"[{label or path}](<{path}>)"


def build(run, old):
    run, old = Path(run).resolve(), Path(old).resolve()
    cfg = read_json(run / "run_config.json")
    assert int(cfg["version"]) >= 3
    assert cfg["status"] == "complete", "Wait for a complete run before summarizing"
    comparison = read_json(run / "v2_v3_comparison.json")
    checks = read_json(run / "independent_validation.json")
    assert Path(comparison["old_run"]).resolve() == old
    assert Path(comparison["new_run"]).resolve() == run
    for key in ("common_support_across_24_rows", "probability_output_has_no_decoded_metrics",
                "drop_ring_contributes_zero", "saved_probabilities_sum_to_one",
                "MC_probabilities_are_64_sample_counts",
                "all_real_and_paired_masks_reconstructed_from_coordinates"):
        assert checks[key] is True, f"Independent validation failed: {key}"
    before_audit = pd.read_csv(old / "recording_audit.csv")
    after_audit = pd.read_csv(run / "recording_audit.csv")
    old_valid = int(before_audit.gaze_valid_n.sum())
    new_valid = int(after_audit.gaze_valid_n.sum())
    coordinate_valid = int(after_audit.gaze_coordinate_valid_n.sum())
    assert new_valid == coordinate_valid == checks["raw_npy_coordinate_only_valid_frames_checked"]
    assert (old_valid, new_valid) == (comparison["gaze_valid_old"], comparison["gaze_valid_new"])
    assert set(before_audit.content_id) == set(after_audit.content_id)

    old_trials, new_trials = (primary(folder / "trial_stage.csv") for folder in (old, run))
    old_errors, new_errors = (primary(folder / "model_trial_errors.csv") for folder in (old, run))
    models = cfg["models"]
    lines = [
        "# Target locking v3：坐标有效性修复与结果变化", "",
        f"新结果：`{run.name}`；历史对照：`{old.name}`。本次从原始帧重新计算，原v2目录保留。", "",
        "原始gaze直接读取 `UserN/npy/<ID>_gaze.npy` 的结构化字段 `gaze_x`、`gaze_y`，按原视频 `Frame_Number` 对齐。"
        "可用条件仅为两个坐标均有限且位于闭区间 `[0,1]`；端点0和1保留。`valid` 不设范围，也不参与筛帧。"
        "目标、双侧DLC、阶段及模型共同帧条件继续独立生效。", "",
        "主分析、半径及平滑敏感性分支均使用上述输入规则。平滑分支仍按原定规则插值短缺口，单独报告；"
        "旧 `gaze3`、`legacy123` 状态码条件仅保留在v2历史结果中，不属于v3分析。", "",
        f"主半径 target/PSM 为 {cfg['target_radius_px']}/{cfg['psm_radius_px']} px；bootstrap "
        f"{cfg['bootstrap_n']:,} 次，seed={cfg['bootstrap_seed']}。", "",
        "## 输入与实际新增分析支持", "",
    ]
    identical_recordings = comparison["identical_recording_inputs"]
    identical_predictions = comparison["identical_prediction_inputs"]
    if identical_recordings and identical_predictions:
        lines += [f"{len(after_audit)} 份录像的 NPY、video_ts 和 DLC 输入SHA256均与v2一致；"
                  "模型预测文件SHA256也全部一致。原帧号、时间戳与逐帧时长经逐记录核对一致，"
                  "可把此次变化归于gaze筛选规则及其下游重算。", ""]
    else:
        lines += ["输入一致性检查发现变化："
                  f"录像输入一致={identical_recordings}，预测输入一致={identical_predictions}。"
                  "本表差异不能只归因于gaze筛选规则；具体文件见比较JSON。", ""]
    lines += [f"在与阶段、target及DLC取交集之前，合格gaze由 **{old_valid:,}** 帧增至 "
              f"**{new_valid:,}** 帧，净增加 **{new_valid-old_valid:,}** 帧。"
              "下表为经过全部条件后的主分析新增支持；配对集合是全量真实集合的子集，两行不可相加。", ""]
    support_rows = []
    for scope, label in (("real", "GT84真实全量"), ("paired", "五模型配对测试")):
        item = comparison["per_scope"][scope]
        support_rows.append([label, item["recordings"], f"{item['added_frames']:,}",
                             f"{item['added_seconds']:.6f}", f"{item['removed_frames']:,}",
                             f"{item['removed_seconds']:.6f}"])
    lines += [table(["集合", "核对录像", "新增帧", "新增秒", "移除帧", "移除秒"], support_rows), "",
              "逐录像变化见 " + relative_link("frame_support_v2_v3_changes.csv") + "。", "",
              "## 主概率分析：任务整体占用变化", "",
              "下表只比较 `probability_mc / main / whole`。每条录像的指标先按原帧时间计算，"
              "随后在参与者内等权平均录像，再对参与者等权平均。各指标独立排除不可估值，n为实际参与者数。"
              "格式为“v2 → v3（Δ；n）”；比例以%显示，Δ为百分点。C为target-only在两类exclusive占用之和中的比例，"
              "不是target总占用。这里的Δ是两版组均值相减，没有新增差异显著性检验。", ""]
    for scope, label, sources in (("real_all", "GT84真实全量", ["gaze"]),
                                  ("paired_test", "共同帧测试集合", ["gaze", *models])):
        rows = []
        for task in TASKS:
            for source in sources:
                a, b = (select(data, task, scope, source) for data in (old_trials, new_trials))
                rows.append([task, source, *(comparison_cell(a, b, metric) for metric in METRICS)])
        lines += [f"### {label}", "", table(["任务", "来源", "target占用 %", "腕部占用 %", "C %"], rows), ""]
    lines += ["## 模型Brier点估计与排序", "",
              "Brier越小越好；下表沿用参与者等权汇总，格式为“v2 → v3（Δ；n）”，保持原量纲。"
              "排序依据未四舍五入的点估计；它不等于模型差异具有统计显著性。", ""]
    rows = []
    for task in TASKS:
        for model in models:
            a, b = (select(data, task, "paired_test", model) for data in (old_errors, new_errors))
            rows.append([task, model, comparison_cell(a, b, "BS_target", 1, 6),
                         comparison_cell(a, b, "BS_gripper", 1, 6)])
    lines += [table(["任务", "模型", "target Brier", "腕部 Brier"], rows), ""]
    rows = []
    for task in TASKS:
        for metric, label in (("BS_target", "target"), ("BS_gripper", "腕部")):
            a, sig_a = ranking(old_errors, task, metric, models)
            b, sig_b = ranking(new_errors, task, metric, models)
            rows.append([task, label, a, b, "相同" if sig_a == sig_b else "改变"])
    lines += [table(["任务", "事件", "v2从低到高", "v3从低到高", "点估计排序"], rows), "",
              "占用和Brier的变化表不能证明所有结论不变。阶段结果、置信区间、BSS、成绩相关、"
              "敏感性与重复练习对比均应以v3重新生成的表和详细报告为准。", "",
              "## 验证与复核入口", ""]
    regression_path = next((p for p in (run / "regression_tests.log", Path(__file__).parent / "regression_tests.log") if p.is_file()), None)
    if regression_path:
        raw = regression_path.read_bytes()
        regression = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        match = re.search(r"Ran (\d+) tests? in", regression)
        passed = bool(match and re.search(r"(?m)^OK\s*$", regression) and "FAILED" not in regression)
        if passed:
            lines += [f"{int(match.group(1))} 项回归测试通过：覆盖任意valid值、闭区间边界、非有限/越界坐标排除、"
                      "乱序及缺口原帧匹配、重复/缺失帧报错、所有当前分支及平滑输入。", ""]
        else:
            lines += ["回归测试日志未提供完整通过证据，需单独检查。", ""]
    else:
        lines += ["本摘要未找到回归测试日志，不代替该项验证。", ""]
    lines += [f"独立验证已从原始NPY重建全部真实和配对掩码，核对 {checks['raw_npy_coordinate_only_valid_frames_checked']:,} "
              f"个坐标合格帧、{checks['native_frame_gaze_trial_rows_checked']:,} 行真实gaze统计；"
              f"另完成 {checks['independent_probability_trial_checks']:,} 项概率试次检查及 "
              f"{checks['raw_sample_frame_regions_checked']:,} 项原始样本帧区域检查。"
              "五模型/四分析分支的共同支持一致；区域概率和为1；概率为64样本计数；"
              "drop ring未纳入；主概率输出未混入点轨迹驻留/切换指标。", ""]
    links = [
        ("detailed_report_zh.md", "v3详细中文分析报告"),
        ("report.md", "v3结果与分支导航"),
        ("v2_v3_comparison.json", "输入一致性和支持变化摘要"),
        ("frame_support_v2_v3_changes.csv", "逐录像支持变化"),
        ("trial_stage_v2_v3_changes.csv", "逐录像/阶段指标变化"),
        ("model_trial_errors_v2_v3_changes.csv", "逐录像/阶段模型误差变化"),
        ("independent_validation.json", "独立验证结果"),
        ("validation.json", "分析内置验证结果"),
        ("regression_tests.log", "回归测试日志"),
        ("input_metadata_validation.json", "样本元数据与训练划分核验"),
        ("run_config.json", "运行配置与代码哈希"),
        ("output_index.csv", "全部输出文件索引"),
    ]
    lines.extend("- " + relative_link(path, label) for path, label in links if (run / path).is_file())
    lines += ["", f"历史对照：[{old.name}](<{old.as_posix()}/report.md>)。", ""]
    destination = run / "correction_summary_zh.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.run, args.old))
