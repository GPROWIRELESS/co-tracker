"""Finalize the coordinate-only revision directly in its authorized output directory."""
from pathlib import Path
import argparse,json,hashlib,re,shutil
from urllib.parse import unquote

DEFAULT_ROOT=Path(r'D:\GitHub_Project\co-tracker\outputs\novice_transition_A_seed42_20261105_analysis_20261106')
WORK=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\transition_A_coordinate_only_work')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    root=args.root.resolve()
    components=['input_audit/validation.json','target_locking/validation.json','gaze_psm_lead/validation.json',
        'gaze_psm_lead/supplement_validation.json','gaze_psm_lead/zero_horizon_validation.json',
        'verification/independent_input_validation.json','verification/independent_result_validation.json',
        'verification/coordinate_only_input_validation.json',
        'verification/lead_completion.json','verification/lead_loo_change_validation.json']
    verifications={}
    for rel in components:
        v=json.loads((root/rel).read_text(encoding='utf8'));verifications[rel]=v
        if not any(k in v for k in ['passed','complete']):
            assert rel=='target_locking/validation.json',rel
            for key in ['anchor_common_support_identical_all_models','four_region_probability_conservation','whole_equals_stage_exposure_sum','source_weighting_handles_stride2','input_arrays_unmodified']:
                assert v[key] is True,(rel,key)
            assert v['valid_column_used_for_selection'] is False and v['scenario_count']==3
        for key in ['passed','complete']:
            if key in v:assert v[key] is True,(rel,key)
    meta=json.loads((root/'input_audit/metadata.json').read_text(encoding='utf8'))
    assert meta['valid_column_used_for_selection'] is False
    audits=json.loads((root/'input_audit/native_audit.json').read_text(encoding='utf8'))
    assert all(a['valid_column_used_for_selection'] is False and a['gaze_valid_n']==a['gaze_coordinate_valid_n'] for a in audits)
    config=json.loads((root/'target_locking/run_config.json').read_text(encoding='utf8'))
    assert set(config['scenarios'])=={'main','r25','r100'}
    assert verifications['target_locking/validation.json']['anchor_common_support_identical_all_models'] is True
    assert verifications['gaze_psm_lead/validation.json']['shared_support_all_preprocessing_checks'] is True
    assert verifications['gaze_psm_lead/zero_horizon_validation.json']['max_abs_anchor_vs_target0_correlation_difference']==0
    source_manifest=json.loads((root/'input_audit/input_manifest.json').read_text(encoding='utf8'))
    for item in source_manifest:assert sha(item['path'])==item['sha256'],item['path']
    for source,destination in [('prepare_inputs.log','prepare_inputs.log'),('independent_input_validation.log','independent_validation.log'),('independent_result_validation.log','independent_results.log')]:
        p=WORK/source
        if p.exists():shutil.copy2(p,root/'execution_logs'/destination)
    reports=list((root/'report').glob('*.html'))+[root/'index.html',root/'analysis.md',root/'summary.md',root/'target_locking/report_zh.md']
    missing=[];links=0
    for path in reports:
        text=path.read_text(encoding='utf8')
        refs=re.findall(r'(?:href|src)="([^"]+)"',text) if path.suffix=='.html' else re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        for rel in refs:
            if rel.startswith(('http:','https:','data:','#')):continue
            rel=unquote(rel.split('#')[0]);target=Path(rel)
            if not target.is_absolute():target=path.parent/target
            links+=1
            if not target.exists():missing.append([str(path),rel])
    assert not missing,missing
    old_manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf8'))
    snapshot=root/'code_snapshot'
    numerical_kernels=['gaze_psm_core.py','gaze_psm_core_legacy.py','gaze_psm_preprocess_v2.py','gaze_psm_reliability_v2.py','gaze_psm_statistics_v2.py','tl_common.py','tl_geometry.py','tl_stats.py']
    unchanged={name:sha(snapshot/name)==old_manifest['code_sha256'][name] for name in numerical_kernels}
    assert all(unchanged.values()),unchanged
    helper=(snapshot/'gaze_psm_pipeline_dlc.py').read_bytes()
    guard=b'    raise RuntimeError("Legacy GT cache entry is disabled in this coordinate-only snapshot; use prepare_inputs.py")'
    nl=b'\r\n' if b'\r\n' in helper else b'\n'
    assert helper.count(guard)==2
    original_helper_hash=old_manifest.get('helper_usage_safety',{}).get('original_sha256',old_manifest['code_sha256']['gaze_psm_pipeline_dlc.py'])
    assert hashlib.sha256(helper.replace(guard+nl,b'')).hexdigest()==original_helper_hash
    helper_usage=dict(disabled_legacy_entries=['extract','main'],helper_functions_unchanged=True,original_sha256=original_helper_hash,
        replacement_input_entry='code_snapshot/prepare_inputs.py')
    new_manifest=dict(analysis='Task A four transition models vs paired GT',date='2026-11-06',
        revision='coordinate_only_in_place',execution_output=str(root),delivery_output=str(root),
        input_log_root=meta['log_root'],source_tools=r'D:\GitHub_Project\co-tracker\tools',
        previous_results_overwritten=True,backup_created=False,raw_data_and_models_unchanged=True,
        gaze_source='User/npy/*_gaze.npy gaze_x/gaze_y',gaze_validity='finite normalized x/y in inclusive [0,1]',
        valid_column_used_for_selection=False,nonfinite_policy='NaN/Inf treated as missing; no interpolation across gaps longer than existing limit',
        native_coordinate_valid_n=sum(a['gaze_valid_n'] for a in audits),removed_status_only_sensitivity='gaze3',
        methods=['heatmap regional probability integration','expected and argmax point decoders','fixed bilateral PSM cross-correlation'],
        split='validation used for checkpoint selection; exploratory',model_horizons_ms=[0,500,1000,1500],
        python=r'C:\ProgramData\anaconda3\envs\sam3\python.exe',numerical_kernels_unchanged=unchanged,helper_usage_safety=helper_usage,
        code_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},
        report='report/analysis_report_zh.html',markdown_reports=['analysis.md','summary.md'],
        main_figures=['figures/tl_decoding_occupancy_zh.png','figures/lead_whole_curves_zh.png'])
    dump(root/'run_manifest.json',new_manifest)
    dump(root/'validation.json',dict(complete=True,revision='coordinate_only_in_place',valid_column_used_for_selection=False,
        native_coordinate_valid_n=new_manifest['native_coordinate_valid_n'],report_links_resolve=True,report_links_checked=links,
        original_input_hashes_unchanged=True,component_validation_files=components,
        numerical_kernels_unchanged=unchanged,previous_analysis_replaced_in_place=True,backup_created=False))
    instructions='''复现说明：坐标有效性修正版

当前目录已按用户要求直接重算并覆盖，没有创建旧结果备份。GT坐标来自User/npy中的NPY，
仅采用有限且归一化x、y均在闭区间[0,1]的点。valid/Status不参与筛选；NaN/Inf视为缺失。
原始NPY、时间文件、DLC和训练模型预测保持不变。gaze3状态敏感性已删除，半径25/50/100 px保留。

Python环境：C:\\ProgramData\\anaconda3\\envs\\sam3\\python.exe
以当前目录为ROOT，依次运行（将覆盖ROOT内相应结果）：
python -B ROOT/code_snapshot/prepare_inputs.py --output ROOT
python -B ROOT/code_snapshot/reproduce_target.py --inputs ROOT/input_audit --output ROOT/target_locking
python -B ROOT/gaze_psm_lead/code_snapshot/reproduce_lead.py --root ROOT --workers 4
python -B ROOT/code_snapshot/independent_validation.py
python -B ROOT/code_snapshot/independent_result_validation.py

然后运行冻结的build_audit_figure.py、render_decoding_occupancy_zh.py、build_lead_main_figure.py、build_report.py。
报告脚本的ROOT指向本目录。build_report.py同步生成HTML/TXT、analysis.md与summary.md。
数值参数保存在target_locking/run_config.json和gaze_psm_lead/run_manifest.json。
原生输入在input_audit/native，模型概率缓存在input_audit/predictions。predictive target-time使用名义horizon。
已保存的future_valid等训练字段仅为训练来源审计，不能重新作为本轮GT状态筛选规则。
冻结gaze_psm_pipeline_dlc.py的旧extract/main建缓存入口已禁用；辅助函数保持原样。
请通过prepare_inputs.py重建输入，避免旧raw gaze或状态筛选入口。
最终用code_snapshot/finalize_package.py更新文件清单，verify_delivery.py核对当前目录指纹。
'''
    (root/'reproduce_zh.txt').write_text(instructions,encoding='utf8')
    artifacts=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name not in ['artifact_manifest.json','delivery_validation.json']]
    dump(root/'artifact_manifest.json',dict(files=artifacts,file_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),revision='coordinate_only_in_place'))
    # Re-read every listed artifact after publishing the manifest.
    for row in artifacts:
        p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    delivery=dict(complete=True,delivered_to=str(root),mode='in_place',revision='coordinate_only',
        verified_files=len(artifacts),bytes=sum(r['bytes'] for r in artifacts),all_manifest_file_hashes_match=True,
        previous_results_overwritten=True,backup_created=False,raw_data_and_models_unchanged=True,
        valid_column_used_for_selection=False,report='report/analysis_report_zh.html',markdown_reports=['analysis.md','summary.md'])
    dump(root/'delivery_validation.json',delivery)
    print(json.dumps(delivery,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
