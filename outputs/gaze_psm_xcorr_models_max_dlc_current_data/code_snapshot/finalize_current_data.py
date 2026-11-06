"""Add current-source audit and compare the fresh calculation to the previous run."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, subprocess, sys

ROOT = Path(__file__).parent/'gaze_psm_xcorr_models_max_dlc_current_data'
AUDIT = Path(__file__).parent.parent/'cache_current_audit'
OLD = Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_models_max_dlc')
SOURCES = ['gt','all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write(path, data):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader();w.writerows(data)


def main():
    global ROOT, AUDIT, OLD
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--audit-root',type=Path,default=AUDIT)
    parser.add_argument('--previous-output',type=Path,default=OLD)
    args=parser.parse_args()
    ROOT,AUDIT,OLD=args.output,args.audit_root,args.previous_output
    audit=json.loads((AUDIT/'audit_summary.json').read_text(encoding='utf-8'))
    assert audit['all_test_arrays_equal'] and audit['n_test_videos']==10
    rechecks=[]
    for item in rows(AUDIT/'current_input_manifest.csv'):
        path=Path(item['path']);stat=path.stat()
        same_stat=stat.st_size==int(item['bytes']) and stat.st_mtime_ns==int(item['mtime_ns'])
        same_content=sha(path)==item['sha256'] if item['sha256'] else None
        assert same_stat and same_content is not False, str(path)
        rechecks.append(dict(content_id=item['content_id'],path=str(path),file_metadata_unchanged=same_stat,
                             content_sha256_verified=same_content))
    target=ROOT/'input_audit'
    shutil.copytree(AUDIT,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    write(target/'post_calculation_input_recheck.csv',rechecks)
    checks=[]
    filenames=['results.json','record_summary.csv','novice_summary.csv','group_summary.csv',
               'bootstrap_peaks.csv','bootstrap_contrasts.csv','unpaired_peaks.csv','contrasts.csv','side_selection.csv']
    paths=[Path(source)/file for source in SOURCES for file in filenames]
    paths += [Path(f) for f in ['paired_record_summary.csv','paired_group_summary.csv','paired_bootstrap.csv',
                               'phase_validity.csv','side_selection.csv','unpaired_pairings.csv']]
    for rel in paths:
        previous,current=sha(OLD/rel),sha(ROOT/rel)
        checks.append(dict(relative_path=str(rel),previous_sha256=previous,current_sha256=current,
                           exact_file_match=previous==current))
    assert all(row['exact_file_match'] for row in checks), [r for r in checks if not r['exact_file_match']]
    write(ROOT/'recalculation_vs_previous.csv',checks)
    summary=dict(current_data_root=audit['current_data_root'],fresh_extraction_reuse_cache=False,
                 current_csv_files_hashed_and_rechecked=336,current_videos_metadata_rechecked=84,
                 test_video_input_arrays_exactly_equal=True,test_videos=10,compared_arrays_per_video=21,
                 all_dataset_completely_equal_videos=83,all_dataset_videos=84,
                 non_test_difference='User9_2_1: gaze only; max difference 0.0108863349 px before smoothing and 0.0037131522 px after smoothing',
                 n_predictions_recalculated=50,n_paired_records=760,
                 previous_result_files_compared=len(checks),all_compared_result_files_exactly_equal=True,
                 all_source_csv_hashes_rechecked_after_calculation=True,
                 video_comparison_scope='Dimensions/fps and file metadata only; no historical source-video content hash is available.')
    (ROOT/'current_data_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    audit_text='''# 当前数据与旧 DLC 缓存的一致性核验

旧分析读取的是 `D:\\GitHub_Project\\co-tracker\\outputs\\gaze_psm_xcorr_dlc\\cache` 中的 NPZ；缓存内部记录原 Linux/NAS 来源。本轮实际从 `D:\\Dataset\\dvss\\Active_Data` 读取原始文件，不依赖缓存的来源标签判断一致性。

## 结果

| 范围 | 结果 |
|---|---|
| 全84份 DLC CSV 的左右坐标、likelihood、帧号和时间 | 逐值精确相等，最大差0 |
| 全84个视频的21类分析数组（原始/平滑gaze与PSM、时间、stage、occurrence、分辨率等） | 83个完全相等；1个非测试视频的4个gaze数组存在微小差异 |
| 本次10个测试视频 | 全部21类数组逐值精确相等，包含NaN位置 |
| 当前来源CSV | 336份计算内容SHA256，并在重新计算结束后再次核验 |
| 当前视频 | 84份独立读取宽高/FPS并核验文件元数据；不宣称与历史视频文件逐字节一致 |
| 使用新提取输入重算50份prediction | 760条配对记录及所有核验的源级/组级统计文件与上次max-DLC结果完全相同 |

## 唯一差异

`User9_2_1` 不在本次10个测试视频中。DLC PSM、时间、stage、分辨率等均一致；仅 gaze 不同：插值后、平滑前的最大坐标差为0.010886334932251884像素，平滑后为0.0037131522313558207像素。旧SAM与旧DLC缓存中的gaze彼此一致。仅凭旧缓存不能确定历史原始gaze CSV是否曾发生精度或内容变化；详细事实见差异诊断报告。此差异不影响本次50份prediction的分析。

## 检查方式与文件

- `audit_and_rebuild.py` 使用原 pipeline 的数据读取、过滤、插值和平滑函数，设 `reuse_cache=False`，从当前 DLC、gaze、stage、video timestamps CSV 重新构建。视频尺寸由ffprobe独立读取，FPS仅用于QC。
- `current_data/cache/` 为此次新建的84份缓存；旧缓存未被覆盖。
- `cache_current_record_comparison.csv` 为逐视频结果；`cache_current_field_comparison.csv` 为1764项逐数组检查。
- `current_input_manifest.csv` 保存当前原CSV的路径、大小、时间和SHA256；`post_calculation_input_recheck.csv` 记录分析结束后的二次核验。
- `all84_native_dlc_audit.json` 等是另一条独立读取路径完成的原生DLC检查。
- 旧 fingerprint 包含Linux绝对路径、文件大小及mtime，并不是原CSV内容哈希。迁移到Windows后fingerprint不同是预期现象；本次结论依据逐数组数值比较。
'''
    (target/'README.md').write_text(audit_text,encoding='utf-8')
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    readme=readme.replace('本目录是用户指定 `outputs/gaze_psm_xcorr_max_dlc` 方法后的正式重算版本。此前 `gaze_psm_xcorr_models` 使用 SAM mask 和不同主分析选侧规则，其数值不作为本版本结论。',
                          '本目录从当前 `D:\\Dataset\\dvss\\Active_Data` 原始文件重新提取输入，再按 `outputs/gaze_psm_xcorr_max_dlc` 方法从头重算50份prediction。未启用prepared/computed/null结果复用。旧缓存及旧输出仅用于独立一致性和结果回归核验。')
    readme=readme.replace('PSM 使用 `gaze_psm_xcorr_dlc/cache` 中独立的左/右 DLC x/y，cache version 5；不按 likelihood 设阈值，jump 阈值仅用于 QC。GT 使用缓存的高频原始 gaze（Status=2）。',
                          'PSM 从当前用户 sam 子目录中的原 DLC CSV 读取独立左右 x/y，重新构建cache version 5；不按 likelihood 设阈值，jump 阈值仅用于 QC。GT 从当前原始 gaze CSV 重新读取并筛选 Status=2。')
    readme=readme.replace('未重新解码原始视频；复现从已有 DLC 缓存开始。',
                          '本次从当前CSV新提取缓存，视频宽高由当前MP4独立读取，未逐帧解码；完整输入审计及重建脚本见 `input_audit/`。')
    readme=readme.replace('python code_snapshot/analyze_model_xcorr_max_dlc.py --output <new_output> --workers 4',
                          'python input_audit/audit_and_rebuild.py\npython code_snapshot/analyze_model_xcorr_max_dlc.py --output <new_output> --source-cache-root input_audit/current_data --workers 4')
    readme += '\n## 当前数据核验与本次重算\n\n优先阅读 `input_audit/README.md`。84份DLC输入完全一致；83/84视频全部分析数组一致，唯一差异为非测试User9_2_1的微小gaze差异。本次10测试视频的21类数组全部精确一致。重新计算后的50份prediction结果与上次max-DLC输出一致，逐文件核验见 `recalculation_vs_previous.csv`，核验摘要见 `current_data_validation.json`。\n\n本目录 `cache/gt_legacy` 和 `cache/gt` 均由本轮当前原始文件新提取；`gt_legacy` 仅是共同支持限制前的兼容命名，并非读入历史NPZ。\n'
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    meta=json.loads((ROOT/'delivery_metadata.json').read_text(encoding='utf-8'))
    meta.update(delivery_root=r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_models_max_dlc_current_data',
                calculation_root=str(ROOT),current_data_root=audit['current_data_root'],
                supersedes_for_this_request='gaze_psm_xcorr_models_max_dlc: same numerical results, now independently rebuilt from current local raw inputs',
                input_audit='input_audit/README.md')
    (ROOT/'delivery_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    validation=json.loads((ROOT/'validation.json').read_text(encoding='utf-8'))
    validation.update(current_data_audit=summary)
    (ROOT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    shutil.copy2(__file__,ROOT/'code_snapshot'/Path(__file__).name)
    manifest=[dict(relative_path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.csv']
    write(ROOT/'artifact_manifest.csv',manifest)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
