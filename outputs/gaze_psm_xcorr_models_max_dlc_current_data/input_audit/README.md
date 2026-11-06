# 当前数据与旧 DLC 缓存的一致性核验

旧分析读取的是 `D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_dlc\cache` 中的 NPZ；缓存内部记录原 Linux/NAS 来源。本轮实际从 `D:\Dataset\dvss\Active_Data` 读取原始文件，不依赖缓存的来源标签判断一致性。

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
