# 模型补充：区域对应与 GT 平滑诊断

全部结果限定为本次16对中的原模型测试交集：Task A 2对/2位源操作者，Task C 1对/1位源操作者。
Task C 只有一个独立源操作者，均值区间为NA。每项数据均提供 active/passive 对应结果。

## 区域对应矩阵

每个配对先按共同有效时间归一化，再按内容内observer、源操作者内内容、操作者依次等权。条件矩阵先对每个配对的实测类别归一化；只有两类gaze对该行均有支持的配对才进入该条件对照，零分母保留NA。矩阵行是实测gaze区域，列是模型解码区域；上排显示joint share，下排显示条件比例，左为active、右为passive。

表中 `reference_region` 是两类参照共用的区域代码索引：`active_*` 列按 active gaze 的该区域统计，`passive_*` 列按 passive gaze 的该区域统计。`active_region` 仅为兼容旧读取代码保留的同值别名，不表示 passive 列使用 active gaze 的分类。

[内容汇总](model_confusion_content_summary.csv) · [操作者汇总](model_confusion_participant_summary.csv) · [组汇总](model_confusion_group_summary.csv)

## GT-smoothed诊断

使用原v3的sigma=1网格核，分别由active/passive实测gaze构造，仅作标签平滑诊断。它不是独立预测器、严格上界或BSS参照；不加入模型排名。

[内容汇总](model_gt_smoothed_content_summary.csv) · [操作者汇总](model_gt_smoothed_participant_summary.csv) · [组汇总](model_gt_smoothed_group_summary.csv)

## 图件

### Task A · decoded_argmax · all_ABCD

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/all_ABCD_task_A_confusion.png)

### Task C · decoded_argmax · all_ABCD

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/all_ABCD_task_C_confusion.png)

### Task A · decoded_argmax · leave_out

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/leave_out_task_A_confusion.png)

### Task C · decoded_argmax · leave_out

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/leave_out_task_C_confusion.png)

### Task A · decoded_argmax · single_task

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/single_task_task_A_confusion.png)

### Task C · decoded_argmax · single_task

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/single_task_task_C_confusion.png)

### Task A · decoded_argmax · spd_finetune

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_finetune_task_A_confusion.png)

### Task C · decoded_argmax · spd_finetune

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_finetune_task_C_confusion.png)

### Task A · decoded_argmax · spd_zero_shot

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_zero_shot_task_A_confusion.png)

### Task C · decoded_argmax · spd_zero_shot

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_zero_shot_task_C_confusion.png)

### Task A · decoded_expectation · all_ABCD

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/all_ABCD_task_A_confusion.png)

### Task C · decoded_expectation · all_ABCD

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/all_ABCD_task_C_confusion.png)

### Task A · decoded_expectation · leave_out

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/leave_out_task_A_confusion.png)

### Task C · decoded_expectation · leave_out

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/leave_out_task_C_confusion.png)

### Task A · decoded_expectation · single_task

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/single_task_task_A_confusion.png)

### Task C · decoded_expectation · single_task

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/single_task_task_C_confusion.png)

### Task A · decoded_expectation · spd_finetune

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_finetune_task_A_confusion.png)

### Task C · decoded_expectation · spd_finetune

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_finetune_task_C_confusion.png)

### Task A · decoded_expectation · spd_zero_shot

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_zero_shot_task_A_confusion.png)

### Task C · decoded_expectation · spd_zero_shot

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_zero_shot_task_C_confusion.png)

### Task A · decoded_sample · all_ABCD

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/all_ABCD_task_A_confusion.png)

### Task C · decoded_sample · all_ABCD

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/all_ABCD_task_C_confusion.png)

### Task A · decoded_sample · leave_out

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/leave_out_task_A_confusion.png)

### Task C · decoded_sample · leave_out

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/leave_out_task_C_confusion.png)

### Task A · decoded_sample · single_task

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/single_task_task_A_confusion.png)

### Task C · decoded_sample · single_task

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/single_task_task_C_confusion.png)

### Task A · decoded_sample · spd_finetune

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_finetune_task_A_confusion.png)

### Task C · decoded_sample · spd_finetune

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_finetune_task_C_confusion.png)

### Task A · decoded_sample · spd_zero_shot

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_zero_shot_task_A_confusion.png)

### Task C · decoded_sample · spd_zero_shot

![Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_zero_shot_task_C_confusion.png)

### Task A · GT_smoothed_diagnostic · 

![Paired active/passive GT-smoothed occupancy diagnostic; no BSS baseline](figures/model_supplement/task_A_gt_smoothed.png)

### Task C · GT_smoothed_diagnostic · 

![Paired active/passive GT-smoothed occupancy diagnostic; no BSS baseline](figures/model_supplement/task_C_gt_smoothed.png)
