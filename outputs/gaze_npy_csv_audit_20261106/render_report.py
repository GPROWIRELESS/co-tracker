import csv,json,html,shutil,hashlib
from pathlib import Path
from collections import Counter
OUT=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\outputs\gaze_npy_csv_audit_20261106')
WORK=Path(__file__).parent
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
ind=json.loads((WORK/'independent_gaze_pair_results.json').read_text(encoding='utf-8'))['summary']
assert (s['pairs_checked'],s['rows_checked'],s['core_equal_pairs'],s['common_equal_pairs'],s['schema_equal_pairs'])==(ind['pairs'],ind['rows'],ind['critical_match_pairs'],ind['shared_field_match_pairs'],ind['schema_equal_pairs'])
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def table(headers,data):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in data)+'</table></div>'
users=rows(OUT/'per_user_summary.csv');pairs=rows(OUT/'pair_summary.csv')
fields=json.loads((OUT/'field_comparison.json').read_text(encoding='utf-8'))
counter=Counter(r['field'] for r in fields if not r['equal'])
new={'User1_6_1_gaze','User2_3_1_gaze','User5_2_1_gaze','User5_9_1_gaze','User11_2_1_gaze'}
links=' '.join(f'<a href="{p}">{label}</a>' for p,label in [('pair_summary.csv','逐文件结果'),('per_user_summary.csv','逐用户结果'),('field_comparison.csv','逐字段差异'),('source_manifest.json','源文件指纹'),('independent_annotation_details.json','目标标注差异详情')])
sections=[]
def add(t,b):sections.append('<section><h2>'+t+'</h2>'+b+'</section>')
add('结论：核心gaze一致，完整文件不一致',
    '<p><b>279组同名NPY/CSV的Frame_Number、Timestamp(ns)、gaze_x、gaze_y、valid全部一致，行数与行序一致。</b>存在stage字段的文件，其stage也全部一致。</p>'+
    '<p>83组Active文件的point_label_mode及目标点坐标存在差异，因此只能确认GT gaze核心内容一致，不能把NPY与CSV作为完整标注文件互换。</p>'+
    table(['数据域','用户数','文件对','核心gaze一致','全部共有字段一致','完整字段集合和值均一致'],[
        ['Active_Data',14,157,157,74,73],['Passive_Data',11,122,122,122,122],['合计','25个用户目录（跨域重复）',279,279,196,195]])+
    '<p>共核对2,864,975行，未发现缺失配对或读取错误。排除了128个历史备份文件；没有修改原始数据或运行新的模型分析。</p>')
add('一致性判定方法','<p>逐行比较同名文件。整数和科学计数法时间戳使用Decimal精确解析，避免float64丢失纳秒精度；浮点按NPY字段的实际存储精度进行往返比较，空白与NaN视为同样的缺失值；字符串直接比较，不截断、不改写阶段名称。</p><p>浮点CSV文本与float32二进制的末位表示差异单独记录，不把正常序列化误差误判为gaze不同。主核验与独立实现得到相同的文件数、行数及一致性结果；558个当前源文件在核验前后指纹均未变化。</p>')
add('逐用户结果',table(['数据域','用户','文件对','核心gaze一致','共有字段一致','完整字段和值一致'],[[r['domain'],r['user'],r['pairs_checked'],r['core_equal_pairs'],r['common_equal_pairs'],r['whole_equal_pairs']] for r in users]))
add('完整文件的差异',
    '<ul><li>83组Active文件：point_label_mode和point_1_x/y至point_5_x/y存在值或缺失情况差异。</li><li>83个Active CSV额外含gaze_x_old、gaze_y_old；NPY没有这两列。</li><li>17个Active NPY额外含purple_string_knot_center；CSV没有此列。</li></ul>'+
    '<p>目标点差异主要是NPY有坐标而CSV为空，但也有CSV有坐标而NPY为空，以及两者都有坐标但数值不同的情况。这里不根据文件修改时间推断哪个版本正确。</p>'+
    table(['当前新模型涉及的录像','gaze核心','stage','目标点／标注模式'],[[r['record'],'一致','一致','不一致'] for r in pairs if r['domain']=='Active_Data' and r['record'] in new])+
    '<p>这些目标标注差异会影响target locking所用的目标区域，不能仅凭gaze一致就认为两种文件可直接替换。完整83组清单与每个字段的示例行见逐字段差异表。</p>')
add('时间戳一致不代表时间戳有效',
    '<p>User1_1_1_gaze与User1_6_1_gaze在NPY和CSV中的Timestamp(ns)虽一致，但各自整段只有一个唯一值；分别有12,594和7,326个非递增间隔。这是两份文件共同具有的时间信息问题，不是NPY/CSV之间的差异。本次未修正时间戳。</p>')
body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NPY与CSV gaze一致性核验</title><style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#f3f5f5;color:#243b47;margin:0}main{max-width:1150px;margin:auto;padding:32px}section{background:white;padding:25px;margin:20px 0;border:1px solid #dfe6e8;border-radius:8px}h1{font-size:30px}h2{font-size:21px;color:#176779}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:8px 12px}th{background:#e9f2f2}.scroll{overflow:auto}a{color:#146d83;margin-right:20px}</style><main><h1>NPY与CSV gaze一致性核验</h1><p>2026-11-06 · D:\\Dataset\\dvss\\Active_Data / Passive_Data · User*\\npy</p><p>'+links+'</p>'+''.join(sections)+'</main></html>'
(OUT/'report_zh.html').write_text(body,encoding='utf-8')
for name in ['independent_gaze_pair_results.json','independent_annotation_details.json']:
    shutil.copy2(WORK/name,OUT/name)
for p in WORK.glob('*.py'):shutil.copy2(p,OUT/p.name)
validation=dict(completed=True,all_279_core_gaze_pairs_equal=True,all_files_completely_equal=False,
                annotation_difference_pairs=83,independent_implementation_agrees=True,source_files_unchanged=s['source_files_unchanged'],
                pairs=279,total_rows=2864975,scope=['Active_Data/User*/npy','Passive_Data/User*/npy'],no_data_repair_or_analysis_rerun=True)
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
artifacts=[]
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name!='artifact_manifest.json':artifacts.append(dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(OUT/'artifact_manifest.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
print(json.dumps(validation,ensure_ascii=True))
