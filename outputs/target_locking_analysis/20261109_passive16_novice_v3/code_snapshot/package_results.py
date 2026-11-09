"""Package the audited analysis, add an offline HTML reader and file inventory."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import re
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'output')
    out=ap.parse_args().output;work=Path(__file__).resolve().parents[1]
    for folder in ['validation','logs','code_snapshot','figure_data']: (out/folder).mkdir(exist_ok=True)
    for path in (work/'code').glob('*.py'):shutil.copy2(path,out/'code_snapshot'/path.name)
    for path in (work/'validation').iterdir():
        if path.is_file():shutil.copy2(path,out/'validation'/path.name)
    validations=[]
    for path in (out/'validation').glob('independent*.json'):
        result=json.loads(path.read_text(encoding='utf-8'))
        assert result.get('status')=='passed',f'Validation failed: {path.name}'
        validations.append(dict(report=path.relative_to(out).as_posix(),status=result['status'],checks=result.get('check_count')))
    required=['independent_validation.json','independent_model_validation.json',
              'independent_human_statistics_validation.json','independent_model_statistics_validation.json']
    assert all((out/'validation'/name).is_file() for name in required),'Human/model measurement and human/model statistics validations required'
    (out/'validation_summary.json').write_text(json.dumps(dict(status='passed',reports=validations),indent=2,ensure_ascii=False),encoding='utf-8')
    for path in work.glob('*.log'):shutil.copy2(path,out/'logs'/path.name)
    run=json.loads((out/'run_config.json').read_text(encoding='utf-8'))
    groups=pd.read_csv(out/'group_stage_summary_long.csv')
    main=groups[(groups.scenario=='main')&(groups.stage=='whole')]
    figures=pd.read_csv(out/'figure_index.csv')
    allfig=figures.to_dict('records')
    supplement=out/'model_supplement_figure_index.csv'
    if supplement.exists():allfig+=pd.read_csv(supplement).to_dict('records')
    examples=pd.read_csv(out/'example_index.csv')
    for row in examples.itertuples():
        allfig.extend([dict(path=row.overlay,description=f'{row.pair_id} 双gaze视频帧',data_source='example_index.csv'),
            dict(path=row.timeline,description=f'{row.pair_id} 双gaze时间序列',data_source=f'figure_data/examples/{row.pair_id}_timeline.csv.gz')])
    allfig=list({f['path']:f for f in allfig}.values())
    # A plot's paired data are accessible alongside the visual, not hidden in code.
    data_cache={}; plot_index=[]
    for fig in allfig:
        rel=fig.get('path',fig.get('figure',''))
        if not rel or not isinstance(rel,str):continue
        assert (out/rel).is_file(),rel
        source=fig.get('data_source','')
        if not isinstance(source,str) or not source:source='group_stage_summary_long.csv'
        if rel.startswith('figures/task_'):
            name=Path(rel).name
            source={'coverage.png':'coverage.csv','performance.png':'performance_participants.csv',
                'rounds.png':'participant_round_summary.csv','region_correspondence.png':'region_joint_summary.csv',
                'agreement.png':'participant_stage_summary.csv'}.get(name,'group_stage_summary_long.csv')
        elif rel.startswith('figures/models/'):
            source='model_brier_skill.csv' if rel.endswith('_bss.png') else 'model_group_stage_summary_long.csv'
        elif rel.startswith('figures/model_supplement/'):
            source='model_gt_smoothed_group_summary.csv' if rel.endswith('_gt_smoothed.png') else 'model_confusion_group_summary.csv'
        if (out/source).is_file() and source.endswith('.csv') and not rel.startswith('figures/examples/'):
            if source not in data_cache:data_cache[source]=pd.read_csv(out/source)
            data=data_cache[source].copy()
            # Filter context, while retaining active/passive/delta together.
            task=next((t for t in ['A','C'] if f'task_{t}' in rel),None)
            if task and 'task' in data:data=data[data.task.eq(task)]
            if '/models/' in rel:
                bits=rel.split('/');branch,model=bits[2],bits[3]
                if 'analysis_branch' in data:data=data[data.analysis_branch.eq(branch)]
                if 'model' in data:data=data[data.model.eq(model)]
            elif '/model_supplement/' in rel and rel.endswith('_confusion.png'):
                bits=rel.split('/');branch=bits[2];model=Path(rel).name.split('_task_')[0]
                data=data[data.analysis_branch.eq(branch)&data.model.eq(model)]
            if 'scenario' in data and not rel.endswith('sensitivity.png'):data=data[data.scenario.eq('main')]
            if 'stage' in data and Path(rel).name in ['agreement.png','sensitivity.png','performance.png','rounds.png','region_correspondence.png','coverage.png']:
                data=data[data.stage.eq('whole')]
            if 'stage' in data and rel.endswith('_confusion.png'):data=data[data.stage.eq('whole')]
            datarel='figure_data/'+rel.removeprefix('figures/').removesuffix('.png')+'.csv'
            (out/datarel).parent.mkdir(parents=True,exist_ok=True)
            data.to_csv(out/datarel,index=False,encoding='utf-8-sig',na_rep='NA')
        else:datarel=source
        plot_index.append(dict(path=rel,description=fig.get('description',Path(rel).stem),data_source=datarel))
    pd.DataFrame(plot_index).to_csv(out/'all_figure_index.csv',index=False,encoding='utf-8-sig')
    def fmt(v):return f'{v:.3f}' if pd.notna(v) else 'NA'
    def record(task,metric,domain):
        g=main[main.task.eq(task)&main.metric.eq(metric)&main.domain.eq(domain)]
        return g.iloc[0] if len(g) else None
    rows=[]
    for task in ['A','C']:
        for metric in ['L_target','L_gripper','C','D']:
            a,p,d=[record(task,metric,domain) for domain in ['active','passive','delta']]
            rows.append(f'<tr><td>{task}</td><td>{metric}</td><td>{fmt(a["mean"])}</td><td>{fmt(p["mean"])}</td><td>{fmt(d["mean"])}</td><td>[{fmt(d.ci_low)}, {fmt(d.ci_high)}]</td><td>{int(d.n_participants)}</td></tr>')
    def a(path,label):return f'<a href="{html.escape(path,quote=True)}">{html.escape(label)}</a>'
    def figblock(fig):
        return f'<figure><a href="{fig["path"]}"><img loading="lazy" src="{fig["path"]}" alt="{html.escape(str(fig["description"]),quote=True)}"></a><figcaption>{html.escape(str(fig["description"]))} · {a(fig["data_source"],"配对绘图数据")}</figcaption></figure>'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Target locking · novice–novice 16对</title>',
        '<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#233342;background:#f4f6f8;margin:0;line-height:1.65}main{max-width:1180px;margin:auto;padding:36px 26px;background:white}h1{font-size:29px}h2{margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:8px}a{color:#195c96}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:14px;color:#596570}summary{cursor:pointer;font-weight:bold;padding:12px;background:#eef3f6}details{margin:14px 0}.lead{font-size:18px}.note{background:#eef4f7;padding:16px;border-radius:6px}nav{display:flex;gap:18px;flex-wrap:wrap}code{background:#eee;padding:2px 5px}</style><main>',
        '<h1>Target locking：Passive 与配对 Active gaze</h1><p class="lead">novice source × novice observer · Task A 7对 / Task C 9对</p>',
        '<nav><a href="#overview">主要结果</a><a href="#human">人类gaze图表</a><a href="#models">模型子集</a><a href="#examples">逐对示例</a><a href="#files">数据与复现</a></nav>',
        '<p class="note">16对、16段源录像、10位源操作者、8位观察者。仅以原始NPY行数相等判定逐行配对；两类gaze共享源视频时间、阶段和区域，在共同有效帧上比较。冻结复用参考v3的空间、时长、驻留和切换指标。置信区间按源操作者bootstrap，条件于当前观察者集合。</p>',
        '<h2 id="overview">主要结果 · 主半径50/50 px · whole</h2><p>以下是源操作者等权均值；Δ=Passive−Active。占用比例取值0–1。whole为原v3纳入阶段的并集。</p>',
        '<table><thead><tr><th>任务</th><th>指标</th><th>Active</th><th>Passive</th><th>Δ</th><th>Δ的95% CI</th><th>操作者n</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>',
        '<p>两任务的passive target占用均低于对应active。C描述排除overlap后的相对target偏好，不能与target占用混为一谈。具体阶段、敏感性及有效支持见下图和完整报告。</p>',
        '<p>'+a('report.md','完整中文分析报告（Markdown）')+' · '+a('group_stage_summary.csv','全部组水平指标与CI')+'</p>',
        '<h2 id="human">人类 gaze 配对分析</h2>']
    for task in ['A','C']:
        parts.append(f'<h3>Task {task}</h3>')
        parts.extend(figblock(f) for f in plot_index if f['path'].startswith(f'figures/task_{task}/'))
    parts+=['<h2 id="models">模型比较子集</h2><p class="note">仅3对：Task A 2对、Task C 1对。沿用原五模型、64-sample概率分析和三种点解码。Task C仅1位源操作者，不提供参与者bootstrap置信区间。模型分别以active和passive为参照；GT平滑仅为诊断。</p>']
    for branch in ['probability_mc','decoded_expectation','decoded_argmax','decoded_sample']:
        fs=[f for f in plot_index if f'/models/{branch}/' in f['path']]
        if fs:parts+=['<details><summary>'+branch+'</summary>']+[figblock(f) for f in fs]+['</details>']
    supp=[f for f in plot_index if '/model_supplement/' in f['path']]
    if supp:parts+=['<details><summary>模型区域对应矩阵与GT平滑诊断</summary>']+[figblock(f) for f in supp]+['</details>']
    parts+=['<h2 id="examples">逐对视频帧与时间序列</h2><p>每对示例帧固定取main共同有效帧的中位索引，不依据结果挑选。Active为蓝色圆点，Passive为橙色三角。</p>']
    for row in examples.itertuples():
        fs=[f for f in plot_index if f['path'] in [row.overlay,row.timeline]]
        parts+=['<details><summary>'+html.escape(row.pair_id)+'</summary>']+[figblock(f) for f in fs]+['</details>']
    files=[('report.md','详细中文报告'),('pair_manifest.csv','16对输入与来源'),('pair_metrics.csv','逐配对指标'),('coverage.csv','有效帧审计'),
        ('group_stage_summary.csv','组水平全部指标'),('model_pair_metrics.csv','模型逐配对指标'),('all_figure_index.csv','完整图与数据索引'),
        ('active_reproduction.csv','active指标复现记录'),('validation_summary.json','独立验证摘要'),('run_config.json','运行配置'),('statistics_manifest.json','统计配置')]
    parts+=['<h2 id="files">数据、验证与复现</h2><ul>']+[f'<li>{a(path,label)}</li>' for path,label in files]+['</ul>']
    for path in sorted((out/'validation').glob('*.json')):parts.append('<p>'+a(path.relative_to(out).as_posix(),path.name)+'</p>')
    parts+=['<p>运行入口与冻结v3内核保存在code_snapshot；命令记录见reproduce.txt。原始数据及原v3目录未修改。</p></main></html>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    (out/'reproduce.txt').write_text('''Environment: C:\\ProgramData\\anaconda3\\envs\\sam3\\python.exe
Use a new writable OUTPUT directory. Original source paths are recorded in run_config.json.
python -B code_snapshot/analyze_passive16.py --output OUTPUT
python -B code_snapshot/model_passive16.py --output OUTPUT
python -B code_snapshot/plot_examples.py --output OUTPUT
python -B code_snapshot/model_supplement_report.py --output OUTPUT
python -B code_snapshot/report_passive16.py --output-dir OUTPUT
See validation scripts for their CLI. The packaging helper uses the staging workspace layout.
''',encoding='utf-8')
    missing=[]
    for path in [out/'report.html',out/'report.md',out/'model_supplement_report.md']:
        content=path.read_text(encoding='utf-8')
        links=(re.findall(r'(?:href|src)="([^"]+)"',content) if path.suffix=='.html'
               else re.findall(r'\]\(([^)]+)\)',content))
        for link in links:
            if link.startswith(('#','http:','https:')):continue
            target=link.split('#')[0]
            if not (path.parent/target).exists():missing.append(dict(document=path.name,target=target))
    assert not missing,missing
    pngs=list((out/'figures').rglob('*.png'))
    for path in pngs:
        with Image.open(path) as im:im.verify()
    assert {p.relative_to(out).as_posix() for p in pngs}=={f['path'] for f in plot_index}
    assert all((out/f['data_source']).is_file() for f in plot_index)
    (out/'final_output_validation.json').write_text(json.dumps(dict(status='passed',
        figures_verified=len(pngs),figure_data_links_verified=len(plot_index),missing_links=missing,
        independent_validation_reports=len(validations),scope_pairs=16,scope_task_counts={'A':7,'C':9},
        visual_review=['task_A/paired_differences.png','task_C/core_metrics.png',
            'models/probability_mc/all_ABCD/task_C_bss.png','model_supplement/task_C_gt_smoothed.png',
            'examples/User1_4_User11_2_1_overlay.png','examples/User1_4_User11_2_1_timeline.png'],
        visual_review_result='readable labels and legends; paired references shown; n=1 model CI absent'),indent=2),encoding='utf-8')
    run.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        generated_figures=len(plot_index),entry_report='report.html',
        implementation_sha256={p.name:sha(p) for p in (out/'code_snapshot').glob('*.py')})
    (out/'run_config.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    inventory=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name!='output_inventory.csv':inventory.append(dict(path=path.relative_to(out).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    pd.DataFrame(inventory).to_csv(out/'output_inventory.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(files=len(inventory),figures=len(plot_index),total_bytes=sum(r['bytes'] for r in inventory))))


if __name__=='__main__':main()
