"""Read-only model evidence aggregation; no fitting or heldout selection."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import stage151 as m
import audit_metrics_20261111 as metrics
g=m.g;OUT=m.HERE/'stage155'

def main():
    # Current request authorizes new research, not historical power-off rearming.
    g.BASE=OUT;rt=g.SafeRuntime();rt.path=OUT/'audit_status.json';rt.marker=g.GUARD/'workers/stage155_audit.json';rt.hold=g.GUARD/'cpu_holds/stage155_audit.json'
    try:
        rt.start();rt.phase('evidence_analysis');parent=g.read(m.HERE/'stage151/fit_seal.json');results={};rng=np.random.default_rng(155)
        for key in ('r8_v1','r8_v2','r8_UniC','r32_v1','r32_v2','r32_UniC'):
            p=m.HERE/'stage151'/f'predictions_{key}.npz';assert g.sha(p)==parent['predictions'][p.name]
            with np.load(p,allow_pickle=False) as d:
                results[key]=metrics.evaluate(d,m.ARMS,list(range(9,28)),m.ARMS[:8],('uni_mtl','task_only','centered_only','conditional_direction'),rng)
            rt.boundary(p)
        # Contrast auxiliary-supervision interaction on MAE; no claim of a causal factorial experiment.
        rows=[dict(cohort=k,**r) for k,v in results.items() for r in v['rows']]
        with (OUT/'ablation_metrics.csv').open('w',encoding='utf-8',newline='') as f:
            fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        g.save(OUT/'ablation_evaluation.json',dict(code=g.sha(__file__),parent=g.sha(m.HERE/'stage151/fit_seal.json'),results=results,new_fits=0,scope='retrospective paired contrasts; no coefficient promotion, 3000 source/seed draws'))
        text=['# Stage155：已有消融的集中判定','', '复用 Stage151 全部封存预测，不训练、不按外折选系数。下表正值为候选相对对照的 MAE 降幅；区间为探索性 95% 来源/种子重采样区间。','', '| 设置 | 加条件项：完整 vs 普通多任务 | 加普通辅助：完整 vs 仅条件项 | 完整 vs 仅主任务 |','|---|---|---|---|']
        for key,v in results.items():
            cells=[]
            for ref in ('uni_mtl','centered_only','task_only'):
                r=next(r for r in v['rows'] if r['method']=='conditional_direction' and r['reference']==ref and r['group']=='all');cells.append(f"{r['gain']:+.6f} [{r['low']:+.6f}, {r['high']:+.6f}]")
            text.append('| '+key+' | '+' | '.join(cells)+' |')
        text+=['','判定：不能把完整模型优于普通多任务的点估计，直接解释成每个组件均有独立可靠贡献。不同系数仅作敏感性分析，不按已查看的外折挑最佳系数。完整 C/N/A/overall 与非重叠 N 见 ablation_metrics.csv。','', '统计审计覆盖：1 聚合/子组方向分别报告；2 不作个体无害推断；3 匹配支持限制适用群体；4 对评分条件化不证明因果消混杂；5 冲突比例与支持量由结果 support 保留；6 使用配对同初始化对照；7 保留全部已完成臂与旧失败；8 不把相关性写成因果；9 bootstrap 只覆盖已训练预测条件下的来源/初始化不确定性；10 多比较区间未校正，不能当确认检验；11 历史外折重复用于探索，不能等同新独立复制。']
        (OUT/'ABLATION_REPORT_ZH.md').write_text('\n'.join(text),encoding='utf-8')
        rt.state.update(status='complete',active_training=False);rt.boundary(OUT/'ablation_evaluation.json')
        print('\n'.join(text[:12]))
    finally:rt.finish()

if __name__=='__main__':main()
