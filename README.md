# 哪些会议证据可以替换？

*Which Meeting Evidence Can Change? Exact Verification under a Context-Benefit Constraint*

这个仓库存放论文的方法实现和数值复现材料。设计会议的检索工具先对每段发言给出局部读数，再用片段内部的短对话补充评分。本文研究，当修订必须保住原有上下文收益，且总改动有预算时，哪些候选仍可能进入前三？算法计算每个候选的允许区间和实现一次替换所需的最小总移动，只向神经验证器查询仍能影响结果的片段。

## 一条命令复现三张主表

在仓库根目录使用 Python 3.10。默认复现使用归档数值和 CPU；首次安装依赖后即可离线运行。

~~~bash
python -m pip install -r requirements.txt
python -m experiments.reproduce
~~~

程序将新结果写入 `runs/`，并对照归档核验。它重算设计会议的八种策略、查询级配对、候选池扩张、七种工作负载的联合枚举和线性规划抽查，还从两种论坛候选池重新计算交换结果。计时表按保存的逐批记录重新汇总。若需在另一台 GPU 上实际重测耗时，见[可选实验](docs/experiments.md)。

| 论文结果 | 此处可核查的证据 |
|---|---|
| 表 I · 分数保护、移动预算、成员锁 | [设计会议完整对照](results/table_1_mechanisms.csv) · [逐查询配对汇总](results/paired_summary.json) · [违规名单的可实现性](results/violation_causes.json) |
| 表 II · 三种调度在不同批量下的服务时间 | [设计会议计时表](results/table_2_timing.csv) · [论坛端到端计时](results/forum_service_time.csv) |
| 表 III · 十二个论坛，标题与全文候选池 | [逐论坛召回和净增命中](results/table_3_forums.csv) · [候选池与重排的分项结果](results/body_rerun.json) |
| 图 1、图 2 | [方法框架](figures/framework.pdf) · [主机制结果](figures/evidence.pdf) · [绘图代码](experiments/figures.py) |

在 847 个设计查询中，固定相同预算并锁定相同成员时，两种策略都保留了 98 个由上下文带入名单的相关片段，均净增 17 个命中。加入分数保护后，至少有一个候选违反收益下界的查询从 150 个降到 0，所需评分从 1,690 次降到 1,247 次。两种策略总体 P@3 都为 0.3526，逐查询比较为 10 个改善、827 个持平、10 个退化。调用次数与实际服务时间分别计量。

## 找到对应实现

| 目录 | 内容 |
|---|---|
| [controller/](controller/) | 逐标签保护区间、中心分配、共同阈值、单交换索引和自适应访问 |
| [experiments/](experiments/) | 主实验回放、配对统计、LP 对照、可选端点重拟合、第一阶段检索与 GPU 计时 |
| [data/](data/) | 经整理的数值输入、查询划分、模型系数及逐请求计时记录 |
| [results/](results/) | 与论文对照的 CSV 和 JSON，以及一个真实会议案例的数值 |
| [provenance/](provenance/) | 原数据和模型版本、原始缓存指纹、[复现入口自动核对的输入 SHA-256](provenance/replay_inputs.sha256) |
| [docs/](docs/) | [方法、样本划分与审计口径](docs/method-and-audits.md) · [数据格式和实验命令](docs/experiments.md) |

默认控制器使用二元标签、90% 收益保护和最多一次换项。复现材料覆盖 2,761 个可枚举查询、924 次短列表 LP 检查，以及单独选取的 130 次跨工作负载 LP 检查。会议标注衡量证据相关性，论坛标注衡量已链接的重复问题。真实团队的人工复核时间尚未测量。
