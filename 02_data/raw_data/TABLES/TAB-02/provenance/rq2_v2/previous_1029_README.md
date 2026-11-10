# 仿真图表：逐项一键运行

对应你指定的 **2026-10-29 PDF**：
`/home/aaa/data2/ZCY/paper1/修正/自己修正的/1029/DOVE__Learning_Diffusion_Policies_from_Accepted_and_Rejected_Actions_in_Human_in_the_Loop_Driving.pdf`。

在 `simulation` 下直接执行下面的命令即可。输入数据已随本目录整理好，常规运行不需要指定旧目录、不需要 GPU，也不需要重新训练。

| 论文项目 | 直接运行 | 主要输出目录 |
| --- | --- | --- |
| 图 1：方法框架 | `bash paper_repro/figure_01_framework.sh` | `paper_repro/outputs/figure_01/` |
| 图 2(a)：MetaDrive 平台 | `bash paper_repro/figure_02a_metadrive_setup.sh` | `paper_repro/outputs/figure_02a/` |
| 图 3(a)：奖励曲线 | `bash paper_repro/figure_03a_reward.sh` | `paper_repro/outputs/figure_03a/` |
| 图 3(b)：接管率曲线 | `bash paper_repro/figure_03b_takeover.sh` | `paper_repro/outputs/figure_03b/` |
| 图 3：同时生成两个子图 | `bash paper_repro/figure_03_training_dynamics.sh` | 上面两个目录 |
| 表 I：超参数 | `bash paper_repro/table_01_hyperparameters.sh` | `paper_repro/outputs/table_01/` |
| 表 II：闭环主实验 | `bash paper_repro/table_02_closed_loop.sh` | `paper_repro/outputs/table_02/` |
| 表 III：actor/critic 诊断 | `bash paper_repro/table_03_actor_critic.sh` | `paper_repro/outputs/table_03/` |
| 表 IV：7 种推理模式 | `bash paper_repro/table_04_inference_modes.sh` | `paper_repro/outputs/table_04/` |
| 上面全部项目 | `bash paper_repro/run_all.sh` | `paper_repro/outputs/` |

本次范围是 `simulation`；图 2(b)、图 4/5、表 V–VII 属于实车部分。

## 输出怎么看

- 图目录保存 PDF、PNG、绘图 CSV 和 `report.json`。两张曲线另有逐条矢量曲线核对结果 `vector_check.csv`。
- 表目录保存重新计算的 CSV、论文参考值、逐项 `comparison.csv` 和 `report.json`。论文参考值不作为计算结果。
- `outputs/all_summary.json` 是总入口的运行结果清单。
- `MATCH` 表示数值在论文显示精度下相同；`DIFFERENT` 表示不相同；`MISSING` 表示没有相应测量记录。

默认运行会生成所有结果并明确列出差异。如果要让缺口或数值差异触发非零退出码：

```bash
bash paper_repro/run_all.sh --strict
```

退出码：`0` 表示所选计算已完成；`1` 表示文件、数据校验或执行错误；严格模式下 `2` 表示已生成报告但存在论文证据缺口/差异。

## 每项数据怎样得到

**图 1、图 2(a)**：导出你提供的论文压缩包内原始 PDF，并生成 PNG 预览；PDF 按 SHA-256 校验。它们是原成图导出，未声称从原始素材重建。

**图 3(a)**：使用恢复的最终绘图聚合数据，绘制 Mean 和 Mean±Std，保持原版坐标、颜色和画布。五条中心线与指定论文 PDF 的矢量顶点核对。保留了两个历史 DOVE 最终输入文件，但完整原始 reward 处理公式未恢复，因此这一步是最终绘图数据复现。

**图 3(b)**：从三种方法各两次运行的六份历史输入重新计算。H-DSAC 百分数乘 0.01；数值裁剪到 [0,1]；按历史窗口平滑；线性插值到 0–50000、步长 500 的网格；按两次运行求均值及样本标准差（ddof=1）。聚合 CSV 与旧结果逐值核对，再核对论文 PDF 的三条中心线。

**表 I**：从提供的论文源码提取超参数表，输出配置 CSV、Markdown、TeX。配置表没有实验测量的均值计算。

**表 II**：保留论文全部 13 行参考值；重算找到的 DOVE 单检查点 100 个回合的原始和历史处理后 reward/cost/success，并记录历史 `reward_new=reward_old*375/max(reward_old)` 的逐回合误差。历史处理后六个显示值能匹配论文，但不能由此验证论文描述的三次独立训练各 20 回合统计。其他方法及 human-step/training-cost 的完整逐 run 数据不在本次恢复记录中。

**表 III**：按冻结的同一组 256 个 intervention pairs、50/50/46/46 个检查点计算。VLB 从保存的 `delta_vlb>0` 重新统计；CPCal-only 使用归档的每对 64 次采样计数；其余三组缺失的 HSR/RMR 已用原冻结模型在 GPU 上补算，并随包保存每对的计数及模型哈希。

HSR 统计采样动作位于 accepted-action 半空间的比例；RMR_0.5 统计采样动作到 rejected action 的距离严格小于正负动作距离一半的比例。critic 列使用同一 256 对和同一检查点集合的 twin-critic 均值次序。

旧 critic 原表使用 512 对及 52/52/48 个检查点；这套旧统计另外输出到 `legacy_critic_denominator_audit.csv`，不混入当前 256 对计算。当前重新计算的 16 个单元格有 **10 个匹配、6 个不匹配**，见 `comparison.csv`。

**表 IV**：逐种推理模式读取五个检查点、各 100 回合，校验唯一 episode ID，合并 500 回合求 success/crash/route_completion/reward 均值。**28 个显示值全部匹配**。但现存记录只包含地图 104，保存的训练配置也使用地图 104，因此报告标为 `NUMBERS_MATCH_PROTOCOL_GAP`，不将其认作已验证的 disjoint held-out 评估。

## 环境、输出位置

常规绘图/计算需要 Python 3、NumPy、Matplotlib、PyMuPDF。依赖见本目录 `requirements.txt`。脚本使用当前环境的 `python3`，也可以显式指定：

```bash
PYTHON_BIN=/path/to/python bash paper_repro/run_all.sh
bash paper_repro/run_all.sh --output-dir /path/to/output
```

支持从任意工作目录运行脚本；默认输出始终在本目录 `outputs/`。路径含空格时加引号。

## 表 III：需要重新从模型采样时

正常出表直接使用已经补算并随包保存的计数。下面入口用于再次执行冻结模型推理，要求原项目中的 demo 和全部所选 checkpoint 仍在，以及 JAX/CUDA 环境可用：

```bash
PYTHON_BIN=/home/aaa/miniconda3/envs/dacer_pvp/bin/python \
bash paper_repro/table_03_resample_models.sh \
  --source-root ../.. \
  --output-dir paper_repro/outputs/resampled_actor

bash paper_repro/table_03_actor_critic.sh \
  --actor-counts-dir paper_repro/outputs/resampled_actor
```

以上相对路径以 `simulation` 为当前目录。默认 GPU 0，可以设置 `CUDA_VISIBLE_DEVICES`。模型逐份核对冻结审计记录中的 SHA-256。该入口不更新模型。`--limit-checkpoints 1 --groups positive_only` 可作单模型运行检查；不完整结果不能用于整行表格。

## 来源与验证

`inputs/sources.csv` 记录每个输入的原路径、原文件哈希、本地文件哈希、所选字段和行数。CSV 字段投影保留原数值文本，未冒充完整原日志。新补算数据标为 `new_frozen_model_inference_not_training`。`inputs/paper_identity.json` 记录指定 PDF/ZIP 的哈希。

每次运行先核对全部输入哈希，再计算或绘图。原有训练和实验脚本未被覆盖。本次不修改论文、不替换历史数据、不提交或推送 Git。

验证命令：

```bash
cd paper_repro
python3 -B -m unittest discover -s tests -v
bash run_all.sh
```
