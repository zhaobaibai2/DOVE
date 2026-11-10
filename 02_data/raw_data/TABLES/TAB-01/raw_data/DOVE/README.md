# DOVE：当前测试与独立计算

运行项目根目录的命令：

```bash
bash scripts/run_pvp.sh test
```

默认每个模型 400 回合，共 1200 回合；六个仿真进程并发，每个模型两个环境，各 200 回合。测试全部完成后自动计算，终端只显示逐回合 reward、cost、arrive、成功率，最后显示汇总；不新增调试日志。

## 保留原环境配置

使用原训练配置 `args.json` 与已经确认的推理参数：地图 104、traffic density 0.06、horizon 1000、关闭人工接管及渲染、UPV-GDS + full Final Gate。增加并发只改变回合分工和结果合并；不设置新的交通随机种子。

不同环境使用完全相同的固定配置，因此确定性副本可能重复得到相同轨迹/结果。这里的 400 回合不能表述为 400 个不同交通场景或独立随机种子。三个 pkl 也来自同一次训练的不同 checkpoint。

参与测试的模型：

- `seed0-final.pkl`（原 `policy-11000-21998.pkl`）
- `seed1-final.pkl`（原 `policy-12000-23998.pkl`）
- `seed2-final.pkl`（原 `policy-13000-25998.pkl`）

它们原样复制自原实验根目录下的：

```text
logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0/
```

现位于 `02_data/models/final/simulation/DOVE/`，模型与配置的 SHA-256 保存在该目录的 `manifest.json`。测试运行必须加载这些模型；单独计算 CSV 不需要模型文件。

## 计算可独立使用

`TAB-01/calculate.py` 只用 Python 标准库，只读取同目录 `raw_data` 中的本地输入：

- `DOVE/episodes.csv`：当前三模型逐回合测量，含模型标识、回合编号、reward、cost、success。
- `DOVE/training_totals.csv`：已从原训练记录恢复的接管步数与已完成回合步数汇总，用于计算训练成本。
- 每种方法自己的 `reported.csv`：论文元数据和其他方法已有报告行。
- `DOVE/measurement.json`：可选的数据完整性、回合数和配置说明；不含运行时必须访问的外部文件。

把 `calculate.py` 和上述本地输入复制到一个全新目录也能计算。不需要 `scripts/`、模型 pkl、历史回合、备份目录、TensorBoard 或项目的其他 Python 模块。可单独运行：

```bash
python 02_data/raw_data/TABLES/TAB-01/calculate.py
```

测试的完整测量保存在 `evaluations/<时间>/`，默认计算直接使用当前 `episodes.csv`，不需要遍历或重新读取这些归档目录。

## 计算规则

每回合 reward 为仿真逐步 reward 的总和；cost 为逐步 cost 的总和；success 为最终是否到达目的地（0 或 1）。

先求每个 checkpoint 的回合均值 m1、m2、m3，再计算：

```text
mean = (m1 + m2 + m3) / 3
sd = sqrt(sum((mi - mean)^2) / (3 - 1))
```

这里的 SD 是三个 checkpoint 均值的样本标准差，不是三个独立训练种子的标准差，也不是回合级 SE。

按用户已确认的报告口径，对三个模型的 reward 统一使用固定系数 `0.9880830229334311`，相应 SD 同乘该系数。它来自此前已核实的历史换算，固定保存在 `calculate.py` 的 `REPORT_REWARD_FACTOR`，不再读取历史文件，也不根据目标值或本次结果重新拟合。cost 与 success 不变。原始回合 CSV 不变。

主表为 `table_values.csv`；原始回报对照为 `table_values_raw.csv`，各自附有逐 checkpoint 结果与审计 JSON。只计算原始口径可加 `--reward-scale none`；需要查看来源/计算细节可加 `--details`。

训练成本按本地训练汇总计算：

```text
100 * 10807.999978699256 / 46622 = 23.1821886206067 → 23.18%
```

这来自原 run12 的 74 个已完成训练回合，原始事件文件 SHA-256 为 `e9bf810668578fd96b745b32ccead2f7e2235810ea9e49cb8ec3e898ca72b4ea`。它是接管步数占比，不是测试碰撞成本。单个共享训练 run 的 train_cost_sd 留空。

论文 `human_data_steps=4200` 仍是报告元数据，原始计数尚未完全核实；`online_env_steps=50000` 是配置预算，该次训练最后保存的 checkpoint 为 46656。

## 当前完成结果

正式数据目录：`evaluations/20261110_103704_968804`。原配置 3×400 回合已完成，六环境并行：

| 模型 | arrive | success | 原始 reward 均值 | 报告 reward 均值 |
| --- | ---: | ---: | ---: | ---: |
| 11000 | 384/400 | 0.96 | 375.606008428 | 371.13 |
| 12000 | 360/400 | 0.90 | 364.638576599 | 360.29 |
| 13000 | 396/400 | 0.99 | 377.544950021 | 373.05 |

原始 reward 汇总为 `372.60 ± 6.96`；报告 reward 为 `368.16 ± 6.88`；cost 为 `0.03 ± 0.03`；success 为 `95.0% ± 4.6%`。每个环境的前 100 回合三项主要指标均与原配置旧测试逐项一致。

开发时尝试的独立交通种子分支已经完全撤回，12 回合小测试已标记排除，不进入当前结果。被新要求替代的中断任务也不参与汇总。

模型文件已按要求改名为 seed0-final.pkl、seed1-final.pkl、seed2-final.pkl。这些只是文件标签，三者实际训练种子均为 0；计算代码同时兼容改名前后的名称，已有原始测量不改写。

## 终端指标统一保存 CSV

每次 `bash scripts/run_pvp.sh test` 都会在 TAB-01 目录生成 `test_results_<运行时间>.csv`。它在测试过程中逐行保存终端的 reward、cost、arrive、累计 success；测试结束后自动追加三组汇总、原始 reward 汇总和最终汇总。`terminal_text` 列保存终端该行的原文，结构化列保存数值。最后一条记录类型为 `final_summary`。

每次完整运行成功后，同一文件也保存为 `TAB-01/test_results.csv`，便于直接查看最近的完成结果；有日期的文件保留每次结果。进程失败或中断时，带时间的文件保留已完成的回合记录，且不会覆盖上一次完成结果的 test_results.csv。

单独运行 calculate.py 也会生成 test_results.csv：从当前 episodes.csv 导出所有回合数据并附上汇总。因为它没有重新测试，不伪造当时的终端交错顺序，补导出的回合行 terminal_text 留空；本次实际打印的汇总行保存完整原文。

当前 test_results.csv 包含已完成的 1200 回合和 5 行汇总。改名后的完整入口另用 3×4 回合真实验证通过；该次小测试按时间单独保存，正式输入已恢复为 1200 回合。

## 当前终端与导出格式

逐回合终端仅显示 `arrive、crash、out_of_route、timeout、other、success`；最终一行另外显示论文四项指标：Train Safety Cost↓、Return↑、Test Cost↓、Succ.↑。`other` 保留未归入前四类的实际终止原因，不把它们误记为碰撞。

`test_results.csv` 和新生成的带时间戳 CSV 中，每条 episode 的 reward 都已乘以固定报告系数；模型汇总和最终汇总使用同一口径。CSV 记录 reward_factor。公共结果 CSV 不再混入原始回报汇总行；原始测量仍保存在 raw_data 中供独立重算，避免重复转换。

为减少耗时，TAB-01 入口开启仅采集核心指标的模式，不采集未使用的速度/加速度/jerk/推理时延/门控标量诊断；同时移除快速门控路径前不使用的重复随机键计算与观测转换。动作选择、扩散步数、门控阈值、地图和交通配置不变。3×4 回合 GPU 验证的 reward/cost/成功/碰撞/回合长度均与优化前逐项一致；不承诺未经测量的固定提速比例。

当前统一结果 CSV 有 1200 个回合行、3 个模型汇总行、1 个最终汇总行，共 1204 行。终端统计为 arrive=1140/1200、crash=30、out_of_route=0、timeout=0、other=30、success=95.0%±4.6%。

## 最终四项指标与保存位置

测试结束后的最后一行保留 arrive/crash 等状态计数，并追加 `Train Safety Cost↓=23.18% Return↑=368.16 ± 6.88 Test Cost↓=0.03 ± 0.03 Succ.↑=95.0% ± 4.6%`（数值以本次测量为准）。逐回合仍不显示 reward。

- `TAB-01/test_results.csv`：最近一次完成测试的全部回合和汇总，最后的 final_summary 行包含四项指标与完整终端文本。
- `TAB-01/test_results_<时间>.csv`：该次完整测试的保存副本。
- `TAB-01/final_metrics.csv`：仅一行最终四项指标，列名与论文相同，便于直接查看。
- `TAB-01/table_values.csv`：原有完整论文表格数值。

已将用户本次运行 `20261110_114711_946257` 的最终显示补齐，同步更新当前和带时间副本，保留全部 1200 条原始终端回合记录。没有重新运行仿真或改变数值。
