# DOVE 三 checkpoint 测试与 TAB-01 计算

## 改动与原因

按用户要求，恢复 run12 的 11000、12000、13000 三个 checkpoint 及原训练配置，复制原 logs50 中每个 checkpoint 的 100 回合测试数据。按用户确认，允许使用同次训练的三个 checkpoint；没有将它们宣称为三个独立训练种子。

`bash scripts/run_pvp.sh test` 默认并行真实测试三个 checkpoint，各 100 回合，全部成功后自动执行 `calculate.py`。终端仅显示逐回合 reward/cost/arrive/成功率及最终汇总，不新增 console/debug 日志或 TensorBoard 输出。模型与测量的必要对应关系保存在结果清单中。

先算三个 checkpoint 各自的回合均值，再算三个均值的平均值与样本 SD。按用户确认，默认对 reward 使用已证实的固定历史系数 0.9880830229334311，同时自动输出原始口径对照表；没有拟合目标值或更改原始 episode 数据。成本和成功率不变，三个指标的 SD 算法一致。

从原训练 TensorBoard 提取 74 个完成回合，重算训练接管百分数 23.1821886206067%。保留原论文单模型 369.23/15.33 的原始与历史处理后数据及独立计算入口。

## 验证

- `bash -n scripts/run_pvp.sh`：通过。
- 修改的 Python 文件 `py_compile`：通过。
- `python scripts/test_dove_table01.py`：4 项通过，涵盖历史数值、缺失/重复模型、损坏/不完整 CSV、错误模型名和非有限指标。
- `bash scripts/run_pvp.sh test`：在 NVIDIA GPU 0 运行 3×100 回合，300 回合完成；本次与原 logs50 的三组 reward/cost/success 一致。
- 真实结果目录：`02_data/raw_data/TABLES/TAB-01/raw_data/DOVE/evaluations/20261110_101213_263214`。
- `bash scripts/run_pvp.sh test --num-episodes 2`：在加入自动计算和精简输出后，真实运行 3×2 回合，通过；终端最后自动显示计算汇总，无 pkl/map/debug 信息。
- `PYTHON_BIN=python bash scripts/run_pvp.sh test --replay`：无需仿真即可验证旧数据入口自动计算，通过。
- 已恢复完整 3×100 实测为当前输入，并重新计算主表，避免将 6 回合接口验证当成最终结果。
- 全部复制来源 SHA-256 验证通过；其他方法的 reported 行逐字段保持不变。
- 已确认新测试结果目录没有 `.log` 文件。

## 当前结果

原始三模型 Return：374.0711019159418 ± 5.763716781474342。

默认报告口径：Return 369.61 ± 5.70；Test Cost 0.03 ± 0.03；Success 0.95 ± 0.04；Train Cost 23.18%。

`table_values.csv` 是默认报告表，`table_values_raw.csv` 是原始口径表，各自附有逐模型结果和审计 JSON。

## 主要文件

- `scripts/run_pvp.sh`
- `scripts/eval_dove_table01.py`
- `scripts/dove_table01.py`
- `scripts/extract_dove_training_evidence.py`
- `scripts/test_dove_table01.py`
- `scripts/eval_pvp_policies_fixed_v3.py`（只增加条件式逐回合进度输出和关闭 TensorBoard 的开关，不改推理/仿真算法）
- `02_data/models/final/simulation/DOVE/`（三个原始 pkl、args、模型清单、说明）
- `02_data/raw_data/TABLES/TAB-01/calculate.py`
- `02_data/raw_data/TABLES/TAB-01/raw_data/DOVE/`（三模型原始测量、单模型历史证据、训练提取、来源清单、详细说明）
- `02_data/raw_data/TABLES/TAB-01/table_values*.csv`、对应审计 JSON。

## 未完全核实的内容

论文 human_data_steps=4200 尚未找到独立原始计数来源，仍明确标记为报告元数据。online_env_steps=50000 是配置预算；该训练最后的 checkpoint 为 46656，已记录区别。训练成本来自同一个共享训练 run，没有独立训练种子的 SD。默认历史 reward 换算只说明 DOVE 的历史口径，不证明其他对比方法采用相同换算。

原备份目录只读，未更改。当前工作目录不在 Git 仓库中，未提交或推送。
