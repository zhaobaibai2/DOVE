# 将测试终端指标和最终汇总统一保存 CSV

用户要求改名后的 sh 入口可靠运行，并将终端显示的所有指标保存到 TAB-01 目录。

## 改动

- 默认真实测试仍为 seed0-final.pkl、seed1-final.pkl、seed2-final.pkl，各 400 回合，六环境，沿用原配置。
- 每次运行在 TAB-01 根目录创建 test_results_<运行时间>.csv，逐回合写入 reward、cost、arrive、累计成功率和终端行原文；组内回合完成编号与实际 episode_id 分开记录。
- 并发写入与终端打印使用相同锁，避免 CSV 行交错；每行写入后关闭文件，部分运行的已完成记录仍保留。
- 结束时 calculate.py 向同一 CSV 追加三组均值、原始回报汇总和最终指标，所有 terminal_text 与实际输出对应。
- 计算成功后原子更新 test_results.csv；保留带时间的运行文件。计算失败不会发布为最近完成结果。
- 独立 calculate.py 默认也导出 test_results.csv，其中包含当前所有测量回合及汇总，无项目模块或历史文件依赖。补导出的回合行没有伪造历史终端原文。
- 未改变模型载荷、原环境配置、推理设置或计算口径。

## 验证

- 使用 GPU 0 运行 `bash scripts/run_pvp.sh test --num-episodes 4`，新名字的三个模型与六环境真实运行成功，自动调用计算成功。
- 验证 CSV 共 17 行：12 个回合、3 个模型汇总、1 个原始回报汇总、1 个最终汇总；逐行终端文本的 reward、arrive、success 与数值列一致，组内完成计数为 1..4，各 episode_id 无缺失。
- 检查原模型 SHA-256、改名后选择与原配置保持不变。
- 验证后恢复正式的 1200 回合为当前输入，并独立重算。当前 test_results.csv 共 1205 行，最后一行为 final_summary，成功总数为 1140。
- 正式结果仍为 reward=368.16±6.88，cost=0.03±0.03，success=95.0%±4.6%，train_cost=23.18%。
- 回归测试新增独立 CSV 完整导出、终端逐行对应及追加汇总不重复回合两项；共 7 项通过。
- Python 和 Shell 语法检查通过。

## 主要文件

scripts/eval_dove_table01.py、scripts/dove_table01.py、scripts/test_dove_table01.py、TAB-01/calculate.py、DOVE 数据说明、TAB-01/test_results.csv 及带运行时间的 CSV。

当前 test_results.csv 使用保留的正式 1200 回合，不将 12 回合接口验证作为正式评估结果。没有重跑全部 1200 回合；改名后已完成三模型小回合真实端到端验证。未来实际运行仍可能受系统资源、驱动或外部文件变更影响，不声称排除一切运行时故障。
