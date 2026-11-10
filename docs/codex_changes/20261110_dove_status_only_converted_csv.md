# 终端只显示到达/碰撞状态，逐回合 CSV 使用统一报告口径

按用户要求，常规终端去掉 reward/cost，显示 arrive、crash、out_of_route、timeout、other 和累计成功率，最终汇总同样只显示状态。未归类终止记录为 other，避免把未到达都当成碰撞。

新结果 CSV 中所有 reward（逐回合、逐模型、最终均值/SD）统一使用已确认的固定系数 0.9880830229334311；增加 reward_factor 与状态计数列。不再在公共 CSV 中混入原始 reward 汇总。原始测量仍保留供重算，未改写原数据或成功率。输出位置仍为 TAB-01/test_results.csv 与按时间保留的运行 CSV。

提速：快速门控分支不再提前计算未使用的随机键；det_action 不再重复转换观测；仅 TAB-01 入口通过 DOVE_CORE_METRICS_ONLY 关闭无关的逐步诊断和车辆平滑性指标采集。完整策略前向、20 步扩散、guidance/final-gate 判定、地图/交通/reset 配置和 400 回合默认值均未改变。其他评估使用者不启用该环境变量时仍保留完整统计路径。

验证：

- 新入口 3×4 回合 GPU 实测成功，自动计算和 CSV 保存正常。
- 与优化前同样 3×4 回合逐项比较 reward、cost、success、crash、out_of_route、timeout、length，完全一致。
- 确认该次实时 CSV 每回合 reward 恰好转换一次，终端文本不含 reward。
- 7 项回归测试通过，含独立计算/追加 CSV、逐条转换与终端文本验证；Python 语法检查通过。
- 验证后恢复原正式 1200 回合数据，重新导出统一口径 CSV 共 1204 行。逐模型均值与回合均值一致，最终均值与 1200 条已转换回报均值一致，终止状态计数合计 1200。
- 当前 arrive=1140、crash=30、out_of_route=0、timeout=0、other=30；成功率 95.0%±4.6%。完整指标仍在 CSV：reward=368.16±6.88，cost=0.03±0.03，train_cost=23.18%。
- 未重跑正式 1200 回合，不声称固定提速比例。

文件：scripts/eval_pvp_policies_fixed_v3.py、scripts/eval_dove_table01.py、scripts/dove_table01.py、scripts/test_dove_table01.py、TAB-01/calculate.py、当前结果 CSV 和 DOVE 说明。
