# 最终终端行显示论文四项指标

用户要求说明保存位置，并让测试最后一行同时显示 Train Safety Cost↓、Return↑、Test Cost↓、Succ.↑。

已修改独立 calculate.py：保留逐回合只显示状态的行为，最终行保留 arrive/crash 等计数并追加四项指标。既有 test_results.csv 的 final_summary 行继续保存对应结构化数值和完整终端文本；增加 final_metrics.csv，仅一行四项论文展示值。sh 自动调用计算的路径不变。

使用用户刚完成的运行 20261110_114711_946257，未重跑仿真。保留并逐字典校验全部 1200 条已有 episode 记录/终端文本，仅刷新三组及最终汇总；同步更新 test_results.csv 和该次带时间 CSV。当前四项是 23.18%、368.16±6.88、0.03±0.03、95.0%±4.6%。统计公式、转换系数和原测量不变。

验证：7 项测试通过，新增断言 final_metrics.csv 的四个列名/展示值与最终终端行逐项一致；Python 语法检查通过；正式 CSV 仍为 1204 行（1200 回合+4 汇总），全部 episode 行保持原样。GPU 未使用。

修改文件：TAB-01/calculate.py、scripts/test_dove_table01.py、DOVE README、本次说明，以及重新计算的结果/汇总 CSV。
