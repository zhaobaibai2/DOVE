# DOVE checkpoint 文件改名

按用户要求，将模型文件依次改名：

| 原文件 | 新文件 |
| --- | --- |
| policy-11000-21998.pkl | seed0-final.pkl |
| policy-12000-23998.pkl | seed1-final.pkl |
| policy-13000-25998.pkl | seed2-final.pkl |

模型内容 SHA-256 不变；原训练配置和推理参数不变。三个模型实际均为同一次 seed-0 训练的中间 checkpoint，seed1/seed2/final 只是文件标签。模型清单保留 original_filename、training_seed=0、checkpoint/sample/update 步数和来源，不声称是独立训练或最后保存的模型。

测试入口使用更新后的模型清单，评估器目录扫描支持新文件名。共享结果校验器和独立 calculate.py 兼容原名与新名；已有原始回合 CSV 和归档清单保持原样。默认报告表与原始表的数值不变，逐 checkpoint 汇总输出使用新标签。

验证：三个文件哈希与原清单逐项一致；实际文件选择和目录扫描通过；旧 1200 回合结果可按新标签校验；独立复制的 calculate.py 对旧名/新名 CSV 产生相同表格；5 项已有验证测试、Python/Shell 语法检查通过。未重跑 GPU 仿真，因模型载荷、仿真和推理参数均未改变。

修改文件：模型目录三个 pkl 的路径和 manifest.json/README.md，scripts/dove_table01.py，scripts/eval_pvp_policies_fixed_v3.py，TAB-01/calculate.py，DOVE 数据说明和重新生成的逐模型汇总标签。
