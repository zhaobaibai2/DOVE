# 三模型各 400 回合，保留原配置并增加环境并发

用户将 1000 回合改为每个模型 400 回合、共 1200 回合，并要求更多并发环境，且明确不得改变原交通配置。

## 实现

- `run_pvp.sh test` 默认每个模型 400 回合。
- 每个模型两个独立仿真进程，各 200 回合，共六环境并行；GPU 0。
- 支持 `--envs-per-model` 和 `--workers` 设置并发。分片只调整回合分工、结果编号与合并，不改变地图/reset/交通设置。
- 三个 checkpoint、原 args.json、地图 104、traffic density 0.06、完整推理参数均沿用已确认的原配置。
- 同配置环境的确定性副本可能重复得到相同轨迹，不宣称为 400 个不同场景或独立随机种子。
- 独立交通随机种子的尝试已完全撤回；开发中 12 回合小测试目录 `20261110_103354_202153` 已标记 excluded，不参与最终汇总。
- 1000 回合与最初三环境 400 回合任务已按用户新要求中止，未激活部分结果。
- 最终统计与默认报告仍同时保留原始回报对照；终端不显示 pkl/map/debug 信息，不新增调试日志。

## 检查

- GPU 检查：RTX 4070 SUPER，当前六环境运行利用率约 96%，显存约 4.7GB/12GB。
- CPU：i5-12400F，6 核 12 线程；内存 32GB。选择六进程而非继续扩大，避免额外进程导致内存不足。
- Python 语法检查通过。
- `python scripts/test_dove_table01.py`：5 项通过，包括分片无缺漏/重复编号、原始数值、错误/不完整结果拒绝。
- 正式运行：`bash scripts/run_pvp.sh test`。
- 正式目录：`02_data/raw_data/TABLES/TAB-01/raw_data/DOVE/evaluations/20261110_103704_968804`。

状态：1200 回合已全部完成。成功数分别 384/400、360/400、396/400；原始 reward=372.5965116825535，SD=6.95962798745231；报告 reward=368.16±6.88，cost=0.03±0.03，success=95.0%±4.6%。六个分片编号完整，且各分片前 100 回合的主要指标与原配置旧测量逐项一致。

自动计算末尾曾因历史目录被移除而报缺失参考文件；测量未丢失，无需重跑。已按用户“所有都独立，不要依赖”要求重写 calculate.py：仅使用 Python 标准库及自身 raw_data 内的当前 episodes.csv、本地训练汇总、报告元数据；报告系数固定在计算代码内，不再运行时读取历史文件、模型目录、备份路径或共享 Python 模块。测试完成后发布独立的 measurement.json，本目录内可选校验数据哈希。

已在全新临时目录中只复制 calculate.py 和本地必要 CSV/measurement.json，独立计算结果与主目录逐字节一致；移除可选 measurement.json 后仍可独立计算。未使用历史目录、脚本目录或模型文件。Python 语法、Shell 语法和 5 项数据验证测试通过。正式结果已激活。

## 文件

`scripts/run_pvp.sh`、`scripts/eval_dove_table01.py`、`scripts/eval_pvp_policies_fixed_v3.py`、`scripts/dove_table01.py`、`scripts/test_dove_table01.py`、`02_data/raw_data/TABLES/TAB-01/calculate.py`、DOVE 数据来源说明及本变更说明。
