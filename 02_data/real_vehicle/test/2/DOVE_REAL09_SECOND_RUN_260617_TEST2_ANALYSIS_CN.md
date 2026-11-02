# DOVEER_REAL09 第二轮实车训练与测试数据分析报告

分析目录：`/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508/src/test/2`

生成时间：`2026-06-18 10:05:34`

本报告只基于远程目录中的已有离线数据做分析，原始训练/测试文件未修改。主要依据包括：

```text
train_logs/train_260617-204943.csv
DOVEER_REAL07_R1_EVAL_FULL_260617-222229/train_logs/train_260617-222229.csv
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/eval_logs/eval_260617-222423.csv
config.json
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/config_eval.json
events.out.tfevents.*
models/dacer_torch_*.pt
buffers/*.pkl
```

## 1. 总体结论

这次第二轮数据比第一轮更完整：主训练从 Phase1 数据采集开始，经过 Phase2 离线预热，再进入 Phase3 PVP 在线学习，最终保存到 `30000` step；随后又做了一个很短的测试启动目录和一个完整的 Route1 eval 目录。

最重要的结论是：

```text
1. 训练链路明显更完整：30475 行主训练 CSV，模型从 500 step 保存到 30000 step，buffer 持续增长。
2. Gate/Final filter 机制比第一轮好很多：Phase3 有效诊断行 gate_accepted 约 74.6%，完整 eval 约 98.5%。
3. 但实时性仍未达标：训练 Phase3 deadline_miss 约 75.4%，完整 eval deadline_miss 约 82.3%。
4. 完整 eval 横向误差仍偏大：|error_distance| p95 约 1.63 m，2 m 以上有 57 行，占 2.17%。
5. route_start/end/completion 仍然没有形成有效闭环统计，因此不能据此 claim autonomous success 或 route completion。
```

所以这次数据可以作为第二轮实车训练与 Route1 seen-check 调试结果；相比第一轮，它更像一次完整训练+冻结测试，但还不应作为论文主表中的正式闭环成功率。

## 2. 文件与产物

主训练 CSV：`train_logs/train_260617-204943.csv`，共 `30475` 行，时间 `2026-06-17 20:49:46` -> `2026-06-17 22:18:09`，持续约 `88.38` 分钟。

短测试目录：`DOVEER_REAL07_R1_EVAL_FULL_260617-222229`，只有 `33` 行、约 `6.080` 秒，更像误启动或短启动检查，不应当作正式 eval。

完整 eval 目录：`DOVEER_REAL07_R1_EVAL_FULL_260617-222419`，共 `2629` 行，时间 `2026-06-17 22:24:36` -> `2026-06-17 22:32:03`，持续约 `7.5` 分钟。

本次额外生成的分析产物：

```text
analysis_outputs/second_run_summary.json
analysis_outputs/train_phase_summary.csv
analysis_outputs/eval_summary.csv
analysis_outputs/tensorboard_summary.json
analysis_outputs/train_runtime_vehicle_loss.png
analysis_outputs/eval_runtime_vehicle_gate.png
analysis_outputs/train_phase_comparison.png
```

模型 checkpoint 共 `60` 个，step 范围 `500` -> `30000`。buffer/segment/demo pkl 共 `63` 个。顶层归档文件有 `4` 个。

## 3. 配置边界

主训练配置：

```text
experiment_name = DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH
config_path     = /home/nvidia/AutoDrive/src/car_dacer_torch/car_dacer_torch/configs/09_route1_train_doveer_full_from_scratch.yaml
phase1_episodes = 7500
phase2_updates  = 60000
offline_bc_updates  = 60000
offline_pvp_updates = 60000
phase3_online_train_enabled = True
phase3_train_in_background  = True
actor_sync_interval = 10
runtime_tb_interval = 10
use_guidance        = True
final_gate_enabled  = True
final_gate_max_uncertainty = 0.5
ddim_steps          = 5
```

完整 eval 配置：

```text
experiment_name = DOVEER_REAL07_R1_EVAL_FULL
config_path     = /home/nvidia/AutoDrive/src/car_dacer_torch/car_dacer_torch/configs/07_route1_eval_doveer_full.yaml
method_name     = DOVE_full
trial_id        = R1_DOVE_001
route_id        = Route1
scenario_id     = route1_seen_check
deterministic   = True
phase3_online_train_enabled = False
phase3_train_in_background  = False
eval.model_path = auto
eval.model_dir  = /home/nvidia/Autodrive/results/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH_260617-204943/models/dacer_torch_00029000.pt
CSV model_path  = /home/nvidia/Autodrive/results/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH_260617-204943/models/dacer_torch_00030000.pt
```

这里有一个需要记录的歧义：`config_eval.json` 里的 `eval.model_dir` 字段显示成了一个 `29000.pt` 文件路径，但 eval CSV 每一行记录的实际 `model_path` 是 `30000.pt`。本报告以 eval CSV 的运行时记录为准，认为完整 eval 实际测试的是 `dacer_torch_00030000.pt`。建议后续把 eval 配置里的 `model_dir/model_path` 字段整理清楚，避免复现实验时误读。

## 4. 主训练阶段进展

| 阶段 | 行数 | iter | 时长 min | deadline_miss | process p95 ms | infer p95 ms | loop p95 ms | intervention | gate 接受率/诊断行 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Data Collection | 7514 | 1 -> 7514 | 13.65 | 42.5% | 157.6 | 152.2 | 161.5 | 99.8% | 0.5% |
| Offline Pretraining | 599 | 7515 -> 8113 | 16.17 | 50.1% | 3194.0 | 111.5 | 3195.6 | 100.0% | 无诊断 |
| PVP Learning | 22362 | 8114 -> 30475 | 58.55 | 75.4% | 231.7 | 224.2 | 238.2 | 55.0% | 74.6% |

阶段含义：

```text
Data Collection: 从空/近空 buffer 开始采集，持续 7514 行。
Offline Pretraining: 599 行，但耗时约 16.2 分钟，训练更新非常重。
PVP Learning: 22362 行，持续约 58.6 分钟，是这次最关键的在线训练阶段。
```

最终主训练行记录：`human_buffer_size=19511.0`，`pvp_buffer_size=29581.0`，`total_samples=49092.0`，`total_interventions=39022.0`。

## 5. 与第一轮训练对照

| 指标 | 第一轮 | 第二轮 |
| --- | --- | --- |
| PVP 行数 | 5993 | 22362 |
| PVP 时长 | 15.46 min | 58.55 min |
| PVP deadline_miss | 90.3% | 75.4% |
| PVP infer p95 | 224.3 ms | 224.2 ms |
| PVP process p95 | 231.8 ms | 231.7 ms |
| PVP abs横向误差 p95 | 1.234 m | 1.366 m |
| PVP car_run_mode=1 占比 | 0.0% | 45.0% |
| PVP intervention 占比 | 99.5% | 55.0% |
| Gate 诊断接受率 | 第一轮 CSV 无诊断行/报告为 0 | 74.6% |

对照结论：

```text
1. 第二轮训练规模明显更大：PVP 行数从约 5993 增加到 22362。
2. 第二轮从 Phase1 采集开始，不再是第一轮那种加载已有 buffer 后直接进入 Phase2/Phase3 的小 run。
3. 推理延迟 p95 几乎没改善，仍在 224 ms 左右。
4. deadline_miss 从第一轮 Phase3 约 90.3% 降到第二轮 Phase3 约 75.4%，但仍远高于 10 Hz 闭环要求。
5. Gate 行为显著改善：第二轮有效诊断行里 gate_accepted 已经不是 0。
6. 横向误差 p95 第二轮更高，这可能和更长、更完整的 Route1 采样/测试有关，也说明车辆控制质量还没有稳定到正式结果级别。
```

## 6. 实时性分析

### 6.1 训练 Phase3

```text
process_time mean = 136.7 ms
process_time p50  = 123.0 ms
process_time p95  = 231.7 ms
process_time p99  = 318.9 ms
infer_time p95    = 224.2 ms
loop_period p95   = 238.2 ms
sensor_age p95    = 355.7 ms
deadline_miss     = 75.4%
```

Phase3 仍不满足 10 Hz。目标应该是 `process_time p95 < 100 ms`，更理想是 `infer_time p95 < 80 ms`，而当前推理 p95 约 224 ms。

### 6.2 完整 eval

| 行数 | iter | 时长 | deadline_miss | process p95 | infer p95 | loop p95 | abs误差p95 | gate 接受率/诊断行 | safety override |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2629 | 122 -> 2750 | 7.5 min | 82.3% | 289.7 ms | 286.3 ms | 306.0 ms | 1.632 m | 98.5% | 0.0% |

完整 eval 更能代表冻结策略闭环测试，但实时性更差：`process_time p95` 约 290 ms，`loop_period p95` 约 306 ms，`deadline_miss` 约 82.3%。CAN 发送不是主瓶颈，`send_time p95` 只有约 2.18 ms。

### 6.3 Phase2 仍会阻塞

Offline Pretraining 阶段虽然只有 599 行，但持续约 16.17 分钟：

```text
train_time mean = 2810.2 ms
train_time p95  = 3348.6 ms
process p95     = 3194.0 ms
loop p95        = 3195.6 ms
```

这说明如果 Phase2 放在车端实时节点里跑，仍然会造成秒级阻塞。正式上车前最好把大规模 Phase2/BC/PVP 预热离线完成，上车只做轻量 Phase3 或 eval。

## 7. car_run_mode 与 intervention

第二轮和第一轮相比，`car_run_mode` 不再全程为 0：训练 Phase3 中 `car_run_mode=1` 占约 45.0%，完整 eval 中 `car_run_mode=1` 占约 82.2%。

但 intervention 语义仍需谨慎：

```text
训练 Phase3 intervention = 55.0%
训练最终 total_interventions / total_samples = 39022.0 / 49092.0
完整 eval running_takeover_rate 最后一行 = 20.15%
```

这里不能简单说策略已经低接管率，因为 eval 的 `running_takeover_rate` 是运行时统计字段，最后约 20.15%，但 CSV 里没有直接的人工接管事件列，只能结合 `car_run_mode`、human command 和车辆模式解释。下一次正式实验前仍建议做 20-30 秒静态/低速校准，确认 `car_run_mode=1` 是否确实代表自动模式。

## 8. Gate / Final Filter 行为

这次最明显的正向变化是 gate。CSV 中 gate 指标只在 `gate_diagnostics_logged=1` 的诊断行上有意义，非诊断行填 0，因此必须用诊断行条件统计。

训练 Phase3 诊断行统计：

```text
diag rows              = 1104 / 22362 (4.9%)
gate_accepted          = 74.6%
gate_fallback          = 25.4%
gate_proxy_ok          = 78.7%
gate_uncertainty_ok    = 95.5%
gate_shift_ok          = 99.8%
gate_action_shift_l2 p95 = 0.198
gate_unc_guided p95      = 0.486
```

完整 eval 诊断行统计：

```text
diag rows              = 131 / 2629 (5.0%)
gate_accepted          = 98.5%
gate_fallback          = 1.5%
gate_proxy_ok          = 98.5%
gate_uncertainty_ok    = 100.0%
gate_shift_ok          = 100.0%
gate_action_shift_l2 p95 = 0.052
gate_unc_guided p95      = 0.167
```

这说明第一轮的主要问题 `uncertainty_ok=0 / gate_accepted=0` 在第二轮已经明显缓解。特别是 eval 中 uncertainty 和 shift 几乎都通过，proxy 也基本通过。

需要注意的是，CSV 中 `guidance_gate/guidance_q_uncertainty/guidance_grad_norm` 仍然全为 0。结合 gate 诊断，当前可以确认 final gate 诊断链路在工作，但不能只靠 `guidance_*` 三个字段证明 guidance 梯度实际数值。后续如果要写论文或做机制消融，建议把 guided proposal 的 applied flag、grad norm、uncertainty 单独稳定记录到 eval CSV。

## 9. 训练指标

Phase3 训练指标：

```text
bc_loss mean/p95        = 0.020 / 0.039
q1_loss mean/p95        = 1.346 / 1.720
q2_loss mean/p95        = 1.364 / 1.756
policy_loss mean/p95    = 0.222 / 0.342
energy_loss mean/p95    = 0.364 / 0.435
energy_violation mean/p95 = 0.375 / 0.448
pair_sat_e mean/p95     = 0.703 / 0.816
iar mean/p95            = 0.284 / 0.438
```

相较第一轮，`pair_sat_e` 的均值从约 0.53 提高到约 0.70，`iar` 从约 0.47 降到约 0.28，这对 DOVE/PVP 排序信号是正向变化。`bc_loss` Phase3 均值约 0.020，说明行为克隆拟合没有崩；`q1/q2_loss` 维持在 1.3 左右，没有看到 NaN 或明显数值爆炸。

## 10. 车辆状态、误差与动作

训练 Phase3：

```text
carspeed mean/p95/max = 3.169 / 5.083 / 5.972 m/s
|error_distance| mean/p95/max = 0.527 / 1.366 / 3.053 m
|error_distance| > 1m = 2877 行 (12.9%)
|error_distance| > 2m = 172 行 (0.8%)
|sent_steer| >= 190 deg = 476 行 (2.1%)
```

完整 eval：

```text
carspeed mean/p95/max = 2.468 / 3.611 / 3.972 m/s
|error_distance| mean/p95/max = 0.649 / 1.632 / 2.423 m
|error_distance| > 1m = 621 行 (23.6%)
|error_distance| > 2m = 57 行 (2.2%)
|sent_steer| >= 190 deg = 122 行 (4.6%)
sent_throttle mean/p95 = 19.29% / 27.00%
sent_brake mean/max = 0.000% / 0.000%
```

完整 eval 中油门比较稳定，刹车输出为 0，但横向误差和转向饱和仍偏高。尤其 `|error_distance| > 1m` 占 23.6%，说明冻结策略还不能算稳定跟踪。

## 11. Safety 与 Route Completion

Safety 字段：

```text
训练 Phase3 safety_override = 0.0%
完整 eval safety_override   = 0.0%
完整 eval emergency_brakes  = 0.0%
完整 eval obstacle_brakes   = 0.0%
完整 eval steer_violations  = 0.0%
```

没有 safety override 不等于无风险。当前配置中 emergency brake 还受到速度阈值影响，而完整 eval 的速度均值约 2.47 m/s、p95 约 3.61 m/s。低速偏离 2 m 以上可能不会触发 emergency brake。

Route completion 字段：

```text
训练 route_start_count/end_count/completion = 0 / 0 / 0.000
eval route_start_count/end_count/completion = 0 / 0 / 0.000
eval is_at_start/is_at_end mean = 0.000 / 0.000
```

这些字段全是 0，所以当前日志不能自动给出 success rate 或 route completion。正式实验需要补齐 route start/end 事件或外部人工标注，否则只能做 latency、误差、接管和 gate 的诊断分析。

## 12. TensorBoard 核对

TensorBoard event 已解析为 `analysis_outputs/tensorboard_summary.json`。主训练 event 有 `104` 个 scalar tag，完整 eval event 有 `61` 个 scalar tag，短测试 event 有 `62` 个 scalar tag。

关键 TensorBoard 采样统计：

```text
train Timing/deadline_miss mean = 0.780
train Timing/process_time_ms p95 = 294.1 ms
train Timing/infer_time_ms p95   = 220.0 ms
train Vehicle/intervention mean  = 0.666
train gate/accepted mean         = 0.558
eval Timing/deadline_miss mean   = 0.823
eval Timing/process_time_ms p95  = 289.7 ms
eval Vehicle/error_distance p95  = 1.632 m
eval gate/accepted mean          = 0.985
```

TensorBoard 与 CSV 的方向一致。差异主要来自采样间隔：TensorBoard 不是每一行都记，CSV 是全量行，因此比例类指标应优先看 CSV，全趋势图可以看 TensorBoard。

## 13. 短测试目录说明

`DOVEER_REAL07_R1_EVAL_FULL_260617-222229` 只有 33 行、6.08 秒，并且保存的是 `train_logs/train_260617-222229.csv` 而不是 `eval_logs`。它的特征是：

```text
deadline_miss = 66.7%
intervention  = 100.0%
process p95   = 338.5 ms
infer p95     = 335.9 ms
|error| max   = 0.453 m
```

它可以作为启动检查记录，但样本太少，不能和完整 eval 混在一起汇总。

## 14. 当前问题优先级

```text
P0: 实时性仍未达标。完整 eval deadline_miss 82.3%，process p95 290 ms。
P0: route completion/start/end 没有有效记录，无法自动给 success rate。
P1: eval config 的 model_dir/model_path 字段存在 29000 与 30000 的歧义。
P1: eval 横向误差偏大，|error| > 1m 占 23.6%，>2m 占 2.17%。
P1: car_run_mode/intervention 语义仍建议做静态标定。
P2: guidance_* 三个字段为 0，机制论文证据还需要更明确的 guided action 日志。
P2: Phase2 在车端实时节点中仍是秒级阻塞，不适合正式闭环。
```

## 15. 建议下一步

建议按以下顺序处理：

```text
1. 先修 eval 配置字段：明确 model_path 指向哪个 checkpoint，不要让 model_dir 写成 .pt 文件。
2. 做 2 分钟延迟诊断：分别跑 unguided、ddim_steps=3、当前 guided，比较 infer/process p95。
3. 把正式 eval 的 route_start/end/completion 或外部人工标注补上，否则无法形成成功率。
4. 保留当前 gate 阈值或轻微收紧；第二轮 gate 已经能接受，不再是第一轮的主要瓶颈。
5. 训练/测试前再次确认 car_run_mode=1 与自动模式的对应关系。
6. 如果要正式 Route2，先在 Route1 seen-check 上把 deadline_miss 降到可接受范围，再冻结 checkpoint。
```

## 16. 一句话结论

第二轮相比第一轮已经把训练规模和 gate 机制跑起来了，尤其 gate acceptance 从几乎不可用变成诊断行高接受率；但系统瓶颈已经转移到实时性和闭环评测标注，当前完整 eval 仍不能作为成功率结果，只能作为 Route1 seen-check 的诊断数据。
