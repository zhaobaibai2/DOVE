# DOVEER_REAL09 2026-06-17 第二轮实车训练与测试汇报结果

分析目录：

```text
/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508/src/test/2
```

参考格式：

```text
/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508/src/test/1/DOVE_REAL09_260617_ANALYSIS_CN.md
```

本报告按第一轮 `DOVE_REAL09_260617_ANALYSIS_CN.md` 的汇报风格补写，重点不是重新提出方案，而是把第二轮训练和测试的结果说清楚。

本报告只基于该目录下已有数据做离线分析，核心依据是：

```text
config.json
train_logs/train_260617-204943.csv
events.out.tfevents.1781700583.miivii-tegra.22909.0
models/dacer_torch_*.pt
buffers/*.pkl
DOVEER_REAL07_R1_EVAL_FULL_260617-222229/train_logs/train_260617-222229.csv
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/config_eval.json
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/eval_logs/eval_260617-222423.csv
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/events.out.tfevents.1781706263.miivii-tegra.206750.0
analysis_outputs/second_run_summary.json
analysis_outputs/tensorboard_summary.json
```

## 1. 总体结论

这次第二轮 run 比第一轮更完整，已经不是单纯的小规模链路调试。它包含一段完整的 Route1 在线训练主 run，以及训练后冻结 checkpoint 的 Route1 seen-check eval。

它能说明：

```text
1. 训练链路完整跑通：Data Collection -> Offline Pretraining -> PVP Learning 都有记录。
2. 主训练 CSV 有 30475 行，持续约 88.38 分钟。
3. checkpoint 从 500 step 保存到 30000 step，共 60 个模型文件。
4. buffer 持续增长，最终 human_buffer_size=19511.0，pvp_buffer_size=29581.0。
5. 第二轮的 gate/filter 不再像第一轮那样几乎全 fallback。
6. 训练 Phase3 有效诊断行 gate_accepted=74.6%。
7. 完整 eval 有效诊断行 gate_accepted=98.5%。
8. eval CSV、TensorBoard event、模型路径、车辆误差、动作、实时性字段都有保存。
```

它暂时不能说明：

```text
1. 不能说明已经满足 10 Hz 实时闭环。
2. 不能说明 Route1 已经稳定自主完成。
3. 不能说明 Route2 泛化已经有效。
4. 不能直接给出 success rate，因为 route_start_count、route_end_count、route_completion_rate 都没有有效触发。
5. 不能直接作为论文主表的实车闭环结果，因为 eval 时 deadline_miss 和横向误差仍然偏高。
```

这次最重要的三个结论是：

```text
1. Gate 机制明显改善：Phase3 诊断行 accepted 约 74.6%，eval 诊断行 accepted 约 98.5%。
2. 实时性仍是最大瓶颈：Phase3 deadline_miss=75.4%，eval deadline_miss=82.3%。
3. eval 横向控制还不稳：eval |error_distance| p95=1.632 m，>1m 占 23.6%，>2m 占 2.2%。
```

因此这次结果应该定位为：

```text
Route1 第二轮完整训练 + Route1 seen-check 冻结评估的工程诊断结果，而不是正式论文成功率结果。
```

## 2. 文件与保存情况

目录下主要文件：

```text
config.json                                                        4630 bytes
train_logs/train_260617-204943.csv                                 24971275 bytes
events.out.tfevents.1781700583.miivii-tegra.22909.0                9685687 bytes
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/eval_logs/eval_260617-222423.csv   24828365 bytes
DOVEER_REAL07_R1_EVAL_FULL_260617-222419/events.out.tfevents.1781706263.miivii-tegra.206750.0   6898573 bytes
```

模型每 500 iter 保存一次，范围是：

```text
model count = 60
first checkpoint = dacer_torch_00000500.pt
last checkpoint  = dacer_torch_00030000.pt
last checkpoint size = 9.46 MB
```

buffer 文件情况：

```text
buffer/segment/demo pkl count = 63
min buffer file size = 1.37 MB
max buffer file size = 82.18 MB
```

同时已经补充生成分析产物：

```text
analysis_outputs/second_run_summary.json
analysis_outputs/train_phase_summary.csv
analysis_outputs/eval_summary.csv
analysis_outputs/tensorboard_summary.json
analysis_outputs/train_runtime_vehicle_loss.png
analysis_outputs/eval_runtime_vehicle_gate.png
analysis_outputs/train_phase_comparison.png
```

## 3. 配置含义

主训练配置关键项：

```json
"experiment_name": "DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH",
"config_path": "/home/nvidia/AutoDrive/src/car_dacer_torch/car_dacer_torch/configs/09_route1_train_doveer_full_from_scratch.yaml",
"phase1_episodes": 7500,
"phase2_updates": 60000,
"offline_bc_updates": 60000,
"offline_pvp_updates": 60000,
"phase3_online_train_enabled": true,
"phase3_train_in_background": true,
"phase3_train_period_s": 0.2,
"phase3_skip_train_on_deadline_miss": true,
"actor_sync_interval": 10,
"runtime_tb_interval": 10
```

算法配置：

```json
"num_timesteps": 10,
"use_ddim": true,
"ddim_steps": 5,
"use_guidance": true,
"guidance_scale": 0.04,
"final_gate_enabled": true,
"final_gate_max_shift": 0.45,
"final_gate_min_q_improve": 0.0,
"final_gate_max_uncertainty": 0.5,
"final_gate_use_uncertainty": true
```

完整 eval 配置关键项：

```json
"experiment_name": "DOVEER_REAL07_R1_EVAL_FULL",
"config_path": "/home/nvidia/AutoDrive/src/car_dacer_torch/car_dacer_torch/configs/07_route1_eval_doveer_full.yaml",
"method_name": "DOVE_full",
"trial_id": "R1_DOVE_001",
"route_id": "Route1",
"scenario_id": "route1_seen_check",
"phase3_online_train_enabled": false,
"phase3_train_in_background": false,
"eval.model_path": "auto",
"eval.model_dir": "/home/nvidia/Autodrive/results/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH_260617-204943/models/dacer_torch_00029000.pt"
```

注意：`config_eval.json` 中 `eval.model_dir` 记录为 `29000.pt`，但 eval CSV 的 `model_path` 每行记录为：

```text
/home/nvidia/Autodrive/results/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH_260617-204943/models/dacer_torch_00030000.pt
```

因此本报告按运行时 CSV 记录，认为完整 eval 实际评估的是 `dacer_torch_00030000.pt`。这个字段歧义后续需要修掉。

## 4. 实际阶段进展

CSV 统计：

```text
主训练总行数: 30475
iter 范围: 1 -> 30475
总时长: 约 88.38 分钟
平均运行频率: 约 5.75 Hz
```

阶段分布：

```text
Data Collection:      7514 rows, iter 1 -> 7514, duration 13.65 min
Offline Pretraining:  599 rows, iter 7515 -> 8113, duration 16.17 min
PVP Learning:         22362 rows, iter 8114 -> 30475, duration 58.55 min
```

这和第一轮不同：第一轮是加载已有 buffer 后做小规模 Phase3；第二轮是真正包含 Phase1 数据采集的大 run。

最终 buffer 和样本：

```text
human_buffer_size final = 19511.0
pvp_buffer_size final   = 29581.0
total_samples final     = 49092.0
total_interventions final = 39022.0
```

## 5. car_run_mode 与 intervention 标签问题

第一轮最大问题之一是 `car_run_mode` 全程为 0，导致 intervention 几乎全为 1。第二轮这里有改善：

```text
Data Collection car_run_mode=1 rate = 0.0%
PVP Learning car_run_mode=1 rate    = 45.0%
Full eval car_run_mode=1 rate       = 82.2%
PVP Learning intervention rate      = 55.0%
Full eval running_takeover_rate last = 20.15%
```

这说明第二轮不是全程人工模式，自动模式数据比例明显增加。

但仍然不能完全放松，因为：

```text
1. PVP Learning 中 intervention 仍有约 55%。
2. eval 中没有直接 intervention 字段，只能看 running_takeover_rate 和 car_run_mode。
3. 如果 car_run_mode 的语义在车端还有映射问题，DOVE pair 标签会受影响。
```

建议下一次正式跑之前仍做 20-30 秒静态/低速确认：手动模式、自动模式、接管、释放各记录一小段 CSV，明确 car_run_mode 与 intervention 的对应关系。

## 6. 实时性分析

### 6.1 Data Collection

```text
process_time mean = 103.0 ms
process_time p95  = 157.6 ms
infer_time p95    = 152.2 ms
loop_period p95   = 161.5 ms
sensor_age p95    = 274.0 ms
deadline_miss     = 42.5%
```

即使在 Data Collection，deadline miss 也有约 42.5%，说明实时性不是只由训练更新造成，推理路径本身也偏重。

### 6.2 Offline Pretraining

```text
process_time mean = 1408.0 ms
process_time p95  = 3194.0 ms
train_time mean   = 2810.2 ms
train_time p95    = 3348.6 ms
loop_period p95   = 3195.6 ms
sensor_age p95    = 3236.0 ms
deadline_miss     = 50.1%
```

Phase2/Offline Pretraining 仍然非常重，`train_time p95` 超过 3 秒。正式实车闭环前不建议在车端实时节点里做大规模 Phase2 更新。

### 6.3 PVP Learning

```text
process_time mean = 136.7 ms
process_time p50  = 123.0 ms
process_time p95  = 231.7 ms
process_time p99  = 318.9 ms
infer_time mean   = 131.9 ms
infer_time p95    = 224.2 ms
loop_period mean  = 142.9 ms
loop_period p95   = 238.2 ms
sensor_age p95    = 355.7 ms
deadline_miss     = 75.4%
```

结论很明确：

```text
当前配置仍不满足 10 Hz 实时闭环。
```

虽然 deadline miss 从第一轮 Phase3 的约 90.3% 降到第二轮约 75.4%，但 `infer_time p95` 基本仍在 224 ms 左右，没有真正解决推理耗时问题。

### 6.4 完整 eval

```text
rows              = 2629
duration          = 7.45 min
process_time mean = 156.4 ms
process_time p95  = 289.7 ms
infer_time p95    = 286.3 ms
loop_period p95   = 306.0 ms
sensor_age p95    = 398.3 ms
deadline_miss     = 82.3%
```

eval 阶段比训练 Phase3 还慢一些，`process p95` 约 290 ms，`deadline_miss` 约 82.3%。这说明冻结评估也没有达到实时闭环要求。

## 7. 推理延迟为什么高

当前配置仍是 full DOVE 路径：

```text
use_ddim = True
ddim_steps = 5
use_guidance = True
final_gate_enabled = True
num_timesteps = 10
num_samples = 100
```

这意味着一个控制 tick 内不是简单 actor forward，而是 diffusion/DDIM 推理、guided proposal、critic/gate 诊断等组合路径。

第二轮相比第一轮的关键事实是：

```text
第一轮 Phase3 infer_time p95 ≈ 224.3 ms
第二轮 Phase3 infer_time p95 = 224.2 ms
完整 eval infer_time p95 = 286.3 ms
```

所以第二轮的 gate 行为改善了，但推理延迟没有本质改善。下一步优化重点应该是推理路径，而不是继续先调训练损失。

## 8. Gate / Filter 行为分析

这里必须注意统计口径：CSV 里 gate 诊断不是每行都记录，只有 `gate_diagnostics_logged=1` 的行才适合计算 accepted/proxy/uncertainty/shift 的真实诊断率。

训练 Phase3 gate 诊断：

```text
diag rows = 1104 / 22362 (4.9%)
gate_accepted       = 74.6%
gate_fallback       = 25.4%
gate_proxy_ok       = 78.7%
gate_uncertainty_ok = 95.5%
gate_shift_ok       = 99.8%
gate_action_shift_l2 mean/p95 = 0.081 / 0.198
gate_unc_guided mean/p95      = 0.139 / 0.486
```

完整 eval gate 诊断：

```text
diag rows = 131 / 2629 (5.0%)
gate_accepted       = 98.5%
gate_fallback       = 1.5%
gate_proxy_ok       = 98.5%
gate_uncertainty_ok = 100.0%
gate_shift_ok       = 100.0%
gate_action_shift_l2 mean/p95 = 0.044 / 0.052
gate_unc_guided mean/p95      = 0.041 / 0.167
```

这和第一轮形成明显对比。第一轮报告中的结论是 gate 基本全 fallback，主要被 uncertainty 拦住。第二轮中，Phase3 的 uncertainty_ok 已经约 95.5%，eval 中 uncertainty_ok 达到 100%。

因此第二轮不应该再把核心问题归因于 gate 太严格。当前核心问题已经转成：

```text
1. 推理/闭环延迟太高。
2. eval 横向误差仍偏大。
3. route completion 没有有效标注。
```

## 9. Guidance 诊断

从 gate 诊断看，guided proposal 与 unguided action 的偏移在第二轮 eval 中很小：

```text
eval gate_action_shift_l2 mean = 0.044
eval gate_action_shift_l2 p95  = 0.052
eval gate_proxy_gain mean      = 0.019
eval gate_proxy_gain p95       = 0.040
eval gate_unc_guided mean      = 0.041
eval gate_unc_guided p95       = 0.167
```

这说明 eval 中 final gate 大多认为 guided action 是可接受的。

但是 CSV 中 `guidance_gate`、`guidance_q_uncertainty`、`guidance_grad_norm` 仍然全为 0。这不一定说明 guidance 没有执行，也可能是日志字段没有填上；但如果要把这个作为论文机制证据，后续必须把 guided proposal 的 applied flag、grad norm、uncertainty 和 final selected action 明确记录。

## 10. 训练指标分析

Phase3 训练指标：

```text
bc_loss mean/p95        = 0.020 / 0.039
q1_loss mean/p95        = 1.346 / 1.720
q2_loss mean/p95        = 1.364 / 1.756
policy_loss mean/p95    = 0.222 / 0.342
rl_loss mean/p95        = -0.060 / -0.000
energy_loss mean/p95    = 0.364 / 0.435
energy_violation mean/p95 = 0.375 / 0.448
pair_sat_e mean/p95     = 0.703 / 0.816
iar mean/p95            = 0.284 / 0.438
```

解读：

```text
1. 训练没有出现 NaN/OOM/数值崩溃迹象。
2. bc_loss 维持在较低水平，说明行为克隆项没有坏掉。
3. pair_sat_e 均值约 0.70，比第一轮约 0.53 更好。
4. iar 均值约 0.28，比第一轮约 0.47 更低，是正向变化。
5. q1/q2 loss 仍在 1.x，critic 不是完全收敛状态，但链路可运行。
```

## 11. 车辆状态与控制质量

训练 Phase3 车辆统计：

```text
carspeed mean/p95/max = 3.169 / 5.083 / 5.972 m/s
|error_distance| mean/p95/max = 0.527 / 1.366 / 3.053 m
|error_distance| > 1m = 2877 ticks, 12.9%
|error_distance| > 2m = 172 ticks, 0.8%
abs yaw error mean/p95 = 0.027 / 0.213
```

完整 eval 车辆统计：

```text
carspeed mean/p95/max = 2.468 / 3.611 / 3.972 m/s
|error_distance| mean/p95/max = 0.649 / 1.632 / 2.423 m
|error_distance| > 1m = 621 ticks, 23.6%
|error_distance| > 2m = 57 ticks, 2.2%
running_avg_abs_lateral_error last = 0.649 m
running_avg_abs_yaw_error last = 0.081
running_avg_abs_jerk last = 1.656
```

这部分是第二轮仍然不能作为正式结果的主要原因：完整 eval 中 `|error_distance| > 1m` 占 23.6%，`>2m` 占 2.17%。如果要作为论文主结果，这个误差水平还需要降低，或者至少需要配合清晰的 route completion 和人工标注解释。

## 12. Safety override 分析

```text
训练 Phase3 safety_override = 0.0%
训练 Phase3 emergency_brakes = 0.0%
训练 Phase3 obstacle_brakes = 0.0%
训练 Phase3 steer_violations = 0.0%
eval safety_override = 0.0%
eval emergency_brakes = 0.0%
eval obstacle_brakes = 0.0%
eval steer_violations = 0.0%
```

这说明 safety manager 没有大量接管，也没有频繁触发 emergency brake。

但这不代表完全没有风险。完整 eval 中存在 57 行 `|error_distance| > 2m`，而 safety 没触发，说明当前 safety 条件对低速偏离并不敏感。结合配置：

```text
max_abs_lateral_error = 2.0
emergency_brake_threshold = 5.0
eval carspeed mean/p95 = 2.468 / 3.611 m/s
```

如果希望低速偏离也保护，需要单独设计低速 lateral deviation guard，不能只依赖当前 emergency brake 阈值。

## 13. 动作分布

训练 Phase3 动作统计：

```text
sent_throttle mean/p95/max = 13.89% / 27.00% / 100.0%
sent_brake mean/p95/max    = 0.181% / 0.000% / 33.00%
|sent_steer| p95/max       = 148.8 / 200.0 deg
|sent_steer| >= 190 deg   = 476 ticks, 2.1%
action_smooth_l1 mean/p95  = 0.049 / 0.109
```

完整 eval 动作统计：

```text
sent_throttle mean/p95/max = 19.29% / 27.00% / 32.00%
sent_brake mean/max        = 0.000% / 0.000%
|sent_steer| p95/max       = 187.5 / 200.0 deg
|sent_steer| >= 190 deg   = 122 ticks, 4.6%
```

eval 中油门较稳定，刹车始终为 0，但转向接近饱和仍有约 4.6%。这和横向误差偏大相互印证：策略在一些片段需要较大转向修正。

## 14. TensorBoard 内容

TensorBoard event 文件已经解析，摘要保存在：

```text
analysis_outputs/tensorboard_summary.json
```

主训练 TensorBoard 关键 tag 数量：

```text
train scalar tags = 104
eval scalar tags = 61
short_eval scalar tags = 62
```

TensorBoard 采样统计中的关键值：

```text
train Timing/deadline_miss mean = 0.780
train Timing/process_time_ms p95 = 294.1 ms
train Timing/infer_time_ms p95 = 220.0 ms
train Vehicle/intervention mean = 0.666
train gate/accepted mean = 0.558
eval Timing/deadline_miss mean = 0.823
eval Timing/process_time_ms p95 = 289.7 ms
eval Vehicle/error_distance p95 = 1.632 m
eval gate/accepted mean = 0.985
```

TensorBoard 与 CSV 趋势一致，但 TensorBoard 是采样记录，CSV 是全量记录。比例类结论以 CSV 为准，趋势图和 tag 完整性以 TensorBoard 为辅助。

## 15. 这次数据中哪些是好的

```text
1. 第二轮训练规模足够大，不再只是 6000 tick 小 run。
2. 从 Phase1 到 Phase3 的训练链路完整跑通。
3. checkpoint、buffer、train CSV、eval CSV、TensorBoard event 都正常保存。
4. car_run_mode 不再全程为 0，自动模式比例明显增加。
5. gate/filter 行为相比第一轮明显改善，accepted 不再为 0。
6. uncertainty_ok 在 Phase3 和 eval 中都大幅改善。
7. 训练指标没有崩，bc_loss、energy、pair_sat_e、iar 都能解释。
8. eval 是冻结模型测试，phase3_online_train_enabled=false，边界比训练 run 更清楚。
9. safety override 没有频繁触发，说明没有被安全层完全拦截。
10. CAN send time 仍然很低，不是主要瓶颈。
```

## 16. 这次数据中哪些是不好的

```text
1. 实时性仍不达标：Phase3 deadline_miss=75.4%，eval deadline_miss=82.3%。
2. eval process p95=289.7 ms，远高于 100 ms。
3. eval |error_distance| p95=1.632 m，横向误差仍偏大。
4. eval |error_distance| > 1m 占 23.6%，>2m 占 2.2%。
5. route_start_count、route_end_count、route_completion_rate 全是 0，无法自动给成功率。
6. eval 配置中 model_dir/model_path 存在 29000 与 30000 的歧义。
7. guidance_* 三个字段仍全为 0，机制日志还不够充分。
8. Phase2 在车端实时节点中仍会造成秒级阻塞。
9. eval 时间只有约 7.5 分钟，仍偏短。
```

## 17. 建议怎么修改

### 17.1 先修 eval 配置字段

当前 eval 配置容易误读：

```text
config eval.model_dir = /home/nvidia/Autodrive/results/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH_260617-204943/models/dacer_torch_00029000.pt
csv model_path        = /home/nvidia/Autodrive/results/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH_260617-204943/models/dacer_torch_00030000.pt
```

建议把 eval 配置改成明确字段：

```text
eval:
  model_path: /home/nvidia/Autodrive/results/.../models/dacer_torch_00030000.pt
  model_dir: null
```

### 17.2 降低推理延迟

下一轮优先做延迟消融，而不是直接长时间跑：

```text
A. 当前 full DOVE: use_guidance=true, final_gate_enabled=true, ddim_steps=5
B. 轻量 DDIM: ddim_steps=3
C. Unguided 延迟基线: use_guidance=false, final_gate_enabled=false
```

目标：

```text
infer_time p95 < 80 ms
process_time p95 < 100 ms
deadline_miss < 5%
```

### 17.3 补 route completion

现在 route completion 无法用，因为：

```text
train route_start/end/completion = 0 / 0 / 0.000
eval route_start/end/completion = 0 / 0 / 0.000
```

必须增加 route start/end 触发，或者在 eval 后人工标注每次是否完成。否则论文主表不能给 success rate。

### 17.4 保留 gate，但不要再盲目放宽

第二轮 gate 已经可用，不是第一瓶颈。下一轮不要再优先放宽 uncertainty；更应该验证接受的 guided action 是否真的改善误差，以及减少 guided 路径的耗时。

### 17.5 Phase2 尽量离线

Phase2 的 `train_time p95` 超过 3 秒。建议正式上车前先离线训练 warm checkpoint，上车后只跑轻量 Phase3 或直接 eval。

## 18. 推荐下一轮实车流程

```text
1. 静态/低速确认 car_run_mode 与 intervention 语义。
2. 修正 eval config 的 checkpoint 字段，确保日志和配置一致。
3. 跑 2 分钟 unguided 延迟基线。
4. 跑 2 分钟 ddim_steps=3 guided 轻量版本。
5. 跑 2 分钟当前 full DOVE，对照 infer/process/deadline_miss。
6. 如果 p95 仍大于 100 ms，先不要跑长 eval，继续优化推理路径。
7. 如果实时性达标，再做 Route1 seen-check 10-20 分钟冻结评估。
8. 补 route_start/end 或人工标注后，再统计 success rate。
9. Route1 稳定后再做 Route2 eval。
```

## 19. 这次报告可用于论文吗

可以用于论文的工程诊断和补充材料：

```text
1. 可以说明实车在线训练 pipeline 可以跑通。
2. 可以说明日志体系完整：runtime、vehicle、action、gate、training、TensorBoard 都有。
3. 可以说明第二轮 gate/filter 已经从第一轮的不可用状态改善。
4. 可以作为实车调试过程、系统瓶颈、工程限制的证据。
```

不建议用于论文主结果 claim：

```text
1. 不能 claim 10 Hz realtime。
2. 不能 claim Route1 autonomous success。
3. 不能 claim Route2 generalization。
4. 不能 claim safety 完整，因为低速大偏差没有触发 override。
5. 不能 claim guided filter 一定提升闭环性能，因为还缺明确消融和 route completion。
```

## 20. 最终一句话

第二轮的最大价值是把第一轮暴露出的 gate 不工作问题基本补上了，并且跑出了完整的 30000 step 训练和冻结 eval；但新的主要瓶颈变成了实时推理延迟、横向误差和 route completion 标注。

```text
一句话总结：第二轮已经从“链路调试”进步到“完整训练+测试诊断”，但还没有达到“论文主表实车结果”的标准。
```
