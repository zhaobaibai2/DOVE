import datetime
import csv
import os
import sys
import time
import json
import copy
import pickle
import yaml
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from car_interfaces.msg import *
import can
import torch
try:
    from pynput import keyboard
except Exception:
    keyboard = None

from .torch_networks import DACERTorchAgent, DACERActionConfig
from .torch_algorithm import PVPDACERTorch, TorchDACERConfig
from .torch_replay_buffer import Experience, TorchPVPBuffer, PhaseManager
from .safety_manager import SafetyManager


OBS_DIM = 344
PI = 3.1415926535

ros_node_name = "car_dacer_torch"

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getcwd() + "/src/utils/")
sys.path.append(os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name))
sys.path.append(os.getcwd() + "/src/%s/%s/utils"%(ros_node_name, ros_node_name))
sys.path.append(os.path.join(_PKG_DIR, "utils"))

try:
    import tjitools
except Exception:
    class _TjiToolsFallback:
        @staticmethod
        def ros_log(name, msg):
            print(f"[{name}] {msg}")

    tjitools = _TjiToolsFallback()


class CarDACERTorch(Node):
    def __init__(self):
        super().__init__(ros_node_name)

        self.config = self._load_config()

        self.subSurrounding = self.create_subscription(
            SurroundingInfoInterface, "surrounding_info_data", self.sub_callback_surrounding, 1
        )

        self.pubCarDACER = self.create_publisher(
            CarRLInterface, "car_dacer_data", 10
        )
        self.timerCarDACER = self.create_timer(0.1, self.pub_callback_car_dacer_torch)

        self.dry_run = bool(self.config.get('hardware', {}).get('dry_run', False))
        self.carControlBus = None
        if not self.dry_run:
            self.carControlBus = can.interface.Bus(
                channel=self.config['hardware']['can_interface'],
                bustype='socketcan'
            )

        self.iteration = 0
        self.rcvMsgSurroundingInfo = None
        self.takeover_recorder = deque(maxlen=2000)
        self._prev_intervention = 0.0

        training_cfg = self.config.get('training', {})
        self.use_one_step_delay = bool(training_cfg.get('use_one_step_delay', False))
        self.phase1_collect_only_intervention = bool(training_cfg.get('phase1_collect_only_intervention', False))
        self.collect_only = bool(training_cfg.get('collect_only', False))
        self.phase3_human_mix_ratio = float(training_cfg.get('phase3_human_mix_ratio', 0.5))
        self.phase2_bc_updates_per_iter = int(training_cfg.get('phase2_bc_updates_per_iter', 1))
        self.log_detail_interval = int(training_cfg.get('log_detail_interval', 10))  # 详细日志打印间隔，防止刷屏
        self.phase3_online_train_enabled = bool(training_cfg.get('phase3_online_train_enabled', True))
        self.phase3_train_in_background = bool(training_cfg.get('phase3_train_in_background', True))
        self.phase3_train_period_s = float(training_cfg.get('phase3_train_period_s', 0.2))
        self.phase3_skip_train_on_deadline_miss = bool(training_cfg.get('phase3_skip_train_on_deadline_miss', True))
        self.phase3_max_infer_time_ms = float(training_cfg.get('phase3_max_infer_time_ms', 80.0))
        self.actor_sync_interval = max(1, int(training_cfg.get('actor_sync_interval', 10)))
        self.runtime_tb_interval = max(1, int(training_cfg.get('runtime_tb_interval', 10)))
        self.runtime_csv_interval = max(1, int(self.config.get('logging', {}).get('runtime_csv_interval', 1)))
        self.runtime_csv_flush_interval = max(1, int(self.config.get('logging', {}).get('runtime_csv_flush_interval', 20)))
        self.action_path = str(training_cfg.get('action_path', 'fast')).lower()
        self.gate_diagnostics_interval = max(1, int(training_cfg.get('gate_diagnostics_interval', 20)))
        self.console_status_interval = max(1, int(training_cfg.get('console_status_interval', 20)))

        self._prev_state = None
        self._prev_action_novice = None
        self._prev_action_behavior = None
        self._prev_intervention_for_exp = None
        self._prev_stop_td = 0.0
        self._collect_only_threshold_saved = False

        # 运行指标缓存：用于 TensorBoard/终端统计（不改变训练逻辑，只做记录）
        self.start_wall_time = time.time()
        self.last_timestamp = 0.0
        self.last_process_time = 0.0
        self.last_loop_period = 0.0
        self.last_infer_time = 0.0
        self.last_train_time = 0.0
        self.last_actor_sync_time = 0.0
        self.last_send_time = 0.0
        self.last_deadline_miss = 0.0
        self.last_sensor_age = 0.0
        self.last_intervention = 0.0
        self.last_state_error_yaw = 0.0
        self.last_state_error_distance = 0.0
        self.last_state_carspeed = 0.0
        self.last_is_at_start = 0.0
        self.last_is_at_end = 0.0

        self.last_action_human = np.zeros(2, dtype=np.float32)
        self.last_action_model = np.zeros(2, dtype=np.float32)
        self.last_action_sent = np.zeros(2, dtype=np.float32)
        self.last_sent_throttle = 0
        self.last_sent_brake = 0
        self.last_sent_steer = 0.0
        self.last_safety_override = 0
        self.last_can_bytes = [0] * 8
        self.last_guidance_info = {}

        self.prev_action_sent_for_smooth = None
        self.last_action_smooth_l1 = 0.0
        self.prev_carspeed_for_jerk = None
        self.prev_accel_for_jerk = None
        self.running_metric_count = 0
        self.sum_carspeed = 0.0
        self.sum_abs_error_distance = 0.0
        self.sum_abs_error_yaw = 0.0
        self.sum_action_smooth_l1 = 0.0
        self.sum_abs_jerk = 0.0
        self.last_speed_accel = 0.0
        self.last_speed_jerk = 0.0
        self.last_min_surrounding_distance = 0.0
        self.prev_is_at_start = 0.0
        self.prev_is_at_end = 0.0
        self.route_start_count = 0
        self.route_end_count = 0

        # 分段数据缓存：用于按 P 暂停时立即落盘当前段
        self.segment_human_buffer = []
        self.segment_pvp_buffer = []
        self.segment_idx = 0

        self.buffer_lock = threading.RLock()
        self.actor_lock = threading.RLock()
        self.learner_lock = threading.RLock()
        self.metrics_lock = threading.RLock()
        self._trainer_stop = threading.Event()
        self._trainer_updates_since_sync = 0
        self._trainer_thread = None

        # 按键 P 暂停/恢复数据采集（参考 car_rl）
        self.is_paused = False
        self.listener = None
        if bool(self.config.get("hardware", {}).get("enable_keyboard_pause", True)) and keyboard is not None:
            try:
                self.listener = keyboard.Listener(on_press=self.on_press)
                self.listener.start()
                tjitools.ros_log(self.get_name(), "Keyboard listener started. Press 'P' to Pause/Resume.")
            except Exception as e:
                self.get_logger().warning(f"Keyboard listener disabled: {e}")

        self._init_logging()
        self._init_safety()
        self._init_algo_and_buffer()
        self._init_async_training()

        tjitools.ros_log(self.get_name(), f"Start Node: {self.get_name()}")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        cwd_default = os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name) + "config.yaml"
        pkg_default = os.path.join(_PKG_DIR, "config.yaml")
        default_config_path = cwd_default if os.path.exists(cwd_default) else pkg_default
        config_path = os.environ.get("DACER_CONFIG_PATH", default_config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["config_path"] = config_path
        logging_cfg = cfg.get("logging", {}) or {}
        base_save = logging_cfg.get("result_root", "./outputs")
        experiment_name = logging_cfg.get("experiment_name", "DACER_TORCH_CAR")
        save_folder = os.path.join(
            base_save,
            f"{experiment_name}_" + datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        cfg["save_folder"] = save_folder
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "buffers"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "train_logs"), exist_ok=True)

        with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        return cfg

    def _init_logging(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.config["save_folder"],
                flush_secs=self.config.get('logging', {}).get('tensorboard_flush_secs', 20),
            )
            self.use_tensorboard = True
        except Exception:
            self.writer = None
            self.use_tensorboard = False

        self.training_metrics = {}
        self._init_train_csv()

    def _init_train_csv(self):
        ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        self._csv_path = os.path.join(self.config['save_folder'], 'train_logs', f"train_{ts}.csv")
        self._csv_f = open(self._csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_header())
        self._csv_writer.writeheader()
        self._csv_f.flush()

    def _csv_header(self):
        return [
            'ts',
            'iter',
            'phase',
            'phase_name',
            'process_time',
            'infer_time',
            'train_time',
            'actor_sync_time',
            'send_time',
            'loop_period',
            'sensor_age',
            'deadline_miss',
            'gearpos',
            'car_run_mode',
            'intervention',
            'human_buffer_size',
            'pvp_buffer_size',
            'total_samples',
            'total_interventions',
            'error_yaw',
            'error_distance',
            'carspeed',
            'is_at_start',
            'is_at_end',
            'route_start_count',
            'route_end_count',
            'route_completion_rate',
            'min_surrounding_distance',
            'speed_accel',
            'speed_jerk',
            'running_avg_speed',
            'running_avg_abs_lateral_error',
            'running_avg_abs_yaw_error',
            'running_avg_abs_jerk',
            'human_action_0',
            'human_action_1',
            'model_action_0',
            'model_action_1',
            'sent_action_0',
            'sent_action_1',
            'sent_throttle_percentage',
            'sent_braking_percentage',
            'sent_steering_angle',
            'action_smooth_l1',
            'safety_override',
            'safety_emergency_brakes',
            'safety_obstacle_brakes',
            'safety_steer_violations',
            'gate_accepted',
            'gate_fallback',
            'gate_action_shift_l2',
            'gate_proxy_gain',
            'gate_delta_c',
            'gate_proxy_ok',
            'gate_uncertainty_ok',
            'gate_shift_ok',
            'gate_unc_guided',
            'gate_diagnostics_logged',
            'guidance_gate',
            'guidance_q_uncertainty',
            'guidance_grad_norm',
            'bc_loss',
            'q1_loss',
            'q2_loss',
            'policy_loss',
            'rl_loss',
            'energy_loss',
            'energy_violation',
            'pair_sat_e',
            'iar',
            'delta_e',
            'action_gap_l2',
            'mean_q1_std',
            'mean_q2_std',
            'can_bytes',
        ]

    def _init_safety(self):
        self.safety_manager = SafetyManager(self.config.get('safety', {}))

    def _init_algo_and_buffer(self):
        hw = self.config.get('hardware', {})
        use_gpu = bool(hw.get('use_gpu', True))
        cuda_mem_fraction = float(hw.get('cuda_mem_fraction', 0.7))

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            try:
                torch.cuda.set_per_process_memory_fraction(cuda_mem_fraction, device=self.device)
            except Exception:
                pass
        else:
            self.device = torch.device('cpu')

        net_cfg = self.config.get('network', {})
        alg_cfg = self.config.get('algorithm', {})

        action_cfg = DACERActionConfig(
            init_alpha=float(alg_cfg.get('init_alpha', 0.1)),
            action_noise_scale=float(alg_cfg.get('action_noise_scale', 0.05)),
            use_ddim=bool(alg_cfg.get('use_ddim', True)),
            ddim_steps=int(alg_cfg.get('ddim_steps', max(1, min(int(alg_cfg.get('num_timesteps', 20)), 5)))),
            ddim_eta=float(alg_cfg.get('ddim_eta', 0.0)),
            use_guidance=bool(alg_cfg.get('use_guidance', True)),
            guidance_scale=float(alg_cfg.get('guidance_scale', 0.04)),
            guidance_uncertainty_kappa=float(alg_cfg.get('guidance_uncertainty_kappa', 1.0)),
            guidance_min_gate=float(alg_cfg.get('guidance_min_gate', 0.05)),
            guidance_sigma_ref=float(alg_cfg.get('guidance_sigma_ref', 1.0)),
            guidance_grad_clip=float(alg_cfg.get('guidance_grad_clip', 1.0)),
            final_gate_enabled=bool(alg_cfg.get('final_gate_enabled', True)),
            final_gate_max_shift=float(alg_cfg.get('final_gate_max_shift', 0.45)),
            final_gate_min_q_improve=float(alg_cfg.get('final_gate_min_q_improve', 0.0)),
            final_gate_max_uncertainty=float(alg_cfg.get('final_gate_max_uncertainty', 0.50)),
            final_gate_use_uncertainty=bool(alg_cfg.get('final_gate_use_uncertainty', True)),
        )

        self.agent = DACERTorchAgent(
            obs_dim=self.config['env']['state_dim'],
            act_dim=self.config['env']['action_dim'],
            hidden_dims=tuple(net_cfg.get('hidden_dims', [256, 256, 256])),
            diffusion_hidden_dims=tuple(net_cfg.get('diffusion_hidden_dims', [256, 256, 256])),
            num_timesteps=int(alg_cfg.get('num_timesteps', 20)),
            target_entropy=float(alg_cfg.get('target_entropy', -2.0)),
            time_dim=int(net_cfg.get('time_dim', 16)),
            activation=str(net_cfg.get('activation', 'relu')),
            use_layer_norm=bool(net_cfg.get('use_layer_norm', True)),
            action_cfg=action_cfg,
            device=self.device,
        ).to(self.device)

        cfg = TorchDACERConfig(
            gamma=float(alg_cfg.get('gamma', 0.99)),
            tau=float(alg_cfg.get('tau', 0.005)),
            lr=float(alg_cfg.get('lr', 1e-4)),
            alpha_lr=float(alg_cfg.get('alpha_lr', 3e-2)),
            delay_update=int(alg_cfg.get('delay_update', 1)),
            delay_alpha_update=int(alg_cfg.get('delay_alpha_update', 1000)),
            reward_scale=float(alg_cfg.get('reward_scale', 1.0)),
            lambda_pv=float(alg_cfg.get('lambda_pv', 1.0)),
            B=float(alg_cfg.get('B', 1.0)),
            lambda_bc=float(alg_cfg.get('lambda_bc', 5.0)),
            reward_free=bool(alg_cfg.get('reward_free', True)),
            phase3_use_bc_boost=bool(alg_cfg.get('phase3_use_bc_boost', True)),
            fix_alpha=bool(alg_cfg.get('fix_alpha', True)),
            target_entropy=float(alg_cfg.get('target_entropy', -2.0)),
            lambda_energy=float(alg_cfg.get('lambda_energy', 0.5)),
            energy_margin=float(alg_cfg.get('energy_margin', 0.5)),
            energy_min_action_gap=float(alg_cfg.get('energy_min_action_gap', 0.03)),
            rejected_action_radius=float(alg_cfg.get('rejected_action_radius', 0.20)),
            critic_objective=str(alg_cfg.get('critic_objective', 'cost')),
        )

        self.algorithm = PVPDACERTorch(self.agent, cfg, device=self.device)
        self.actor_agent = copy.deepcopy(self.agent).to(self.device)
        self.actor_agent.eval()

        train_cfg = self.config.get('training', {})
        buffer_max_size = int(train_cfg.get('buffer_max_size', train_cfg['replay_batch_size'] * 100))
        human_buffer_max_size = int(train_cfg.get('human_buffer_max_size', max(1, buffer_max_size // 2)))
        self.buffer = TorchPVPBuffer(
            max_size=buffer_max_size,
            human_max_size=human_buffer_max_size,
            obs_dim=self.config['env']['state_dim'],
            act_dim=self.config['env']['action_dim'],
        )

        self.phase_manager = PhaseManager()
        self.phase_manager.phase1_threshold = int(self.config['training']['phase1_episodes'])
        self.phase_manager.phase2_threshold = int(self.config['training']['phase2_updates'])
        
        # 加载已有buffer数据（如果配置了路径）
        self._load_existing_buffer()

    def _init_async_training(self):
        if self.phase3_train_in_background:
            self._trainer_thread = threading.Thread(
                target=self._phase3_train_loop,
                name="dacer_phase3_trainer",
                daemon=True,
            )
            self._trainer_thread.start()
            self.get_logger().info("[AsyncTrain] Phase3 background trainer started")

    def _load_existing_buffer(self):
        """加载已有buffer数据并直接进入Phase2"""
        train_cfg = self.config.get('training', {}) or {}
        demo_data_path = os.environ.get('DACER_BUFFER_PATH') or train_cfg.get('demo_data_path')
        demo_data_dir = os.environ.get('DACER_BUFFER_DIR') or train_cfg.get('demo_data_dir')
        auto_buffer_requested = isinstance(demo_data_dir, str) and demo_data_dir.lower() == 'auto'
        if isinstance(demo_data_dir, str) and demo_data_dir.lower() == 'auto':
            demo_data_dir = self._auto_find_latest_buffer_dir()
        
        loaded_any = False
        
        # 方案1：加载单个文件
        if demo_data_path and os.path.exists(demo_data_path):
            try:
                self.get_logger().info(f"[加载Buffer] 正在加载单个文件: {demo_data_path}")
                self.buffer.load(demo_data_path)
                loaded_any = True
            except Exception as e:
                self.get_logger().error(f"[加载Buffer] 单文件加载失败: {e}")
        
        # 方案2：批量加载目录下的所有buffer文件
        elif demo_data_dir and os.path.exists(demo_data_dir):
            try:
                self.get_logger().info(f"[加载Buffer] 正在批量加载目录: {demo_data_dir}")
                self._load_multiple_buffers(demo_data_dir)
                loaded_any = True
            except Exception as e:
                self.get_logger().error(f"[加载Buffer] 批量加载失败: {e}")
        
        elif auto_buffer_requested:
            self.get_logger().warning("[加载Buffer] demo_data_dir=auto 但未找到 Route1 buffers，将从 Phase1 重新采集")
        elif demo_data_path or demo_data_dir:
            self.get_logger().warning(
                f"[加载Buffer] 指定的 buffer 路径不存在，跳过加载: path={demo_data_path}, dir={demo_data_dir}"
            )
        else:
            self.get_logger().info("[加载Buffer] 未指定 demo_data_path/demo_data_dir/DACER_BUFFER_*，从 Phase1 开始")
        
        # 如果成功加载了数据，设置为Phase2
        if loaded_any:
            self.phase_manager.current_phase = 2
            self.phase_manager.phase1_episodes = self.phase_manager.phase1_threshold  # 标记Phase1已完成
            
            buffer_stats = self.buffer.get_statistics()
            self.get_logger().info(
                f"[加载Buffer] 加载完成 - Human: {buffer_stats['human_size']} "
                f"PVP: {buffer_stats['pvp_size']} | 直接进入Phase2离线更新"
            )
            
            tjitools.ros_log(self.get_name(), "Loaded existing buffer(s) and entered Phase2")
        else:
            self.get_logger().info("[初始化] 未加载任何buffer数据，从Phase1开始")

    def _load_multiple_buffers(self, directory: str):
        """批量加载目录下的所有buffer文件"""
        import glob
        
        # 搜索所有buffer_*.pkl文件
        pattern = os.path.join(directory, "buffer_*.pkl")
        buffer_files = glob.glob(pattern)
        
        # 也搜索segment文件（如果有的话）
        segment_pattern = os.path.join(directory, "segment_*.pkl")
        segment_files = glob.glob(segment_pattern)
        
        all_files = buffer_files + segment_files
        
        if not all_files:
            self.get_logger().warning(f"[批量加载] 目录中未找到buffer文件: {directory}")
            return
        
        # 按文件名排序（通常包含时间戳）
        all_files.sort()
        
        self.get_logger().info(f"[批量加载] 找到 {len(all_files)} 个文件")
        
        loaded_count = 0
        total_human = 0
        total_pvp = 0
        
        for file_path in all_files:
            try:
                self.get_logger().info(f"[批量加载] 正在加载: {os.path.basename(file_path)}")

                base = os.path.basename(file_path)
                if base.startswith('segment_'):
                    with open(file_path, 'rb') as f:
                        payload = pickle.load(f)
                    human_list = payload.get('human', [])
                    pvp_list = payload.get('pvp', [])

                    for exp in list(human_list):
                        self.buffer.add_human(exp)
                        total_human += 1
                    for exp in list(pvp_list):
                        self.buffer.add_pvp(exp)
                        total_pvp += 1

                    loaded_count += 1
                    self.get_logger().info(
                        f"[批量加载] 完成: {base} (Human: {len(human_list)}, PVP: {len(pvp_list)})"
                    )
                else:
                    # 创建临时buffer来加载单个文件
                    temp_buffer = TorchPVPBuffer(
                        max_size=self.buffer.max_size,
                        human_max_size=self.buffer.human_max_size,
                        obs_dim=self.buffer.obs_dim,
                        act_dim=self.buffer.act_dim,
                    )
                    temp_buffer.load(file_path)

                    # 合并到主buffer
                    for exp in list(temp_buffer.human_buffer):
                        self.buffer.add_human(exp)
                        total_human += 1

                    for exp in list(temp_buffer.pvp_buffer):
                        self.buffer.add_pvp(exp)
                        total_pvp += 1

                    loaded_count += 1
                    self.get_logger().info(
                        f"[批量加载] 完成: {base} (Human: {len(temp_buffer.human_buffer)}, PVP: {len(temp_buffer.pvp_buffer)})"
                    )
                
            except Exception as e:
                self.get_logger().error(f"[批量加载] 文件加载失败 {file_path}: {e}")
                continue
        
        self.get_logger().info(f"[批量加载] 总结: 加载了 {loaded_count}/{len(all_files)} 个文件 "
                             f"(总Human: {total_human}, 总PVP: {total_pvp})")

    def _auto_find_latest_buffer(self) -> str:
        """自动搜索最新的buffer文件"""
        import glob
        
        # 搜索常见的buffer目录
        search_paths = [
            "./outputs/DACER_TORCH_CAR_*/buffers/",
            "./outputs/*/buffers/",
            "./buffers/",
            "../buffers/",
        ]
        
        for search_path in search_paths:
            expanded_paths = glob.glob(search_path)
            for path in expanded_paths:
                if os.path.exists(path):
                    buffer_files = glob.glob(os.path.join(path, "buffer_*.pkl"))
                    if buffer_files:
                        # 按修改时间排序，选择最新的
                        latest_file = max(buffer_files, key=os.path.getmtime)
                        return latest_file
        
        return None

    def _auto_find_latest_buffer_dir(self) -> str:
        """自动搜索最近一次 Route1 采集输出的 buffers 目录。"""
        import glob

        search_paths = [
            "./outputs/DOVEER_REAL01_R1_HIL_COLLECT*/buffers",
            "./outputs/*R1_HIL_COLLECT*/buffers",
            "./outputs/*/buffers",
        ]

        candidates = []
        for pattern in search_paths:
            for path in glob.glob(pattern):
                if not os.path.isdir(path):
                    continue
                files = []
                files.extend(glob.glob(os.path.join(path, "buffer_*.pkl")))
                files.extend(glob.glob(os.path.join(path, "segment_*.pkl")))
                if files:
                    candidates.append((max(os.path.getmtime(f) for f in files), path))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    def on_press(self, key):
        """监听按键：P 键切换暂停状态。"""
        try:
            if key.char == 'p' or key.char == 'P':
                self.is_paused = not self.is_paused
                status = "PAUSED" if self.is_paused else "RESUMED"
                tjitools.ros_log(self.get_name(), f"System {status}")

                # 当切换到暂停时，立即把当前段数据保存下来
                if self.is_paused:
                    self._save_segment_data()
        except AttributeError:
            pass

    def _log_step_detail(self, intervention: float, action_behavior: np.ndarray, action_novice: np.ndarray):
        """打印采样输入/模型输出/人工动作，便于对齐数据。"""
        msg = self.rcvMsgSurroundingInfo
        throttle_h, brake_h, steer_h = (
            msg.throttle_percentage,
            msg.braking_percentage,
            msg.steerangle,
        )

        th_pred, br_pred, steer_pred = self.process_action(action_novice)
        state_preview = {
            "iter": self.iteration,
            "yaw_err": float(msg.error_yaw),
            "dist_err": float(msg.error_distance),
            "speed": float(msg.carspeed),
            "run_mode": int(msg.car_run_mode),
        }

        self.get_logger().info(
            "[采样明细] {iter} | run_mode:{run_mode} inter:{inter:.1f}\n"
            "  state(yaw:{yaw_err:.2f}, dist:{dist_err:.2f}, v:{speed:.2f})\n"
            "  Human(th:{th_h:.0f}, br:{br_h:.0f}, steer:{st_h:.1f}) | Model(th:{th_p:.0f}, br:{br_p:.0f}, steer:{st_p:.1f})".format(
                inter=intervention,
                th_h=throttle_h,
                br_h=brake_h,
                st_h=steer_h,
                th_p=th_pred,
                br_p=br_pred,
                st_p=steer_pred,
                **state_preview,
            )
        )

    def _log_console_status(self):
        with self.buffer_lock:
            buffer_stats = self.buffer.get_statistics()
        phase_info = self.phase_manager.get_phase_info()
        with self.metrics_lock:
            tm = dict(self.training_metrics)
        gi = self.last_guidance_info or {}

        takeover_rate = float(np.mean(np.array(self.takeover_recorder)) * 100.0) if len(self.takeover_recorder) > 0 else 0.0
        human_th = self._safe_msg_get('throttle_percentage', 0.0)
        human_br = self._safe_msg_get('braking_percentage', 0.0)
        human_st = self._safe_msg_get('steerangle', 0.0)
        model_th, model_br, model_st = self.process_action(self.last_action_model)
        critic_loss = 0.5 * (float(tm.get('q1_loss', 0.0)) + float(tm.get('q2_loss', 0.0)))
        mgn = float(gi.get('guidance/grad_norm', gi.get('gate/action_shift_l2', 0.0)))

        self.get_logger().info(
            "[状态] iter={it} phase={ph}/{phn} p2={p2}/{p2t} p3={p3} "
            "buf(H/P)={hb}/{pb} takeover={tk:.1f}% inter={inter:.0f} "
            "v={v:.2f} ey={ey:.2f} epsi={epsi:.3f} "
            "Human(th/br/st)={hth:.0f}/{hbr:.0f}/{hst:.1f} "
            "Model(th/br/st)={mth:.0f}/{mbr:.0f}/{mst:.1f} "
            "Sent(th/br/st)={sth:.0f}/{sbr:.0f}/{sst:.1f} "
            "loss(actor/critic/rl/bc/energy)={pl:.4f}/{cl:.4f}/{rl:.4f}/{bc:.4f}/{el:.4f} "
            "dE={de:.4f} pair={pair:.3f} IAR={iar:.3f} MGN={mgn:.4f} "
            "gate(acc/fb/shift/proxy/unc)={gacc:.2f}/{gfb:.2f}/{gsh:.3f}/{gpg:.4f}/{gunc:.4f} "
            "time(infer/proc/train)={infer:.1f}/{proc:.1f}/{train:.1f}ms"
            .format(
                it=int(self.iteration),
                ph=phase_info.get('current_phase', ''),
                phn=phase_info.get('phase_name', ''),
                p2=int(phase_info.get('phase2_updates', 0)),
                p2t=int(phase_info.get('phase2_threshold', 0)),
                p3=int(phase_info.get('phase3_iterations', 0)),
                hb=int(buffer_stats.get('human_size', 0)),
                pb=int(buffer_stats.get('pvp_size', 0)),
                tk=takeover_rate,
                inter=float(self.last_intervention),
                v=float(self.last_state_carspeed),
                ey=float(self.last_state_error_distance),
                epsi=float(self.last_state_error_yaw),
                hth=float(human_th or 0.0),
                hbr=float(human_br or 0.0),
                hst=float(human_st or 0.0),
                mth=float(model_th),
                mbr=float(model_br),
                mst=float(model_st),
                sth=float(self.last_sent_throttle),
                sbr=float(self.last_sent_brake),
                sst=float(self.last_sent_steer),
                pl=float(tm.get('policy_loss', 0.0)),
                cl=critic_loss,
                rl=float(tm.get('rl_loss', 0.0)),
                bc=float(tm.get('bc_loss', 0.0)),
                el=float(tm.get('energy_loss', 0.0)),
                de=float(tm.get('delta_e', tm.get('critic/objective_gap', 0.0))),
                pair=float(tm.get('pair_sat_e', 0.0)),
                iar=float(tm.get('iar', 0.0)),
                mgn=mgn,
                gacc=float(gi.get('gate/accepted', 0.0)),
                gfb=float(gi.get('gate/fallback', 0.0)),
                gsh=float(gi.get('gate/action_shift_l2', 0.0)),
                gpg=float(gi.get('gate/proxy_gain', 0.0)),
                gunc=float(gi.get('gate/unc_guided', gi.get('guidance/q_uncertainty', 0.0))),
                infer=float(self.last_infer_time) * 1000.0,
                proc=float(self.last_process_time) * 1000.0,
                train=float(self.last_train_time) * 1000.0,
            )
        )

    def _save_segment_data(self):
        """将当前分段的采样数据落盘，便于按场景切片保存。"""
        with self.buffer_lock:
            human_segment = list(self.segment_human_buffer)
            pvp_segment = list(self.segment_pvp_buffer)

        if not human_segment and not pvp_segment:
            self.get_logger().info("[P段保存] 当前段没有数据可保存，跳过。")
            return

        buffers_dir = os.path.join(self.config['save_folder'], 'buffers')
        os.makedirs(buffers_dir, exist_ok=True)

        file_name = f"segment_{self.segment_idx:04d}_iter_{self.iteration:08d}.pkl"
        file_path = os.path.join(buffers_dir, file_name)

        payload = {
            'human': human_segment,
            'pvp': pvp_segment,
            'iteration': self.iteration,
            'phase': self.phase_manager.current_phase,
        }

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(payload, f)
            self.get_logger().info(f"[P段保存] 已保存分段数据 -> {file_path}")
            tjitools.ros_log(self.get_name(), f"Segment saved: {file_name}")
            self.segment_idx += 1
        except Exception as e:
            self.get_logger().error(f"[P段保存] 保存失败: {e}")

        # 重置当前段缓存
        with self.buffer_lock:
            self.segment_human_buffer.clear()
            self.segment_pvp_buffer.clear()

    def sub_callback_surrounding(self, msgSurroundingInfo: SurroundingInfoInterface):
        self.rcvMsgSurroundingInfo = msgSurroundingInfo
        self._last_sensor_receive_time = time.time()

    def get_state(self) -> np.ndarray:
        if self.rcvMsgSurroundingInfo is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        state = []

        errorYaw = float(self.rcvMsgSurroundingInfo.error_yaw) / PI
        errorYaw = np.clip(errorYaw, -1.0, 1.0)
        state.append(errorYaw)

        errorDistance = float(self.rcvMsgSurroundingInfo.error_distance) / 5.0
        errorDistance = np.clip(errorDistance, -1.0, 1.0)
        state.append(errorDistance)

        carspeed = float(self.rcvMsgSurroundingInfo.carspeed) / 15.0
        carspeed = np.clip(carspeed, 0.0, 1.0)
        state.append(carspeed)

        turn_state = float(self.rcvMsgSurroundingInfo.turn_signals)
        turn_state = (turn_state + 1.0) / 2.0
        state.append(turn_state)

        radar_data = list(self.rcvMsgSurroundingInfo.surroundinginfo)
        expected_radar_len = 240
        if len(radar_data) < expected_radar_len:
            radar_data.extend([1.0] * (expected_radar_len - len(radar_data)))
        elif len(radar_data) > expected_radar_len:
            radar_data = radar_data[:expected_radar_len]
        state.extend(radar_data)

        path = list(self.rcvMsgSurroundingInfo.path_rfu)
        expected_path_len = 100
        if len(path) < expected_path_len:
            path.extend([0.0] * (expected_path_len - len(path)))
        elif len(path) > expected_path_len:
            path = path[:expected_path_len]
        path = np.array(path, dtype=np.float32) / 30.0
        path = np.clip(path, -1.0, 1.0)
        state.extend(list(path))

        final_state = np.array(state, dtype=np.float32)
        if len(final_state) != OBS_DIM:
            fixed = np.zeros(OBS_DIM, dtype=np.float32)
            n = min(len(final_state), OBS_DIM)
            fixed[:n] = final_state[:n]
            final_state = fixed

        return final_state

    def process_action(self, action: np.ndarray) -> tuple:
        x = float(action[0])
        y = float(action[1])

        if x > 0:
            throttle_percentage = int(round(x * 100))
            braking_percentage = 0
        else:
            throttle_percentage = 0
            braking_percentage = int(round(abs(x) * 100))

        steering_angle = y * 200.0
        return throttle_percentage, braking_percentage, steering_angle

    def send_action(self, action: np.ndarray):
        send_t0 = time.time()

        throttle_percentage, braking_percentage, steering_angle = self.process_action(action)
        actual_action = np.asarray(action, dtype=np.float32).copy()

        state = self.get_state()
        safety_override = 0
        if not self.safety_manager.check_action_safety(action, state):
            throttle_percentage = 0
            braking_percentage = 100
            steering_angle = 0.0
            actual_action = np.array([-1.0, 0.0], dtype=np.float32)
            safety_override = 1

        # 记录真实下发等价动作；如果 safety 覆盖，这里是急刹而不是原始 policy action。
        self.last_action_sent = actual_action
        self.last_sent_throttle = int(throttle_percentage)
        self.last_sent_brake = int(braking_percentage)
        self.last_sent_steer = float(steering_angle)
        self.last_safety_override = int(safety_override)

        gearpos = 0x03
        enableSignal = 1
        ultrasonicSwitch = 0
        dippedHeadlight = 0
        contourLamp = 0

        brakeEnable = 1 if braking_percentage != 0 else 0
        alarmLamp = 0
        turnSignalControl = 0
        appInsert = 0

        Byte7 = appInsert << 7 | turnSignalControl << 1 | alarmLamp
        Byte6 = int(braking_percentage) << 1 | brakeEnable
        Byte5 = (int(steering_angle) & 0xFF00) >> 8
        Byte4 = int(steering_angle) & 0x00FF
        Byte3 = 0
        Byte2 = ((int(throttle_percentage) * 10) & 0xFF00) >> 8
        Byte1 = (int(throttle_percentage) * 10) & 0x00FF
        Byte0 = gearpos << 6 | enableSignal << 5 | ultrasonicSwitch << 4 | dippedHeadlight << 1 | contourLamp

        canData = [Byte0, Byte1, Byte2, Byte3, Byte4, Byte5, Byte6, Byte7]
        self.last_can_bytes = list(canData)
        carControlMsg = can.Message(
            arbitration_id=self.config['hardware']['can_id'],
            data=canData,
            extended_id=False,
        )
        if not self.dry_run and self.carControlBus is not None:
            self.carControlBus.send(carControlMsg)
        self.last_send_time = float(time.time() - send_t0)

    def reverse_process_action(self, throttle_percentage: float, braking_percentage: float, steering_angle: float) -> np.ndarray:
        if braking_percentage != 0:
            x = -float(braking_percentage / 100)
        else:
            x = float(throttle_percentage / 100)
        x = np.clip(x, -1.0, 1.0)

        y = float(steering_angle) / 200.0
        y = np.clip(y, -1.0, 1.0)

        return np.array([x, y], dtype=np.float32)

    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        t0 = time.time()
        obs = torch.from_numpy(np.expand_dims(state, axis=0).astype(np.float32)).to(self.device)
        use_fast_gate = self.action_path not in ('debug', 'full_debug', 'diagnostic')
        log_gate_info = (not use_fast_gate) or (self.iteration % self.gate_diagnostics_interval == 0)
        with self.actor_lock:
            try:
                action, info = self.actor_agent.get_action_with_gate(
                    obs,
                    deterministic=False,
                    add_noise=False,
                    fast=use_fast_gate,
                    log_info=log_gate_info,
                )
            except Exception as e:
                self.get_logger().warning(f"[DOVE-ER] guided action failed, fallback to unguided sampler: {e}")
                action = self.actor_agent.get_action(obs, deterministic=False, add_noise=False)
                info = {"gate/accepted": 0.0, "gate/fallback": 1.0}
        self.last_infer_time = float(time.time() - t0)
        if info:
            self.last_guidance_info = dict(info)
        else:
            self.last_guidance_info = {"gate/diagnostics_logged": 0.0}
        action = action.detach().cpu().numpy().flatten().astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def _current_intervention(self) -> float:
        try:
            return 0.0 if int(self.rcvMsgSurroundingInfo.car_run_mode) == 1 else 1.0
        except Exception:
            return 0.0

    def _current_behavior_action(self) -> np.ndarray:
        return self.reverse_process_action(
            self.rcvMsgSurroundingInfo.throttle_percentage,
            self.rcvMsgSurroundingInfo.braking_percentage,
            self.rcvMsgSurroundingInfo.steerangle,
        )

    def _finish_sample_tick(
        self,
        *,
        state: np.ndarray,
        intervention: float,
        action_behavior: np.ndarray,
        action_novice: np.ndarray,
    ) -> None:
        self.last_action_human = np.asarray(action_behavior, dtype=np.float32).copy()
        self.last_action_model = np.asarray(action_novice, dtype=np.float32).copy()
        self.send_action(action_novice)

        if self.prev_action_sent_for_smooth is None:
            self.last_action_smooth_l1 = 0.0
        else:
            self.last_action_smooth_l1 = float(np.mean(np.abs(self.last_action_sent - self.prev_action_sent_for_smooth)))
        self.prev_action_sent_for_smooth = self.last_action_sent.copy()

        self.last_intervention = float(intervention)
        self.last_state_error_yaw = float(getattr(self.rcvMsgSurroundingInfo, 'error_yaw', 0.0))
        self.last_state_error_distance = float(getattr(self.rcvMsgSurroundingInfo, 'error_distance', 0.0))
        self.last_state_carspeed = float(getattr(self.rcvMsgSurroundingInfo, 'carspeed', 0.0))
        self.last_is_at_start = float(getattr(self.rcvMsgSurroundingInfo, 'is_at_start', 0.0))
        self.last_is_at_end = float(getattr(self.rcvMsgSurroundingInfo, 'is_at_end', 0.0))
        self._update_vehicle_rollup_metrics()

    def _sample_one_step_delay(self):
        state = self.get_state()
        batch_data = []

        intervention = self._current_intervention()
        self.takeover_recorder.append(intervention)

        action_behavior = self._current_behavior_action()

        takeover_start = (intervention == 1.0 and self._prev_intervention == 0.0)
        takeover_end = (intervention == 0.0 and self._prev_intervention == 1.0)
        stop_td = 1.0 if (takeover_start or takeover_end) else 0.0

        if (
            self._prev_state is not None
            and self._prev_action_novice is not None
            and self._prev_action_behavior is not None
            and self._prev_intervention_for_exp is not None
        ):
            exp = Experience.from_pvp(
                obs=self._prev_state,
                next_obs=state,
                reward=0.0,
                done=False,
                a_behavior=self._prev_action_behavior,
                a_novice=self._prev_action_novice,
                a_human=self._prev_action_behavior,
                intervention=self._prev_intervention_for_exp,
                stop_td=self._prev_stop_td,
            )
            batch_data.append(exp)

        action_novice = self._compute_action(state)
        self._finish_sample_tick(
            state=state,
            intervention=intervention,
            action_behavior=action_behavior,
            action_novice=action_novice,
        )

        self._prev_state = state.copy()
        self._prev_action_novice = action_novice.copy()
        self._prev_action_behavior = action_behavior.copy()
        self._prev_intervention_for_exp = float(intervention)
        self._prev_stop_td = float(stop_td)
        self._prev_intervention = intervention

        return batch_data

    def _sample_no_delay(self):
        state = self.get_state()
        intervention = self._current_intervention()
        self.takeover_recorder.append(intervention)
        action_behavior = self._current_behavior_action()

        takeover_start = (intervention == 1.0 and self._prev_intervention == 0.0)
        takeover_end = (intervention == 0.0 and self._prev_intervention == 1.0)
        stop_td = 1.0 if (takeover_start or takeover_end) else 0.0

        action_novice = self._compute_action(state)
        exp = Experience.from_pvp(
            obs=state,
            next_obs=state.copy(),
            reward=0.0,
            done=False,
            a_behavior=action_behavior,
            a_novice=action_novice,
            a_human=action_behavior,
            intervention=intervention,
            stop_td=stop_td,
        )
        self._finish_sample_tick(
            state=state,
            intervention=intervention,
            action_behavior=action_behavior,
            action_novice=action_novice,
        )

        self._prev_state = state.copy()
        self._prev_action_novice = action_novice.copy()
        self._prev_action_behavior = action_behavior.copy()
        self._prev_intervention_for_exp = float(intervention)
        self._prev_stop_td = float(stop_td)
        self._prev_intervention = intervention
        return [exp]

    def _update_vehicle_rollup_metrics(self):
        dt = self.last_loop_period if self.last_loop_period > 1e-6 else 0.1
        speed = float(self.last_state_carspeed)

        if self.prev_carspeed_for_jerk is None:
            accel = 0.0
            jerk = 0.0
        else:
            accel = (speed - float(self.prev_carspeed_for_jerk)) / dt
            if self.prev_accel_for_jerk is None:
                jerk = 0.0
            else:
                jerk = (accel - float(self.prev_accel_for_jerk)) / dt

        self.prev_carspeed_for_jerk = speed
        self.prev_accel_for_jerk = accel
        self.last_speed_accel = float(accel)
        self.last_speed_jerk = float(jerk)

        radar = np.asarray(list(getattr(self.rcvMsgSurroundingInfo, 'surroundinginfo', [])), dtype=np.float32)
        if radar.size > 0:
            finite = radar[np.isfinite(radar)]
            self.last_min_surrounding_distance = float(np.min(finite)) if finite.size > 0 else 0.0
        else:
            self.last_min_surrounding_distance = 0.0

        self.running_metric_count += 1
        self.sum_carspeed += speed
        self.sum_abs_error_distance += abs(float(self.last_state_error_distance))
        self.sum_abs_error_yaw += abs(float(self.last_state_error_yaw))
        self.sum_action_smooth_l1 += abs(float(self.last_action_smooth_l1))
        self.sum_abs_jerk += abs(float(jerk))
        if self.last_is_at_start > 0.5 and self.prev_is_at_start <= 0.5:
            self.route_start_count += 1
        if self.last_is_at_end > 0.5 and self.prev_is_at_end <= 0.5:
            self.route_end_count += 1
        self.prev_is_at_start = float(self.last_is_at_start)
        self.prev_is_at_end = float(self.last_is_at_end)

    def _vehicle_rollup_metrics(self) -> dict:
        n = max(1, int(self.running_metric_count))
        route_trials = max(1, int(self.route_start_count))
        return {
            'avg_speed': self.sum_carspeed / n,
            'avg_abs_lateral_error': self.sum_abs_error_distance / n,
            'avg_abs_yaw_error': self.sum_abs_error_yaw / n,
            'avg_action_smooth_l1': self.sum_action_smooth_l1 / n,
            'avg_abs_jerk': self.sum_abs_jerk / n,
            'route_completion_rate': float(self.route_end_count) / float(route_trials),
        }

    def _safe_msg_get(self, name: str, default=None):
        try:
            return getattr(self.rcvMsgSurroundingInfo, name)
        except Exception:
            return default

    def _write_runtime_csv(self, now_ts: float):
        if self.iteration % self.runtime_csv_interval != 0:
            return

        with self.buffer_lock:
            buffer_stats = self.buffer.get_statistics()
        phase_info = self.phase_manager.get_phase_info()
        safety_stats = self.safety_manager.get_safety_stats()
        rollup = self._vehicle_rollup_metrics()
        gi = self.last_guidance_info or {}
        with self.metrics_lock:
            tm = dict(self.training_metrics)

        row = {
            'ts': float(now_ts),
            'iter': int(self.iteration),
            'phase': phase_info.get('current_phase', ''),
            'phase_name': phase_info.get('phase_name', ''),
            'process_time': float(self.last_process_time),
            'infer_time': float(self.last_infer_time),
            'train_time': float(self.last_train_time),
            'actor_sync_time': float(self.last_actor_sync_time),
            'send_time': float(self.last_send_time),
            'loop_period': float(self.last_loop_period),
            'sensor_age': float(self.last_sensor_age),
            'deadline_miss': float(self.last_deadline_miss),
            'gearpos': self._safe_msg_get('gearpos', None),
            'car_run_mode': self._safe_msg_get('car_run_mode', None),
            'intervention': float(self.last_intervention),
            'human_buffer_size': int(buffer_stats.get('human_size', 0)),
            'pvp_buffer_size': int(buffer_stats.get('pvp_size', 0)),
            'total_samples': int(buffer_stats.get('total_samples', 0)),
            'total_interventions': int(buffer_stats.get('total_interventions', 0)),
            'error_yaw': float(self.last_state_error_yaw),
            'error_distance': float(self.last_state_error_distance),
            'carspeed': float(self.last_state_carspeed),
            'is_at_start': float(self.last_is_at_start),
            'is_at_end': float(self.last_is_at_end),
            'route_start_count': int(self.route_start_count),
            'route_end_count': int(self.route_end_count),
            'route_completion_rate': float(rollup['route_completion_rate']),
            'min_surrounding_distance': float(self.last_min_surrounding_distance),
            'speed_accel': float(self.last_speed_accel),
            'speed_jerk': float(self.last_speed_jerk),
            'running_avg_speed': float(rollup['avg_speed']),
            'running_avg_abs_lateral_error': float(rollup['avg_abs_lateral_error']),
            'running_avg_abs_yaw_error': float(rollup['avg_abs_yaw_error']),
            'running_avg_abs_jerk': float(rollup['avg_abs_jerk']),
            'human_action_0': float(self.last_action_human[0]),
            'human_action_1': float(self.last_action_human[1]),
            'model_action_0': float(self.last_action_model[0]),
            'model_action_1': float(self.last_action_model[1]),
            'sent_action_0': float(self.last_action_sent[0]),
            'sent_action_1': float(self.last_action_sent[1]),
            'sent_throttle_percentage': int(self.last_sent_throttle),
            'sent_braking_percentage': int(self.last_sent_brake),
            'sent_steering_angle': float(self.last_sent_steer),
            'action_smooth_l1': float(self.last_action_smooth_l1),
            'safety_override': int(self.last_safety_override),
            'safety_emergency_brakes': int(safety_stats.get('emergency_brakes', 0)),
            'safety_obstacle_brakes': int(safety_stats.get('obstacle_brakes', 0)),
            'safety_steer_violations': int(safety_stats.get('steer_violations', 0)),
            'gate_accepted': float(gi.get('gate/accepted', 0.0)),
            'gate_fallback': float(gi.get('gate/fallback', 0.0)),
            'gate_action_shift_l2': float(gi.get('gate/action_shift_l2', 0.0)),
            'gate_proxy_gain': float(gi.get('gate/proxy_gain', 0.0)),
            'gate_delta_c': float(gi.get('gate/delta_c', 0.0)),
            'gate_proxy_ok': float(gi.get('gate/proxy_ok', 0.0)),
            'gate_uncertainty_ok': float(gi.get('gate/uncertainty_ok', 0.0)),
            'gate_shift_ok': float(gi.get('gate/shift_ok', 0.0)),
            'gate_unc_guided': float(gi.get('gate/unc_guided', 0.0)),
            'gate_diagnostics_logged': float(gi.get('gate/diagnostics_logged', 0.0)),
            'guidance_gate': float(gi.get('guidance/gate', 0.0)),
            'guidance_q_uncertainty': float(gi.get('guidance/q_uncertainty', 0.0)),
            'guidance_grad_norm': float(gi.get('guidance/grad_norm', 0.0)),
            'bc_loss': float(tm.get('bc_loss', 0.0)),
            'q1_loss': float(tm.get('q1_loss', 0.0)),
            'q2_loss': float(tm.get('q2_loss', 0.0)),
            'policy_loss': float(tm.get('policy_loss', 0.0)),
            'rl_loss': float(tm.get('rl_loss', 0.0)),
            'energy_loss': float(tm.get('energy_loss', 0.0)),
            'energy_violation': float(tm.get('energy_violation', 0.0)),
            'pair_sat_e': float(tm.get('pair_sat_e', 0.0)),
            'iar': float(tm.get('iar', 0.0)),
            'delta_e': float(tm.get('delta_e', tm.get('critic/objective_gap', 0.0))),
            'action_gap_l2': float(tm.get('action_gap_l2', 0.0)),
            'mean_q1_std': float(tm.get('mean_q1_std', 0.0)),
            'mean_q2_std': float(tm.get('mean_q2_std', 0.0)),
            'can_bytes': json.dumps(list(self.last_can_bytes), ensure_ascii=False),
        }
        try:
            self._csv_writer.writerow(row)
            if self.iteration % self.runtime_csv_flush_interval == 0:
                self._csv_f.flush()
        except Exception as e:
            self.get_logger().warning(f"Failed to write train CSV: {e}")

    def train_algorithm(self) -> dict:
        with self.buffer_lock:
            stats = self.buffer.get_statistics()

        if self.phase_manager.current_phase == 1:
            return {}

        if self.phase_manager.current_phase == 2:
            bs = int(self.config['training']['replay_batch_size'])
            bc_updates = max(1, self.phase2_bc_updates_per_iter)
            metrics = {}
            train_t0 = time.time()
            with self.learner_lock:
                for _ in range(bc_updates):
                    with self.buffer_lock:
                        human_exps = self.buffer.sample_human(bs)
                    if not human_exps:
                        break

                    obs = torch.as_tensor(
                        np.stack([e.obs for e in human_exps], axis=0),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    act = torch.as_tensor(
                        np.stack([e.actions_behavior for e in human_exps], axis=0),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    m = self.algorithm.train_offline_bc(obs, act)
                    metrics.update(m)
                    self.phase_manager.update_progress(updates=1)
            self.last_train_time = float(time.time() - train_t0)
            if not metrics:
                return {}
# [新增] 打印 Phase 2 的 BC Loss
            self.get_logger().info(f"Phase2 Training | BC Loss: {metrics.get('bc_loss', 0.0):.4f} | Progress: {self.phase_manager.phase2_updates}/{self.phase_manager.phase2_threshold}")


            if self.phase_manager.should_transition_to_pvp():
                old = self.phase_manager.current_phase
                self._sync_actor_from_learner()
                self.phase_manager.transition_to_pvp()
                info = self.phase_manager.get_phase_info()
                self.get_logger().info(f"=== PHASE TRANSITION: {old} -> {info['current_phase']} ({info['phase_name']}) ===")

            return metrics

        if self.phase_manager.current_phase == 3:
            return self._train_phase3_once()

        return {}

    def _train_phase3_once(self) -> dict:
        bs = int(self.config['training']['replay_batch_size'])
        with self.buffer_lock:
            exps = self.buffer.sample_pvp_mixed(bs, human_ratio=self.phase3_human_mix_ratio)
        batch = self.buffer.to_pvp_batch(exps, device=self.device)
        if batch is None:
            return {}

        train_t0 = time.time()
        with self.learner_lock:
            metrics = self.algorithm.train_pvp(batch)
        self.last_train_time = float(time.time() - train_t0)
        self.phase_manager.update_progress(iterations=1)

        self._trainer_updates_since_sync += 1
        if self._trainer_updates_since_sync >= self.actor_sync_interval:
            self._sync_actor_from_learner()
            self._trainer_updates_since_sync = 0

        self.get_logger().info(
            f"Phase3 PVP | Actor: {metrics.get('policy_loss', 0):.4f} | "
            f"Critic: {(metrics.get('q1_loss', 0) + metrics.get('q2_loss', 0)) * 0.5:.4f} | "
            f"RL: {metrics.get('rl_loss', 0):.4f} | "
            f"BC: {metrics.get('bc_loss', 0):.4f} | "
            f"Energy: {metrics.get('energy_loss', 0):.4f} | "
            f"dE: {metrics.get('delta_e', 0):.4f} | "
            f"Pair: {metrics.get('pair_sat_e', 0):.3f} | "
            f"IAR: {metrics.get('iar', 0):.3f} | "
            f"Train: {self.last_train_time * 1000.0:.1f}ms"
        )
        return metrics

    def _sync_actor_from_learner(self):
        sync_t0 = time.time()
        with self.learner_lock:
            state = copy.deepcopy(self.agent.state_dict())
        with self.actor_lock:
            self.actor_agent.load_state_dict(state)
            self.actor_agent.eval()
        self.last_actor_sync_time = float(time.time() - sync_t0)

    def _phase3_train_loop(self):
        next_train_time = time.time()
        while not self._trainer_stop.is_set():
            if (
                not self.phase3_online_train_enabled
                or self.phase_manager.current_phase != 3
            ):
                time.sleep(0.02)
                next_train_time = time.time()
                continue

            now = time.time()
            if now < next_train_time:
                time.sleep(min(0.02, next_train_time - now))
                continue

            with self.buffer_lock:
                stats = self.buffer.get_statistics()
            if stats['pvp_size'] < int(self.config['training']['replay_batch_size']):
                time.sleep(0.02)
                continue

            if self.phase3_skip_train_on_deadline_miss and (
                self.last_deadline_miss > 0.5
                or self.last_infer_time * 1000.0 > self.phase3_max_infer_time_ms
            ):
                next_train_time = time.time() + float(self.phase3_train_period_s)
                time.sleep(0.02)
                continue

            try:
                metrics = self._train_phase3_once()
                if metrics:
                    with self.metrics_lock:
                        self.training_metrics.update(metrics)
                next_train_time = time.time() + max(0.001, self.phase3_train_period_s)
            except Exception as e:
                self.get_logger().error(f"[AsyncTrain] Phase3 training failed: {e}")
                next_train_time = time.time() + max(0.1, self.phase3_train_period_s)
                time.sleep(0.1)

    def _save_demo_data(self) -> str:
        path = os.path.join(self.config['save_folder'], 'buffers', f"demo_data_{self.iteration:08d}.pkl")
        with self.buffer_lock:
            data = list(self.buffer.human_buffer)
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(data, f)
        return path

    def _save_model(self):
        model_path = os.path.join(self.config['save_folder'], 'models', f"dacer_torch_{self.iteration:08d}.pt")
        try:
            with self.learner_lock:
                self.algorithm.save(model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to save model: {e}")

        buffer_path = os.path.join(self.config['save_folder'], 'buffers', f"buffer_{self.iteration:08d}.pkl")
        try:
            with self.buffer_lock:
                self.buffer.save(buffer_path)
        except Exception as e:
            self.get_logger().error(f"Failed to save buffer: {e}")

    def _log_metrics(self):
        with self.buffer_lock:
            buffer_stats = self.buffer.get_statistics()
        phase_info = self.phase_manager.get_phase_info()
        safety_stats = self.safety_manager.get_safety_stats()
        rollup = self._vehicle_rollup_metrics()

        if len(self.takeover_recorder) > 0:
            takeover_rate = float(np.mean(np.array(self.takeover_recorder)) * 100.0)
        else:
            takeover_rate = 0.0

# [新增] 打印统计摘要
        with self.metrics_lock:
            tm = dict(self.training_metrics)
        gi = self.last_guidance_info or {}
        self.get_logger().info(
            f"\n=== Stats @ Iter {self.iteration} ===\n"
            f"  Buffer (Human/PVP): {buffer_stats['human_size']}/{buffer_stats['pvp_size']}\n"
            f"  Safety (Brakes/Steer): {safety_stats['emergency_brakes']}/{safety_stats['steer_violations']}\n"
            f"  Takeover Rate: {takeover_rate:.2f}%\n"
            f"  Train Steps (Phase2/Phase3): {phase_info.get('phase2_updates', 0)}/{phase_info.get('phase3_iterations', 0)}\n"
            f"  Loss actor/critic/rl/bc/energy: "
            f"{float(tm.get('policy_loss', 0.0)):.4f}/"
            f"{0.5 * (float(tm.get('q1_loss', 0.0)) + float(tm.get('q2_loss', 0.0))):.4f}/"
            f"{float(tm.get('rl_loss', 0.0)):.4f}/"
            f"{float(tm.get('bc_loss', 0.0)):.4f}/"
            f"{float(tm.get('energy_loss', 0.0)):.4f}\n"
            f"  dE/PairSat/IAR/MGN: "
            f"{float(tm.get('delta_e', tm.get('critic/objective_gap', 0.0))):.4f}/"
            f"{float(tm.get('pair_sat_e', 0.0)):.3f}/"
            f"{float(tm.get('iar', 0.0)):.3f}/"
            f"{float(gi.get('guidance/grad_norm', gi.get('gate/action_shift_l2', 0.0))):.4f}\n"
            f"  Gate acc/fallback/shift/proxy/unc: "
            f"{float(gi.get('gate/accepted', 0.0)):.2f}/"
            f"{float(gi.get('gate/fallback', 0.0)):.2f}/"
            f"{float(gi.get('gate/action_shift_l2', 0.0)):.3f}/"
            f"{float(gi.get('gate/proxy_gain', 0.0)):.4f}/"
            f"{float(gi.get('gate/unc_guided', gi.get('guidance/q_uncertainty', 0.0))):.4f}\n"
            f"=============================="
        )

        if self.use_tensorboard:
            with self.metrics_lock:
                metrics_items = list(self.training_metrics.items())
            for k, v in metrics_items:
                try:
                    self.writer.add_scalar(k, float(v), self.iteration)
                except Exception:
                    pass

            self.writer.add_scalar('Buffer/human_size', buffer_stats['human_size'], self.iteration)
            self.writer.add_scalar('Buffer/pvp_size', buffer_stats['pvp_size'], self.iteration)
            self.writer.add_scalar('Phase/current', phase_info['current_phase'], self.iteration)
            self.writer.add_scalar('Safety/emergency_brakes', safety_stats['emergency_brakes'], self.iteration)
            self.writer.add_scalar('Safety/steer_violations', safety_stats['steer_violations'], self.iteration)
            self.writer.add_scalar('takeover_rate', takeover_rate, self.iteration)

            # 运行/车辆关键指标
            self.writer.add_scalar('Timing/process_time_ms', float(self.last_process_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/infer_time_ms', float(self.last_infer_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/can_send_time_ms', float(self.last_send_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/loop_period_ms', float(self.last_loop_period) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/sensor_age_ms', float(self.last_sensor_age) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/deadline_miss', float(self.last_deadline_miss), self.iteration)
            self.writer.add_scalar('Time/elapsed_s', float(time.time() - self.start_wall_time), self.iteration)

            self.writer.add_scalar('Vehicle/carspeed', float(self.last_state_carspeed), self.iteration)
            self.writer.add_scalar('Vehicle/error_yaw', float(self.last_state_error_yaw), self.iteration)
            self.writer.add_scalar('Vehicle/error_distance', float(self.last_state_error_distance), self.iteration)
            self.writer.add_scalar('Vehicle/intervention', float(self.last_intervention), self.iteration)
            self.writer.add_scalar('Vehicle/is_at_start', float(self.last_is_at_start), self.iteration)
            self.writer.add_scalar('Vehicle/is_at_end', float(self.last_is_at_end), self.iteration)
            self.writer.add_scalar('Vehicle/route_start_count', float(self.route_start_count), self.iteration)
            self.writer.add_scalar('Vehicle/route_end_count', float(self.route_end_count), self.iteration)
            self.writer.add_scalar('Vehicle/route_completion_rate', float(rollup['route_completion_rate']), self.iteration)
            self.writer.add_scalar('Vehicle/min_surrounding_distance', float(self.last_min_surrounding_distance), self.iteration)
            self.writer.add_scalar('Vehicle/speed_accel', float(self.last_speed_accel), self.iteration)
            self.writer.add_scalar('Vehicle/speed_jerk', float(self.last_speed_jerk), self.iteration)
            self.writer.add_scalar('Vehicle/avg_speed', float(rollup['avg_speed']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_abs_lateral_error', float(rollup['avg_abs_lateral_error']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_abs_yaw_error', float(rollup['avg_abs_yaw_error']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_abs_jerk', float(rollup['avg_abs_jerk']), self.iteration)

            # 动作对齐：human / model / sent
            self.writer.add_scalar('Action/human_x', float(self.last_action_human[0]), self.iteration)
            self.writer.add_scalar('Action/human_y', float(self.last_action_human[1]), self.iteration)
            self.writer.add_scalar('Action/model_x', float(self.last_action_model[0]), self.iteration)
            self.writer.add_scalar('Action/model_y', float(self.last_action_model[1]), self.iteration)
            self.writer.add_scalar('Action/sent_throttle', float(self.last_sent_throttle), self.iteration)
            self.writer.add_scalar('Action/sent_brake', float(self.last_sent_brake), self.iteration)
            self.writer.add_scalar('Action/sent_steer', float(self.last_sent_steer), self.iteration)
            self.writer.add_scalar('Safety/safety_override', float(self.last_safety_override), self.iteration)
            for i, b in enumerate(self.last_can_bytes):
                self.writer.add_scalar(f'CAN/byte_{i}', float(b), self.iteration)

            for k, v in self.last_guidance_info.items():
                try:
                    self.writer.add_scalar(str(k), float(v), self.iteration)
                except Exception:
                    pass

            # 舒适度代理：动作变化幅度（越小越平滑）
            self.writer.add_scalar('Comfort/action_smooth_l1', float(self.last_action_smooth_l1), self.iteration)

    def pub_callback_car_dacer_torch(self):
        msg = CarRLInterface()
        now_ts = time.time()
        msg.timestamp = now_ts

        if self.last_timestamp > 0.0:
            self.last_loop_period = float(now_ts - self.last_timestamp)
        self.last_timestamp = float(now_ts)
        self.last_sensor_age = float(now_ts - getattr(self, '_last_sensor_receive_time', now_ts))

        if self.rcvMsgSurroundingInfo is None:
            return

        # 暂停时不采样/不训练/不发送动作，便于阶段一按场景采集
        if self.is_paused:
            return

        self.iteration += 1


# [新增] 打印当前 Iter 和 Phase，方便确认程序在跑
        if self.iteration % self.console_status_interval == 0:
            self._log_console_status()



        if int(self.rcvMsgSurroundingInfo.gearpos) != 2:
            # Phase2 仅离线 BC 训练，不发送 AI 动作，保持人工控制以保障安全与实时性
            if self.phase_manager.current_phase != 2:
                exps = self._sample_one_step_delay() if self.use_one_step_delay else self._sample_no_delay()

                for exp in exps:
                    if self.phase_manager.current_phase == 1:
                        if self.phase1_collect_only_intervention and exp.interventions <= 0.5:
                            continue
                        with self.buffer_lock:
                            self.buffer.add_human(exp)
                            self.buffer.add_pvp(exp)
                            # 记录分段数据，便于按 P 保存
                            self.segment_human_buffer.append(exp)
                            self.segment_pvp_buffer.append(exp)
                        self.phase_manager.update_progress(episodes=1)

                        if self.phase_manager.should_transition_to_phase2():
                            if self.collect_only:
                                if not self._collect_only_threshold_saved:
                                    self._save_demo_data()
                                    self._collect_only_threshold_saved = True
                                    self.get_logger().info(
                                        "=== COLLECT-ONLY: Phase1 target reached; buffer saved, continuing data collection without Phase2/Phase3 ==="
                                    )
                            else:
                                self._save_demo_data()
                                old = self.phase_manager.current_phase
                                self.phase_manager.transition_to_phase2()
                                info = self.phase_manager.get_phase_info()
                                self.get_logger().info(f"=== PHASE TRANSITION: {old} -> {info['current_phase']} ({info['phase_name']}) ===")
                    else:
                        with self.buffer_lock:
                            if exp.interventions > 0.5:
                                self.buffer.add_human(exp)
                                self.segment_human_buffer.append(exp)
                            self.buffer.add_pvp(exp)
                            self.segment_pvp_buffer.append(exp)

                # 打印采样明细（节流），与 car_rl 一致地看到模型输出 vs 实际输出
                if self.iteration % self.log_detail_interval == 0 and exps:
                    # 使用第一个 exp 的行为动作作为参考
                    ref_exp = exps[0]
                    self._log_step_detail(
                        intervention=ref_exp.interventions,
                        action_behavior=ref_exp.actions_behavior,
                        action_novice=ref_exp.actions_novice,
                    )

        with self.buffer_lock:
            stats = self.buffer.get_statistics()

        can_train = False
        if self.phase_manager.current_phase == 2:
            can_train = stats['human_size'] >= int(self.config['training']['buffer_warm_size'])
        elif self.phase_manager.current_phase == 3:
            can_train = stats['pvp_size'] >= int(self.config['training']['replay_batch_size'])

        phase3_callback_training = not (
            self.phase_manager.current_phase == 3 and self.phase3_train_in_background
        )
        train_allowed = can_train and phase3_callback_training and (
            self.phase_manager.current_phase == 2 or self.phase3_online_train_enabled
        )
        if self.iteration % int(self.config['training']['update_interval']) == 0 and train_allowed:
            metrics = self.train_algorithm()
            if metrics:
                with self.metrics_lock:
                    self.training_metrics.update(metrics)

        if self.iteration % int(self.config['training']['log_save_interval']) == 0:
            self._log_metrics()

        if self.iteration % int(self.config['training']['apprfunc_save_interval']) == 0:
            self._save_model()

        msg.process_time = time.time() - now_ts
        self.last_process_time = float(msg.process_time)
        self.last_deadline_miss = 1.0 if self.last_process_time > 0.1 else 0.0
        self.pubCarDACER.publish(msg)
        self._write_runtime_csv(now_ts)

        if self.use_tensorboard and self.iteration % self.runtime_tb_interval == 0:
            try:
                self.writer.add_scalar('Timing/process_time_ms', self.last_process_time * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/infer_time_ms', self.last_infer_time * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/train_time_ms', self.last_train_time * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/actor_sync_time_ms', self.last_actor_sync_time * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/can_send_time_ms', self.last_send_time * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/loop_period_ms', self.last_loop_period * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/sensor_age_ms', self.last_sensor_age * 1000.0, self.iteration)
                self.writer.add_scalar('Timing/deadline_miss', self.last_deadline_miss, self.iteration)
                self.writer.add_scalar('Time/timestamp', float(now_ts), self.iteration)
                self.writer.add_scalar('Time/elapsed_s', float(time.time() - self.start_wall_time), self.iteration)
                self.writer.add_scalar('Vehicle/carspeed', float(self.last_state_carspeed), self.iteration)
                self.writer.add_scalar('Vehicle/error_yaw', float(self.last_state_error_yaw), self.iteration)
                self.writer.add_scalar('Vehicle/error_distance', float(self.last_state_error_distance), self.iteration)
                self.writer.add_scalar('Vehicle/intervention', float(self.last_intervention), self.iteration)
                rollup = self._vehicle_rollup_metrics()
                self.writer.add_scalar('Vehicle/is_at_start', float(self.last_is_at_start), self.iteration)
                self.writer.add_scalar('Vehicle/is_at_end', float(self.last_is_at_end), self.iteration)
                self.writer.add_scalar('Vehicle/route_start_count', float(self.route_start_count), self.iteration)
                self.writer.add_scalar('Vehicle/route_end_count', float(self.route_end_count), self.iteration)
                self.writer.add_scalar('Vehicle/route_completion_rate', float(rollup['route_completion_rate']), self.iteration)
                self.writer.add_scalar('Vehicle/min_surrounding_distance', float(self.last_min_surrounding_distance), self.iteration)
                self.writer.add_scalar('Vehicle/speed_accel', float(self.last_speed_accel), self.iteration)
                self.writer.add_scalar('Vehicle/speed_jerk', float(self.last_speed_jerk), self.iteration)
                self.writer.add_scalar('Vehicle/avg_speed', float(rollup['avg_speed']), self.iteration)
                self.writer.add_scalar('Vehicle/avg_abs_lateral_error', float(rollup['avg_abs_lateral_error']), self.iteration)
                self.writer.add_scalar('Vehicle/avg_abs_yaw_error', float(rollup['avg_abs_yaw_error']), self.iteration)
                self.writer.add_scalar('Vehicle/avg_abs_jerk', float(rollup['avg_abs_jerk']), self.iteration)
                self.writer.add_scalar('Action/human_x', float(self.last_action_human[0]), self.iteration)
                self.writer.add_scalar('Action/human_y', float(self.last_action_human[1]), self.iteration)
                self.writer.add_scalar('Action/model_x', float(self.last_action_model[0]), self.iteration)
                self.writer.add_scalar('Action/model_y', float(self.last_action_model[1]), self.iteration)
                self.writer.add_scalar('Action/sent_x', float(self.last_action_sent[0]), self.iteration)
                self.writer.add_scalar('Action/sent_y', float(self.last_action_sent[1]), self.iteration)
                self.writer.add_scalar('Action/sent_throttle', float(self.last_sent_throttle), self.iteration)
                self.writer.add_scalar('Action/sent_brake', float(self.last_sent_brake), self.iteration)
                self.writer.add_scalar('Action/sent_steer', float(self.last_sent_steer), self.iteration)
                self.writer.add_scalar('Comfort/action_smooth_l1', float(self.last_action_smooth_l1), self.iteration)
                self.writer.add_scalar('Safety/safety_override', float(self.last_safety_override), self.iteration)
                for i, b in enumerate(self.last_can_bytes):
                    self.writer.add_scalar(f'CAN/byte_{i}', float(b), self.iteration)
                for k, v in self.last_guidance_info.items():
                    self.writer.add_scalar(str(k), float(v), self.iteration)
            except Exception:
                pass

        tjitools.ros_log(self.get_name(), 'Publish car_dacer_torch msg !!!')

    def destroy_node(self):
        self._trainer_stop.set()
        if self._trainer_thread is not None and self._trainer_thread.is_alive():
            self._trainer_thread.join(timeout=2.0)
        try:
            if self.listener is not None:
                self.listener.stop()
        except Exception:
            pass
        try:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass
        try:
            self._csv_f.flush()
            self._csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    rosNode = CarDACERTorch()
    try:
        rclpy.spin(rosNode)
    finally:
        rosNode.destroy_node()
        rclpy.shutdown()
