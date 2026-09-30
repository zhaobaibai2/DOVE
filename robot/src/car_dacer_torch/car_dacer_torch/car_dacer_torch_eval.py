import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import can
import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node

from car_interfaces.msg import CarRLInterface, SurroundingInfoInterface

from .safety_manager import SafetyManager
from .torch_algorithm import PVPDACERTorch, TorchDACERConfig
from .torch_networks import DACERTorchAgent, DACERActionConfig


OBS_DIM = 344
PI = 3.1415926535


class CarDACERTorchEval(Node):
    def __init__(self, *, noisy: bool):
        super().__init__('car_dacer_torch_eval')

        self.noisy = bool(noisy)
        self.config = self._load_config()
        self.eval_csv_flush_interval = max(
            1,
            int(self.config.get('logging', {}).get('eval_csv_flush_interval', 20)),
        )
        eval_cfg = self.config.get('eval', {}) or {}
        self.action_path = str(eval_cfg.get('action_path', 'fast')).lower()
        self.gate_diagnostics_interval = max(1, int(eval_cfg.get('gate_diagnostics_interval', 20)))

        self.subSurrounding = self.create_subscription(
            SurroundingInfoInterface, 'surrounding_info_data', self.sub_callback_surrounding, 1
        )
        self.pubCarDACER = self.create_publisher(
            CarRLInterface, 'car_dacer_data', 10
        )
        self.timer = self.create_timer(0.1, self._on_timer)

        self.rcvMsgSurroundingInfo: Optional[SurroundingInfoInterface] = None
        self.iteration = 0
        self.start_wall_time = time.time()
        self.last_loop_timestamp = 0.0
        self._last_sensor_receive_time = 0.0
        self.last_sensor_age = 0.0
        self.takeover_recorder = []
        self.takeover_max_len = 2000
        self.last_guidance_info: Dict[str, float] = {}
        self.prev_speed_for_jerk: Optional[float] = None
        self.prev_accel_for_jerk: Optional[float] = None
        self.metric_count = 0
        self.sum_speed = 0.0
        self.sum_abs_lateral_error = 0.0
        self.sum_abs_yaw_error = 0.0
        self.sum_abs_jerk = 0.0
        self.sum_abs_sent_accel_cmd = 0.0
        self.sum_abs_sent_steer = 0.0
        self.safety_override_count = 0
        self.deadline_miss_count = 0
        self.prev_is_at_start = 0.0
        self.prev_is_at_end = 0.0
        self.route_start_count = 0
        self.route_end_count = 0

        self._init_safety()
        self._init_device()
        self._init_algo()
        self._init_can()
        self._init_log_file()
        self._init_tensorboard()

        self.get_logger().info(self._startup_banner())

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.config['save_folder'],
                flush_secs=self.config.get('logging', {}).get('tensorboard_flush_secs', 20),
            )
            self.use_tensorboard = True
        except Exception:
            self.writer = None
            self.use_tensorboard = False

    def _startup_banner(self) -> str:
        mode = 'NOISY' if self.noisy else 'DETERMINISTIC'
        model_path = self._model_path
        return (
            f"=== DACER TORCH EVAL START ===\n"
            f"  mode: {mode}\n"
            f"  model: {model_path}\n"
            f"  log: {self._csv_path}\n"
            f"  can_interface: {self.config.get('hardware', {}).get('can_interface')}\n"
            f"  can_id: {self.config.get('hardware', {}).get('can_id')}\n"
            f"==============================="
        )

    def _load_config(self) -> dict:
        ros_node_name = 'car_dacer_torch'
        default_config_path = os.getcwd() + f"/src/{ros_node_name}/{ros_node_name}/" + 'config.yaml'
        config_path = os.environ.get('DACER_CONFIG_PATH', default_config_path)
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg['config_path'] = config_path
        logging_cfg = cfg.get('logging', {}) or {}
        base_save = logging_cfg.get('result_root', './outputs')
        experiment_name = logging_cfg.get('experiment_name', 'DACER_TORCH_EVAL')
        save_folder = os.path.join(
            base_save,
            f'{experiment_name}_' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
        )
        cfg['save_folder'] = save_folder
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'eval_logs'), exist_ok=True)

        with open(os.path.join(save_folder, 'config_eval.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        return cfg

    def _init_safety(self):
        self.safety_manager = SafetyManager(self.config.get('safety', {}))

    def _init_device(self):
        hw = self.config.get('hardware', {})
        use_gpu = bool(hw.get('use_gpu', True))
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _resolve_model_path(self) -> str:
        eval_cfg = self.config.get('eval', {}) or {}
        model_path = eval_cfg.get('model_path')
        if isinstance(model_path, str) and model_path.lower() == 'auto':
            auto_path = self._auto_find_latest_model()
            if auto_path:
                return auto_path
        if model_path and os.path.exists(model_path):
            return str(model_path)

        model_dir = eval_cfg.get('model_dir') or './outputs/DACER_TORCH_CAR/models/'
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        candidates = []
        for name in os.listdir(model_dir):
            if not name.endswith('.pt'):
                continue
            if not name.startswith('dacer_torch_'):
                continue
            full = os.path.join(model_dir, name)
            if os.path.isfile(full):
                candidates.append(full)

        if not candidates:
            raise FileNotFoundError(f"no model found under: {model_dir}")

        return max(candidates, key=os.path.getmtime)

    def _auto_find_latest_model(self) -> Optional[str]:
        import glob

        eval_cfg = self.config.get('eval', {}) or {}
        method_name = str(eval_cfg.get('method_name', '')).lower()
        if 'positive' in method_name or 'pos' in method_name:
            patterns = [
                "./outputs/DOVEER_REAL03_R1_ADAPT_POS_DBP*/models/*.pt",
                "./outputs/*POS*DBP*/models/*.pt",
            ]
        else:
            patterns = [
                "./outputs/DOVEER_REAL09_R1_FULL_ONLINE_FROM_SCRATCH*/models/*.pt",
                "./outputs/DOVEER_REAL06_R1_ONLINE_FULL*/models/*.pt",
                "./outputs/DOVEER_REAL02_R1_ADAPT_FULL*/models/*.pt",
                "./outputs/*FULL_ONLINE_FROM_SCRATCH*/models/*.pt",
                "./outputs/*ADAPT_FULL*/models/*.pt",
                "./outputs/*ONLINE_FULL*/models/*.pt",
            ]

        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
        candidates = [p for p in candidates if os.path.isfile(p) and p.endswith('.pt')]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    def _init_algo(self):
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
        self._model_path = self._resolve_model_path()
        self.algorithm.load(self._model_path)
        self.algorithm.agent.eval()

        eval_cfg = self.config.get('eval', {}) or {}
        self.eval_deterministic = bool(eval_cfg.get('deterministic', True))
        self.eval_noise_std = float(eval_cfg.get('noise_std', 0.05))

    def _init_can(self):
        hw = self.config.get('hardware', {})
        self.dry_run = bool(hw.get('dry_run', False))
        self.carControlBus = None
        if not self.dry_run:
            self.carControlBus = can.interface.Bus(
                channel=hw.get('can_interface', 'can1'),
                bustype='socketcan',
            )

    def _init_log_file(self):
        ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        self._csv_path = os.path.join(self.config['save_folder'], 'eval_logs', f"eval_{ts}.csv")
        self._csv_f = open(self._csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_header())
        self._csv_writer.writeheader()
        self._csv_f.flush()

    def _csv_header(self):
        return [
            'method_name',
            'trial_id',
            'route_id',
            'scenario_id',
            'checkpoint_id',
            'weather',
            'traffic_density',
            'obstacle_setup',
            'ts',
            'iter',
            'process_time',
            'infer_time',
            'send_time',
            'loop_period',
            'sensor_age',
            'deadline_miss',
            'gearpos',
            'car_run_mode',
            'is_at_start',
            'is_at_end',
            'error_yaw',
            'error_distance',
            'carspeed',
            'turn_signals',
            'human_throttle_percentage',
            'human_braking_percentage',
            'human_steerangle',
            'state_vec',
            'action_raw_0',
            'action_raw_1',
            'action_sent_0',
            'action_sent_1',
            'sent_throttle_percentage',
            'sent_braking_percentage',
            'sent_steering_angle',
            'safety_override',
            'can_bytes',
            'gate_accepted',
            'gate_fallback',
            'gate_action_shift_l2',
            'gate_q_lcb_improve',
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
            'safety_emergency_brakes',
            'safety_obstacle_brakes',
            'safety_steer_violations',
            'min_surrounding_distance',
            'speed_accel',
            'speed_jerk',
            'running_avg_speed',
            'running_avg_abs_lateral_error',
            'running_avg_abs_yaw_error',
            'running_avg_abs_jerk',
            'running_takeover_rate',
            'running_safety_override_rate',
            'running_deadline_miss_rate',
            'route_start_count',
            'route_end_count',
            'route_completion_rate',
            'surroundinginfo',
            'path_rfu',
            'model_path',
            'mode',
        ]

    def sub_callback_surrounding(self, msg: SurroundingInfoInterface):
        self.rcvMsgSurroundingInfo = msg
        self._last_sensor_receive_time = time.time()

    def get_state(self) -> np.ndarray:
        if self.rcvMsgSurroundingInfo is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        msg = self.rcvMsgSurroundingInfo
        state = []

        errorYaw = float(getattr(msg, 'error_yaw', 0.0)) / PI
        errorYaw = np.clip(errorYaw, -1.0, 1.0)
        state.append(errorYaw)

        errorDistance = float(getattr(msg, 'error_distance', 0.0)) / 5.0
        errorDistance = np.clip(errorDistance, -1.0, 1.0)
        state.append(errorDistance)

        carspeed = float(getattr(msg, 'carspeed', 0.0)) / 15.0
        carspeed = np.clip(carspeed, 0.0, 1.0)
        state.append(carspeed)

        turn_state = float(getattr(msg, 'turn_signals', -1.0))
        turn_state = (turn_state + 1.0) / 2.0
        state.append(turn_state)

        radar_data = list(getattr(msg, 'surroundinginfo', []))
        expected_radar_len = 240
        if len(radar_data) < expected_radar_len:
            radar_data.extend([1.0] * (expected_radar_len - len(radar_data)))
        elif len(radar_data) > expected_radar_len:
            radar_data = radar_data[:expected_radar_len]
        state.extend(radar_data)

        path = list(getattr(msg, 'path_rfu', []))
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

    def process_action(self, action: np.ndarray) -> Tuple[int, int, float]:
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

    def _pack_can(self, throttle_percentage: int, braking_percentage: int, steering_angle: float) -> Tuple[can.Message, list]:
        gearpos = 0x03
        enableSignal = 1
        ultrasonicSwitch = 0
        dippedHeadlight = 0
        contourLamp = 0

        brakeEnable = 1 if int(braking_percentage) != 0 else 0
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

        msg = can.Message(
            arbitration_id=self.config.get('hardware', {}).get('can_id', 0x210),
            data=canData,
            extended_id=False,
        )
        return msg, canData

    def _compute_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        t0 = time.time()
        obs = torch.from_numpy(np.expand_dims(state, axis=0).astype(np.float32)).to(self.device)
        use_fast_gate = self.action_path not in ('debug', 'full_debug', 'diagnostic')
        log_gate_info = (not use_fast_gate) or (self.iteration % self.gate_diagnostics_interval == 0)
        try:
            action, info = self.algorithm.get_action_with_info(
                obs,
                deterministic=self.eval_deterministic,
                add_noise=False,
                fast=use_fast_gate,
                log_info=log_gate_info,
            )
        except Exception as e:
            self.get_logger().warning(f"[DOVE-ER Eval] guided action failed, fallback to unguided sampler: {e}")
            action = self.algorithm.get_action(obs, deterministic=self.eval_deterministic, add_noise=False)
            info = {"gate/accepted": 0.0, "gate/fallback": 1.0}
        infer_time = float(time.time() - t0)
        if info:
            self.last_guidance_info = dict(info)
        else:
            self.last_guidance_info = {"gate/diagnostics_logged": 0.0}
        action = action.detach().cpu().numpy().flatten().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        action_sent = action.copy()
        if self.noisy:
            noise = np.random.randn(*action_sent.shape).astype(np.float32) * float(self.eval_noise_std)
            action_sent = np.clip(action_sent + noise, -1.0, 1.0)

        return action, action_sent, infer_time

    def _send_action(self, action_sent: np.ndarray, state: np.ndarray) -> Tuple[int, int, float, bool, list, np.ndarray]:
        throttle_percentage, braking_percentage, steering_angle = self.process_action(action_sent)
        safety_override = False
        actual_action = np.asarray(action_sent, dtype=np.float32).copy()

        if not self.safety_manager.check_action_safety(action_sent, state):
            throttle_percentage = 0
            braking_percentage = 100
            steering_angle = 0.0
            actual_action = np.array([-1.0, 0.0], dtype=np.float32)
            safety_override = True

        can_msg, can_bytes = self._pack_can(throttle_percentage, braking_percentage, steering_angle)

        if not self.dry_run and self.carControlBus is not None:
            self.carControlBus.send(can_msg)

        return throttle_percentage, braking_percentage, steering_angle, safety_override, can_bytes, actual_action

    def _safe_get(self, msg: Any, name: str, default: Any = None) -> Any:
        try:
            return getattr(msg, name)
        except Exception:
            return default

    def _update_eval_rollup(
        self,
        *,
        speed: float,
        lateral_error: float,
        yaw_error: float,
        loop_period: float,
        sent_accel_cmd: float,
        sent_steer: float,
        safety_override: bool,
        deadline_miss: float,
        is_at_start: float,
        is_at_end: float,
    ) -> Dict[str, float]:
        dt = loop_period if loop_period > 1e-6 else 0.1
        if self.prev_speed_for_jerk is None:
            accel = 0.0
            jerk = 0.0
        else:
            accel = (speed - float(self.prev_speed_for_jerk)) / dt
            if self.prev_accel_for_jerk is None:
                jerk = 0.0
            else:
                jerk = (accel - float(self.prev_accel_for_jerk)) / dt

        self.prev_speed_for_jerk = float(speed)
        self.prev_accel_for_jerk = float(accel)

        self.metric_count += 1
        self.sum_speed += float(speed)
        self.sum_abs_lateral_error += abs(float(lateral_error))
        self.sum_abs_yaw_error += abs(float(yaw_error))
        self.sum_abs_jerk += abs(float(jerk))
        self.sum_abs_sent_accel_cmd += abs(float(sent_accel_cmd))
        self.sum_abs_sent_steer += abs(float(sent_steer))
        self.safety_override_count += int(1 if safety_override else 0)
        self.deadline_miss_count += int(1 if deadline_miss > 0.5 else 0)
        if is_at_start > 0.5 and self.prev_is_at_start <= 0.5:
            self.route_start_count += 1
        if is_at_end > 0.5 and self.prev_is_at_end <= 0.5:
            self.route_end_count += 1
        self.prev_is_at_start = float(is_at_start)
        self.prev_is_at_end = float(is_at_end)

        n = max(1, int(self.metric_count))
        route_trials = max(1, int(self.route_start_count))
        return {
            'speed_accel': float(accel),
            'speed_jerk': float(jerk),
            'running_avg_speed': self.sum_speed / n,
            'running_avg_abs_lateral_error': self.sum_abs_lateral_error / n,
            'running_avg_abs_yaw_error': self.sum_abs_yaw_error / n,
            'running_avg_abs_jerk': self.sum_abs_jerk / n,
            'running_avg_abs_sent_accel_cmd': self.sum_abs_sent_accel_cmd / n,
            'running_avg_abs_sent_steer': self.sum_abs_sent_steer / n,
            'running_safety_override_rate': float(self.safety_override_count) / n,
            'running_deadline_miss_rate': float(self.deadline_miss_count) / n,
            'route_completion_rate': float(self.route_end_count) / float(route_trials),
        }

    def _on_timer(self):
        now_ts = time.time()
        loop_period = float(now_ts - self.last_loop_timestamp) if self.last_loop_timestamp > 0.0 else 0.0
        self.last_loop_timestamp = float(now_ts)
        msg_pub = CarRLInterface()
        msg_pub.timestamp = now_ts

        if self.rcvMsgSurroundingInfo is None:
            return

        self.iteration += 1
        self.last_sensor_age = float(now_ts - self._last_sensor_receive_time) if self._last_sensor_receive_time > 0.0 else 0.0
        state = self.get_state()

        gearpos = self._safe_get(self.rcvMsgSurroundingInfo, 'gearpos', None)
        if gearpos is not None:
            try:
                if int(gearpos) == 2:
                    return
            except Exception:
                pass

        action_raw, action_sent, infer_time = self._compute_action(state)
        send_t0 = time.time()
        sent_th, sent_br, sent_steer, safety_override, can_bytes, action_sent_actual = self._send_action(action_sent, state)
        action_sent = action_sent_actual
        send_time = float(time.time() - send_t0)

        process_time = time.time() - now_ts
        deadline_miss = 1.0 if process_time > 0.1 else 0.0
        msg_pub.process_time = process_time
        self.pubCarDACER.publish(msg_pub)
        gi = self.last_guidance_info
        speed = float(self._safe_get(self.rcvMsgSurroundingInfo, 'carspeed', 0.0))
        lateral_error = float(self._safe_get(self.rcvMsgSurroundingInfo, 'error_distance', 0.0))
        yaw_error = float(self._safe_get(self.rcvMsgSurroundingInfo, 'error_yaw', 0.0))
        surrounding = list(self._safe_get(self.rcvMsgSurroundingInfo, 'surroundinginfo', []))
        surrounding_arr = np.asarray(surrounding, dtype=np.float32)
        finite_surrounding = surrounding_arr[np.isfinite(surrounding_arr)]
        min_surrounding_distance = float(np.min(finite_surrounding)) if finite_surrounding.size > 0 else 0.0
        car_run_mode = self._safe_get(self.rcvMsgSurroundingInfo, 'car_run_mode', 1)
        is_at_start = float(self._safe_get(self.rcvMsgSurroundingInfo, 'is_at_start', 0.0))
        is_at_end = float(self._safe_get(self.rcvMsgSurroundingInfo, 'is_at_end', 0.0))
        try:
            intervention = 0.0 if int(car_run_mode) == 1 else 1.0
        except Exception:
            intervention = 0.0
        self.takeover_recorder.append(float(intervention))
        if len(self.takeover_recorder) > int(self.takeover_max_len):
            self.takeover_recorder = self.takeover_recorder[-int(self.takeover_max_len):]
        takeover_rate = float(np.mean(np.asarray(self.takeover_recorder)) * 100.0) if self.takeover_recorder else 0.0
        rollup = self._update_eval_rollup(
            speed=speed,
            lateral_error=lateral_error,
            yaw_error=yaw_error,
            loop_period=float(loop_period),
            sent_accel_cmd=float(action_sent[0]),
            sent_steer=float(action_sent[1]),
            safety_override=bool(safety_override),
            deadline_miss=float(deadline_miss),
            is_at_start=is_at_start,
            is_at_end=is_at_end,
        )
        eval_cfg = self.config.get('eval', {}) or {}
        safety_stats = self.safety_manager.get_safety_stats()

        row = {
            'method_name': eval_cfg.get('method_name', ''),
            'trial_id': eval_cfg.get('trial_id', ''),
            'route_id': eval_cfg.get('route_id', ''),
            'scenario_id': eval_cfg.get('scenario_id', ''),
            'checkpoint_id': eval_cfg.get('checkpoint_id', os.path.basename(str(self._model_path))),
            'weather': eval_cfg.get('weather', ''),
            'traffic_density': eval_cfg.get('traffic_density', ''),
            'obstacle_setup': eval_cfg.get('obstacle_setup', ''),
            'ts': now_ts,
            'iter': self.iteration,
            'process_time': process_time,
            'infer_time': float(infer_time),
            'send_time': float(send_time),
            'loop_period': float(loop_period),
            'sensor_age': float(self.last_sensor_age),
            'deadline_miss': float(deadline_miss),
            'gearpos': self._safe_get(self.rcvMsgSurroundingInfo, 'gearpos', None),
            'car_run_mode': car_run_mode,
            'is_at_start': is_at_start,
            'is_at_end': is_at_end,
            'error_yaw': yaw_error,
            'error_distance': lateral_error,
            'carspeed': speed,
            'turn_signals': self._safe_get(self.rcvMsgSurroundingInfo, 'turn_signals', None),
            'human_throttle_percentage': self._safe_get(self.rcvMsgSurroundingInfo, 'throttle_percentage', None),
            'human_braking_percentage': self._safe_get(self.rcvMsgSurroundingInfo, 'braking_percentage', None),
            'human_steerangle': self._safe_get(self.rcvMsgSurroundingInfo, 'steerangle', None),
            'state_vec': json.dumps(state.tolist(), ensure_ascii=False),
            'action_raw_0': float(action_raw[0]),
            'action_raw_1': float(action_raw[1]),
            'action_sent_0': float(action_sent[0]),
            'action_sent_1': float(action_sent[1]),
            'sent_throttle_percentage': int(sent_th),
            'sent_braking_percentage': int(sent_br),
            'sent_steering_angle': float(sent_steer),
            'safety_override': int(1 if safety_override else 0),
            'can_bytes': json.dumps(can_bytes, ensure_ascii=False),
            'gate_accepted': float(gi.get('gate/accepted', 0.0)),
            'gate_fallback': float(gi.get('gate/fallback', 0.0)),
            'gate_action_shift_l2': float(gi.get('gate/action_shift_l2', 0.0)),
            'gate_q_lcb_improve': float(gi.get('gate/q_lcb_improve', 0.0)),
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
            'safety_emergency_brakes': int(safety_stats.get('emergency_brakes', 0)),
            'safety_obstacle_brakes': int(safety_stats.get('obstacle_brakes', 0)),
            'safety_steer_violations': int(safety_stats.get('steer_violations', 0)),
            'min_surrounding_distance': min_surrounding_distance,
            'speed_accel': rollup['speed_accel'],
            'speed_jerk': rollup['speed_jerk'],
            'running_avg_speed': rollup['running_avg_speed'],
            'running_avg_abs_lateral_error': rollup['running_avg_abs_lateral_error'],
            'running_avg_abs_yaw_error': rollup['running_avg_abs_yaw_error'],
            'running_avg_abs_jerk': rollup['running_avg_abs_jerk'],
            'running_takeover_rate': takeover_rate,
            'running_safety_override_rate': rollup['running_safety_override_rate'],
            'running_deadline_miss_rate': rollup['running_deadline_miss_rate'],
            'route_start_count': self.route_start_count,
            'route_end_count': self.route_end_count,
            'route_completion_rate': rollup['route_completion_rate'],
            'surroundinginfo': json.dumps(surrounding, ensure_ascii=False),
            'path_rfu': json.dumps(list(self._safe_get(self.rcvMsgSurroundingInfo, 'path_rfu', [])), ensure_ascii=False),
            'model_path': self._model_path,
            'mode': 'noisy' if self.noisy else 'deterministic',
        }
        self._csv_writer.writerow(row)
        if self.iteration % self.eval_csv_flush_interval == 0:
            self._csv_f.flush()

        # 评估指标：TensorBoard 每个控制 tick 记录，便于复现实车延迟和 gate 行为。
        if self.use_tensorboard:
            self.writer.add_scalar('Timing/process_time_ms', float(process_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/infer_time_ms', float(infer_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/can_send_time_ms', float(send_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/loop_period_ms', float(loop_period) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/sensor_age_ms', float(self.last_sensor_age) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/deadline_miss', float(deadline_miss), self.iteration)
            self.writer.add_scalar('Timing/deadline_miss_rate', float(rollup['running_deadline_miss_rate']), self.iteration)
            self.writer.add_scalar('Time/elapsed_s', float(time.time() - self.start_wall_time), self.iteration)
            self.writer.add_scalar('Vehicle/carspeed', speed, self.iteration)
            self.writer.add_scalar('Vehicle/error_yaw', yaw_error, self.iteration)
            self.writer.add_scalar('Vehicle/error_distance', lateral_error, self.iteration)
            self.writer.add_scalar('Vehicle/is_at_start', is_at_start, self.iteration)
            self.writer.add_scalar('Vehicle/is_at_end', is_at_end, self.iteration)
            self.writer.add_scalar('Vehicle/route_start_count', float(self.route_start_count), self.iteration)
            self.writer.add_scalar('Vehicle/route_end_count', float(self.route_end_count), self.iteration)
            self.writer.add_scalar('Vehicle/route_completion_rate', float(rollup['route_completion_rate']), self.iteration)
            self.writer.add_scalar('Vehicle/min_surrounding_distance', min_surrounding_distance, self.iteration)
            self.writer.add_scalar('Vehicle/speed_accel', float(rollup['speed_accel']), self.iteration)
            self.writer.add_scalar('Vehicle/speed_jerk', float(rollup['speed_jerk']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_speed', float(rollup['running_avg_speed']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_abs_lateral_error', float(rollup['running_avg_abs_lateral_error']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_abs_yaw_error', float(rollup['running_avg_abs_yaw_error']), self.iteration)
            self.writer.add_scalar('Vehicle/avg_abs_jerk', float(rollup['running_avg_abs_jerk']), self.iteration)
            self.writer.add_scalar('Vehicle/takeover_rate', float(takeover_rate), self.iteration)
            self.writer.add_scalar('takeover_rate', float(takeover_rate), self.iteration)

            self.writer.add_scalar('Action/raw_x', float(action_raw[0]), self.iteration)
            self.writer.add_scalar('Action/raw_y', float(action_raw[1]), self.iteration)
            self.writer.add_scalar('Action/sent_x', float(action_sent[0]), self.iteration)
            self.writer.add_scalar('Action/sent_y', float(action_sent[1]), self.iteration)
            self.writer.add_scalar('Action/avg_abs_sent_x', float(rollup['running_avg_abs_sent_accel_cmd']), self.iteration)
            self.writer.add_scalar('Action/avg_abs_sent_y', float(rollup['running_avg_abs_sent_steer']), self.iteration)
            self.writer.add_scalar('Action/sent_throttle', float(sent_th), self.iteration)
            self.writer.add_scalar('Action/sent_brake', float(sent_br), self.iteration)
            self.writer.add_scalar('Action/sent_steer', float(sent_steer), self.iteration)
            self.writer.add_scalar('Safety/safety_override', float(1.0 if safety_override else 0.0), self.iteration)
            self.writer.add_scalar('Safety/safety_override_rate', float(rollup['running_safety_override_rate']), self.iteration)
            for i, b in enumerate(can_bytes):
                self.writer.add_scalar(f'CAN/byte_{i}', float(b), self.iteration)
            for k, v in self.last_guidance_info.items():
                try:
                    self.writer.add_scalar(str(k), float(v), self.iteration)
                except Exception:
                    pass

    def destroy_node(self):
        try:
            self._csv_f.flush()
            self._csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def _main(noisy: bool):
    rclpy.init()
    node = CarDACERTorchEval(noisy=noisy)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_det():
    _main(noisy=False)


def main_noisy():
    _main(noisy=True)
