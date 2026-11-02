#!/usr/bin/env python3
"""
MetaDrive Fixed Maps Environment with PVP Support
基于原始MetaDriveEnv，添加PVP功能，支持固定seed地图和中间车道重生
"""
import logging
import random
from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym

# 添加本地 metadrive 路径
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
metadrive_path = os.path.join(project_root, "metadrive")
if os.path.isdir(metadrive_path):
    sys.path.insert(0, project_root)

from metadrive import MetaDriveEnv
from relax.utils.takeover_policy import create_takeover_policy

logger = logging.getLogger(__name__)


class MetaDriveFixedPVPEnv(MetaDriveEnv):
    """
    MetaDrive Fixed Maps Environment with PVP (Human-in-the-Loop) Support
    
    基于原始 MetaDriveEnv，添加：
    - 手动控制支持 (keyboard, gamepad, steering wheel)
    - 接管策略用于人工干预
    - PVP 数据收集 (a_n, a_h, a_b)
    - 干预检测和标记
    - 固定seed地图 (40, 41, 42, 43)
    - 中间车道重生
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 提取 PVP 特定配置
        self.controller_type = config.get('controller', 'keyboard') if config else 'keyboard'
        self.manual_control = config.get('manual_control', False) if config else False
        self.enable_takeover = self.manual_control or config.get('enable_takeover', False) if config else False
        
        # 🔄 地图轮换配置
        self.enable_map_rotation = config.get('enable_map_rotation', False) if config else False
        self.available_maps = ['C', 'r', 'T', 'S']  # 四个基本地图
        self.current_map_index = 0
        
        # 保存原始地图类型配置
        if self.enable_map_rotation:
            self.map_type = self.available_maps[0]  # 轮换模式下从第一个开始
            config['map'] = self.map_type  # 更新config
        else:
            self.map_type = config.get('map', 'C') if config else 'C'
        
        # 中间车道控制
        self.force_middle_lane = config.get('force_middle_lane', True)  # 强制中间车道
        
        # 初始化基础环境
        super().__init__(config)
        
        # 如果启用接管，设置接管策略
        if self.enable_takeover:
            self.takeover_policy = create_takeover_policy(self, self.controller_type)
        else:
            self.takeover_policy = None
            
        # PVP 状态跟踪
        self.last_takeover = False
        self.agent_action = None
        self.step_count = 0
        # 添加统计跟踪
        self.total_takeover_steps = 0
        self.total_steps = 0
        self.takeover_start_count = 0
        # TTFT (Time-To-First Takeover) 跟踪
        self.first_takeover_step = None
        self.episode_first_takeover = None
        # 🎯 TTFT相关统计
        self.ttft_censored = False  # 是否右删失（无接管）
        
        logger.info(f"MetaDriveFixedPVPEnv initialized with controller: {self.controller_type}, map: {self.map_type}")
        if self.enable_takeover:
            logger.info(f"Takeover policy enabled - AI training with human intervention")
    
    def default_config(self):
        """带有 PVP 设置的默认配置"""
        cfg = super().default_config()
        
        # 添加 PVP 特定设置
        cfg.update({
            "manual_control": False,            # 环境本身不处于纯手动控制模式
            "enable_takeover": self.manual_control,  # 启用接管功能（由你的 MetaDriveFixedPVPEnv 实现/读取）
            "controller": self.controller_type,
            "use_render": True,                # 训练也保持渲染（用于可视化/接管）
            "map": self.map_type,             # 指定地图类型
            "start_seed": 0,             # 固定种子
            "num_scenarios": 1,                 # ✅ 固定场景（不在 reset 时轮换 seed）
            "traffic_density": 0.3,
            "horizon": 1000,
            "force_destroy": True,
            # ✅ 关键：不再随机选择出生车道
            "random_spawn_lane_index": False if self.force_middle_lane else True,
            "vehicle_config": {},  # 默认车辆配置
        })
        
        return cfg
        
    def reset(self, seed=None, options=None):
        """重置环境和 PVP 状态"""
        # 🔄 地图轮换逻辑
        if self.enable_map_rotation:
            # 轮换到下一个地图
            self.current_map_index = (self.current_map_index + 1) % len(self.available_maps)
            self.map_type = self.available_maps[self.current_map_index]
            
            # 更新环境配置中的地图类型
            self.config["map"] = self.map_type
            
            # 🔄 直接修改map_config - 这是MetaDrive的核心地图配置
            try:
                # 更新map_config中的配置
                if "map_config" in self.config:
                    self.config["map_config"]["config"] = self.map_type
                
                # 强制重新创建地图 - 使用最根本的方法
                if hasattr(self, 'engine'):
                    # 清理所有地图相关对象
                    if hasattr(self.engine, 'task_manager'):
                        self.engine.task_manager.destroy_all_objects()
                    
                    # 强制重新加载地图
                    if hasattr(self.engine, 'force_close'):
                        self.engine.force_close = True
                        
                    # 重新设置地图配置
                    self.engine.current_seed = getattr(self, 'current_seed', 0) + 1
                    
                # 删除地图缓存
                if hasattr(self, '_map_cache'):
                    self._map_cache.clear()
                    
                # 强制使用新地图
                self.config.update({
                    "map": self.map_type,
                    "start_seed": getattr(self, 'current_seed', 0) + 1,
                    "force_reload": True,
                })
                        
            except Exception as e:
                # 如果重新加载失败，至少确保配置是正确的
                pass
            
            # 打印地图切换信息
            print(f"🗺️  Switched to map: {self.map_type} (index {self.current_map_index})")
            # 添加验证信息
            print(f"🔧 Map config updated to: {self.config.get('map', 'unknown')}")
            print(f"🌍 Map seed: {self.config.get('start_seed', 'unknown')}")
            print(f"🗺️  Map config: {self.config.get('map_config', {}).get('config', 'unknown')}")
        
        # 重置 PVP 状态
        self.last_takeover = False
        self.agent_action = None
        self.step_count = 0
        # 重置统计
        self.total_takeover_steps = 0
        self.total_steps = 0
        self.takeover_start_count = 0
        # 重置TTFT跟踪
        self.first_takeover_step = None
        self.episode_first_takeover = None
        # 🎯 重置TTFT统计
        self.ttft_censored = False
        
        if self.takeover_policy:
            self.takeover_policy.reset()
        
        # 重置基础环境 - 修复options参数兼容性
        try:
            # 新版本gymnasium API
            obs, info = super().reset(seed=seed, options=options)
        except TypeError:
            # 旧版本API，不支持options参数
            obs, info = super().reset(seed=seed)
        
        # 强制设置到中间车道（如果启用）- 确保在每次reset后都执行
        if self.force_middle_lane:
            self._force_middle_lane_position()
            # 再次验证确保成功
            import time
            time.sleep(0.1)  # 给设置一点时间生效
            self._force_middle_lane_position()  # 再次强制设置
            
        # 添加 PVP 信息到 info 字典
        info.update({
            'takeover': False,
            'takeover_start': False,
            'raw_action': np.array([0.0, 0.0], dtype=np.float32),  # 人类输入或 agent action fallback
            'agent_action': np.array([0.0, 0.0], dtype=np.float32),  # Agent action (a_n)
            'behavior_action': np.array([0.0, 0.0], dtype=np.float32),  # 执行的动作 (a_b)
            'controller_type': self.controller_type,
            'crash': False,
            'out_of_road': False,
            'success': False,
        })
        
        return obs, info
    
    def _force_middle_lane_position(self):
        """强制车辆重生在中间车道"""
        try:
            # 获取车辆当前位置
            current_pos = self.agent.position
            
            # 对于大多数地图，中间车道的Y坐标通常是0或接近0
            # 保持X坐标不变，只调整Y坐标到中间车道
            middle_lane_y = 0.0
            
            # 设置车辆到中间车道
            import numpy as np
            new_position = np.array([float(current_pos[0]), float(middle_lane_y)])
            self.agent.set_position(new_position)
            
            # 验证设置是否成功
            actual_pos = self.agent.position
            success = abs(actual_pos[1] - middle_lane_y) < 0.1
            
            # logger.info(f"Vehicle forced to middle lane: ({actual_pos[0]:.1f}, {actual_pos[1]:.1f}) {'✅' if success else '❌'}")
                    
        except Exception as e:
            logger.warning(f"Failed to force middle lane position: {e}")
            # 如果设置失败，使用默认重生位置
        
    def step(self, action):
        """
        带有 PVP 支持的步进
        
        Args:
            action: Agent action (a_n)
            
        Returns:
            obs, reward, terminated, truncated, info with PVP data
        """
        self.agent_action = action.copy()
        self.step_count += 1
        
        if self.takeover_policy:
            # 获取接管决策和人类动作
            takeover_result = self.takeover_policy.act(action)
            
            # 提取 PVP 数据
            behavior_action = takeover_result['action']
            human_action = takeover_result['raw_action']
            takeover = takeover_result['takeover']
            takeover_start = takeover_result['takeover_start']
            
            # 更新接管状态
            prev_takeover = self.last_takeover
            self.last_takeover = takeover
            
            # 更新接管统计
            if takeover:
                self.total_takeover_steps += 1
            if takeover_start:
                self.takeover_start_count += 1
                
                # 🎯 记录第一次接管时间 (TTFT) - 学习进度指标
                if self.first_takeover_step is None:
                    self.first_takeover_step = self.step_count
                    self.episode_first_takeover = self.step_count
                    logger.info(f"[TTFT] 第一次接管发生在第 {self.step_count} 步")
            
        else:
            # 没有手动控制 - 直接使用 agent 动作
            behavior_action = action
            human_action = action.copy()  # 实际上是 agent action 的 fallback
            takeover = False
            takeover_start = False
            prev_takeover = self.last_takeover
            
        # 更新总步数
        self.total_steps += 1
        
        # 使用行为动作步进环境
        obs, reward, terminated, truncated, base_info = super().step(behavior_action)
        
        # 添加 PVP 信息到 info
        info = base_info.copy()
        info.update({
            'takeover': takeover,
            'takeover_start': takeover_start,
            'takeover_end': prev_takeover and (not takeover),  # 添加 takeover_end
            'raw_action': human_action,  # 人类输入或 agent action fallback
            'agent_action': self.agent_action,  # 原始 agent 动作
            'behavior_action': behavior_action,  # 实际执行的动作
        })
        
        # 🎯 添加TTFT信息到info
        if self.episode_first_takeover is not None:
            info['ttft'] = self.episode_first_takeover
        else:
            # 如果episode还在进行中且无接管，显示当前状态
            if not terminated and not truncated:
                info['ttft'] = self.step_count  # 当前步数作为临时TTFT
            else:
                # episode结束且无接管，标记为右删失
                self.ttft_censored = True
                info['ttft'] = self.step_count  # episode长度作为TTFT（右删失）
        
        # 添加结果检测
        if terminated or truncated:
            # 从 base_info 或环境状态检查 crash, out_of_road, success
            info['crash'] = base_info.get('crash', False) or getattr(self, 'is_crashed', False)
            info['out_of_road'] = base_info.get('out_of_road', False) or getattr(self, 'out_of_road', False)
            info['success'] = bool(base_info.get('success', False) or base_info.get('arrive_dest', False))
        else:
            info['crash'] = False
            info['out_of_road'] = False
            info['success'] = False
        
        # 📊 添加速度和性能信息到info字典
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            try:
                # 获取车辆速度
                if hasattr(self.vehicle, 'speed'):
                    info['speed'] = float(self.vehicle.speed)
                elif hasattr(self.vehicle, 'velocity'):
                    velocity = self.vehicle.velocity
                    if isinstance(velocity, (list, tuple, np.ndarray)) and len(velocity) >= 2:
                        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                        info['speed'] = float(speed)
                        info['velocity'] = [float(velocity[0]), float(velocity[1])]
                else:
                    # 备用：从位置变化估算速度
                    info['speed'] = 0.0
            except Exception as e:
                logger.debug(f"Failed to get vehicle speed: {e}")
                info['speed'] = 0.0
        else:
            info['speed'] = 0.0
        
        # 记录接管事件
        if takeover_start:
            logger.info(f"Takeover started at step {self.step_count}")
        elif prev_takeover and not takeover:
            logger.info(f"Takeover ended at step {self.step_count}")
            
        return obs, reward, terminated, truncated, info
        
    def render(self, *args, **kwargs):
        """渲染带有 PVP 覆盖信息"""
        result = super().render(*args, **kwargs)
        
        # 如果启用手动控制，添加 PVP 状态覆盖
        if self.manual_control and self.config.get("use_render", False):
            self._render_pvp_overlay()
            
        return result
        
    def _render_pvp_overlay(self):
        """在屏幕上渲染 PVP 状态覆盖"""
        if not hasattr(self, 'engine') or self.engine is None:
            return
            
        try:
            from metadrive.engine.core.onscreen_message import ScreenMessage
            
            # 创建状态文本
            takeover_status = "TAKEOVER" if self.last_takeover else "AUTO"
            controller_text = f"Controller: {self.controller_type.upper()}"
            takeover_text = f"Status: {takeover_status}"
            
            # 在屏幕上显示
            if hasattr(self.engine, 'show_message'):
                self.engine.show_message([
                    controller_text,
                    takeover_text,
                    "Press any button to toggle takeover" if self.manual_control else "",
                ], duration=0.1)  # 显示 0.1 秒
                
        except Exception as e:
            # 如果覆盖不工作，静默失败
            pass
            
    def get_pvp_info(self) -> Dict[str, Any]:
        """获取当前 PVP 状态信息"""
        return {
            'controller_type': self.controller_type,
            'manual_control': self.manual_control,
            'takeover': self.last_takeover,
            'step_count': self.step_count,
            'has_takeover_policy': self.takeover_policy is not None,
        }


def create_metadrive_fixed_pvp_env(config: Dict[str, Any] = None) -> MetaDriveFixedPVPEnv:
    """
    工厂函数创建带有适当配置的 MetaDriveFixedPVPEnv
    
    Args:
        config: 环境配置字典
        
    Returns:
        配置好的 MetaDriveFixedPVPEnv 实例
    """
    if config is None:
        config = {}
        
    # 确保必需的 PVP 配置
    pvp_config = {
        "manual_control": config.get("manual_control", True),
        "controller": config.get("controller", "keyboard"),
        "use_render": config.get("use_render", True),
        # 固定地图配置
        "start_seed": config.get("start_seed", 40),
        "num_scenarios": 1,
        "traffic_density": config.get("traffic_density", 0.3),
        "map": config.get("map", "C"),  # 默认环形地图
        "horizon": 1000,
        "force_destroy": True,
        "vehicle_config": {
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,
        }
    }
    
    # 与任何额外配置合并
    final_config = {**config, **pvp_config}
    
    env = MetaDriveFixedPVPEnv(final_config)
    
    logger.info(f"Created MetaDriveFixedPVPEnv with config: {final_config}")
    
    return env


# Gym 注册以便使用
def register_metadrive_fixed_pvp_env():
    """向 gymnasium 注册 MetaDriveFixedPVPEnv"""
    try:
        gym.register(
            id='MetaDriveFixedPVP-v0',
            entry_point='relax_env.metadrive_fixed_pvp_env:create_metadrive_fixed_pvp_env',
            max_episode_steps=1000,
        )
        logger.info("MetaDriveFixedPVP-v0 registered successfully")
    except gym.error.Error:
        logger.info("MetaDriveFixedPVP-v0 already registered")


def create_training_env(seed=40, map_type="C"):
    """创建训练环境 - 兼容原有接口"""
    config = {
        "start_seed": seed,
        "map": map_type,
        "manual_control": True,  # 启用 PVP
        "enable_takeover": True,  # 启用接管
        "controller": "keyboard",
        "use_render": True,
    }
    return create_metadrive_fixed_pvp_env(config)


def test_environment():
    """测试环境 - 只有固定地图CRTS，带PVP功能"""
    seeds = [40, 41, 42, 43]
    map_types = ["C", "r", "T", "S"]  # 只有固定地图，不包含X
    
    print("=== MetaDrive Fixed PVP 环境测试 (CRTS + PVP) ===")
    print("MetaDriveFixedPVPEnv")
    print("地图类型: C(环形), r(道路), T(T型), S(S型)")
    print("PVP功能: 手动控制 + 接管策略\n")

    for i, seed in enumerate(seeds):
        print(f"🗺️ 测试地图 {i+1} - Seed: {seed} - Map: {map_types[i]}")
        print("-" * 40)

        try:
            env = create_training_env(seed, map_types[i])

            # 重置环境
            obs, info = env.reset(seed=seed)

            print(f"✅ 环境创建成功")
            print(f"   观察空间: {obs.shape}")
            print(f"   动作空间: {env.action_space}")
            print(f"   PVP信息: {env.get_pvp_info()}")

            # 显示地图信息
            if hasattr(env, 'engine') and hasattr(env.engine, 'current_map'):
                current_map = env.engine.current_map
                print(f"   地图类型: {type(current_map).__name__}")

            # 显示交通信息
            if hasattr(env, 'engine') and hasattr(env.engine, 'traffic_manager'):
                traffic_mgr = env.engine.traffic_manager
                if hasattr(traffic_mgr, 'get_vehicles'):
                    vehicles = traffic_mgr.get_vehicles()
                    print(f"   NPC车辆数: {len(vehicles)}")

            # 简单测试运行
            print(f"   🎮 测试运行:")
            total_reward = 0
            for step in range(10):
                # 随机动作
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if step % 5 == 0:
                    print(f"      步骤 {step}: 奖励={reward:.3f}, 接管={info.get('takeover', False)}")

                if terminated or truncated:
                    print(f"      Episode结束: {step} 步")
                    break

            print(f"   总奖励: {total_reward:.3f}")
            env.close()
            print(f"✅ 地图 {seed} 测试完成\n")

        except Exception as e:
            print(f"❌ 地图 {seed} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    test_environment()
    
    print("🎯 训练环境创建函数:")
    print("   create_training_env() - 返回配置好的训练环境")
    print("   create_metadrive_fixed_pvp_env() - 创建 PVP 环境")
    print("   可在训练脚本中导入使用")
    print("   地图类型: C(环形), r(道路), T(T型), S(S型)")
    print("   PVP功能: 手动控制 + 接管策略 + 三动作数据收集")
