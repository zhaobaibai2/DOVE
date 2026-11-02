"""
Human-in-the-Loop Environment
基于expB1/pvp/experiments/metadrive/human_in_the_loop_env.py
但使用expB0的接管、指标记录和动作流接入系统
"""

import copy
import time
from collections import deque
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils.math import safe_clip

# 导入expB0的接管和指标系统
try:
    from relax.utils.takeover_policy import create_takeover_policy
except ImportError as e:
    # 如果导入失败，使用简单的接管策略
    print(f"Failed to import takeover_policy: {e}")
    def create_takeover_policy(env, controller_type):
        print(f"Using dummy takeover policy for controller: {controller_type}")
        return None

ScreenMessage.SCALE = 0.1

logger = logging.getLogger(__name__)

# 使用expB1的配置，但替换接管相关部分，并缩小种子范围
HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting (来自expB1，但缩小范围):
    "out_of_route_done": True,  # Raise done if out of route.
    "crash_done": True,  # Raise done if crash.
    "num_scenarios": 20,  # 训练地图 100–119
    "start_seed": 100,  # 训练地图 100–119
    "traffic_density": 0.06,

    # Reward and cost setting (来自expB1):
    "cost_to_reward": True,  # Cost will be negated and added to the reward.
    "cos_similarity": False,  # If True, the takeover cost will be the cos sim between a_h and a_n.

    # Set up the control device (使用expB0的控制器系统):
    "manual_control": False,  # 环境本身不处于纯手动控制模式
    "enable_takeover": True,  # 启用接管功能
    "controller": "steering_wheel",  # 使用expB0支持的控制器
    "only_takeover_start_cost": False,  # If True, only return a cost when takeover starts.

    # Visualization (来自expB1):
    "vehicle_config": {
        "show_dest_mark": True,  # Show the destination in a cube.
        "show_line_to_dest": True,  # Show the line to the destination.
        "show_line_to_navi_mark": True,  # Show the line to next navigation checkpoint.
    }
}


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    Human-in-the-Loop Environment
    基于expB1的human_in_the_loop_env.py，但使用expB0的接管和指标系统
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # 初始化expB0的PVP统计变量
        self.total_steps = 0
        self.total_takeover_cost = 0
        self.total_cost = 0
        self.takeover = False
        self.takeover_recorder = deque(maxlen=2000)
        self.agent_action = None
        self.in_pause = False
        self.start_time = time.time()
        
        # expB0的PVP统计
        self.total_takeover_steps = 0
        self.takeover_start_count = 0
        self.first_takeover_step = None
        self.episode_first_takeover = None
        self.ttft_censored = False
        
        # 顺序重置功能
        self.sequential_reset = False
        self.current_map_index = 0
        self.map_seeds = []
        
        # 合并配置
        if config is None:
            config = {}
        merged_config = HUMAN_IN_THE_LOOP_ENV_CONFIG.copy()
        merged_config.update(config)
        
        # 检查是否启用顺序重置
        sequential_reset = config.get("sequential_reset", False)
        if sequential_reset:
            start_seed = config.get("start_seed", 100)
            num_scenarios = config.get("num_scenarios", 20)
            self.map_seeds = list(range(start_seed, start_seed + num_scenarios))
            self.current_map_index = 0
            self.sequential_reset = True
            logger.info(f"Sequential reset enabled: maps {self.map_seeds}")
        else:
            self.sequential_reset = False
        
        # 移除sequential_reset配置项，避免MetaDrive报错
        if "sequential_reset" in merged_config:
            del merged_config["sequential_reset"]
        
        # 调用父类初始化
        super().__init__(merged_config)
        
        # 强制设置出界和碰撞终止
        self.config["out_of_route_done"] = True
        self.config["crash_done"] = True
        logger.info("强制设置: out_of_route_done=True, crash_done=True")
        
        # 初始化expB0的接管策略
        self._init_takeover_policy()
        
        logger.info(f"HumanInTheLoopEnv initialized with controller: {self.controller_type}")
        if self.sequential_reset:
            logger.info(f"Sequential reset mode: {self.map_seeds}")
        else:
            logger.info(f"Map seeds: {self.config['start_seed']} - {self.config['start_seed'] + self.config['num_scenarios'] - 1}")
    
    def _init_takeover_policy(self):
        """初始化expB0的接管策略"""
        try:
            # 动态导入接管策略
            self.takeover_policy = create_takeover_policy(
                self, self.config.get("controller", "keyboard")
            )
            logger.info("Takeover policy initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize takeover policy: {e}")
            import traceback
            traceback.print_exc()
            self.takeover_policy = None
    
    def default_config(self):
        """默认配置"""
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        return config
    
    def reset(self, seed=None, options=None):
        """重置环境 - 支持顺序重置或随机重置"""
        # 重置expB0统计变量
        self.takeover = False
        self.agent_action = None
        self.total_takeover_cost = 0
        self.total_cost = 0
        self.takeover_recorder.clear()
        
        # 重置PVP统计
        self.total_takeover_steps = 0
        self.takeover_start_count = 0
        self.first_takeover_step = None
        self.episode_first_takeover = None
        self.ttft_censored = False
        
        # 如果启用顺序重置，使用指定的种子
        if self.sequential_reset:
            target_seed = self.map_seeds[self.current_map_index]
            logger.info(f"Sequential reset: using seed {target_seed} (index {self.current_map_index}/{len(self.map_seeds)-1})")
            
            # 强制使用指定种子重置
            obs, info = super().reset(seed=target_seed)
            
            # 更新到下一个地图
            self.current_map_index = (self.current_map_index + 1) % len(self.map_seeds)
        else:
            # 调用父类reset - MetaDrive会自动从种子池中随机选择种子
            obs, info = super().reset(seed=seed)
            
            # 记录实际使用的种子
            actual_seed = info.get('env_seed', 'Unknown')
            logger.info(f"Reset with random seed: {actual_seed} (from pool {self.config['start_seed']}-{self.config['start_seed'] + self.config['num_scenarios'] - 1})")

        zero_action = np.zeros(self.action_space.shape, dtype=np.float32)
        info = info.copy()
        info.update({
            "takeover": False,
            "takeover_start": False,
            "takeover_end": False,
            "agent_action": zero_action.copy(),
            "raw_action": zero_action.copy(),
            "human_action": zero_action.copy(),
            "behavior_action": zero_action.copy(),
            "controller_type": self.controller_type,
            "takeover_cost": 0.0,
            "total_takeover_cost": 0.0,
            "total_cost": 0.0,
            "total_steps": 0,
            "takeover_rate": 0.0,
            "total_takeover_steps": 0,
            "takeover_start_count": 0,
            "first_takeover_step": None,
        })
        return obs, info

    def set_sequential_reset(self, enable: bool):
        """动态启用或禁用顺序重置"""
        if enable and not self.sequential_reset:
            # 启用顺序重置
            start_seed = self.config.get("start_seed", 100)
            num_scenarios = self.config.get("num_scenarios", 20)
            self.map_seeds = list(range(start_seed, start_seed + num_scenarios))
            self.current_map_index = 0
            self.sequential_reset = True
            logger.info(f"Sequential reset ENABLED: {self.map_seeds}")
        elif not enable and self.sequential_reset:
            # 禁用顺序重置
            self.sequential_reset = False
            self.map_seeds = []
            self.current_map_index = 0
            logger.info("Sequential reset DISABLED: switching to random mode")
        else:
            return

    def step(self, actions):
        """环境步进，使用expB0的接管和指标系统"""
        self.agent_action = np.asarray(copy.copy(actions), dtype=np.float32)
        last_takeover = bool(self.takeover)
        
        # 使用expB0的接管策略处理动作
        if self.takeover_policy is not None and self.config.get("enable_takeover", False):
            result = self.takeover_policy.act(actions)
            processed_action = np.asarray(result['action'], dtype=np.float32)
            human_action = np.asarray(result.get('raw_action', actions), dtype=np.float32)
            takeover = bool(result['takeover'])
            takeover_start = bool(result.get('takeover_start', False))
            
            # 更新接管状态
            self.takeover = takeover
            
            # 检测接管开始
            if not last_takeover and self.takeover:
                self.takeover_start_count += 1
                if self.first_takeover_step is None:
                    self.first_takeover_step = self.total_steps
                    self.episode_first_takeover = self.total_steps
                    logger.info(f"[TTFT] 第一次接管发生在第 {self.total_steps} 步")
            
            # 更新接管步数
            if self.takeover:
                self.total_takeover_steps += 1
            
            behavior_action = processed_action
        else:
            self.takeover = False
            takeover = False
            takeover_start = False
            human_action = self.agent_action.copy()
            behavior_action = self.agent_action.copy()
        
        takeover_end = bool(last_takeover and not self.takeover)
        
        # 调用父类step
        obs, reward, terminated, truncated, base_info = super().step(behavior_action)
        info = base_info.copy()
        info.setdefault("raw_action", human_action.copy())
        info.setdefault("human_action", human_action.copy())
        info.setdefault("behavior_action", behavior_action.copy())
        info.setdefault("agent_action", self.agent_action.copy())
        info.setdefault("takeover_start", takeover_start)
        info.setdefault("takeover_end", takeover_end)
        info.setdefault("takeover", self.takeover)
        
        # 🔄 强制检查出界和碰撞终止
        if not terminated and not truncated:
            # 检查各种可能的出界键名
            out_of_road = (
                info.get("out_of_road", False) or 
                info.get("out_of_route", False) or
                info.get("is_out_of_road", False)
            )
            # 检查各种可能的碰撞键名
            crash = (
                info.get("crash", False) or
                info.get("collision", False) or
                info.get("is_crash", False)
            )
            
            if out_of_road:
                print(f"[FORCE] 出界检测到，强制终止episode")
                terminated = True
            if crash:
                print(f"[FORCE] 碰撞检测到，强制终止episode")
                terminated = True
        
        # 添加expB0的接管和指标信息
        takeover_cost = self.get_takeover_cost(info)
        self.total_takeover_cost += takeover_cost
        current_cost = float(info.get("cost", 0.0))
        self.total_cost += current_cost
        self.takeover_recorder.append(self.takeover)
        takeover_rate = np.mean(np.asarray(self.takeover_recorder, dtype=np.float32)) if self.takeover_recorder else 0.0

        info.setdefault("action", behavior_action)
        
        info.update({
            "takeover": self.takeover,
            "takeover_start": takeover_start,
            "takeover_end": takeover_end,
            "agent_action": self.agent_action.copy(),
            "raw_action": human_action.copy(),
            "human_action": human_action.copy(),
            "behavior_action": behavior_action.copy(),
            "controller_type": self.controller_type,
            "takeover_cost": takeover_cost,
            "total_takeover_cost": self.total_takeover_cost,
            "total_cost": self.total_cost,
            "total_steps": self.total_steps,
            "takeover_rate": float(takeover_rate),
            "total_takeover_steps": self.total_takeover_steps,
            "takeover_start_count": self.takeover_start_count,
            "first_takeover_step": self.first_takeover_step,
        })
        
        # 暂停功能
        while self.in_pause:
            self.engine.taskMgr.step()
        
        # 渲染界面信息
        if self.config.get("use_render", False):
            self._render_interface()
        
        self.total_steps += 1
        
        return obs, reward, terminated, truncated, info

    def _render_interface(self):
        """渲染界面信息，使用expB0的指标"""
        try:
            super().render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(list(self.takeover_recorder))) * 100) if self.takeover_recorder else "0.00%",
                    "PVP Steps": f"{self.total_takeover_steps}/{self.total_steps}",
                    "TTFT": self.first_takeover_step if self.first_takeover_step is not None else "N/A",
                    "Pause": "Press E",
                }
            )
        except Exception as e:
            logger.warning(f"Failed to render interface: {e}")

    def stop(self):
        """切换暂停状态"""
        self.in_pause = not self.in_pause
        logger.info(f"Pause {'enabled' if self.in_pause else 'disabled'}")

    def setup_engine(self):
        """设置引擎，添加键盘控制"""
        super().setup_engine()
        # 添加E键暂停功能
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        """计算接管成本，使用expB0的方法"""
        if not self.config.get("cos_similarity", False):
            return 1.0 if self.takeover else 0.0
        
        # 使用余弦相似度计算接管成本
        takeover_action = safe_clip(np.array(info.get("raw_action", [0, 0])), -1, 1)
        agent_action = safe_clip(np.array(self.agent_action), -1, 1)
        
        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        
        return 1 - cos_dist if self.takeover else 0.0

    def get_pvp_stats(self) -> Dict[str, Any]:
        """获取expB0的PVP统计信息"""
        if self.total_steps > 0:
            takeover_rate = self.total_takeover_steps / self.total_steps
        else:
            takeover_rate = 0.0
        
        return {
            "takeover_rate": takeover_rate,
            "total_takeover_steps": self.total_takeover_steps,
            "total_steps": self.total_steps,
            "takeover_start_count": self.takeover_start_count,
            "first_takeover_step": self.first_takeover_step,
            "episode_first_takeover": self.episode_first_takeover,
            "ttft_censored": self.ttft_censored,
            "total_takeover_cost": self.total_takeover_cost,
            "total_cost": self.total_cost,
        }
    
    @property
    def controller_type(self):
        """获取控制器类型"""
        return self.config.get("controller", "keyboard")
    
    @property
    def enable_takeover(self):
        """是否启用接管"""
        return self.config.get("enable_takeover", False)


def create_human_in_the_loop_env(
    controller: str = "steering_wheel",
    manual_control: bool = True,
    use_render: bool = True,
    start_seed: int = 100,
    num_scenarios: int = 20,
    traffic_density: float = 0.06,
    **kwargs
) -> HumanInTheLoopEnv:
    """
    创建Human-in-the-Loop环境
    
    Args:
        controller: 控制器类型
        manual_control: 是否启用手动控制
        use_render: 是否使用渲染
        start_seed: 起始种子
        num_scenarios: 场景数量
        traffic_density: 交通密度
        **kwargs: 其他配置参数
    
    Returns:
        HumanInTheLoopEnv实例
    """
    config = {
        "manual_control": False,  # 环境本身不处于纯手动控制模式
        "enable_takeover": manual_control,  # 启用接管功能
        "controller": controller,
        "use_render": use_render,
        "start_seed": start_seed,
        "num_scenarios": num_scenarios,
        "traffic_density": traffic_density,
    }
    
    # 添加其他配置
    config.update(kwargs)
    
    # 创建环境
    env = HumanInTheLoopEnv(config)
    
    logger.info(f"Created HumanInTheLoopEnv")
    logger.info(f"Controller: {controller}")
    logger.info(f"Manual control: {manual_control}")
    logger.info(f"Map seeds: {start_seed} - {start_seed + num_scenarios - 1}")
    logger.info(f"Traffic density: {traffic_density}")
    
    return env


if __name__ == "__main__":
    # 测试环境
    env = create_human_in_the_loop_env(
        controller="steering_wheel",
        manual_control=True,
        use_render=True,
        start_seed=100,
        num_scenarios=20,
    )
    
    obs, info = env.reset()
    print("Environment reset successfully!")
    print(f"Controller: {env.controller_type}")
    print(f"Takeover enabled: {env.enable_takeover}")
    print(f"Map seeds: {env.config['start_seed']} - {env.config['start_seed'] + env.config['num_scenarios'] - 1}")
    
    try:
        while True:
            action = [0, 0]  # 零动作，让接管策略处理
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode finished! Stats: {env.get_pvp_stats()}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()
        print(f"Final stats: {env.get_pvp_stats()}")
