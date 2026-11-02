"""
SharedControlMonitor for PVP data collection

Canonical PVP action semantics (问题30)
-----------------------------------------
action_agent / action_novice:
    raw action from learning agent (a_n)
    
action_human:
    human control input (a_h), extracted from info['human_action'] or info['raw_action']
    
action_behavior:
    action actually executed in environment (a_b), extracted from info['behavior_action'] or info['action']

Priority rules (问题4):
- action_behavior: info['behavior_action'] > info['action'] > action_agent
- action_human: info['human_action'] > info['raw_action'] > action_agent
- intervention: info['takeover']
- takeover_start: info['takeover_start']
- takeover_end: info['takeover_end']
"""
import logging
import os
import pickle
import builtins
from collections import deque
from typing import Dict, List, Any, Optional

import gymnasium as gym
import numpy as np

from relax.utils.experience import Experience

logger = logging.getLogger(__name__)


class SharedControlMonitor(gym.Wrapper):
    """
    Store shared control data from multiple episodes for PVP training.
    Adapted from PVP-main to work with DACER Experience format.
    
    问题5：所有 Experience 创建统一使用 Experience.create_pvp_experience 工厂方法。
    """
    def __init__(self, env: gym.Env, folder: str = 'recorded_data', prefix: str = 'data', save_freq: int = 1000, 
                 pre_takeover_window: int = 0):
        super(SharedControlMonitor, self).__init__(env)
        self.data = {
            'observation': [],
            'next_observation': [],  # Add explicit next_observation tracking
            'action_agent': [],      # The action from learning agent (novice action a_n)
            'action_behavior': [],    # The action applied to environment (a_b)
            'action_human': [],       # The action from human intervention (a_h)
            'reward': [],
            'done': [],
            'terminated': [],
            'truncated': [],
            'intervention': [],      # Intervention flag (0/1)
            'takeover_start': [],    # Takeover start flag
            'takeover_end': [],      # Takeover end flag
            'info': [],
            'episode_count': [],
            'step_count': [],
        }
        self.step_count = 0
        self.last_save_step = 0
        self.save_freq = save_freq
        self.folder = folder
        self.prefix = prefix
        self.episode_count = 0
        self.last_observation = None
        
        # Optional soft-context ablation: store recent transitions only when explicitly enabled.
        # The main DOVEER EnergyRank objective uses true intervention pairs only;
        # pre-takeover samples are never used as rejected-action pairs by default.
        self.pre_takeover_window = max(0, int(pre_takeover_window))
        self.recent_transitions = deque(maxlen=self.pre_takeover_window)
        self.pre_takeover_experiences = []  # Optional pre-takeover soft-context experiences
        
    def step(self, action):
        """Override step to record PVP data"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Fixed: Safe done combination for both scalar and array inputs
        def safe_done(terminated, truncated):
            if isinstance(terminated, np.ndarray) or isinstance(truncated, np.ndarray):
                if isinstance(terminated, np.ndarray):
                    term_array = terminated
                else:
                    term_array = np.array([terminated])
                    
                if isinstance(truncated, np.ndarray):
                    trunc_array = truncated
                else:
                    trunc_array = np.array([truncated])
                    
                return np.logical_or(term_array, trunc_array)
            else:
                return bool(terminated) or bool(truncated)
        
        done = safe_done(terminated, truncated)
        
        # Record PVP data
        self._record_step(self.last_observation, observation, action, reward, terminated, truncated, info, done)
        self._check_and_save_data()
        self.last_observation = observation
        
        if done:
            self.episode_count += 1
            
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Override reset to handle new gymnasium API"""
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_observation = obs
        return obs, info

    def _record_step(self, observation, next_observation, action, reward, terminated, truncated, info, done):
        """Record a single step of PVP data with BC-Boost enhancement"""
        if observation is not None:
            self.data['observation'].append(observation)
        if next_observation is not None:
            self.data['next_observation'].append(next_observation)
        self.data['reward'].append(reward)
        self.data['terminated'].append(terminated)
        self.data['truncated'].append(truncated)
        self.data['done'].append(done)  # Use computed safe_done instead of terminated or truncated
        self.data['episode_count'].append(self.episode_count)
        self.data['step_count'].append(self.step_count)

        # 问题4：Extract PVP-specific data from info using canonical priority rules
        action_agent = np.asarray(action).copy()
        if 'behavior_action' in info:
            action_behavior = np.asarray(info['behavior_action'])
        elif 'action' in info:
            action_behavior = np.asarray(info['action'])
        else:
            action_behavior = action_agent.copy()
        if 'human_action' in info:
            action_human = np.asarray(info['human_action'])
        elif 'raw_action' in info:
            action_human = np.asarray(info['raw_action'])
        else:
            action_human = action_agent.copy()

        self.data['action_behavior'].append(action_behavior.copy())
        self.data['action_agent'].append(action_agent.copy())
        self.data['action_human'].append(action_human.copy())
        
        # Intervention flags
        intervention = float(info.get('takeover', False))
        takeover_start = float(info.get('takeover_start', False))
        takeover_end = float(info.get('takeover_end', False))
        
        self.data['intervention'].append(intervention)
        self.data['takeover_start'].append(takeover_start)
        self.data['takeover_end'].append(takeover_end)
        self.data['info'].append(info)  # 添加 info 记录
        
        # Optional soft-context ablation: keep a short pre-takeover buffer only
        # when pre_takeover_window > 0. These samples are not part of the main
        # EnergyRank intervention-pair objective.
        current_transition = {
            'observation': observation,
            'next_observation': next_observation,
            'action_behavior': action_behavior.copy(),
            'action_novice': action_agent.copy(),
            'action_human': action_human.copy(),
            'reward': reward,
            'done': terminated or truncated,
            'intervention': intervention,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        
        # Store current transition in recent buffer only for the optional ablation.
        if self.pre_takeover_window > 0:
            self.recent_transitions.append(current_transition)
        
        # If enabled, takeover_start can generate optional pre-takeover BC data.
        if takeover_start and self.pre_takeover_window > 0:
            self._generate_pre_takeover_experiences(info)
        
        self.step_count += 1

    def _generate_pre_takeover_experiences(self, takeover_info: Dict[str, Any]):
        """Generate optional pre-takeover soft-context BC experiences.
        
        These samples are an ablation-only context augmentation. They do not
        define rejected autonomous actions and therefore must not be used in
        the main EnergyRank intervention-pair loss.
        """
        if self.pre_takeover_window <= 0 or len(self.recent_transitions) == 0:
            return
            
        # Get the expert (human) action that will be applied during takeover
        if 'human_action' in takeover_info:
            expert_action = np.asarray(takeover_info['human_action'])
        elif 'raw_action' in takeover_info:
            expert_action = np.asarray(takeover_info['raw_action'])
        elif self.recent_transitions and self.recent_transitions[-1]['action_human'] is not None:
            expert_action = np.asarray(self.recent_transitions[-1]['action_human'])
        else:
            return

        # Generate experiences for each recent transition, labeled with expert action
        for i, transition in enumerate(self.recent_transitions):
            if transition['observation'] is None or transition['next_observation'] is None:
                continue
            # Create expert-labeled experience
            # Use the expert action for both behavior and human fields (since expert is demonstrating)
            expert_experience = Experience.create_pvp_experience(
                obs=transition['observation'],
                a_novice=transition['action_novice'],
                a_human=expert_action,
                a_behavior=expert_action,
                reward=transition['reward'],
                next_obs=transition['next_observation'],
                done=transition['done'],
                intervention=0.0,
                stop_td=1.0 if i == len(self.recent_transitions) - 1 else 0.0,
                is_pre_takeover=1.0,
            )
            
            # Add to pre-takeover experiences with special metadata
            self.pre_takeover_experiences.append({
                'experience': expert_experience,
                'original_step': transition['step_count'],
                'takeover_step': self.step_count,
                'window_position': i,  # Position in the pre-takeover window (0 = oldest)
                'outcome': self._extract_outcome(takeover_info)
            })
        
        logger.info(f"Generated {len(self.recent_transitions)} pre-takeover expert experiences at step {self.step_count}")

    def _extract_outcome(self, info: Dict[str, Any]) -> int:
        """Extract outcome type from info for BC-Boost weighting"""
        if info.get('crash', False):
            return 1  # crash
        elif info.get('out_of_road', False):
            return 2  # out_of_road
        elif info.get('success', False) or info.get('arrive_dest', False):
            return 3  # success
        else:
            return 0  # normal

    def get_pre_takeover_experiences(self) -> List[Experience]:
        """Get all generated pre-takeover experiences"""
        return [item['experience'] for item in self.pre_takeover_experiences]
    
    def get_pre_takeover_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for pre-takeover experiences"""
        return self.pre_takeover_experiences.copy()
    
    def clear_pre_takeover_experiences(self):
        """Clear stored pre-takeover experiences"""
        self.pre_takeover_experiences.clear()
        logger.info("Cleared pre-takeover experiences")

    def _check_and_save_data(self):
        """Check if it's time to save data"""
        if self.step_count - self.last_save_step >= self.save_freq:
            self._save_data(num_save_steps=self.step_count - self.last_save_step)
            self.last_save_step = self.step_count

    def _save_data(self, num_save_steps):
        """Save collected data to pickle file"""
        # Convert lists to numpy arrays before saving
        save_data = {}
        for key in self.data:
            if len(self.data[key]) == 0:
                continue
            data_array = np.array(self.data[key])
            if data_array.shape[-1] == 1:
                data_array = data_array.reshape(-1)
            
            # 🔄 安全检查：避免断言错误
            if data_array.shape[0] != num_save_steps:
                logger.warning(f"Data length mismatch for {key}: expected {num_save_steps}, got {data_array.shape[0]}")
                # 使用实际数据长度
                num_save_steps = data_array.shape[0]
            
            save_data[key] = data_array

        file_name = f"{self.prefix}_step_{self.last_save_step}_{self.step_count}.pkl"
        file_path = os.path.join(self.folder, file_name)
        os.makedirs(self.folder, exist_ok=True)
        with builtins.open(file_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"PVP trajectory data from step {self.last_save_step} to {self.step_count} "
            f"(totally {self.step_count - self.last_save_step} steps) saved at {file_path}"
        )
        self.data = {key: [] for key in self.data}  # Reinitialize data as empty lists

    def get_pvp_experience_batch(self) -> List[Experience]:
        """Convert recorded data to DACER Experience format"""
        experiences = []
        
        if len(self.data['observation']) == 0:
            return experiences
            
        for i in range(len(self.data['observation'])):
            obs = self.data['observation'][i]
            action_behavior = self.data['action_behavior'][i]
            action_novice = self.data['action_agent'][i]
            action_human = self.data['action_human'][i]
            reward = self.data['reward'][i]
            done = self.data['done'][i]
            # Use explicit next_observation if available, otherwise fall back to observation[i+1]
            if i < len(self.data['next_observation']):
                next_obs = self.data['next_observation'][i]
            else:
                next_obs = self.data['observation'][i + 1] if i + 1 < len(self.data['observation']) else obs
            intervention = self.data['intervention'][i]
            takeover_start = self.data['takeover_start'][i]
            takeover_end = self.data['takeover_end'][i] if i < len(self.data['takeover_end']) else False
            
            # 统一stop_td定义：支持takeover_end边界
            # 优先使用环境显式提供的 takeover_end，再回退到序列推断
            if not takeover_end and i > 0:  # 不是第一步
                prev_intervention = self.data['intervention'][i-1]
                takeover_end = (prev_intervention == 1.0 and intervention == 0.0)
            
            # 统一stop_td定义：takeover_start OR takeover_end
            stop_td = 1.0 if (takeover_start or takeover_end) else 0.0
            
            exp = Experience.create_pvp_experience(
                obs=obs,
                a_novice=action_novice,
                a_human=action_human,
                a_behavior=action_behavior,
                reward=reward,
                next_obs=next_obs,
                done=done,
                intervention=intervention,
                stop_td=stop_td,
                is_pre_takeover=0.0,
            )
            experiences.append(exp)
            
        return experiences

    def __del__(self):
        """析构函数：程序退出时自动保存剩余数据"""
        try:
            if hasattr(self, 'data') and len(self.data.get('observation', [])) > 0:
                remaining_steps = len(self.data['observation'])
                if remaining_steps > 0:
                    logger.info(f"[AutoSave] 程序退出，自动保存剩余 {remaining_steps} 步数据")
                    self._save_data(num_save_steps=remaining_steps)
        except Exception as e:
            logger.error(f"[AutoSave] 自动保存失败: {e}")

    def close(self):
        """手动关闭时保存数据"""
        if hasattr(self, 'data') and len(self.data.get('observation', [])) > 0:
            remaining_steps = len(self.data['observation'])
            if remaining_steps > 0:
                logger.info(f"[Close] 手动保存剩余 {remaining_steps} 步数据")
                self._save_data(num_save_steps=remaining_steps)
        super().close()
