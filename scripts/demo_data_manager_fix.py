#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_data_manager.py - 演示数据管理器

功能：
1. 保存演示数据到文件
2. 从文件加载演示数据
3. 数据格式转换和验证
4. 支持增量保存和加载

使用方法：
# 保存数据
python demo_data_manager.py --action save --log_path logs/xxx --output demo_data.pkl

# 加载数据到训练
python demo_data_manager.py --action load --demo_file demo_data.pkl --log_path logs/xxx
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# 确保能找到当前实验目录下的本地包
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Local package imports (只导入必要的模块)
from relax_env.metadrive_fixed_pvp_env import MetaDriveFixedPVPEnv
from relax.trainer.pvp_off_policy import PVPOffPolicyTrainer
from relax.algorithm.pvp_dacer import PVPDACER
from relax.network.dacer import create_dacer_net
from relax.utils.experience import Experience


class DemoDataManager:
    """演示数据管理器"""
    
    def __init__(self):
        self.data_format_version = "1.0"
        
    def _extract_env_info(self, env) -> Dict[str, Any]:
        """提取环境信息"""
        try:
            env_info = {
                "env_type": type(env).__name__,
                "observation_space": {
                    "shape": env.observation_space.shape,
                    "dtype": str(env.observation_space.dtype)
                },
                "action_space": {
                    "shape": env.action_space.shape,
                    "dtype": str(env.action_space.dtype),
                    "low": float(env.action_space.low[0]) if hasattr(env.action_space, 'low') else None,
                    "high": float(env.action_space.high[0]) if hasattr(env.action_space, 'high') else None
                }
            }
            
            # 尝试获取更多环境配置
            if hasattr(env, 'config'):
                env_info["config"] = dict(env.config)
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'config'):
                env_info["config"] = dict(env.unwrapped.config)
                
            return env_info
        except Exception as e:
            print(f"[Warning] 无法提取环境信息: {e}")
            return {"error": str(e)}
    
    def extract_demo_data(self, trainer, human_only: bool = False) -> Dict[str, Any]:
        """从trainer提取演示数据
        
        Args:
            trainer: 训练器对象
            human_only: 如果为True，只提取人类演示数据（用于1a阶段）
        """
        print("[DataManager] 正在提取演示数据...")
        
        # 获取buffer中的演示数据
        buffer = trainer.buffer
        
        # 创建环境信息（如果trainer没有env，使用默认信息）
        if hasattr(trainer, 'env') and trainer.env is not None:
            env_info = self._extract_env_info(trainer.env)
        else:
            # 使用默认环境信息
            env_info = {
                "env_type": "MetaDriveFixedPVPEnv",
                "observation_space": {
                    "shape": [259],  # MetaDrive观察空间维度
                    "dtype": "float32"
                },
                "action_space": {
                    "shape": [2],
                    "dtype": "float32",
                    "low": -1.0,
                    "high": 1.0
                }
            }
        
        # 创建演示数据结构
        demo_data = {
            "metadata": {
                "total_samples": 0,
                "human_buffer_size": 0,
                "novice_buffer_size": 0,
                "extraction_time": str(Path.cwd()),
                "env_info": env_info,
                "human_only": human_only  # 标记是否只包含人类数据
            },
            "human_buffer": [],
            "novice_buffer": []
        }
        
        if human_only:
            # 只提取人类演示数据（用于1a阶段）
            print("[DataManager] 模式：只保存人类演示数据（1a阶段）")
            if hasattr(buffer, 'human_buffer'):
                print(f"[DataManager] 提取 {len(buffer.human_buffer)} 条人工演示数据")
                try:
                    human_buf = buffer.human_buffer
                    # 直接访问buffer的属性
                    for i in range(len(human_buf)):
                        try:
                            # 创建experience字典
                            experience_dict = {
                                "obs": self._convert_to_numpy(human_buf.obs[i]),
                                "action": self._convert_to_numpy(human_buf.actions_human[i]),  # 人工动作
                                "reward": float(human_buf.reward[i]) if hasattr(human_buf, 'reward') and np.isscalar(human_buf.reward[i]) else float(human_buf.reward[i].item()) if hasattr(human_buf, 'reward') else 0.0,
                                "next_obs": self._convert_to_numpy(human_buf.next_obs[i]) if hasattr(human_buf, 'next_obs') else None,
                                "done": bool(human_buf.done[i]) if hasattr(human_buf, 'done') else False,
                                "info": {},
                                "actions_human": self._convert_to_numpy(human_buf.actions_human[i]),
                                "actions_novice": self._convert_to_numpy(human_buf.actions_novice[i]) if hasattr(human_buf, 'actions_novice') else None,
                                "actions_behavior": self._convert_to_numpy(human_buf.actions_behavior[i]) if hasattr(human_buf, 'actions_behavior') else None,
                                "interventions": self._convert_to_numpy(human_buf.interventions[i]) if hasattr(human_buf, 'interventions') else None,
                                "stop_td": bool(human_buf.stop_td[i]) if hasattr(human_buf, 'stop_td') else False
                            }
                            demo_data["human_buffer"].append(experience_dict)
                        except Exception as e:
                            print(f"[Warning] 跳过无效的人工演示数据 {i}: {e}")
                            continue
                except Exception as e:
                    print(f"[Error] 提取人工演示数据失败: {e}")
            
            # 更新元数据
            demo_data["metadata"]["human_buffer_size"] = len(demo_data["human_buffer"])
            demo_data["metadata"]["total_samples"] = len(demo_data["human_buffer"])
            demo_data["metadata"]["novice_buffer_size"] = 0  # 人类模式，无新手数据
            
        else:
            # 提取所有数据（人类+新手）
            print("[DataManager] 模式：保存所有演示数据（人类+新手）")
            
            # 提取人类演示数据
            if hasattr(buffer, 'human_buffer'):
                print(f"[DataManager] 提取 {len(buffer.human_buffer)} 条人工演示数据")
                try:
                    human_buf = buffer.human_buffer
                    # 直接访问buffer的属性
                    for i in range(len(human_buf)):
                        try:
                            # 创建experience字典
                            experience_dict = {
                                "obs": self._convert_to_numpy(human_buf.obs[i]),
                                "action": self._convert_to_numpy(human_buf.actions_human[i]),  # 人工动作
                                "reward": float(human_buf.reward[i]) if hasattr(human_buf, 'reward') and np.isscalar(human_buf.reward[i]) else float(human_buf.reward[i].item()) if hasattr(human_buf, 'reward') else 0.0,
                                "next_obs": self._convert_to_numpy(human_buf.next_obs[i]) if hasattr(human_buf, 'next_obs') else None,
                                "done": bool(human_buf.done[i]) if hasattr(human_buf, 'done') else False,
                                "info": {},
                                "actions_human": self._convert_to_numpy(human_buf.actions_human[i]),
                                "actions_novice": self._convert_to_numpy(human_buf.actions_novice[i]) if hasattr(human_buf, 'actions_novice') else None,
                                "actions_behavior": self._convert_to_numpy(human_buf.actions_behavior[i]) if hasattr(human_buf, 'actions_behavior') else None,
                                "interventions": self._convert_to_numpy(human_buf.interventions[i]) if hasattr(human_buf, 'interventions') else None,
                                "stop_td": bool(human_buf.stop_td[i]) if hasattr(human_buf, 'stop_td') else False
                            }
                            demo_data["human_buffer"].append(experience_dict)
                        except Exception as e:
                            print(f"[Warning] 跳过无效的人工演示数据 {i}: {e}")
                            continue
                except Exception as e:
                    print(f"[Error] 提取人工演示数据失败: {e}")
            
            # 提取新手数据
            if hasattr(buffer, 'novice_buffer'):
                print(f"[DataManager] 提取 {len(buffer.novice_buffer)} 条新手数据")
                try:
                    novice_buf = buffer.novice_buffer
                    # 直接访问buffer的属性
                    for i in range(len(novice_buf)):
                        try:
                            # 创建experience字典
                            experience_dict = {
                                "obs": self._convert_to_numpy(novice_buf.obs[i]),
                                "action": self._convert_to_numpy(novice_buf.actions_novice[i]),  # 新手动作
                                "reward": float(novice_buf.reward[i]) if hasattr(novice_buf, 'reward') and np.isscalar(novice_buf.reward[i]) else float(novice_buf.reward[i].item()) if hasattr(novice_buf, 'reward') else 0.0,
                                "next_obs": self._convert_to_numpy(novice_buf.next_obs[i]) if hasattr(novice_buf, 'next_obs') else None,
                                "done": bool(novice_buf.done[i]) if hasattr(novice_buf, 'done') else False,
                                "info": {},
                                "actions_human": self._convert_to_numpy(novice_buf.actions_human[i]) if hasattr(novice_buf, 'actions_human') else None,
                                "actions_novice": self._convert_to_numpy(novice_buf.actions_novice[i]),
                                "actions_behavior": self._convert_to_numpy(novice_buf.actions_behavior[i]) if hasattr(novice_buf, 'actions_behavior') else None,
                                "interventions": self._convert_to_numpy(novice_buf.interventions[i]) if hasattr(novice_buf, 'interventions') else None,
                                "stop_td": bool(novice_buf.stop_td[i]) if hasattr(novice_buf, 'stop_td') else False
                            }
                            demo_data["novice_buffer"].append(experience_dict)
                        except Exception as e:
                            print(f"[Warning] 跳过无效的新手数据 {i}: {e}")
                            continue
                except Exception as e:
                    print(f"[Error] 提取新手数据失败: {e}")
            
            # 更新元数据
            demo_data["metadata"]["human_buffer_size"] = len(demo_data["human_buffer"])
            demo_data["metadata"]["novice_buffer_size"] = len(demo_data["novice_buffer"])
            demo_data["metadata"]["total_samples"] = len(demo_data["human_buffer"]) + len(demo_data["novice_buffer"])
        
        print(f"[DataManager] 数据提取完成:")
        print(f"  - 人工演示: {len(demo_data['human_buffer'])} 条")
        print(f"  - 新手数据: {len(demo_data['novice_buffer'])} 条")
        print(f"  - 总计: {len(demo_data['human_buffer']) + len(demo_data['novice_buffer'])} 条")
        
        return demo_data
    
    def _extract_env_info(self, env) -> Dict[str, Any]:
        """提取环境信息"""
        try:
            # 检查env是否为None
            if env is None:
                return {
                    "env_type": "MetaDriveFixedPVPEnv",
                    "observation_space": {
                        "shape": [259],  # MetaDrive观察空间维度
                        "dtype": "float32"
                    },
                    "action_space": {
                        "shape": [2],
                        "dtype": "float32",
                        "low": -1.0,
                        "high": 1.0
                    }
                }
            
            # 获取实际的观察和动作空间
            obs_space = getattr(env, 'observation_space', None)
            action_space = getattr(env, 'action_space', None)
            
            env_info = {
                "env_type": type(env).__name__,
                "observation_space": {
                    "shape": list(obs_space.shape) if obs_space is not None else [259],
                    "dtype": str(obs_space.dtype) if obs_space is not None else "float32"
                },
                "action_space": {
                    "shape": list(action_space.shape) if action_space is not None else [2],
                    "dtype": str(action_space.dtype) if action_space is not None else "float32",
                    "low": float(action_space.low[0]) if action_space is not None and hasattr(action_space, 'low') else -1.0,
                    "high": float(action_space.high[0]) if action_space is not None and hasattr(action_space, 'high') else 1.0
                }
            }
            
            # 尝试获取更多环境配置
            if hasattr(env, 'config'):
                env_info["config"] = dict(env.config)
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'config'):
                env_info["config"] = dict(env.unwrapped.config)
                
            return env_info
        except Exception as e:
            print(f"[Warning] 无法提取环境信息: {e}")
            return {"error": str(e)}
    
    def _extract_buffer_config(self, buffer) -> Dict[str, Any]:
        """提取buffer配置"""
        try:
            config = {
                "capacity": getattr(buffer, 'capacity', None),
                "batch_size": getattr(buffer, 'batch_size', None)
            }
            
            # 获取更多配置信息
            for attr in ['human_capacity', 'novice_capacity', 'alpha', 'beta']:
                if hasattr(buffer, attr):
                    config[attr] = getattr(buffer, attr)
                    
            return config
        except Exception as e:
            print(f"[Warning] 无法提取buffer配置: {e}")
            return {"error": str(e)}
    
    def _convert_experience_to_dict(self, experience) -> Dict[str, Any]:
        """将Experience对象转换为字典"""
        try:
            # 处理Experience对象
            if hasattr(experience, '__dict__'):
                return {
                    "obs": self._convert_to_numpy(experience.obs),
                    "action": self._convert_to_numpy(experience.action),
                    "reward": float(experience.reward) if hasattr(experience, 'reward') else None,
                    "next_obs": self._convert_to_numpy(experience.next_obs) if hasattr(experience, 'next_obs') else None,
                    "done": bool(experience.done) if hasattr(experience, 'done') else None,
                    "info": dict(experience.info) if hasattr(experience, 'info') else {},
                    "actions_human": self._convert_to_numpy(experience.actions_human) if hasattr(experience, 'actions_human') else None,
                    "actions_novice": self._convert_to_numpy(experience.actions_novice) if hasattr(experience, 'actions_novice') else None,
                    "interventions": self._convert_to_numpy(experience.interventions) if hasattr(experience, 'interventions') else None,
                    "stop_td": bool(experience.stop_td) if hasattr(experience, 'stop_td') else None
                }
            else:
                # 如果不是Experience对象，尝试直接转换
                return {
                    "data": self._convert_to_numpy(experience),
                    "type": str(type(experience))
                }
        except Exception as e:
            print(f"[Warning] 转换experience失败: {e}")
            return {"error": str(e), "raw_data": str(experience)}
    
    def _convert_to_numpy(self, data) -> Any:
        """将JAX/numpy数据转换为标准numpy"""
        try:
            if data is None:
                return None
            
            # JAX数组转换
            if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                if 'jax' in str(type(data)):
                    return np.array(data)
                else:
                    return np.asarray(data)
            
            # 列表/元组转换
            elif isinstance(data, (list, tuple)):
                return [self._convert_to_numpy(item) for item in data]
            
            # 字典转换
            elif isinstance(data, dict):
                return {k: self._convert_to_numpy(v) for k, v in data.items()}
            
            # 其他类型直接返回
            else:
                return data
                
        except Exception as e:
            print(f"[Warning] 数据转换失败: {e}")
            return data
    
    def save_demo_data(self, demo_data: Dict[str, Any], output_path: str) -> bool:
        """保存演示数据到文件"""
        try:
            output_path = Path(output_path)
            
            # 检查输出路径是否是目录，如果是则添加默认文件名
            if output_path.is_dir():
                output_path = output_path / "demo_data.pkl"
                print(f"[DataManager] 输出路径是目录，使用: {output_path}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存主数据文件
            with open(output_path, 'wb') as f:
                pickle.dump(demo_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 保存元数据JSON文件（便于查看）
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(demo_data['metadata'], f, indent=2)
            
            print(f"[DataManager] 数据已保存:")
            print(f"  - 主文件: {output_path}")
            print(f"  - 元数据: {metadata_path}")
            print(f"  - 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"[Error] 保存数据失败: {e}")
            return False
    
    def load_demo_data(self, demo_file: str) -> Optional[Dict[str, Any]]:
        """从文件加载演示数据"""
        try:
            demo_file = Path(demo_file)
            if not demo_file.exists():
                print(f"[Error] 文件不存在: {demo_file}")
                return None
            
            print(f"[DataManager] 正在加载演示数据: {demo_file}")
            
            with open(demo_file, 'rb') as f:
                demo_data = pickle.load(f)
            
            # 验证数据格式
            if not self._validate_demo_data(demo_data):
                print(f"[Error] 数据格式验证失败")
                return None
            
            print(f"[DataManager] 数据加载完成:")
            print(f"  - 版本: {demo_data.get('version', 'unknown')}")
            print(f"  - 人工演示: {len(demo_data.get('human_buffer', []))} 条")
            print(f"  - 新手数据: {len(demo_data.get('novice_buffer', []))} 条")
            print(f"  - 提取时间: {demo_data.get('metadata', {}).get('extraction_time', 'unknown')}")
            
            return demo_data
            
        except Exception as e:
            print(f"[Error] 加载数据失败: {e}")
            return None
    
    def _validate_demo_data(self, demo_data: Dict[str, Any]) -> bool:
        """验证演示数据格式"""
        try:
            # 检查必要字段
            required_fields = ['metadata', 'human_buffer', 'novice_buffer']
            for field in required_fields:
                if field not in demo_data:
                    print(f"[Error] 缺少必要字段: {field}")
                    return False
            
            # 检查数据类型
            if not isinstance(demo_data['human_buffer'], list):
                print(f"[Error] human_buffer应该是list类型")
                return False
            
            if not isinstance(demo_data['novice_buffer'], list):
                print(f"[Error] novice_buffer应该是list类型")
                return False
            
            # 检查是否有数据
            total_samples = len(demo_data['human_buffer']) + len(demo_data['novice_buffer'])
            if total_samples == 0:
                print(f"[Warning] 没有找到任何演示数据")
                return False
            
            return True
            
        except Exception as e:
            print(f"[Error] 数据验证失败: {e}")
            return False
    
    def load_demo_data_to_trainer(self, trainer, demo_data: Dict[str, Any]) -> bool:
        """将演示数据加载到trainer的buffer中"""
        try:
            print("[DataManager] 正在将演示数据加载到trainer...")
            
            buffer = trainer.buffer
            
            # 加载human_buffer
            if demo_data['human_buffer']:
                print(f"[DataManager] 加载 {len(demo_data['human_buffer'])} 条人工演示数据")
                for i, demo_dict in enumerate(demo_data['human_buffer']):
                    try:
                        experience = self._convert_dict_to_experience(demo_dict)
                        if experience is not None:
                            buffer.human_buffer.add(experience)
                    except Exception as e:
                        print(f"[Warning] 跳过human_buffer[{i}]: {e}")
            
            # 加载novice_buffer
            if demo_data['novice_buffer']:
                print(f"[DataManager] 加载 {len(demo_data['novice_buffer'])} 条新手数据")
                for i, demo_dict in enumerate(demo_data['novice_buffer']):
                    try:
                        experience = self._convert_dict_to_experience(demo_dict)
                        if experience is not None:
                            buffer.novice_buffer.add(experience)
                    except Exception as e:
                        print(f"[Warning] 跳过novice_buffer[{i}]: {e}")
            
            print(f"[DataManager] 数据加载完成:")
            print(f"  - Buffer总大小: {len(buffer)}")
            print(f"  - Human buffer: {len(buffer.human_buffer) if hasattr(buffer, 'human_buffer') else 'N/A'}")
            print(f"  - Novice buffer: {len(buffer.novice_buffer) if hasattr(buffer, 'novice_buffer') else 'N/A'}")
            
            return True
            
        except Exception as e:
            print(f"[Error] 加载数据到trainer失败: {e}")
            return False
    
    def _convert_dict_to_experience(self, demo_dict: Dict[str, Any]):
        """将字典转换回Experience对象"""
        try:
            from relax.utils.experience import Experience
            
            # 确保数据是numpy格式
            obs = self._ensure_numpy(demo_dict.get('obs'))
            action = self._ensure_numpy(demo_dict.get('action'))
            reward = demo_dict.get('reward', 0.0)
            next_obs = self._ensure_numpy(demo_dict.get('next_obs'))
            done = demo_dict.get('done', False)
            
            # PVP相关字段
            actions_human = self._ensure_numpy(demo_dict.get('actions_human'))
            actions_novice = self._ensure_numpy(demo_dict.get('actions_novice'))
            actions_behavior = self._ensure_numpy(demo_dict.get('actions_behavior', action))  # 默认使用action
            interventions = self._ensure_numpy(demo_dict.get('interventions'))
            stop_td = demo_dict.get('stop_td', False)
            is_pre_takeover = demo_dict.get('is_pre_takeover', 0.0)  # 默认不是pre-takeover
            is_demo = demo_dict.get('is_demo', 0.0)
            if 'pair_ok' in demo_dict:
                pair_ok = demo_dict.get('pair_ok')
            else:
                try:
                    gap = np.linalg.norm((np.asarray(actions_human if actions_human is not None else actions_behavior) - np.asarray(actions_novice)).reshape(-1))
                    has_valid_pair = gap > 1e-8
                except Exception:
                    has_valid_pair = False
                pair_ok = float(
                    demo_dict.get('interventions', 0.0) is not None
                    and np.max(np.asarray(interventions, dtype=np.float32)) > 0.5
                    and float(is_pre_takeover) <= 0.5
                    and float(is_demo) <= 0.5
                    and has_valid_pair
                )
            
            # 创建Experience对象
            return Experience(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                next_obs=next_obs,
                actions_behavior=actions_behavior,
                actions_novice=actions_novice,
                actions_human=actions_human,
                interventions=interventions,
                stop_td=stop_td,
                is_pre_takeover=is_pre_takeover,
                is_demo=is_demo,
                pair_ok=pair_ok,
            )
            
        except Exception as e:
            print(f"[Warning] 转换dict到experience失败: {e}")
            return None
    
    def _ensure_numpy(self, data) -> Any:
        """确保数据是numpy格式"""
        if data is None:
            return None
        
        try:
            if isinstance(data, np.ndarray):
                return data
            elif hasattr(data, 'shape'):  # JAX array
                return np.array(data)
            elif isinstance(data, (list, tuple)):
                return np.array(data)
            else:
                return data
        except Exception:
            return data


def save_demo_data_action(args):
    """保存演示数据"""
    try:
        print(f"[DataManager] 开始保存演示数据...")
        print(f"  - 日志路径: {args.log_path}")
        print(f"  - 输出文件: {args.output}")
        
        # 检查日志路径
        log_path = Path(args.log_path)
        if not log_path.exists():
            print(f"[Error] 日志路径不存在: {log_path}")
            return False
        
        # 查找buffer文件（优先查找两阶段训练的演示数据）
        possible_files = [
            log_path / "buffer_stage1a_demo_only.pkl",  # 两阶段训练的演示数据
            log_path / "buffer_after_stage1a.pkl",     # 旧版本
            log_path / "buffer_demo_emergency_save.pkl",  # 紧急保存
            log_path / "buffer.pkl",                    # 单阶段训练
            log_path / "replay_buffer.pkl"
        ]
        
        buffer_file = None
        for file in possible_files:
            if file.exists():
                buffer_file = file
                print(f"[DataManager] 找到buffer文件: {file}")
                break
        
        if buffer_file is None:
            print(f"[Error] 未找到任何buffer文件")
            print(f"请确保训练已经运行并生成了buffer文件")
            return False
        
        # 加载buffer数据并提取演示数据
        try:
            import pickle
            with open(buffer_file, 'rb') as f:
                buffer = pickle.load(f)
            
            # 创建DemoDataManager实例并提取数据
            manager = DemoDataManager()
            
            # 创建虚拟trainer用于提取数据
            class DummyTrainer:
                def __init__(self, buffer, env=None):
                    self.buffer = buffer
                    self.env = env  # 传递环境信息（如果有的话）
            
            # 尝试从buffer中获取环境信息（如果trainer没有env）
            env_for_extraction = None
            if hasattr(buffer, 'env') and buffer.env is not None:
                env_for_extraction = buffer.env
            
            dummy_trainer = DummyTrainer(buffer, env_for_extraction)
            demo_data = manager.extract_demo_data(dummy_trainer, human_only=getattr(args, 'human_only', False))
            
            # 保存演示数据
            success = manager.save_demo_data(demo_data, args.output)
            
        except Exception as e:
            print(f"[Error] 加载buffer失败: {e}")
            return False
        
        if success:
            print(f"[DataManager] ✅ 演示数据保存成功!")
            print(f"现在你可以使用以下命令加载数据:")
            print(f"python scripts/train_pvp_dacer_metadrive_off.py --demo_file {args.output} --skip_demo_collection")
        else:
            print(f"[DataManager] ❌ 演示数据保存失败!")
        
        return success
        
    except Exception as e:
        print(f"[Error] 保存演示数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_demo_data_action(args):
    """加载演示数据到训练"""
    try:
        print(f"[DataManager] 开始加载演示数据...")
        print(f"  - 演示文件: {args.demo_file}")
        print(f"  - 目标路径: {args.log_path}")
        
        # 检查文件
        demo_file = Path(args.demo_file)
        if not demo_file.exists():
            print(f"[Error] 演示文件不存在: {demo_file}")
            return False
        
        # 加载演示数据
        manager = DemoDataManager()
        demo_data = manager.load_demo_data(args.demo_file)
        if demo_data is None:
            print(f"[Error] 加载演示数据失败")
            return False
        
        # 保存到训练目录，使用训练脚本期望的文件名
        output_file = Path(args.log_path) / "integrated_demo_data.pkl"
        manager.save_demo_data(demo_data, output_file)
        
        print(f"[DataManager] ✅ 演示数据加载成功!")
        print(f"数据已保存到: {output_file}")
        print(f"现在可以在训练时使用这些数据")
        
        return True
        
    except Exception as e:
        print(f"[Error] 加载演示数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="演示数据管理器")
    parser.add_argument("--action", type=str, choices=["save", "load"], required=True,
                       help="操作类型: save(保存) 或 load(加载)")
    
    # 保存相关参数
    parser.add_argument("--log_path", type=str, help="训练日志路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--human_only", action='store_true', help="只保存人类演示数据（用于1a阶段）")
    
    # 加载相关参数  
    parser.add_argument("--demo_file", type=str, help="演示数据文件路径")
    
    args = parser.parse_args()
    
    if args.action == "save":
        if not args.log_path or not args.output:
            print("[Error] 保存操作需要指定 --log_path 和 --output")
            return
        save_demo_data_action(args)
    
    elif args.action == "load":
        if not args.demo_file or not args.log_path:
            print("[Error] 加载操作需要指定 --demo_file 和 --log_path")
            return
        load_demo_data_action(args)


def load_integrated_demo_data(demo_file: str, trainer) -> bool:
    """加载集成的演示数据到trainer"""
    try:
        print(f"[DataManager] Loading integrated demo data from: {demo_file}")
        
        # 创建演示数据管理器
        manager = DemoDataManager()
        
        # 加载演示数据
        demo_data = manager.load_demo_data(demo_file)
        if demo_data is None:
            print(f"[Error] Failed to load demo data from {demo_file}")
            return False
        
        # 加载到trainer
        success = manager.load_demo_data_to_trainer(trainer, demo_data)
        
        if success:
            print(f"[DataManager] Successfully loaded integrated demo data")
            print(f"   - Total buffer size: {len(trainer.buffer)}")
            print(f"   - Human buffer: {len(trainer.buffer.human_buffer)}")
            print(f"   - Novice buffer: {len(trainer.buffer.novice_buffer)}")
            return True
        else:
            print(f"[Error] Failed to load demo data to trainer")
            return False
            
    except Exception as e:
        print(f"[Error] Exception in load_integrated_demo_data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
