from collections import defaultdict
from typing import Callable

from numba import njit, types as nt
import numpy as np

class Accumulator:
    __slots__ = ("prefix", "buffer")

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.buffer = defaultdict(list)

    def add(self, key, value):
        self.buffer[key].append(value)

    def add_vec(self, key, value):
        self.buffer[key].extend(value)

    def add_all(self, data: dict):
        for key, value in data.items():
            self.add(key, value)

    def reset(self):
        self.buffer.clear()

    def log(self, log_fn: Callable[[str, float], None]):
        for key, values in self.buffer.items():
            key = key if not self.prefix else f"{self.prefix}/{key}"
            
            # Handle different types of values appropriately
            try:
                if len(values) == 0:
                    continue
                    
                # Check if values are numeric
                first_val = values[0]
                if isinstance(first_val, (int, float, np.integer, np.floating)):
                    # Numeric values - compute mean
                    value = sum(values) / len(values)
                elif isinstance(first_val, (np.ndarray, list, tuple)):
                    # Array-like values - compute mean of all elements
                    all_values = []
                    for v in values:
                        if isinstance(v, (np.ndarray, list, tuple)):
                            all_values.extend(np.array(v).flatten().tolist())
                        else:
                            all_values.append(float(v))
                    value = np.mean(all_values) if all_values else 0.0
                elif isinstance(first_val, dict):
                    # Dict values - compute mean of numeric values
                    all_values = []
                    for v in values:
                        if isinstance(v, dict):
                            numeric_vals = [float(val) for val in v.values() 
                                          if isinstance(val, (int, float, np.integer, np.floating))]
                            all_values.extend(numeric_vals)
                    value = np.mean(all_values) if all_values else 0.0
                else:
                    # Other types - skip or use count
                    value = len(values)
                    
                # Ensure final value is a scalar float
                if hasattr(value, 'item'):
                    value = float(value.item())
                elif np.ndim(value) > 0:
                    value = float(np.mean(value))
                else:
                    value = float(value)
                    
            except Exception as e:
                # Ultimate fallback
                value = 0.0
                
            log_fn(key, value)

class SampleLog:
    __slots__ = ("sample_step", "sample_episode", "episode_return", "episode_length", "accumulator", "takeover_steps", "total_steps", "episode_takeover_count", "episode_ttft", "ttft_history", "ttft_censored_count", "crash_count", "out_of_road_count", "success_count", "total_episodes", "speed_sum", "speed_count")

    def __init__(self):
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.accumulator = Accumulator("sample")
        # 添加接管统计
        self.takeover_steps = 0
        self.total_steps = 0
        self.episode_takeover_count = 0
        # TTFT (Time-To-First Takeover) 跟踪
        self.episode_ttft = None
        self.ttft_history = []
        # 🎯 TTFT右删失统计
        self.ttft_censored_count = 0  # 右删失episode数量
        # 🛡️ 安全和性能指标
        self.crash_count = 0
        self.out_of_road_count = 0
        self.success_count = 0
        self.total_episodes = 0
        self.speed_sum = 0.0
        self.speed_count = 0

    def add(self, reward: float, terminated: bool, truncated: bool, info: dict):
        self.episode_return += reward
        self.episode_length += 1
        self.sample_step += 1
        
        # 统计接管信息
        self.total_steps += 1
        takeover = info.get('takeover', False)
        takeover_start = info.get('takeover_start', False)
        
        # 🛡️ 统计安全和性能指标
        if info.get('crash', False):
            self.crash_count += 1
        if info.get('out_of_road', False):
            self.out_of_road_count += 1
        if info.get('success', False) or info.get('arrive_dest', False):
            self.success_count += 1
        
        # 📊 统计速度（如果可用）
        if 'speed' in info:
            self.speed_sum += float(info['speed'])
            self.speed_count += 1
        elif 'velocity' in info:
            # 从velocity计算速度
            velocity = info['velocity']
            if isinstance(velocity, (list, tuple, np.ndarray)) and len(velocity) >= 2:
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                self.speed_sum += speed
                self.speed_count += 1
        
        if takeover:
            self.takeover_steps += 1
        if takeover_start:
            self.episode_takeover_count += 1
            
            # 🎯 记录第一次接管时间 (TTFT)
            if self.episode_ttft is None:
                self.episode_ttft = self.sample_step
        
        # 调试输出 - 清理
        # 移除频繁的调试输出

        done = terminated or truncated
        if done:
            self.sample_episode += 1
            self.total_episodes += 1
            self.accumulator.add("episode_return", float(self.episode_return))
            self.accumulator.add("episode_length", self.episode_length)
            # 记录接管统计
            if self.total_steps > 0:
                takeover_rate = self.takeover_steps / self.total_steps
                self.accumulator.add("takeover_rate", takeover_rate)
            self.accumulator.add("takeover_count", self.episode_takeover_count)
            self.accumulator.add("takeover_steps", self.takeover_steps)
            self.accumulator.add("total_steps", self.total_steps)
            
            # 🎯 记录TTFT到TensorBoard（学习进度指标）
            if self.episode_ttft is not None:
                self.accumulator.add("ttft", float(self.episode_ttft))
                self.ttft_history.append(self.episode_ttft)
            else:
                # 如果没有接管，记录episode长度作为TTFT（右删失）
                self.accumulator.add("ttft", float(self.episode_length))
                self.ttft_history.append(self.episode_length)
                self.ttft_censored_count += 1
            
            # 🛡️ 记录安全和性能指标到TensorBoard
            if self.total_episodes > 0:
                success_rate = self.success_count / self.total_episodes
                crash_rate = self.crash_count / self.total_episodes
                out_of_road_rate = self.out_of_road_count / self.total_episodes
                
                self.accumulator.add("success_rate", success_rate)
                self.accumulator.add("crash_rate", crash_rate)
                self.accumulator.add("out_of_road_rate", out_of_road_rate)
                
                # 📊 记录平均速度
                if self.speed_count > 0:
                    avg_speed = self.speed_sum / self.speed_count
                    self.accumulator.add("average_speed", avg_speed)
                
                # 📊 记录人工控制指标
                human_control_steps = self.takeover_steps
                human_control_ratio = human_control_steps / self.total_steps if self.total_steps > 0 else 0.0
                
                self.accumulator.add("human_control_steps", float(human_control_steps))
                self.accumulator.add("human_control_ratio", human_control_ratio)
                self.accumulator.add("takeovers_per_episode", float(self.episode_takeover_count))
            
            # 重置episode统计
            self.episode_return = 0.0
            self.episode_length = 0
            self.takeover_steps = 0
            self.total_steps = 0
            self.episode_takeover_count = 0
            self.episode_ttft = None
            # 🛡️ 不重置累计统计（保持整个训练过程的统计）
            # crash_count, out_of_road_count, success_count 保持累计

        return done

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, float(v), self.sample_step))
        self.accumulator.reset()


class VectorSampleLog:
    __slots__ = ("num_envs", "sample_step", "sample_episode", "episode_return", "episode_length", "accumulator")

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.accumulator = Accumulator("sample")

    def add(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict):
        self.episode_return += reward
        self.episode_length += 1
        self.sample_step += self.num_envs

        done = terminated | truncated
        done_count = np.count_nonzero(done)

        self.sample_episode += done_count
        self.accumulator.add_vec("episode_return", self.episode_return[done].tolist())
        self.accumulator.add_vec("episode_length", self.episode_length[done].tolist())
        self.episode_return[done] = 0.0
        self.episode_length[done] = 0

        return done_count > 0

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, float(v), self.sample_step))
        self.accumulator.reset()

class VectorFragmentSampleLog:
    __slots__ = ("num_envs", "fragment_length", "sample_step", "sample_episode", "episode_return", "episode_length", "accumulator")

    def __init__(self, num_envs: int, fragment_length: int):
        self.num_envs = num_envs
        self.fragment_length = fragment_length
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.accumulator = Accumulator("sample")

    def add(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict):
        done_count, complete_episode_return, complete_episode_length = process_fragment(reward, terminated, truncated, self.episode_return, self.episode_length, self.num_envs, self.fragment_length)
        self.sample_step += self.num_envs * self.fragment_length
        self.sample_episode += done_count
        self.accumulator.add_vec("episode_return", complete_episode_return.tolist())
        self.accumulator.add_vec("episode_length", complete_episode_length.tolist())
        return done_count > 0

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, float(v), self.sample_step))
        self.accumulator.reset()

@njit([(nt.float64[:, ::1], nt.boolean[:, ::1], nt.boolean[:, ::1], nt.float64[::1], nt.int64[::1], nt.int64, nt.int64)], cache=True)
def process_fragment(reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, episode_return: np.ndarray, episode_length: np.ndarray, num_envs: int, fragment_length: int):
    assert reward.shape == terminated.shape == truncated.shape == (num_envs, fragment_length)

    done = terminated | truncated
    done_count = np.count_nonzero(done)

    if done_count > 0:
        complete_episode_return = np.empty((done_count,), dtype=np.float64)
        complete_episode_length = np.empty((done_count,), dtype=np.int64)
        ptr = 0
        for i in range(num_envs):
            initial_return = episode_return[i]
            initial_length = episode_length[i]
            left = 0
            for j in range(fragment_length):
                if done[i, j]:
                    right = j + 1
                    complete_episode_return[ptr] = reward[i, left:right].sum() + initial_return
                    complete_episode_length[ptr] = right - left + initial_length
                    ptr += 1
                    left = right
                    initial_return = 0.0
                    initial_length = 0
            episode_return[i] = initial_return + reward[i, left:].sum()
            episode_length[i] = fragment_length - left + initial_length
    else:
        episode_return += reward.sum(axis=-1)
        episode_length += fragment_length

    return done_count, complete_episode_return, complete_episode_length

class UpdateLog:
    __slots__ = ("update_step", "accumulator")

    def __init__(self):
        self.update_step = 0
        self.accumulator = Accumulator("update")

    def add(self, metrics: dict):
        self.update_step += 1
        self.accumulator.add_all(metrics)

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, float(v), self.update_step))
        self.accumulator.reset()


class Interval:
    __slots__ = ("interval", "last_step")

    def __init__(self, interval: int):
        self.interval = interval
        self.last_step = 0

    def check(self, step: int) -> bool:
        if step - self.last_step >= self.interval:
            self.last_step = step
            return True
        return False
