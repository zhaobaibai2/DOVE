import numpy as np


class SafetyManager:
    def __init__(self, config: dict):
        self.takeover_threshold = config.get('takeover_threshold', 0.8)
        self.max_steering_angle = config.get('max_steering_angle', 200)
        self.emergency_brake_threshold = config.get('emergency_brake_threshold', 5.0)
        self.min_safe_distance = config.get('min_safe_distance', 2.0)
        self.max_abs_lateral_error = config.get('max_abs_lateral_error', self.min_safe_distance)
        self.min_obstacle_distance_brake = config.get('min_obstacle_distance_brake', 0.0)
        self.obstacle_brake_speed = config.get('obstacle_brake_speed', self.emergency_brake_threshold)
        self.emergency_stop = False

        self.emergency_brakes = 0
        self.steer_violations = 0
        self.obstacle_brakes = 0

    def check_action_safety(self, action: np.ndarray, state: np.ndarray) -> bool:
        if len(action) >= 2:
            steer_angle_deg = float(action[1]) * 200.0
            if abs(steer_angle_deg) > self.max_steering_angle:
                self.steer_violations += 1
                return False

        if len(state) >= 3:
            norm_error_distance = float(state[1])
            carspeed_norm = float(state[2])

            raw_error_distance = norm_error_distance * 5.0
            raw_carspeed = carspeed_norm * 15.0

            if abs(raw_error_distance) > self.max_abs_lateral_error and raw_carspeed > self.emergency_brake_threshold:
                self.emergency_brakes += 1
                return False

        if len(state) >= 244 and self.min_obstacle_distance_brake > 0.0:
            obstacle_vec = np.asarray(state[4:244], dtype=np.float32)
            finite = obstacle_vec[np.isfinite(obstacle_vec)]
            if finite.size > 0:
                min_obstacle_distance = float(np.min(finite))
                raw_carspeed = float(state[2]) * 15.0 if len(state) >= 3 else 0.0
                if min_obstacle_distance < self.min_obstacle_distance_brake and raw_carspeed > self.obstacle_brake_speed:
                    self.obstacle_brakes += 1
                    return False

        return True

    def get_safety_stats(self) -> dict:
        return {
            'emergency_brakes': self.emergency_brakes,
            'steer_violations': self.steer_violations,
            'obstacle_brakes': self.obstacle_brakes,
            'emergency_stop': self.emergency_stop,
        }
