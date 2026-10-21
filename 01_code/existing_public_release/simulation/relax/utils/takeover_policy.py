"""
Takeover Policy for Human-in-the-Loop Control
Adapted from PVP-main for DACER integration
"""
import numpy as np
from typing import Dict, Any, Optional


def _safe_local_attr(obj, name: str, default=None):
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return default


def _get_env_agent_vehicle(env):
    base = env
    for _ in range(5):
        next_env = _safe_local_attr(base, "env", None)
        if next_env is None:
            break
        base = next_env

    agent = _safe_local_attr(base, "agent", None)
    if agent is not None:
        return agent

    return _safe_local_attr(base, "vehicle", None)


class TakeoverPolicyWithoutBrake:
    """
    Takeover policy that switches between agent and human control.
    
    Control logic:
    - Hold takeover button -> takeover (intervention = 1)
    - Release takeover button -> end takeover (intervention = 0)
    - During takeover: human action is passed through
    - During normal operation: agent action is passed through
    """
    
    def __init__(self, 
                 env, 
                 takeover_button: str = "A",  # Button for takeover toggle
                 deadzone: float = 0.1):
        """
        Args:
            env: The environment
            takeover_button: Which button triggers takeover
            deadzone: Deadzone for joystick inputs
        """
        self.env = env
        self.takeover_button = takeover_button
        self.deadzone = deadzone
        self.takeover = False
        self.last_button_state = False
        self.current_human_action = np.array([0.0, 0.0], dtype=np.float32)  # [steer, throttle/brake]
        self._previously_pressed = set()  # FIXED: Initialize button state tracking
        
    def act(self, agent_action: np.ndarray) -> Dict[str, Any]:
        """
        Decide which action to use based on takeover state.
        
        Args:
            agent_action: Action from learning agent (a_n)
            
        Returns:
            Dict with:
            - 'action': action actually executed in env (a_b)
            - 'behavior_action': alias of executed action (a_b)
            - 'agent_action': action proposed by agent (a_n)
            - 'human_action': current human input (a_h)
            - 'raw_action': kept as backward-compatible alias of human_action
            - 'takeover': Current takeover state
            - 'takeover_start': Takeover just started
        """
        # Get human input from environment
        human_action = self._get_human_input()
        
        # Check for takeover button - NEW:持续按压式接管
        button_pressed = self._get_button_state(self.takeover_button)
        takeover_start = False
        takeover_end = False

        # NEW LOGIC: 只有持续按住按钮时才接管
        if button_pressed and not self.last_button_state:
            # 按钮刚按下 - 开始接管
            self.takeover = True
            takeover_start = True
            print(f"[INFO] Takeover started (button pressed)!")
        elif not button_pressed and self.last_button_state:
            # 按钮刚松开 - 结束接管
            self.takeover = False
            takeover_end = True
            print(f"[INFO] Takeover ended (button released)!")
        # 如果按钮持续按住或持续松开，保持当前状态
            
        self.last_button_state = button_pressed
        
        # Store current human action
        self.current_human_action = human_action.copy()
        
        # Determine behavior action
        if self.takeover:
            # Human control during takeover
            behavior_action = human_action
        else:
            # Agent control during normal operation
            behavior_action = agent_action
            
        return {
            'action': behavior_action,
            'behavior_action': behavior_action,
            'agent_action': agent_action,
            'human_action': human_action,
            'raw_action': human_action,  # backward-compatible alias: always means human input
            'takeover': self.takeover,
            'takeover_start': takeover_start,
            'takeover_end': takeover_end,
        }
    
    def _get_human_input(self) -> np.ndarray:
        """
        Get human input from controller using correct MetaDrive APIs.
        """
        # Method 1: Try to use MetaDrive's KeyboardController directly
        try:
            from metadrive.engine.core.manual_controller import KeyboardController
            vehicle = _get_env_agent_vehicle(self.env)
            if vehicle is not None:
                # Create a controller instance if needed
                if not hasattr(self, '_keyboard_controller'):
                    self._keyboard_controller = KeyboardController(pygame_control=False)
                
                action = self._keyboard_controller.process_input(vehicle)
                return np.array(action, dtype=np.float32)
        except Exception as e:
            pass
        
        # Method 2: Try to use Panda3D input system directly
        try:
            from panda3d.core import InputDevice
            if hasattr(self.env, 'engine') and self.env.engine:
                devices = self.env.engine.devices.getDevices(InputDevice.DeviceClass.gamepad)
                if devices:
                    device = devices[0]
                    steer = device.findAxis(InputDevice.Axis.left_x).value
                    throttle = device.findAxis(InputDevice.Axis.right_trigger).value
                    brake = device.findAxis(InputDevice.Axis.left_trigger).value
                    return np.array([steer, throttle - brake], dtype=np.float32)
        except Exception as e:
            pass
        
        # Method 3: Try to access MetaDrive's input system
        try:
            if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'inputs'):
                inputs = self.env.engine.inputs
                steer = 0.0
                throttle = 0.0
                
                if hasattr(inputs, 'isSet'):
                    if inputs.isSet('turnLeft'):
                        steer -= 1.0
                    if inputs.isSet('turnRight'):
                        steer += 1.0
                    if inputs.isSet('forward'):
                        throttle += 1.0
                    if inputs.isSet('reverse'):
                        throttle -= 1.0
                    
                    return np.array([steer, throttle], dtype=np.float32)
        except Exception as e:
            pass
        
        # Method 4: Fallback - try old API (for compatibility)
        if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'get_input_manager'):
            input_mgr = self.env.engine.get_input_manager()
            if input_mgr is not None:
                # Try to get steering and throttle
                steer = getattr(input_mgr, 'get_steering', lambda: 0.0)()
                throttle = getattr(input_mgr, 'get_throttle', lambda: 0.0)()
                brake = getattr(input_mgr, 'get_brake', lambda: 0.0)()
                
                # Combine throttle and brake (positive for throttle, negative for brake)
                combined_throttle = throttle - brake
                
                return np.array([steer, combined_throttle], dtype=np.float32)
        
        # Final fallback - return zero action
        import warnings
        warnings.warn("All input methods failed, returning zero action. Check controller connection.")
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def _get_button_state(self, button: str) -> bool:
        """
        Get state of specific button using correct MetaDrive APIs.
        For keyboard policy, only allow SPACE key for takeover toggle to prevent accidental activation.
        """
        # Method 1: Try to use Panda3D input system
        try:
            from panda3d.core import InputDevice
            if hasattr(self.env, 'engine') and self.env.engine:
                devices = self.env.engine.devices.getDevices(InputDevice.DeviceClass.gamepad)
                if devices:
                    device = devices[0]
                    # Check any button press for takeover
                    for i in range(20):  # Check first 20 buttons
                        if device.findButton(i).value:
                            return True
        except Exception as e:
            pass
        
        # Method 2: Try to access MetaDrive's input system
        try:
            if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'inputs'):
                inputs = self.env.engine.inputs
                if hasattr(inputs, 'isSet'):
                    # Only allow dedicated takeover keys for keyboard policy
                    if inputs.isSet('takeover') or inputs.isSet('space'):
                        return True
                    # For keyboard policy, DO NOT allow movement keys to trigger takeover
                    # This prevents accidental toggling when using arrow keys for control
        except Exception as e:
            pass
        
        # Method 3: Fallback - try old API
        if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'get_input_manager'):
            input_mgr = self.env.engine.get_input_manager()
            if input_mgr is not None:
                # Try to get button state
                get_button = getattr(input_mgr, 'get_button', lambda b: False)
                # Check only dedicated takeover keys
                for btn in ['SPACE', 'ENTER']:  # Removed movement keys
                    if get_button(btn):
                        return True
        
        return False
    
    def reset(self):
        """Reset takeover state for new episode"""
        self.takeover = False
        self.last_button_state = False
        self.current_human_action = np.array([0.0, 0.0], dtype=np.float32)


class SteeringWheelPolicy(TakeoverPolicyWithoutBrake):
    """
    Steering wheel specific takeover policy.
    使用拨片进行接管控制
    """
    
    def __init__(self, env):
        super().__init__(env, takeover_button="paddle", deadzone=0.05)
        self._last_steer = 0.0
        self._last_throttle = 0.0
        # 拨片状态跟踪
        self.left_paddle_pressed = False
        self.right_paddle_pressed = False
        
    def _get_human_input(self) -> np.ndarray:
        """Get input from steering wheel (G29) using correct APIs"""
        # Method 1: Try to use MetaDrive's SteeringWheelController
        try:
            from metadrive.engine.core.manual_controller import SteeringWheelController
            vehicle = _get_env_agent_vehicle(self.env)
            if vehicle is not None:
                if not hasattr(self, '_wheel_controller'):
                    self._wheel_controller = SteeringWheelController()
                
                action = self._wheel_controller.process_input(vehicle)
                return np.array(action, dtype=np.float32)
        except Exception as e:
            pass
        
        # Method 2: Direct Panda3D input for steering wheel
        try:
            from panda3d.core import InputDevice
            if hasattr(self.env, 'engine') and self.env.engine:
                devices = self.env.engine.devices.getDevices(InputDevice.DeviceClass.gamepad)
                if devices:
                    device = devices[0]
                    # G29 specific axis mapping
                    steer = -device.findAxis(InputDevice.Axis.left_x).value  # Invert for natural steering
                    throttle = -device.findAxis(InputDevice.Axis.right_y).value  # Right pedal
                    brake = -device.findAxis(InputDevice.Axis.left_y).value     # Left pedal
                    clutch = -device.findAxis(InputDevice.Axis.right_z).value  # Clutch
                    
                    # Convert to [steer, throttle/brake] format
                    combined_throttle = throttle * (1 - clutch) - brake * (1 - clutch)
                    
                    # Apply deadzone
                    if abs(steer) < self.deadzone:
                        steer = 0.0
                    if abs(combined_throttle) < self.deadzone:
                        combined_throttle = 0.0
                    
                    # Store for paddle detection
                    self._last_steer = steer
                    self._last_throttle = combined_throttle
                    
                    return np.array([steer, combined_throttle], dtype=np.float32)
        except Exception as e:
            pass
        
        # Fallback to parent implementation
        return super()._get_human_input()
    
    def _get_button_state(self, button: str) -> bool:
        """Get button state from G29 - 专门检测拨片按压状态"""
        try:
            from metadrive.engine.core.manual_controller import SteeringWheelController
            vehicle = _get_env_agent_vehicle(self.env)
            if vehicle is not None:
                if not hasattr(self, '_wheel_controller'):
                    self._wheel_controller = SteeringWheelController()
                
                controller = self._wheel_controller
                if hasattr(controller, 'joystick') and controller.joystick:
                    joystick = controller.joystick
                    
                    # 检查所有按钮状态
                    num_buttons = joystick.get_numbuttons()
                    pressed_buttons = []
                    for i in range(min(num_buttons, 25)):  # 检查前25个按钮
                        if joystick.get_button(i):
                            pressed_buttons.append(i)
                    
                    # G29拨片实际对应按钮4(左拨片)和5(右拨片)
                    left_paddle = joystick.get_button(4) if num_buttons > 4 else False
                    right_paddle = joystick.get_button(5) if num_buttons > 5 else False
                    
                    # 更新拨片状态
                    self.left_paddle_pressed = left_paddle
                    self.right_paddle_pressed = right_paddle
                    
                    # 任一拨片按下即触发接管
                    return left_paddle or right_paddle
        except Exception as e:
            pass
        
        # 备用方法：使用Panda3D输入系统检测拨片
        try:
            from panda3d.core import InputDevice
            if hasattr(self.env, 'engine') and self.env.engine:
                devices = self.env.engine.devices.getDevices(InputDevice.DeviceClass.gamepad)
                if devices:
                    device = devices[0]
                    
                    # 检查所有按钮状态
                    num_buttons = device.getNumButtons()
                    pressed_buttons = []
                    for i in range(min(num_buttons, 25)):  # 检查前25个按钮
                        btn = device.findButton(i)
                        if btn and btn.value:
                            pressed_buttons.append(i)
                    
                    # 调试输出
                    if pressed_buttons:
                        print(f"[DEBUG] Panda3D检测到按钮: {pressed_buttons}")
                    
                    # 检测拨片按钮（实际是按钮4、5）
                    left_paddle = device.findButton(4).value if device.findButton(4) else False
                    right_paddle = device.findButton(5).value if device.findButton(5) else False
                    
                    # 更新拨片状态
                    self.left_paddle_pressed = left_paddle
                    self.right_paddle_pressed = right_paddle
                    
                    # 调试输出
                    if left_paddle or right_paddle:
                        print(f"[DEBUG] Panda3D拨片检测: 左拨片={left_paddle}, 右拨片={right_paddle}")
                    
                    return left_paddle or right_paddle
        except Exception as e:
            pass
        
        # 如果检测不到拨片，回退到通用按钮检测
        fallback = super()._get_button_state(button)
        return fallback


class GamepadPolicy(TakeoverPolicyWithoutBrake):
    """
    Gamepad (Xbox) specific takeover policy.
    """
    
    def __init__(self, env):
        super().__init__(env, takeover_button="A", deadzone=0.1)
        
    def _get_human_input(self) -> np.ndarray:
        """Get input from Xbox controller using correct APIs"""
        # Method 1: Try to use Panda3D input system directly
        try:
            from panda3d.core import InputDevice
            if hasattr(self.env, 'engine') and self.env.engine:
                devices = self.env.engine.devices.getDevices(InputDevice.DeviceClass.gamepad)
                if devices:
                    device = devices[0]
                    # Xbox controller axis mapping
                    steer = device.findAxis(InputDevice.Axis.left_x).value
                    throttle = device.findAxis(InputDevice.Axis.right_trigger).value
                    brake = device.findAxis(InputDevice.Axis.left_trigger).value
                    
                    # Convert to [steer, throttle/brake] format
                    combined_throttle = throttle - brake
                    
                    # Apply deadzone
                    if abs(steer) < self.deadzone:
                        steer = 0.0
                    if abs(combined_throttle) < self.deadzone:
                        combined_throttle = 0.0
                        
                    return np.array([steer, combined_throttle], dtype=np.float32)
        except Exception as e:
            pass
        
        # Fallback to parent implementation
        return super()._get_human_input()
    
    def _get_button_state(self, button: str) -> bool:
        """Get button state from Xbox controller using correct APIs"""
        # Method 1: Try to use Panda3D input system
        try:
            from panda3d.core import InputDevice
            if hasattr(self.env, 'engine') and self.env.engine:
                devices = self.env.engine.devices.getDevices(InputDevice.DeviceClass.gamepad)
                if devices:
                    device = devices[0]
                    # Check Xbox buttons - any button can trigger takeover
                    for i in range(0, 15):  # Xbox controllers have ~15 buttons
                        if device.findButton(i).value:
                            return True
                    
                    # Check d-pad
                    hat = device.getHat(0)
                    if hat != (0, 0):
                        return True
        except Exception as e:
            pass
        
        # Fallback to parent implementation
        return super()._get_button_state(button)


class KeyboardPolicy(TakeoverPolicyWithoutBrake):
    """
    Keyboard specific takeover policy.
    """
    
    def __init__(self, env):
        super().__init__(env, takeover_button="SPACE", deadzone=0.1)
        
    def _get_human_input(self) -> np.ndarray:
        """Get input from keyboard using correct APIs"""
        # Method 1: Try to use MetaDrive's KeyboardController directly
        try:
            from metadrive.engine.core.manual_controller import KeyboardController
            vehicle = _get_env_agent_vehicle(self.env)
            if vehicle is not None:
                if not hasattr(self, '_keyboard_controller'):
                    self._keyboard_controller = KeyboardController(pygame_control=False)
                
                action = self._keyboard_controller.process_input(vehicle)
                return np.array(action, dtype=np.float32)
        except Exception as e:
            pass
        
        # Method 2: Try to access MetaDrive's input system directly
        try:
            if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'inputs'):
                inputs = self.env.engine.inputs
                steer = 0.0
                throttle = 0.0
                
                if hasattr(inputs, 'isSet'):
                    # Arrow keys for steering
                    if inputs.isSet('turnLeft'):
                        steer = -1.0
                    elif inputs.isSet('turnRight'):
                        steer = 1.0
                        
                    # Up/Down for throttle/brake
                    if inputs.isSet('forward'):
                        throttle = 1.0
                    elif inputs.isSet('reverse'):
                        throttle = -1.0
                        
                    return np.array([steer, throttle], dtype=np.float32)
        except Exception as e:
            pass
        
        # Fallback to parent implementation
        return super()._get_human_input()
    
    def _get_button_state(self, button: str) -> bool:
        """Get key state from keyboard using correct APIs"""
        # Method 1: Try to access MetaDrive's input system
        try:
            if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'inputs'):
                inputs = self.env.engine.inputs
                if hasattr(inputs, 'isSet'):
                    # Only allow dedicated takeover keys for keyboard policy
                    # Space bar (primary takeover key)
                    if inputs.isSet('takeover') or inputs.isSet('space'):
                        return True
                    
                    # Control keys (secondary takeover keys)
                    if any(inputs.isSet(k) for k in ['enter', 'shift', 'control']):
                        return True
                    
                    # DO NOT allow movement keys to trigger takeover
                    # This prevents accidental toggling when using arrow keys for control
        except Exception as e:
            pass
        
        # Fallback to parent implementation
        return super()._get_button_state(button)


def create_takeover_policy(env, controller_type: str):
    """
    Factory function to create appropriate takeover policy.
    
    Args:
        env: The environment
        controller_type: 'keyboard', 'gamepad', or 'steering_wheel'
        
    Returns:
        TakeoverPolicy instance
    """
    if controller_type == 'steering_wheel':
        return SteeringWheelPolicy(env)
    elif controller_type == 'gamepad':
        return GamepadPolicy(env)
    elif controller_type == 'keyboard':
        return KeyboardPolicy(env)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
