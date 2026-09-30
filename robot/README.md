# Physical UGV implementation

`src/car_dacer_torch/` is the ROS 2/PyTorch implementation of DOVE used by the physical-vehicle package. It contains the diffusion policy, proxy-risk critics, EnergyRank/CPCal updates, action guidance/gating, replay buffer, online node, offline training entry point, and the DOVE route configurations.

`src/car_interfaces/` supplies the custom messages imported by the DOVE node. The vehicle's localization, perception, low-level CAN bridge, and route-control stack are platform software and are not bundled here.

## Build

Use the ROS 2 distribution and PyTorch build installed on the target vehicle. Install Python dependencies from `requirements.txt`, then build the two included packages:

```bash
cd robot
rosdep install --from-paths src --ignore-src -r -y
python -m pip install -r requirements.txt
colcon build --packages-up-to car_dacer_torch
source install/setup.bash
```

The YAML files use `./outputs` for model and log paths instead of paths from the original workstation. Set `DACER_CONFIG_PATH` to a selected configuration under `src/car_dacer_torch/car_dacer_torch/configs/` before launching a ROS 2 entry point. Review the CAN interface, topic contracts, route, and hardware settings for the target vehicle. No vehicle logs, replay buffers, or model checkpoints are included.
