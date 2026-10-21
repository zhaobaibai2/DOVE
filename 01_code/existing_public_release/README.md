# DOVE source code

This directory collects the code related to the paper *DOVE: Learning Diffusion Policies from Accepted and Rejected Actions*. It contains the MetaDrive/JAX implementation and the ROS 2/PyTorch implementation used on the physical UGV.

## Layout

```text
simulation/
  relax/                 DOVE training algorithm, networks, replay buffer and trainer
  relax_env/             MetaDrive human-in-the-loop environments
  scripts/               training, policy evaluation and demonstration-data utilities
  experiments/           paper evaluation and ablation command generators
robot/
  src/car_dacer_torch/   ROS 2 DOVE node, PyTorch algorithm, networks and configs
  src/car_interfaces/    ROS messages consumed and published by the DOVE node
```


