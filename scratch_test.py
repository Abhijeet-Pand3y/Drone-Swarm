import torch
from SwarmProject.src.mid_level_controller import MidLevelController

ctrl = MidLevelController(num_envs=4, num_agents=3,device="cuda:0")

print("mode:", ctrl.mode)
print("odometer:", ctrl.odometer)
print("target shape:", ctrl.target.shape)
print("base_xyz:", ctrl.base_xyz)

# test reset
ctrl.odometer[:] = 99.0
ctrl.mode[:] = 1
print("\nbefore reset:", ctrl.mode, ctrl.odometer)

ctrl.reset(torch.tensor([0, 2]))
print("after reset (envs 0,2):", ctrl.mode, ctrl.odometer)
