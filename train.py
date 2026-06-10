"""
Train PPO agent for drone coverage using SKRL 2.1.0.

    cd ~/Drone/IsaacLab
    ./isaaclab.sh -p ~/Drone/SwarmProject/train.py --headless --num_envs 256
    ./isaaclab.sh -p ~/Drone/SwarmProject/train.py --headless --num_envs 64   # low VRAM
"""

import argparse
import sys
import os

parser = argparse.ArgumentParser(description="Train drone coverage PPO")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=5000, help="PPO update iterations")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--episode_length", type=float, default=300.0, help="Episode length in seconds")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from datetime import datetime

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from drone_env import DroneSwarmEnv, DroneSwarmEnvCfg


# ======================================================================
# Policy network (Gaussian — continuous actions)
# ======================================================================
class PolicyNet(GaussianMixin, Model):
    """MLP policy: obs (283) → hidden → action mean (2).
    GaussianMixin handles sampling, log_prob, and clipping.
    """
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
        )

        obs_dim = observation_space.shape[0]    # 283
        act_dim = action_space.shape[0]         # 2

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, act_dim),
        )
        # learnable log_std — state-independent, shared across all observations
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}

# ======================================================================
# Value network (Deterministic — state value)
# ======================================================================
class ValueNet(DeterministicMixin, Model):
    """MLP value function: obs (283) → hidden → scalar value (1)."""
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        obs_dim = observation_space.shape[0]    # 283

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# ======================================================================
# Main
# ======================================================================
def main():
    device = "cuda:0"
    set_seed(args.seed)

    # --- create env ---
    cfg = DroneSwarmEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_length
    env = DroneSwarmEnv(cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"  num_envs:         {args.num_envs}")
    print(f"  episode_length:   {args.episode_length}s")
    print(f"  max_iterations:   {args.max_iterations}")
    print(f"  obs_space:        {env.observation_space}")
    print(f"  action_space:     {env.action_space}")
    print(f"  device:           {device}")
    print(f"{'='*60}\n")

    # --- models ---
    policy = PolicyNet(env.observation_space, env.action_space, device)
    value = ValueNet(env.observation_space, env.action_space, device)
    models = {"policy": policy, "value": value}

    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value.parameters())
    print(f"Policy params: {policy_params:,}  |  Value params: {value_params:,}")
    print(f"Total: {policy_params + value_params:,}\n")

    # --- memory ---
    rollout_steps = 32
    memory = RandomMemory(memory_size=rollout_steps, num_envs=args.num_envs, device=device)

    # --- PPO config (SKRL 2.1.0 field names) ---
    log_dir = os.path.join(
        os.path.dirname(__file__), "logs",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_coverage"
    )
    os.makedirs(log_dir, exist_ok=True)

    ppo_cfg = {
        # rollout
        "rollouts": rollout_steps,

        # learning
        "learning_epochs": 8,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "learning_rate_scheduler": None,

        # preprocessing (observation normalization)
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},

        # PPO clipping and losses
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.01,      # encourage exploration (default 0.0 is too low)
        "value_loss_scale": 0.5,
        "grad_norm_clip": 1.0,
        "kl_threshold": 0.0,             # disabled

        # misc
        "random_timesteps": 0,
        "learning_starts": 0,

        # experiment / logging
        "experiment": {
            "directory": log_dir,
            "experiment_name": "",
            "write_interval": 50,
            "checkpoint_interval": 500,
        },
    }

    # --- PPO agent ---
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # --- trainer ---
    total_timesteps = args.max_iterations * rollout_steps
    trainer_cfg = {
        "timesteps": total_timesteps,
        "headless": True,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print(f"Training: {args.max_iterations} iters × {rollout_steps} steps = {total_timesteps:,} timesteps")
    print(f"With {args.num_envs} envs → {total_timesteps * args.num_envs:,} total transitions")
    print(f"Logs: {log_dir}\n")

    # --- train ---
    trainer.train()

    print(f"\nTraining complete. Checkpoints: {log_dir}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()