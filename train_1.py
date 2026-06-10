"""
Train PPO agent for drone coverage using SKRL.

    cd ~/Drone/IsaacLab
    ./isaaclab.sh -p ~/Drone/SwarmProject/train.py --headless --num_envs 256
    ./isaaclab.sh -p ~/Drone/SwarmProject/train.py --headless --num_envs 64   # low VRAM

Logs and checkpoints saved to: ~/Drone/SwarmProject/logs/
"""

import argparse
import sys
import os

# --- argparse BEFORE AppLauncher ---
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

# --- Now safe to import everything ---
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

# project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from drone_env import DroneSwarmEnv, DroneSwarmEnvCfg


# ======================================================================
# Policy network (Gaussian — continuous actions)
# ======================================================================
class PolicyNet(GaussianMixin, Model):
    """MLP policy: obs (283) → hidden → action mean (2).
    
    GaussianMixin adds learnable log_std and handles sampling,
    log_prob computation, and action clipping.
    """
    def __init__(self, observation_space, action_space, device,
                    clip_actions=False, clip_log_std=True,
                    min_log_std=-20.0, max_log_std=2.0):
            Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
            GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std)

            obs_dim = self.num_observations   # 283
            act_dim = self.num_actions         # 2

            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, act_dim),
            )
            # learnable log std — one per action dim, shared across envs
            self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def compute(self, inputs, role):
        """Forward pass.
        
        Args:
            inputs: dict with "states" key → (N, 283) observation tensor
            role: "policy" (unused, for SKRL compatibility)
        Returns:
            (action_mean, log_std, extra_outputs)
        """
        return self.net(inputs["states"]), self.log_std_parameter, {}


# ======================================================================
# Value network (Deterministic — state value)
# ======================================================================
class ValueNet(DeterministicMixin, Model):
    """MLP value function: obs (283) → hidden → scalar value (1)."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
            DeterministicMixin.__init__(self, clip_actions=clip_actions)

            obs_dim = self.num_observations   # 283

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
        """Forward pass.
        
        Args:
            inputs: dict with "states" key → (N, 283)
            role: "value" (unused)
        Returns:
            (value, extra_outputs)
        """
        return self.net(inputs["states"]), {}


# ======================================================================
# Main
# ======================================================================
def main():
    device = "cuda:0"

    # --- seed ---
    set_seed(args.seed)

    # --- create env ---
    cfg = DroneSwarmEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_length
    env = DroneSwarmEnv(cfg)

    # wrap for SKRL
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"  num_envs:         {args.num_envs}")
    print(f"  episode_length:   {args.episode_length}s")
    print(f"  max_iterations:   {args.max_iterations}")
    print(f"  obs_space:        {env.observation_space}")
    print(f"  action_space:     {env.action_space}")
    print(f"  seed:             {args.seed}")
    print(f"  device:           {device}")
    print(f"{'='*60}\n")

    # --- instantiate models ---
    policy = PolicyNet(env.observation_space, env.action_space, device)
    value = ValueNet(env.observation_space, env.action_space, device)

    models = {"policy": policy, "value": value}

    # print model sizes
    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value.parameters())
    print(f"Policy parameters:  {policy_params:,}")
    print(f"Value parameters:   {value_params:,}")
    print(f"Total parameters:   {policy_params + value_params:,}\n")

    # --- memory (rollout buffer) ---
    rollout_steps = 32  # steps collected per env before each PPO update
    memory = RandomMemory(memory_size=rollout_steps, num_envs=args.num_envs, device=device)

    # --- PPO config ---
    ppo_cfg = {}

    # rollout
    ppo_cfg["rollouts"] = rollout_steps

    # learning
    ppo_cfg["learning_epochs"] = 8
    ppo_cfg["mini_batches"] = 4          # 8192 transitions / 4 = 2048 per mini-batch
    ppo_cfg["discount_factor"] = 0.99
    ppo_cfg["gae_lambda"] = 0.95             # GAE lambda
    ppo_cfg["learning_rate"] = 3e-4
    ppo_cfg["learning_rate_scheduler"] = None

    # observation normalization (running mean/std)
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}   
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # PPO-specific
    ppo_cfg["ratio_clip"] = 0.2
    ppo_cfg["value_clip"] = 0.2
    # ppo_cfg["clip_predicted_values"] = True
    ppo_cfg["entropy_loss_scale"] = 0.01     # encourage exploration
    ppo_cfg["value_loss_scale"] = 0.5
    ppo_cfg["grad_norm_clip"] = 1.0
    ppo_cfg["kl_threshold"] = 0              # disabled

    # logging and checkpointing
    log_dir = os.path.join(
        os.path.dirname(__file__), "logs",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_coverage"
    )
    os.makedirs(log_dir, exist_ok=True)

    ppo_cfg["experiment"] = {
        "directory": log_dir,
        "experiment_name": "",
        "write_interval": 50,           # log every 50 iterations
        "checkpoint_interval": 500,     # save model every 500 iterations
    }

    # --- create PPO agent ---
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # resume from checkpoint if specified
    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # --- trainer ---
    # timesteps = total env steps per training run
    # iterations × rollout_steps = timesteps
    total_timesteps = args.max_iterations * rollout_steps
    trainer_cfg = {
        "timesteps": total_timesteps,
        "headless": True,
    }

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print(f"Starting training: {args.max_iterations} iterations × {rollout_steps} steps = {total_timesteps:,} timesteps")
    print(f"With {args.num_envs} envs: {total_timesteps * args.num_envs:,} total transitions")
    print(f"Logs: {log_dir}\n")

    # --- train ---
    trainer.train()

    print(f"\nTraining complete. Checkpoints saved to: {log_dir}")

    # --- cleanup ---
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
