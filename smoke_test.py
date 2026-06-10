"""
Smoke test for DroneSwarmEnv.

Run with GUI (watch the drone):
    cd ~/Drone/IsaacLab
    ./isaaclab.sh -p ~/Drone/SwarmProject/smoke_test.py

Run headless (quick verification):
    cd ~/Drone/IsaacLab
    ./isaaclab.sh -p ~/Drone/SwarmProject/smoke_test.py --headless
"""

import argparse
import sys
import os

# --- Isaac Sim bootstrap (MUST come before any isaaclab imports) ---
parser = argparse.ArgumentParser(description="DroneSwarmEnv smoke test")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs")
parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# --- Now safe to import isaaclab and project modules ---
import torch

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from drone_env import DroneSwarmEnv, DroneSwarmEnvCfg


def main():
    # create env config — small for smoke test
    cfg = DroneSwarmEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # create env
    env = DroneSwarmEnv(cfg)

    print(f"\n{'='*60}")
    print(f"DroneSwarmEnv created successfully!")
    print(f"  num_envs:          {env.num_envs}")
    print(f"  action_space:      {cfg.action_space}")
    print(f"  observation_space: {cfg.observation_space}")
    print(f"  arena_size:        {cfg.arena_size}m")
    print(f"  grid:              {env.coverage_map.grid_size}x{env.coverage_map.grid_size}")
    print(f"  device:            {env.device}")
    print(f"{'='*60}\n")

    # reset all envs
    obs, info = env.reset()
    print(f"Initial obs shape: {obs['policy'].shape}")
    print(f"Initial obs range: [{obs['policy'].min().item():.3f}, {obs['policy'].max().item():.3f}]")

    # run with random actions
    for step in range(args.steps):
        # random actions in [-1, 1]
        actions = torch.rand(env.num_envs, cfg.action_space, device=env.device) * 2 - 1

        obs, rewards, terminated, truncated, info = env.step(actions)

        # print stats every 50 steps
        if (step + 1) % 50 == 0:
            coverage = env.coverage_map.get_coverage_pct()
            modes = env.controller.mode.squeeze(1)
            active_count = (modes == 0).sum().item()
            print(f"Step {step+1:4d} | "
                  f"reward: [{rewards.min().item():+.3f}, {rewards.max().item():+.3f}] | "
                  f"coverage: [{coverage.min().item():.1%}, {coverage.max().item():.1%}] | "
                  f"active: {active_count}/{env.num_envs} | "
                  f"terminated: {terminated.sum().item()} | "
                  f"truncated: {truncated.sum().item()}")

    # final stats
    print(f"\n{'='*60}")
    print(f"Smoke test complete — {args.steps} steps, no crashes.")
    final_coverage = env.coverage_map.get_coverage_pct()
    print(f"Final coverage: [{final_coverage.min().item():.1%}, {final_coverage.max().item():.1%}]")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()