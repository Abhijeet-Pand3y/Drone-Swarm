"""
Evaluate a trained PPO policy and visualize coverage behavior.

    cd ~/Drone/IsaacLab
    ./isaaclab.sh -p ~/Drone/SwarmProject/evaluate.py --checkpoint <path_to_agent.pt> --steps 3000
"""

import argparse
import sys
import os

parser = argparse.ArgumentParser(description="Evaluate trained drone coverage policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint .pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--episode_length", type=float, default=300.0)
parser.add_argument("--max_range", type=float, default=None, help="Override max range (set high to disable return)")
parser.add_argument("--scan_speed", type=float, default=None, help="Override scan speed")
parser.add_argument("--prescanned", type=float, default=None, help="Override pre-scan fraction (0.0=clean map)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from skrl.agents.torch.ppo import PPO
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from drone_env import DroneSwarmEnv, DroneSwarmEnvCfg
from mid_level_controller import MidLevelController as MLC


# --- model definitions MUST match train.py exactly ---

class PolicyNet(GaussianMixin, Model):
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
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, act_dim),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def compute(self, inputs, role):
        # SKRL 2.1.0: log_std in outputs dict, not separate return
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class ValueNet(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


def main():
    device = "cuda:0"
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "viz_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- create env ---
    cfg = DroneSwarmEnvCfg()
    
    if args.max_range is not None:
        cfg.max_range = args.max_range
        cfg.max_range_min = args.max_range
        cfg.max_range_max = args.max_range
        cfg.randomize_max_range = False

    if args.scan_speed is not None:
        cfg.scan_speed = args.scan_speed

    if args.prescanned is not None:
        cfg.prescanned_fraction_min = args.prescanned
        cfg.prescanned_fraction_max = args.prescanned
        cfg.randomize_prescanned = args.prescanned > 0.0
    else:
        # default: clean map for eval
        cfg.randomize_prescanned = False
        cfg.prescanned_fraction_min = 0.0
        cfg.prescanned_fraction_max = 0.0

    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_length
    env = DroneSwarmEnv(cfg)

    arena = cfg.arena_size
    cell = cfg.cell_size
    grid_size = int(arena / cell)

    # --- create agent and load checkpoint ---
    # hardcode spaces to match what SKRL wrapper provided during training
    obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(283,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    policy = PolicyNet(obs_space, act_space, device)
    value = ValueNet(obs_space, act_space, device)
    models = {"policy": policy, "value": value}

    ppo_cfg = {
        "rollouts": 32,
        "learning_epochs": 8,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": obs_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {"directory": "", "experiment_name": "",
                       "write_interval": 0, "checkpoint_interval": 0},
    }

    agent = PPO(
        models=models,
        cfg=ppo_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)

    # set to eval mode
    policy.eval()
    value.eval()

    print(f"Running {args.steps} steps with trained policy...")

    # --- data buffers ---
    positions_log = []
    coverage_log = []
    coverage_pct_log = []
    reward_log = []
    episode_boundaries = []

    snapshot_interval = max(1, args.steps // 20)

    # --- run ---
    obs, info = env.reset()
    obs_tensor = obs["policy"] if isinstance(obs, dict) else obs

    for step in range(args.steps):
        # deterministic action: use mean directly, no sampling noise
        with torch.no_grad():
            mean, outputs = policy.compute({"observations": obs_tensor}, role="policy")
            actions = mean.clamp(-1, 1)

        obs, rewards, terminated, truncated, info = env.step(actions)
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs

        # log (env 0 only)
        pos_w = env.drone.data.root_pos_w[0]
        pos_local = (pos_w - env.scene.env_origins[0]).cpu().numpy()
        positions_log.append(pos_local[:2].copy())

        cov_pct = env.coverage_map.get_coverage_pct()[0].item()
        coverage_pct_log.append(cov_pct)
        reward_log.append(rewards[0].item())

        if terminated[0].item() or truncated[0].item():
            episode_boundaries.append(step)

        if step % snapshot_interval == 0 or step == args.steps - 1:
            grid_snap = env.coverage_map.grid[0].cpu().numpy().copy()
            coverage_log.append((step, grid_snap))

        if (step + 1) % 200 == 0:
            print(f"  step {step+1}: coverage={cov_pct:.1%}, reward={rewards[0].item():+.3f}")

    positions = np.array(positions_log)
    coverage_pcts = np.array(coverage_pct_log)
    rewards_arr = np.array(reward_log)

    print(f"\nFinal coverage: {coverage_pcts[-1]:.1%}")
    print(f"Max coverage reached: {coverage_pcts.max():.1%}")
    print(f"Total reward: {rewards_arr.sum():.2f}")
    print(f"Episodes completed: {len(episode_boundaries)}")

    # =====================================================================
    # PLOT
    # =====================================================================
    cov_cmap = LinearSegmentedColormap.from_list("coverage",
        ["#1a1a2e", "#0d4f4f", "#2d8a4e", "#a8d848", "#ffee58"])

    fig = plt.figure(figsize=(22, 14), facecolor="#1a1a2e")
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35, height_ratios=[1.2, 1, 1])

    # --- trajectory + heatmap ---
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_facecolor("#1a1a2e")

    final_grid = coverage_log[-1][1]
    im = ax1.imshow(final_grid.T, origin="lower", extent=[0, arena, 0, arena],
                    cmap=cov_cmap, vmin=0, vmax=1, alpha=0.8, aspect="equal")

    for i in range(grid_size + 1):
        ax1.axhline(y=i * cell, color="#ffffff", alpha=0.1, linewidth=0.5)
        ax1.axvline(x=i * cell, color="#ffffff", alpha=0.1, linewidth=0.5)

    # color trajectory by time (blue=early, red=late), skip episode teleports
    for i in range(1, len(positions)):
        if i in episode_boundaries or (i > 0 and i-1 in episode_boundaries):
            continue
        t_frac = i / len(positions)
        color = (t_frac, 0.3, 1.0 - t_frac)
        ax1.plot(positions[i-1:i+1, 0], positions[i-1:i+1, 1],
                 color=color, linewidth=0.6, alpha=0.7)

    ax1.scatter(positions[0, 0], positions[0, 1], color="#4ecdc4", s=100,
                zorder=5, marker="^", label="spawn", edgecolors="white", linewidth=1.5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], color="#ff6b6b", s=100,
                zorder=5, marker="s", label="final", edgecolors="white", linewidth=1.5)
    ax1.scatter(0, 0, color="#ffd93d", s=150, zorder=5, marker="*",
                label="base", edgecolors="white", linewidth=1.5)

    arena_rect = patches.Rectangle((0, 0), arena, arena, linewidth=2,
                                    edgecolor="#4ecdc4", facecolor="none", linestyle="--")
    ax1.add_patch(arena_rect)
    ax1.set_xlim(-2, arena + 2)
    ax1.set_ylim(-2, arena + 2)
    ax1.set_xlabel("X (m)", color="white", fontsize=11)
    ax1.set_ylabel("Y (m)", color="white", fontsize=11)
    ax1.set_title("Trained Policy — Trajectory + Coverage", color="white",
                   fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, facecolor="#2a2a4a",
               edgecolor="#4ecdc4", labelcolor="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#4ecdc4")
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label("Scan Progress", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # --- coverage progress ---
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(coverage_pcts * 100, color="#4ecdc4", linewidth=1.5)
    ax2.axhline(y=95, color="#ffd93d", linestyle="--", alpha=0.7, label="95% threshold")
    for eb in episode_boundaries:
        ax2.axvline(x=eb, color="#ff6b6b", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("Step", color="white", fontsize=10)
    ax2.set_ylabel("Coverage %", color="white", fontsize=10)
    ax2.set_title("Coverage Progress", color="white", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, facecolor="#2a2a4a", edgecolor="#4ecdc4", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.set_ylim(0, 100)
    for spine in ax2.spines.values():
        spine.set_color("#4ecdc4")

    # --- reward ---
    ax3 = fig.add_subplot(gs[1, 2:4])
    ax3.set_facecolor("#1a1a2e")
    window = min(50, len(rewards_arr))
    if window > 1:
        rolling = np.convolve(rewards_arr, np.ones(window)/window, mode="valid")
        ax3.plot(rolling, color="#ff6b6b", linewidth=1.2, label=f"avg ({window})")
    ax3.plot(rewards_arr, color="#ff6b6b", alpha=0.15, linewidth=0.5)
    ax3.axhline(y=0, color="white", alpha=0.3, linewidth=0.5)
    for eb in episode_boundaries:
        ax3.axvline(x=eb, color="#4ecdc4", alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("Step", color="white", fontsize=10)
    ax3.set_ylabel("Reward", color="white", fontsize=10)
    ax3.set_title("Per-Step Reward", color="white", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, facecolor="#2a2a4a", edgecolor="#4ecdc4", labelcolor="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_color("#4ecdc4")

    # --- coverage snapshots ---
    n_snaps = 4
    snap_indices = np.linspace(0, len(coverage_log)-1, n_snaps, dtype=int)
    for i, idx in enumerate(snap_indices):
        ax = fig.add_subplot(gs[2, i])
        ax.set_facecolor("#1a1a2e")
        step_num, grid = coverage_log[idx]
        ax.imshow(grid.T, origin="lower", extent=[0, arena, 0, arena],
                  cmap=cov_cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Step {step_num}", color="white", fontsize=10, fontweight="semibold")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#4ecdc4")

    ckpt_name = os.path.basename(args.checkpoint)
    fig.suptitle(
        f"Project Alpha — Trained Policy Evaluation  |  {ckpt_name}\n"
        f"Steps: {args.steps}  |  Max Coverage: {coverage_pcts.max():.1%}  |  "
        f"Reward: {rewards_arr.sum():.1f}  |  Episodes: {len(episode_boundaries)}",
        color="white", fontsize=14, fontweight="bold", y=0.99
    )

    output_path = os.path.join(output_dir, f"eval_{ckpt_name.replace('.pt','')}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nVisualization saved: {output_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()