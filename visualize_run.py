"""
Run DroneSwarmEnv headless, log trajectory data, produce visualization.

    cd ~/Drone/IsaacLab
    ./isaaclab.sh -p ~/Drone/SwarmProject/visualize_run.py --steps 500

Generates: ~/Drone/SwarmProject/viz_output/run_summary.png
"""

import argparse
import sys
import os

parser = argparse.ArgumentParser(description="DroneSwarmEnv visual verification")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (1 for clean visualization)")
parser.add_argument("--steps", type=int, default=500, help="Steps to run")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from drone_env import DroneSwarmEnv, DroneSwarmEnvCfg
from mid_level_controller import MidLevelController as MLC


def main():
    # default output dir
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "viz_output")
    os.makedirs(output_dir, exist_ok=True)

    # create env — single env for clean viz
    cfg = DroneSwarmEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = DroneSwarmEnv(cfg)

    arena = cfg.arena_size
    cell = cfg.cell_size
    grid_size = int(arena / cell)

    print(f"Running {args.steps} steps, arena {arena}m, grid {grid_size}x{grid_size}")

    # --- data buffers ---
    positions_log = []       # (steps, 2) drone xy positions
    coverage_log = []        # (snapshots, grid, grid) coverage snapshots
    coverage_pct_log = []    # (steps,) coverage percentage over time
    reward_log = []          # (steps,) per-step reward
    mode_log = []            # (steps,) drone mode

    snapshot_interval = max(1, args.steps // 20)  # ~20 coverage snapshots

    # reset
    obs, info = env.reset()

    for step in range(args.steps):
        # random actions
        actions = torch.rand(env.num_envs, cfg.action_space, device=env.device) * 2 - 1
        obs, rewards, terminated, truncated, info = env.step(actions)

        # log data (env 0 only)
        pos_w = env.drone.data.root_pos_w[0]             # (3,) world
        pos_local = (pos_w - env.scene.env_origins[0]).cpu().numpy()
        positions_log.append(pos_local[:2].copy())

        cov_pct = env.coverage_map.get_coverage_pct()[0].item()
        coverage_pct_log.append(cov_pct)

        reward_log.append(rewards[0].item())
        mode_log.append(env.controller.mode[0, 0].item())

        # coverage snapshot
        if step % snapshot_interval == 0 or step == args.steps - 1:
            grid_snap = env.coverage_map.grid[0].cpu().numpy().copy()
            coverage_log.append((step, grid_snap))

        if (step + 1) % 100 == 0:
            print(f"  step {step+1}: coverage={cov_pct:.1%}, reward={rewards[0].item():+.3f}")

    # convert to numpy
    positions = np.array(positions_log)    # (steps, 2)
    coverage_pcts = np.array(coverage_pct_log)
    rewards_arr = np.array(reward_log)
    modes_arr = np.array(mode_log)

    print(f"\nFinal coverage: {coverage_pcts[-1]:.1%}")
    print(f"Total reward: {rewards_arr.sum():.2f}")
    print(f"Generating visualization...")

    # =====================================================================
    # PLOT REFACTOR: Fixed GridSpec Layout Engine
    # =====================================================================
    fig = plt.figure(figsize=(22, 14), facecolor='#1a1a2e')
    
    # 3 rows, 4 columns uniform spacing framework
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # custom colormap for coverage: dark blue (0) → green (0.5) → yellow (1.0)
    cov_cmap = LinearSegmentedColormap.from_list("coverage",
        ["#1a1a2e", "#0d4f4f", "#2d8a4e", "#a8d848", "#ffee58"])

    # --- 1. Top-down trajectory + final coverage heatmap (Large Left Framework) ---
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Spans Rows 0-1, Columns 0-1
    ax1.set_facecolor("#1a1a2e")

    final_grid = coverage_log[-1][1]
    im = ax1.imshow(final_grid.T, origin="lower", extent=[0, arena, 0, arena],
                    cmap=cov_cmap, vmin=0, vmax=1, alpha=0.8, aspect="equal")

    # grid lines
    for i in range(grid_size + 1):
        ax1.axhline(y=i * cell, color="#ffffff", alpha=0.1, linewidth=0.5)
        ax1.axvline(x=i * cell, color="#ffffff", alpha=0.1, linewidth=0.5)

    # trajectory paths
    ax1.plot(positions[:, 0], positions[:, 1], color="#ff6b6b", linewidth=0.8, alpha=0.6, label="trajectory")
    ax1.scatter(positions[0, 0], positions[0, 1], color="#4ecdc4", s=100, zorder=5,
                marker="^", label="spawn", edgecolors="white", linewidth=1.5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], color="#ff6b6b", s=100, zorder=5,
                marker="s", label="final pos", edgecolors="white", linewidth=1.5)

    # base station
    ax1.scatter(0, 0, color="#ffd93d", s=150, zorder=5, marker="*",
                label="base", edgecolors="white", linewidth=1.5)

    # arena boundary geofence
    arena_rect = patches.Rectangle((0, 0), arena, arena, linewidth=2,
                                    edgecolor="#4ecdc4", facecolor="none", linestyle="--")
    ax1.add_patch(arena_rect)

    ax1.set_xlim(-2, arena + 2)
    ax1.set_ylim(-2, arena + 2)
    ax1.set_xlabel("X (m)", color="white", fontsize=11)
    ax1.set_ylabel("Y (m)", color="white", fontsize=11)
    ax1.set_title("Drone Trajectory + Coverage Heatmap", color="white", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, facecolor="#2a2a4a", edgecolor="#4ecdc4", labelcolor="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#4ecdc4")

    # isolated colorbar placement
    cbar = plt.colorbar(im, ax=ax1, shrink=0.75, pad=0.04)
    cbar.set_label("Scan Progress", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # --- 2. Coverage % over time (Top Right Double-Width) ---
    ax2 = fig.add_subplot(gs[0, 2:4])  # Row 0, Spans Columns 2-3
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(coverage_pcts * 100, color="#4ecdc4", linewidth=1.5)
    ax2.axhline(y=95, color="#ffd93d", linestyle="--", alpha=0.7, label="95% threshold")
    ax2.set_xlabel("Step", color="white", fontsize=10)
    ax2.set_ylabel("Coverage %", color="white", fontsize=10)
    ax2.set_title("Coverage Progress", color="white", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, facecolor="#2a2a4a", edgecolor="#4ecdc4", labelcolor="white")
    ax2.tick_params(colors="white")
    ax2.set_ylim(0, 100)
    for spine in ax2.spines.values():
        spine.set_color("#4ecdc4")

    # --- 3. Per-step reward (Middle Right Double-Width) ---
    ax3 = fig.add_subplot(gs[1, 2:4])  # Row 1, Spans Columns 2-3
    ax3.set_facecolor("#1a1a2e")
    window = min(20, len(rewards_arr))
    if window > 1:
        rolling = np.convolve(rewards_arr, np.ones(window)/window, mode="valid")
        ax3.plot(rolling, color="#ff6b6b", linewidth=1.2, label=f"rolling avg ({window})")
    ax3.plot(rewards_arr, color="#ff6b6b", alpha=0.2, linewidth=0.5)
    ax3.axhline(y=0, color="white", alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("Step", color="white", fontsize=10)
    ax3.set_ylabel("Reward", color="white", fontsize=10)
    ax3.set_title("Per-Step Reward", color="white", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, facecolor="#2a2a4a", edgecolor="#4ecdc4", labelcolor="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_color("#4ecdc4")

    # --- 4. Coverage snapshots (Bottom Row Equal Partition) ---
    n_snaps = 4
    snap_indices = np.linspace(0, len(coverage_log)-1, n_snaps, dtype=int)
    for i, idx in enumerate(snap_indices):
        ax = fig.add_subplot(gs[2, i])  # Row 2, unique dedicated columns 0, 1, 2, 3
        ax.set_facecolor("#1a1a2e")
        step_num, grid = coverage_log[idx]
        ax.imshow(grid.T, origin="lower", extent=[0, arena, 0, arena],
                  cmap=cov_cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Step {step_num}", color="white", fontsize=10, fontweight="semibold")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#4ecdc4")

    # --- uniform master header block ---
    fig.suptitle(
        f"Project Alpha — Smoke Test Visualization  |\n"
        f"Steps: {args.steps}  |  Final Coverage: {coverage_pcts[-1]:.1%}  |  Total Accumulated Reward: {rewards_arr.sum():.2f}",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    # Export configuration using clean boundary scaling margins
    output_path = os.path.join(output_dir, "run_summary.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nVisualization saved successfully: {output_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()