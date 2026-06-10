"""
DroneSwarmEnv — DirectRLEnv for autonomous drone coverage.

Phase 0: single Crazyflie drone, velocity-controlled, coverage scanning.
Wires together CoverageMap, MidLevelController, ObservationBuilder, RewardComputer.

Usage:
    env = DroneSwarmEnv(DroneSwarmEnvCfg())
"""

from __future__ import annotations
from collections.abc import Sequence

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG

from coverage_map import CoverageMap
from mid_level_controller import MidLevelController as MLC
from observation import ObservationBuilder
from reward import RewardComputer


# ==========================================================================
# Config
# ==========================================================================

@configclass
class DroneSwarmEnvCfg(DirectRLEnvCfg):
    """Configuration for the drone swarm coverage environment."""

    # --- env ---
    decimation = 4                    # physics substeps per RL step
    episode_length_s = 120.0          # max episode time (seconds)
    action_space = 2                  # (dx, dy)
    observation_space = 283           # locked obs dim
    state_space = 0                   # no asymmetric critic state
    num_agents = 1                    # Phase 0: single drone per env
    

    max_range_min: float = 150.0
    max_range_max: float = 500.0
    randomize_max_range: bool = True

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=4,
    )

    # --- robot ---
    # modified Crazyflie: disable gravity for velocity-controlled flight
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    robot_cfg.spawn.rigid_props.disable_gravity = True

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=60.0,              # must exceed arena_size to avoid env overlap
        replicate_physics=True,
    )

    # --- arena ---
    arena_size: float = 50.0           # square arena side length (meters)
    cell_size: float = 5.0             # coverage grid cell size (= sensor footprint)

    # --- controller ---
    max_range: float = 200.0           # odometer distance before return trigger
    safety_factor: float = 0.8         # return at this fraction of max_range
    radius_cap: float = 5.0            # max step size per action (meters)
    scan_speed: float = 1.5            # drone speed (m/s)
    arrival_threshold: float = 1.0     # distance to base to count as docked (meters)

    # --- coverage ---
    scan_range: float = 5.0            # physical scan radius (meters)
    scan_rate: float = 0.2             # progress per step of presence (1/dwell_steps)
    coverage_threshold: float = 0.95   # cells at/above this progress count as covered

    # --- reward ---
    coverage_weight: float = 1.0
    completion_weight: float = 0.3
    time_penalty: float = 0.01

    prescanned_fraction: float = 0.0      # 0.0 = off, 0.5 = 50% pre-scanned
    prescanned_fraction_min: float = 0.7  # randomize within range each episode
    prescanned_fraction_max: float = 0.85  # so policy sees varied starting states
    randomize_prescanned: bool = True


# ==========================================================================
# Environment
# ==========================================================================

class DroneSwarmEnv(DirectRLEnv):
    """
    DirectRLEnv for autonomous drone coverage scanning.

    Step orchestration (inherited from DirectRLEnv base):
        1. _pre_physics_step(actions)    → controller.step → velocity setpoints
        2. [decimation substeps]         → _apply_action writes velocity to sim
        3. _get_dones()                  → mark_scanned, deltas, done conditions
        4. _reset_idx(done_envs)         → reset contract
        5. _get_observations()           → obs_builder.build (post-reset state)
        6. _get_rewards()                → reward.compute (pre-reset deltas)

    Coordinate contract: all modules receive env-LOCAL coordinates [0, arena_size].
    World→local conversion happens here, once, before calling any module.
    """

    cfg: DroneSwarmEnvCfg

    def __init__(self, cfg: DroneSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- create modules ---
        self.coverage_map = CoverageMap(
            num_envs=self.num_envs,
            arena_size=cfg.arena_size,
            cell_size=cfg.cell_size,
            device=self.device,
        )
        self.controller = MLC(
            num_envs=self.num_envs,
            num_agents=cfg.num_agents,
            device=self.device,
            base_position=(0.0, 0.0),
            max_range=cfg.max_range,
            safety_factor=cfg.safety_factor,
            radius_cap=cfg.radius_cap,
            scan_speed=cfg.scan_speed,
            arrival_threshold=cfg.arrival_threshold,
            arena_min=(0.0, 0.0),
            arena_max=(cfg.arena_size, cfg.arena_size),
        )
        self.obs_builder = ObservationBuilder(
            num_envs=self.num_envs,
            num_agents=cfg.num_agents,
            device=self.device,
            arena_size=cfg.arena_size,
            max_speed=cfg.scan_speed + 0.5,   # slight headroom over scan_speed
            max_range=cfg.max_range,
        )
        self.reward_computer = RewardComputer(
            num_envs=self.num_envs,
            num_agents=cfg.num_agents,
            device=self.device,
            coverage_weight=cfg.coverage_weight,
            completion_weight=cfg.completion_weight,
            time_penalty=cfg.time_penalty,
        )

        # --- per-step state (stored between _get_dones and _get_rewards) ---
        self._progress_delta = torch.zeros(self.num_envs, device=self.device)
        self._completion_count = torch.zeros(self.num_envs, device=self.device)
        self._velocity_setpoint = torch.zeros(self.num_envs, 3, device=self.device)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Create the Crazyflie drone and ground plane."""
        self.drone = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["drone"] = self.drone
        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _world_to_local(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Convert world-frame positions to env-local [0, arena_size] coordinates.
        
        Args:
            world_pos: (num_envs, 3) world positions
        Returns:
            (num_envs, 3) local positions
        """
        return world_pos - self.scene.env_origins

    def _local_to_world(self, local_pos: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Convert env-local positions to world-frame.
        
        Args:
            local_pos: (len(env_ids), 3) local positions
            env_ids:   (len(env_ids),) env indices
        Returns:
            (len(env_ids), 3) world positions
        """
        return local_pos + self.scene.env_origins[env_ids]

    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion (w, x, y, z).
        
        Args:
            quat: (num_envs, 4) quaternion [w, x, y, z]
        Returns:
            (num_envs,) yaw in radians
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # ------------------------------------------------------------------
    # Step orchestration
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process policy actions through the controller to get velocity setpoints.

        Called once per RL step, before the decimation physics loop.

        Args:
            actions: (num_envs, 2) — (dx, dy) in [-1, 1]
        """
        # read current drone state (world coords)
        pos_w = self.drone.data.root_pos_w            # (num_envs, 3)
        pos_local = self._world_to_local(pos_w)       # (num_envs, 3)

        # expand to agent dim for controller: (num_envs, 1, ...)
        actions_ag = actions.unsqueeze(1)               # (num_envs, 1, 2)
        pos_ag = pos_local.unsqueeze(1)                 # (num_envs, 1, 3)

        # controller computes velocity setpoints + updates mode/odometer
        velocity = self.controller.step(actions_ag, pos_ag)   # (num_envs, 1, 3)

        # store for _apply_action (squeeze agent dim for sim)
        self._velocity_setpoint = velocity.squeeze(1)   # (num_envs, 3)

    def _apply_action(self) -> None:
        """Write velocity setpoint to the sim. Called every physics substep.

        Sets root velocity directly — no forces, no PD controller. The drone
        moves at exactly the commanded velocity. Gravity is disabled on the
        Crazyflie so there's no fighting between velocity writes and gravity.
        """
        # (num_envs, 6) = [lin_vel(3), ang_vel(3)]
        vel_6d = torch.zeros(self.num_envs, 6, device=self.device)
        vel_6d[:, :3] = self._velocity_setpoint
        # ang_vel stays zero — no yaw control via physics
        self.drone.write_root_velocity_to_sim(vel_6d)

    # ------------------------------------------------------------------
    # Dones — also computes coverage deltas (runs BEFORE reset and rewards)
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute done flags and coverage deltas.

        Reads post-physics state, runs mark_scanned for ACTIVE drones, stores
        deltas for _get_rewards (which runs after _reset_idx).

        Returns:
            (terminated, truncated) — both (num_envs,) bool
        """
        # read current state
        pos_w = self.drone.data.root_pos_w
        pos_local = self._world_to_local(pos_w)

        # --- mark scanned for ACTIVE drones only ---
        modes = self.controller.mode.squeeze(1)          # (num_envs,)
        active_mask = (modes == MLC.MODE_ACTIVE)

        # reset deltas
        self._progress_delta.zero_()
        self._completion_count.zero_()

        if active_mask.any():
            active_ids = torch.where(active_mask)[0]     # env indices of active drones
            active_pos = pos_local[active_ids, :2]       # (N_active, 2) — xy only

            progress, completion = self.coverage_map.mark_scanned(
                env_ids=active_ids,
                positions=active_pos,
                scan_range=self.cfg.scan_range,
                scan_rate=self.cfg.scan_rate,
                threshold=self.cfg.coverage_threshold,
            )
            # mark_scanned returns per-env (num_envs,) — deltas are nonzero only for active envs
            self._progress_delta = progress
            self._completion_count = completion

        # --- crash detection (NaN / velocity blowup) ---
        vel_w = self.drone.data.root_lin_vel_w           # (num_envs, 3)
        has_nan = torch.isnan(pos_w).any(dim=-1) | torch.isnan(vel_w).any(dim=-1)
        vel_blowup = torch.norm(vel_w, dim=-1) > 100.0   # extreme velocity threshold
        crash_mask = has_nan | vel_blowup
        if crash_mask.any():
            self.controller.mark_dead(crash_mask.unsqueeze(1))  # (num_envs, 1) for agent dim

        # --- done conditions ---
        # terminated: coverage complete
        terminated = self.coverage_map.is_fully_covered(threshold=self.cfg.coverage_threshold)
        # also terminate if all agents in env are docked or dead
        all_inactive = (
            (modes == MLC.MODE_DOCKED) | (modes == MLC.MODE_DEAD)
        )
        terminated = terminated | all_inactive

        # truncated: episode timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute per-env reward from stored deltas.

        Uses progress_delta and completion_count computed in _get_dones.
        For Phase 0 (1 agent/env), per-env == per-agent.

        Returns:
            (num_envs,) reward tensor
        """
        modes = self.controller.mode.squeeze(1)   # (num_envs,)
        return self.reward_computer.compute(
            progress_delta=self._progress_delta,
            completion_count=self._completion_count,
            modes=modes,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Build the observation vector. Runs AFTER _reset_idx.

        For reset envs: state reflects the new episode (clean map, fresh spawn).
        For continuing envs: state reflects post-physics, post-mark_scanned.

        Returns:
            {"policy": (num_envs, 283)} observation dict
        """
        # read post-reset state
        pos_w = self.drone.data.root_pos_w
        vel_w = self.drone.data.root_lin_vel_w
        quat = self.drone.data.root_quat_w

        pos_local = self._world_to_local(pos_w)
        yaw = self._quat_to_yaw(quat)

        # controller state (squeeze agent dim for Phase 0)
        modes = self.controller.mode.squeeze(1)
        odometer = self.controller.odometer.squeeze(1)
        dist_to_base = self.controller.dist_to_base.squeeze(1)

        # env_ids: identity mapping for Phase 0 (each env has 1 agent)
        env_ids = torch.arange(self.num_envs, device=self.device)

        max_range = self.controller.max_range.squeeze(1)

        obs = self.obs_builder.build(
            positions=pos_local,
            velocities=vel_w,
            yaw=yaw,
            modes=modes,
            odometer=odometer,
            dist_to_base=dist_to_base,
            env_ids=env_ids,
            coverage_map=self.coverage_map,
            max_range=max_range,
        )

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset specific environments.

        Implements the reset contract:
            1. super()._reset_idx — resets episode buffers
            2. respawn drone at random position in arena
            3. controller.reset + seed_positions
            4. coverage_map.reset

        Args:
            env_ids: environments to reset
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        super()._reset_idx(env_ids)
        num_resets = len(env_ids)

        # --- random spawn position in arena [0, arena_size] ---
        random_x = torch.rand(num_resets, device=self.device) * self.cfg.arena_size
        random_y = torch.rand(num_resets, device=self.device) * self.cfg.arena_size
        local_pos = torch.stack([
            random_x,
            random_y,
            torch.full((num_resets,), MLC.SCAN_HEIGHT, device=self.device),
        ], dim=-1)   # (num_resets, 3)

        # convert to world frame for sim
        world_pos = self._local_to_world(local_pos, env_ids)

        # Radomize max_range
        random_range = torch.rand(num_resets, device=self.device) * (self.cfg.max_range_max - self.cfg.max_range_min) + self.cfg.max_range_min
        self.controller.set_max_range(env_ids, random_range)

        # build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
        root_state = torch.zeros(num_resets, 13, device=self.device)
        root_state[:, :3] = world_pos
        root_state[:, 3] = 1.0       # quaternion w (identity rotation)
        # vel stays zero

        self.drone.write_root_state_to_sim(root_state, env_ids)

        # --- reset modules (reset contract) ---
        self.controller.reset(env_ids)
        # seed with local positions (agent dim for controller)
        self.controller.seed_positions(local_pos.unsqueeze(1), env_ids)
        self.coverage_map.reset(env_ids)


        if self.cfg.randomize_prescanned and self.cfg.prescanned_fraction_max > 0.0:
            num_cells = self.coverage_map.grid_size ** 2
            for i, env_id in enumerate(env_ids):
                # random fraction per episode
                fraction = torch.rand(1).item() * (
                    self.cfg.prescanned_fraction_max - self.cfg.prescanned_fraction_min
                ) + self.cfg.prescanned_fraction_min
                num_prescanned = int(num_cells * fraction)
                # random cells set to 1.0 (fully scanned)
                flat_indices = torch.randperm(num_cells, device=self.device)[:num_prescanned]
                self.coverage_map.grid[env_id].view(-1)[flat_indices] = 1.0
            

            # pct = self.coverage_map.get_coverage_pct(threshold=0.95)
            # print(f"[reset] pre-scanned coverage: min={pct.min().item():.1%} max={pct.max().item():.1%}")

        # clear per-step deltas for reset envs
        self._progress_delta[env_ids] = 0.0
        self._completion_count[env_ids] = 0.0

        