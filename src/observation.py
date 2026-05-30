import torch
import math
import torch.nn.functional as F
from mid_level_controller import MidLevelController as MLC


class ObservationBuilder:
    """
    Assembles the per-agent observation vector from sim state, controller state,
    and the coverage map. Pure assembly — receives clean env-LOCAL coordinates
    (the env does world->local conversion before calling this).

    Output layout (fixed forever, 283 dims):
        ego                    14   pos(3) vel(3) heading_sincos(2) mode_onehot(4)
                                    remaining_budget_ratio(1) dist_to_base_ratio(1)
        full_map              100   from CoverageMap.get_full_map
        local_window          121   zeroed Phase 0-1, from get_local_window in Phase 2+
        neighbor_slots         44   K=4 x M=11, zeroed Phase 0
                                    per neighbor: rel_pos(3) dist(1) rel_vel(3) mode_onehot(4)
        validity_mask           4   zeroed Phase 0

    Budget fields (both / arena_diagonal — drone-independent so they're
    commensurable and the policy can derive usable budget itself):
        remaining_budget_ratio = (max_range - odometer) / arena_diagonal  (travel left)
        dist_to_base_ratio     = dist_to_base_xy / arena_diagonal         (cost to fly home)
        usable budget = remaining_budget_ratio - dist_to_base_ratio  (policy learns this)
    """

    OBS_DIM = 283

    # per-neighbor field layout (M=11): rel_pos(3) dist(1) rel_vel(3) mode_onehot(4)
    K = 4 # Number of neighbors
    M = 11 # Dims per neighbor 

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: str,
        arena_size: float = 50.0,
        max_speed: float = 2.0,
        max_range: float = 200.0,
        local_range: float = 25.0,
        local_output_size: int = 11,
    ):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.arena_size = arena_size
        self.arena_diagonal = math.sqrt(2) * arena_size
        self.max_speed = max_speed
        self.max_range = max_range
        self.local_range = local_range
        self.local_output_size = local_output_size

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _mode_onehot(self, modes: torch.Tensor) -> torch.Tensor:
            """
            Convert integer flight modes into a continuous 4-class one-hot tensor.
            
            Args:
                modes: (A,) Long tensor matching MLC.MODE_* constants {0, 1, 2, 3}
            Returns:
                (A, 4) Float tensor representation
            """
            # Ensure correct long integer formatting before one-hot generation
            modes_long = modes.to(torch.long)
            return F.one_hot(modes_long, num_classes=4).float()

    def _heading_sincos(self, yaw: torch.Tensor) -> torch.Tensor:
            """
            Convert directional yaw angles into a continuous sine/cosine coordinate pair.
            
            Args:
                yaw: (A,) Float tensor containing angles in radians
            Returns:
                (A, 2) Float tensor where column 0 is sin(yaw) and column 1 is cos(yaw)
            """
            sin_yaw = torch.sin(yaw)
            cos_yaw = torch.cos(yaw)
            return torch.stack((sin_yaw, cos_yaw), dim=-1)

    # ------------------------------------------------------------------
    # ego block (14)
    # ------------------------------------------------------------------
    def _build_ego(self, positions, velocities, yaw, modes, odometer, dist_to_base):
        """
        All inputs per-agent flattened (A, ...), env-LOCAL coords.
            positions:    (A, 3)
            velocities:   (A, 3)
            yaw:          (A,)
            modes:        (A,)
            odometer:     (A,)
            dist_to_base: (A,)   xy-only distance to base
        returns: (A, 14)

        Budget fields normalized by arena_diagonal (drone-independent) so they're
        commensurable and the policy can derive usable_budget = budget - dist.
        """
        norm_pos = positions / self.arena_size              # (A,3)
        norm_vel = velocities / self.max_speed              # (A,3)
        heading  = self._heading_sincos(yaw)                # (A,2)
        mode_oh  = self._mode_onehot(modes)                 # (A,4)

        remaining = (self.max_range - odometer) / self.arena_diagonal   # (A,)
        dist_r    = dist_to_base / self.arena_diagonal                  # (A,)

        return torch.cat([
            norm_pos,
            norm_vel,
            heading,
            mode_oh,
            remaining.unsqueeze(-1),    # (A,1)
            dist_r.unsqueeze(-1),       # (A,1)
        ], dim=-1)                       # (A,14)

    # ------------------------------------------------------------------
    # neighbor block (K*M + K)  — zeroed in Phase 0
    # ------------------------------------------------------------------
    def _build_neighbors(self, positions, velocities, modes, env_ids) -> tuple:
        """
        Phase 0: single agent per env, no neighbors. Return zeros.
        Phase 1+: find K nearest neighbors within comm range, build M-dim slot each.

        returns:
            neighbor_slots: (A, K*M)
            validity_mask:  (A, K)
        """
        A = positions.shape[0]
        
        # Initialize the flattened neighbor slot array (K=4, M=11 -> 44 dimensions)
        neighbor_slots = torch.zeros(
            (A, self.K * self.M), 
            dtype=positions.dtype, 
            device=positions.device
        )
        
        # Initialize the categorical validity tracking mask (K=4 -> 4 dimensions)
        validity_mask = torch.zeros(
            (A, self.K), 
            dtype=positions.dtype, 
            device=positions.device
        )
        
        return neighbor_slots, validity_mask

    # ------------------------------------------------------------------
    # gating — the per-field observation gate (front end)
    # ------------------------------------------------------------------
    def _apply_gating(self, ego, full_map, local_window, modes):
        """
        Per-field observation gate (front end). Modifies fields based on mode.

        RETURNING / DOCKED: sentinel-fill scan fields (full_map, local_window → 1.0),
                            leave ego fully live so the critic sees a real present drone.
        DEAD:               sentinel-fill scan fields AND zero ego EXCEPT the mode
                            one-hot (cols 8:12), so the slot stays self-describing
                            as "dead" even though DEAD is excluded from the critic.

        ego layout (14): pos(0:3) vel(3:6) heading(6:8) mode(8:12) budget(12) dist(13)

        Gates the COVERAGE POLICY's input only. The reactive avoidance layer
        (controller-side, Phase 1+) reads ungated neighbor data separately.

        returns gated (ego, full_map, local_window)
        """
        active = modes == MLC.MODE_ACTIVE          # (A,)
        dead   = modes == MLC.MODE_DEAD            # (A,)
        scan_inactive = ~active                    # returning / docked / dead

        # sentinel scan fields for any non-active drone
        full_map[scan_inactive]     = 1.0
        local_window[scan_inactive] = 1.0

        # DEAD: zero ego except the mode one-hot (cols 8:12)
        ego[dead, 0:8]   = 0.0     # pos, vel, heading
        ego[dead, 12:14] = 0.0     # budget, dist
        # cols 8:12 (mode one-hot) left intact

        return ego, full_map, local_window

    # ------------------------------------------------------------------
    # top-level assembly
    # ------------------------------------------------------------------
    
    def build(self, positions, velocities, yaw, modes, odometer, dist_to_base, env_ids, coverage_map):
        """
        Assemble the full per-agent observation vector.

        All per-agent inputs are flattened (A, ...) and in env-LOCAL coordinates
        (the env does world->local conversion before calling).

            positions:    (A, 3)
            velocities:   (A, 3)
            yaw:          (A,)
            modes:        (A,)   from controller
            odometer:     (A,)   from controller
            dist_to_base: (A,)   from controller (xy-only)
            env_ids:      (A,)   agent -> env mapping
            coverage_map: CoverageMap instance
        returns: (A, OBS_DIM) == (A, 283)
        """
        A = positions.shape[0]

        # 1. ego block (14)
        ego = self._build_ego(positions, velocities, yaw, modes, odometer, dist_to_base)

        # 2. full coverage map (100) — per-agent copy of its env's map
        full_map = coverage_map.get_full_map(env_ids)

        # 3. local window (121) — zeroed in Phase 0-1, activated in Phase 2+
        local_window = torch.zeros(
            (A, self.local_output_size * self.local_output_size),
            dtype=positions.dtype,
            device=positions.device,
        )

        # 4. neighbor slots (K*M) + validity mask (K) — zeroed in Phase 0
        neighbor_slots, validity_mask = self._build_neighbors(
            positions, velocities, modes, env_ids
        )

        # 5. per-field gating (modifies ego / full_map / local_window in place)
        ego, full_map, local_window = self._apply_gating(
            ego, full_map, local_window, modes
        )

        # 6. concatenate into the fixed-width observation
        obs = torch.cat([
            ego,             # (A, 14)
            full_map,        # (A, 100)
            local_window,    # (A, 121)
            neighbor_slots,  # (A, 44)
            validity_mask,   # (A, 4)
        ], dim=-1)           # (A, 283)
        
        assert obs.shape[-1] == self.OBS_DIM, \
        f"obs width {obs.shape[-1]} != OBS_DIM {self.OBS_DIM}"

        return obs