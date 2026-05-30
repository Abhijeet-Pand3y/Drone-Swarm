import torch


class MidLevelController:
    """
    Bridge between RL policy and physics.
    Translates (dx, dy) policy actions into velocity setpoints.
    Manages mode state machine: ACTIVE → RETURNING → DOCKED → DEAD.
    Owns Z-axis control — policy never controls altitude.
    """

    # --- Mode constants ---
    MODE_ACTIVE    = 0
    MODE_RETURNING = 1
    MODE_DOCKED    = 2
    MODE_DEAD      = 3

    # --- Heights (controller-owned, never RL-controlled) ---
    SCAN_HEIGHT   = 5.0
    RETURN_HEIGHT = 15.0
    DOCK_HEIGHT   = 0.0

    # --- Action space ---
    ACTION_DIM = 2

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: str,
        base_position: tuple = (0.0, 0.0),
        max_range: float = 200.0,
        safety_factor: float = 0.8,
        radius_cap: float = 5.0,
        scan_speed: float = 1.5,
        arrival_threshold: float = 0.5,
        arena_min: tuple = (0.0, 0.0),      # MOVED HERE — into the signature
        arena_max: tuple = (50.0, 50.0),    # MOVED HERE
    ):
        # store args
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.max_range = max_range
        self.safety_factor = safety_factor
        self.radius_cap = radius_cap
        self.scan_speed = scan_speed
        self.arrival_threshold = arrival_threshold

        # arena bounds for geofence clamping (xy only — z is controller-owned)
        self.arena_min = torch.tensor(arena_min, device=device)   # (2,)
        self.arena_max = torch.tensor(arena_max, device=device)   # (2,)

        # derived constants
        self.base_xyz = torch.tensor(
            [base_position[0], base_position[1], self.RETURN_HEIGHT],
            device=device
        )
        self.return_trigger_distance = max_range * safety_factor

        # state tensors
        self.mode = torch.full(
            (num_envs, num_agents),
            self.MODE_ACTIVE,
            dtype=torch.long,
            device=device
        )
        self.odometer = torch.zeros((num_envs, num_agents), dtype=torch.float, device=device)
        self.target = torch.zeros((num_envs, num_agents, 3), dtype=torch.float, device=device)
        self.prev_pos = torch.zeros((num_envs, num_agents, 3), dtype=torch.float, device=device)
        self.dist_to_base = torch.zeros((num_envs, num_agents), dtype=torch.float, device=device)


    def reset(self, env_ids):
        """Reset state for given envs. Modifies in place.

        Note: prev_pos and target must be set to actual drone position
        by the environment after this call.
        """
        self.mode[env_ids] = self.MODE_ACTIVE
        self.odometer[env_ids] = 0.0
        self.target[env_ids] = 0.0
        self.prev_pos[env_ids] = 0.0

    def step(self, policy_action: torch.Tensor, current_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            policy_action: (num_envs, num_agents, 2) — (dx, dy) in [-1, 1]
            current_pos:   (num_envs, num_agents, 3) — drone world positions from sim
        Returns:
            velocity_setpoint: (num_envs, num_agents, 3) — target velocity for PD layer
        """

        # --- 1. Update odometer (ACTIVE drones only) ---
        distance_moved = torch.norm(current_pos - self.prev_pos, dim=-1)  # (num_envs, num_agents)
        active = (self.mode == self.MODE_ACTIVE)                          # CHANGED
        self.odometer += distance_moved * active.float()                 # CHANGED: freeze non-active
        self.prev_pos = current_pos.clone()

        # --- 2. Return trigger: ACTIVE → RETURNING ---
        mask_return = (self.mode == self.MODE_ACTIVE) & (self.odometer >= self.return_trigger_distance)
        self.mode[mask_return] = self.MODE_RETURNING

        # --- 3. Compute target per mode ---
        active_mask    = self.mode == self.MODE_ACTIVE      # (num_envs, num_agents)
        returning_mask = self.mode == self.MODE_RETURNING

        # default target = stay put (covers DOCKED and DEAD)
        target = current_pos.clone()

        # ACTIVE target: current xy + scaled action, z = SCAN_HEIGHT
        active_target = current_pos.clone()
        active_target[..., 0:2] = current_pos[..., 0:2] + policy_action * self.radius_cap
        active_target[..., 2]   = self.SCAN_HEIGHT
        target = torch.where(active_mask.unsqueeze(-1), active_target, target)

        # RETURNING target: base station at RETURN_HEIGHT
        target = torch.where(returning_mask.unsqueeze(-1), self.base_xyz, target)

        self.raw_target = target.clone()

        # geofence: clamp target xy into arena bounds (z untouched)
        target[..., 0] = target[..., 0].clamp(self.arena_min[0], self.arena_max[0])
        target[..., 1] = target[..., 1].clamp(self.arena_min[1], self.arena_max[1])

        self.target = target

        
        # --- 4. Arrival check: RETURNING → DOCKED ---
        dist_to_base = torch.norm(current_pos - self.base_xyz, dim=-1)   # (num_envs, num_agents) 3D
        mask_docked = (self.mode == self.MODE_RETURNING) & (dist_to_base <= self.arrival_threshold)
        self.mode[mask_docked] = self.MODE_DOCKED

        # store xy-only distance to base for the observation builder
        # (vertical is mode-determined, not part of "how far to fly home";
        #  a real impl can add a fixed buffer for the climb/descent)
        base_xy = self.base_xyz[:2]                                       # (2,)
        self.dist_to_base = torch.norm(
            current_pos[..., :2] - base_xy, dim=-1
        ) 

        # --- 5. Velocity setpoint toward target ---
        direction = target - current_pos                                 # (num_envs, num_agents, 3)
        dist = torch.norm(direction, dim=-1, keepdim=True)               # (num_envs, num_agents, 1)
        direction_norm = direction / (dist + 1e-8)                       # avoid div by zero
        velocity = direction_norm * self.scan_speed

        # zero out velocity for DOCKED and DEAD drones                   # CHANGED
        inactive_mask = (
            (self.mode == self.MODE_DOCKED) | (self.mode == self.MODE_DEAD)
        ).unsqueeze(-1)                                                   # CHANGED
        velocity = torch.where(inactive_mask, torch.zeros_like(velocity), velocity)

        return velocity

    def seed_positions(self, positions: torch.Tensor, env_ids: torch.Tensor | None = None):
        """
        Seed prev_pos and target with drones' actual spawn positions.
        MUST be called by the env immediately after reset(), before the first
        step(), or the first odometer update adds a bogus jump from the origin.

        Args:
            positions: (num_envs, num_agents, 3) if env_ids is None,
                    else (len(env_ids), num_agents, 3)
            env_ids:   optional subset of envs being reset
        """
        base_xy = self.base_xyz[:2]
        if env_ids is None:
            self.prev_pos = positions.clone()
            self.target = positions.clone()
            self.dist_to_base = torch.norm(positions[..., :2] - base_xy, dim=-1)
        else:
            self.prev_pos[env_ids] = positions.clone()
            self.target[env_ids] = positions.clone()
            self.dist_to_base[env_ids] = torch.norm(positions[..., :2] - base_xy, dim=-1)




    def mark_dead(self, dead_mask: torch.Tensor):                        
        """
        Flag drones as DEAD (crashed / out-of-bounds). Terminal until reset.
        The environment decides what counts as death (out-of-bounds, ground
        collision, NaN); the controller just records it.

        Args:
            dead_mask: (num_envs, num_agents) bool — True where the drone died
        """
        self.mode[dead_mask] = self.MODE_DEAD