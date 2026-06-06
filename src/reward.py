import torch
from mid_level_controller import MidLevelController as MLC


class RewardComputer:
    """
    Computes per-agent reward each step. Pure computation — receives state from
    the env (deltas from CoverageMap, modes from controller), returns (A,) reward.

    Computes honest reward for ACTIVE / RETURNING / DOCKED. DEAD → 0.
    Actor/critic loss masking lives in the training loop, NOT here.

    Phase 0 active terms: coverage_delta, completion_bonus, time_penalty.
    All other terms present but weight-gated to 0.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: str,
        coverage_weight: float = 1.0,
        completion_weight: float = 0.3,
        time_penalty: float = 0.01,
        wall_weight: float = 0.0,         # gated — ACTIVE wall-push, Phase 0 off
        milestone_weight: float = 0.0,    # gated — env crosses 95%, Phase 0 off
        safe_return_weight: float = 0.0,  # gated — dock after contributing, off
        collision_weight: float = 0.0,    # gated — Phase 1+ stub
    ):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        self.coverage_weight = coverage_weight
        self.completion_weight = completion_weight
        self.time_penalty = time_penalty
        self.wall_weight = wall_weight
        self.milestone_weight = milestone_weight
        self.safe_return_weight = safe_return_weight
        self.collision_weight = collision_weight

    def compute(
        self,
        progress_delta,      # (A,) per-agent scan-progress added this step
        completion_count,    # (A,) per-agent cells newly completed this step
        modes,               # (A,) controller modes
        wall_overshoot=None,    # (A,) ACTIVE wall-push magnitude (gated)
        milestone_crossed=None, # (A,) bool, env crossed completion threshold (gated)
        docked_contrib=None,    # (A,) bool, docked after contributing (gated)
        collision=None,         # (A,) collision signal (gated, Phase 1+)
    ) -> torch.Tensor:
        """
        All inputs per-agent flattened (A,). Returns (A,) reward.

        Honest reward for ACTIVE / RETURNING / DOCKED; DEAD forced to 0.
        Actor/critic loss masking is applied later in the training loop, not here.
        """
        active = (modes == MLC.MODE_ACTIVE)

        # --- active terms ---
        # dense coverage: progress added this step
        reward = self.coverage_weight * progress_delta

        # completion bonus: cells that crossed the threshold this step
        reward = reward + self.completion_weight * completion_count

        # time penalty: ACTIVE drones only (productive-urgency pressure;
        # returning/docked drones aren't under scanning time pressure)
        reward = reward - self.time_penalty * active.float()

        # --- gated terms (each guarded so a None input is safe when weight is 0) ---
        if self.wall_weight != 0.0:
            reward = reward - self.wall_weight * wall_overshoot

        if self.milestone_weight != 0.0:
            reward = reward + self.milestone_weight * milestone_crossed.float()

        if self.safe_return_weight != 0.0:
            reward = reward + self.safe_return_weight * docked_contrib.float()

        if self.collision_weight != 0.0:
            reward = reward - self.collision_weight * collision

        # --- DEAD gets zero reward ---
        dead = (modes == MLC.MODE_DEAD)
        reward = torch.where(dead, torch.zeros_like(reward), reward)

        return reward