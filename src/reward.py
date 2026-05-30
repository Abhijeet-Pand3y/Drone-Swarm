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
        # --- gated-term inputs, ignored while weights are 0 ---
        wall_overshoot=None,    # (A,) how far raw_target exceeded arena (ACTIVE)
        milestone_crossed=None, # (A,) bool, env crossed completion threshold this step
        docked_contrib=None,    # (A,) bool, drone docked after contributing coverage
        collision=None,         # (A,) collision signal (Phase 1+)
    ) -> torch.Tensor:
        """
        All inputs per-agent flattened (A,). Returns (A,) reward.

        Note: progress_delta / completion_count arrive per-AGENT here. The env is
        responsible for mapping CoverageMap's per-ENV deltas to per-agent (Phase 0:
        env delta -> its single agent; Phase 1: per-drone under Model 2).
        """
        # TODO 1 — active terms
        #   coverage = coverage_weight * progress_delta
        #   completion = completion_weight * completion_count
        #   time = time_penalty (subtract)
        #   reward = coverage + completion - time

        # TODO 2 — gated terms (each multiplied by its weight; 0 disables)
        #   if wall_weight:      reward -= wall_weight * wall_overshoot
        #   if milestone_weight: reward += milestone_weight * milestone_crossed
        #   if safe_return:      reward += safe_return_weight * docked_contrib
        #   if collision:        reward -= collision_weight * collision
        #   (write these so a None input is safe when the weight is 0)

        # TODO 3 — mask DEAD to zero reward
        #   dead = modes == MLC.MODE_DEAD
        #   reward[dead] = 0.0

        # return reward  # (A,)
        pass