"""
Test suite for RewardComputer.

    python test_reward.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from reward import RewardComputer
from mid_level_controller import MidLevelController as MLC


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_reward(**kw):
    return RewardComputer(num_envs=2, num_agents=1, device=DEVICE, **kw)


# ----------------------------------------------------------------------
# init
# ----------------------------------------------------------------------
def test_init_defaults():
    r = make_reward()
    assert r.coverage_weight == 1.0
    assert r.completion_weight == 0.3
    assert r.time_penalty == 0.01
    assert r.wall_weight == 0.0
    assert r.collision_weight == 0.0
    print("PASS test_init_defaults")


# ----------------------------------------------------------------------
# coverage delta — the core dense signal
# ----------------------------------------------------------------------
def test_coverage_reward_basic():
    """Productive step: coverage_delta > 0 → positive reward."""
    r = make_reward()
    progress = torch.tensor([0.2, 0.0], device=DEVICE)   # agent 0 scanned, agent 1 idle
    completion = torch.zeros(2, device=DEVICE)
    modes = torch.full((2,), MLC.MODE_ACTIVE, dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    # agent 0: 1.0*0.2 + 0.3*0 - 0.01 = 0.19
    assert abs(reward[0].item() - 0.19) < 1e-6, reward[0].item()
    # agent 1: 1.0*0.0 + 0.3*0 - 0.01 = -0.01
    assert abs(reward[1].item() - (-0.01)) < 1e-6, reward[1].item()
    print("PASS test_coverage_reward_basic")


def test_coverage_with_completion():
    """Completing a cell adds the completion bonus."""
    r = make_reward()
    progress = torch.tensor([0.1], device=DEVICE)   # clamp-accounted final step
    completion = torch.tensor([1.0], device=DEVICE)  # one cell completed
    modes = torch.tensor([MLC.MODE_ACTIVE], dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    # 1.0*0.1 + 0.3*1.0 - 0.01 = 0.39
    assert abs(reward[0].item() - 0.39) < 1e-6, reward[0].item()
    print("PASS test_coverage_with_completion")


def test_coverage_zero_on_no_progress():
    """No progress, no completion → only time penalty."""
    r = make_reward()
    progress = torch.zeros(1, device=DEVICE)
    completion = torch.zeros(1, device=DEVICE)
    modes = torch.tensor([MLC.MODE_ACTIVE], dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    assert abs(reward[0].item() - (-0.01)) < 1e-6, reward[0].item()
    print("PASS test_coverage_zero_on_no_progress")


# ----------------------------------------------------------------------
# time penalty — ACTIVE only
# ----------------------------------------------------------------------
def test_time_penalty_active_only():
    """ACTIVE pays time penalty. RETURNING/DOCKED do not."""
    r = make_reward()
    progress = torch.zeros(3, device=DEVICE)
    completion = torch.zeros(3, device=DEVICE)
    modes = torch.tensor([MLC.MODE_ACTIVE, MLC.MODE_RETURNING, MLC.MODE_DOCKED],
                         dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    assert abs(reward[0].item() - (-0.01)) < 1e-6, f"active should pay time penalty, got {reward[0].item()}"
    assert abs(reward[1].item() - 0.0) < 1e-6, f"returning should not pay time penalty, got {reward[1].item()}"
    assert abs(reward[2].item() - 0.0) < 1e-6, f"docked should not pay time penalty, got {reward[2].item()}"
    print("PASS test_time_penalty_active_only")


# ----------------------------------------------------------------------
# DEAD masking
# ----------------------------------------------------------------------
def test_dead_gets_zero():
    """DEAD reward is forced to 0 regardless of what terms compute."""
    r = make_reward()
    progress = torch.tensor([0.2], device=DEVICE)  # would normally be positive
    completion = torch.tensor([1.0], device=DEVICE)
    modes = torch.tensor([MLC.MODE_DEAD], dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    assert reward[0].item() == 0.0, f"dead must get 0, got {reward[0].item()}"
    print("PASS test_dead_gets_zero")


# ----------------------------------------------------------------------
# returning drone honest reward
# ----------------------------------------------------------------------
def test_returning_honest_near_zero():
    """RETURNING drone: no coverage (not scanning), no time penalty → ~0 naturally."""
    r = make_reward()
    progress = torch.zeros(1, device=DEVICE)    # not scanning, delta = 0
    completion = torch.zeros(1, device=DEVICE)
    modes = torch.tensor([MLC.MODE_RETURNING], dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    assert abs(reward[0].item()) < 1e-6, f"returning should be ~0, got {reward[0].item()}"
    print("PASS test_returning_honest_near_zero")


# ----------------------------------------------------------------------
# gated terms — None inputs safe when weight=0
# ----------------------------------------------------------------------
def test_gated_terms_safe_with_none():
    """All gated weights=0, inputs=None → no crash."""
    r = make_reward()  # all gated weights 0 by default
    progress = torch.zeros(2, device=DEVICE)
    completion = torch.zeros(2, device=DEVICE)
    modes = torch.full((2,), MLC.MODE_ACTIVE, dtype=torch.long, device=DEVICE)
    # all gated inputs left as None (default)
    reward = r.compute(progress, completion, modes)
    assert reward.shape == (2,), reward.shape
    print("PASS test_gated_terms_safe_with_none")


def test_wall_penalty_when_active():
    """Wall penalty fires when weight > 0 and input provided."""
    r = make_reward(wall_weight=1.0)
    progress = torch.zeros(2, device=DEVICE)
    completion = torch.zeros(2, device=DEVICE)
    modes = torch.full((2,), MLC.MODE_ACTIVE, dtype=torch.long, device=DEVICE)
    overshoot = torch.tensor([3.0, 0.0], device=DEVICE)  # agent 0 pushed 3m past wall
    reward = r.compute(progress, completion, modes, wall_overshoot=overshoot)
    # agent 0: 0 + 0 - 0.01 - 1.0*3.0 = -3.01
    assert abs(reward[0].item() - (-3.01)) < 1e-4, reward[0].item()
    # agent 1: 0 + 0 - 0.01 - 0 = -0.01
    assert abs(reward[1].item() - (-0.01)) < 1e-4, reward[1].item()
    print("PASS test_wall_penalty_when_active")


# ----------------------------------------------------------------------
# output shape
# ----------------------------------------------------------------------
def test_output_shape():
    r = make_reward()
    A = 5
    progress = torch.zeros(A, device=DEVICE)
    completion = torch.zeros(A, device=DEVICE)
    modes = torch.full((A,), MLC.MODE_ACTIVE, dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    assert reward.shape == (A,), reward.shape
    print("PASS test_output_shape")


# ----------------------------------------------------------------------
# custom weights
# ----------------------------------------------------------------------
def test_custom_weights():
    """Different coverage weight changes the reward proportionally."""
    r = make_reward(coverage_weight=2.0, time_penalty=0.0)
    progress = torch.tensor([0.2], device=DEVICE)
    completion = torch.zeros(1, device=DEVICE)
    modes = torch.tensor([MLC.MODE_ACTIVE], dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    # 2.0 * 0.2 = 0.4
    assert abs(reward[0].item() - 0.4) < 1e-6, reward[0].item()
    print("PASS test_custom_weights")


# ----------------------------------------------------------------------
# multi-agent independence
# ----------------------------------------------------------------------
def test_multi_agent_independent():
    """Each agent's reward depends only on its own inputs."""
    r = make_reward()
    progress = torch.tensor([0.2, 0.0, 0.1], device=DEVICE)
    completion = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    modes = torch.tensor([MLC.MODE_ACTIVE, MLC.MODE_ACTIVE, MLC.MODE_ACTIVE],
                         dtype=torch.long, device=DEVICE)
    reward = r.compute(progress, completion, modes)
    # agent 0: 1.0*0.2 - 0.01 = 0.19
    assert abs(reward[0].item() - 0.19) < 1e-5, reward[0].item()
    # agent 1: 0 - 0.01 = -0.01
    assert abs(reward[1].item() - (-0.01)) < 1e-5, reward[1].item()
    # agent 2: 1.0*0.1 + 0.3*1.0 - 0.01 = 0.39
    assert abs(reward[2].item() - 0.39) < 1e-5, reward[2].item()
    print("PASS test_multi_agent_independent")


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running RewardComputer tests on device: {DEVICE}\n")
    tests = [
        test_init_defaults,
        test_coverage_reward_basic,
        test_coverage_with_completion,
        test_coverage_zero_on_no_progress,
        test_time_penalty_active_only,
        test_dead_gets_zero,
        test_returning_honest_near_zero,
        test_gated_terms_safe_with_none,
        test_wall_penalty_when_active,
        test_output_shape,
        test_custom_weights,
        test_multi_agent_independent,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'='*55}")
    print(f"All {len(tests)} tests passed." if failed == 0 else f"{failed}/{len(tests)} failed.")
