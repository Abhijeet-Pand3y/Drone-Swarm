"""
Test suite for ObservationBuilder.

Run from wherever observation.py, coverage_map.py, mid_level_controller.py
are importable (adjust imports below if your layout uses subpackages).

    python test_observation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from observation import ObservationBuilder as OB
from coverage_map import CoverageMap
from mid_level_controller import MidLevelController as MLC




DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ARENA = 50.0
CELL = 5.0
MAX_SPEED = 2.0
MAX_RANGE = 200.0


def make_builder(num_envs=2, num_agents=1):
    return OB(num_envs=num_envs, num_agents=num_agents, device=DEVICE,
              arena_size=ARENA, max_speed=MAX_SPEED, max_range=MAX_RANGE,
              local_range=25.0, local_output_size=11)


def make_cmap(num_envs=2):
    return CoverageMap(num_envs=num_envs, arena_size=ARENA, cell_size=CELL, device=DEVICE)


def make_inputs(A, env_ids, mode=MLC.MODE_ACTIVE):
    """Build a clean set of per-agent inputs."""
    positions    = torch.tensor([[25.0, 25.0, 5.0]] * A, device=DEVICE)
    velocities   = torch.zeros((A, 3), device=DEVICE)
    yaw          = torch.zeros(A, device=DEVICE)
    modes        = torch.full((A,), mode, dtype=torch.long, device=DEVICE)
    odometer     = torch.zeros(A, device=DEVICE)
    dist_to_base = torch.zeros(A, device=DEVICE)
    return positions, velocities, yaw, modes, odometer, dist_to_base, env_ids


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def test_mode_onehot():
    ob = make_builder()
    modes = torch.tensor([0, 1, 2, 3], device=DEVICE)
    oh = ob._mode_onehot(modes)
    assert oh.shape == (4, 4)
    assert torch.equal(oh[0], torch.tensor([1., 0., 0., 0.], device=DEVICE))
    assert torch.equal(oh[3], torch.tensor([0., 0., 0., 1.], device=DEVICE))
    print("PASS test_mode_onehot")


def test_heading_sincos():
    ob = make_builder()
    yaw = torch.tensor([0.0], device=DEVICE)
    sc = ob._heading_sincos(yaw)
    assert sc.shape == (1, 2)
    # yaw=0 → sin=0, cos=1
    assert abs(sc[0, 0].item() - 0.0) < 1e-6
    assert abs(sc[0, 1].item() - 1.0) < 1e-6
    print("PASS test_heading_sincos")


def test_arena_diagonal():
    ob = make_builder()
    expected = (2 ** 0.5) * ARENA  # ~70.71
    assert abs(ob.arena_diagonal - expected) < 1e-4, ob.arena_diagonal
    print("PASS test_arena_diagonal")


# ----------------------------------------------------------------------
# ego block
# ----------------------------------------------------------------------
def test_ego_shape():
    ob = make_builder()
    A = 3
    pos = torch.zeros((A, 3), device=DEVICE)
    vel = torch.zeros((A, 3), device=DEVICE)
    yaw = torch.zeros(A, device=DEVICE)
    modes = torch.zeros(A, dtype=torch.long, device=DEVICE)
    odo = torch.zeros(A, device=DEVICE)
    dtb = torch.zeros(A, device=DEVICE)
    ego = ob._build_ego(pos, vel, yaw, modes, odo, dtb)
    assert ego.shape == (A, 14), ego.shape
    print("PASS test_ego_shape")


def test_ego_position_normalized():
    ob = make_builder()
    pos = torch.tensor([[50.0, 25.0, 5.0]], device=DEVICE)  # x at full arena
    vel = torch.zeros((1, 3), device=DEVICE)
    yaw = torch.zeros(1, device=DEVICE)
    modes = torch.zeros(1, dtype=torch.long, device=DEVICE)
    odo = torch.zeros(1, device=DEVICE)
    dtb = torch.zeros(1, device=DEVICE)
    ego = ob._build_ego(pos, vel, yaw, modes, odo, dtb)
    # pos x = 50/50 = 1.0
    assert abs(ego[0, 0].item() - 1.0) < 1e-6, ego[0, 0]
    # pos y = 25/50 = 0.5
    assert abs(ego[0, 1].item() - 0.5) < 1e-6, ego[0, 1]
    print("PASS test_ego_position_normalized")


def test_ego_budget_fields():
    ob = make_builder()
    pos = torch.zeros((1, 3), device=DEVICE)
    vel = torch.zeros((1, 3), device=DEVICE)
    yaw = torch.zeros(1, device=DEVICE)
    modes = torch.zeros(1, dtype=torch.long, device=DEVICE)
    odo = torch.tensor([50.0], device=DEVICE)         # used 50m of 200m range
    dtb = torch.tensor([35.355], device=DEVICE)        # ~ half diagonal
    ego = ob._build_ego(pos, vel, yaw, modes, odo, dtb)
    diag = ob.arena_diagonal
    # remaining_budget = (200 - 50) / diag
    exp_budget = (MAX_RANGE - 50.0) / diag
    # dist_ratio = 35.355 / diag
    exp_dist = 35.355 / diag
    assert abs(ego[0, 12].item() - exp_budget) < 1e-3, (ego[0, 12].item(), exp_budget)
    assert abs(ego[0, 13].item() - exp_dist) < 1e-3, (ego[0, 13].item(), exp_dist)
    print("PASS test_ego_budget_fields")


def test_ego_budget_can_exceed_one():
    """max_range > arena_diagonal → remaining budget ratio > 1.0, not clamped."""
    ob = make_builder()
    pos = torch.zeros((1, 3), device=DEVICE)
    vel = torch.zeros((1, 3), device=DEVICE)
    yaw = torch.zeros(1, device=DEVICE)
    modes = torch.zeros(1, dtype=torch.long, device=DEVICE)
    odo = torch.zeros(1, device=DEVICE)               # full range left
    dtb = torch.zeros(1, device=DEVICE)
    ego = ob._build_ego(pos, vel, yaw, modes, odo, dtb)
    # remaining = 200/70.7 = ~2.83  > 1.0
    assert ego[0, 12].item() > 1.0, ego[0, 12].item()
    print("PASS test_ego_budget_can_exceed_one")


# ----------------------------------------------------------------------
# neighbor stub
# ----------------------------------------------------------------------
def test_neighbors_zeroed():
    ob = make_builder()
    A = 3
    pos = torch.zeros((A, 3), device=DEVICE)
    vel = torch.zeros((A, 3), device=DEVICE)
    modes = torch.zeros(A, dtype=torch.long, device=DEVICE)
    env_ids = torch.arange(A, device=DEVICE)
    slots, mask = ob._build_neighbors(pos, vel, modes, env_ids)
    assert slots.shape == (A, OB.K * OB.M), slots.shape
    assert mask.shape == (A, OB.K), mask.shape
    assert slots.sum().item() == 0.0
    assert mask.sum().item() == 0.0
    print("PASS test_neighbors_zeroed")


# ----------------------------------------------------------------------
# full build
# ----------------------------------------------------------------------
def test_build_shape():
    ob = make_builder(num_envs=2, num_agents=1)
    cmap = make_cmap(num_envs=2)
    env_ids = torch.tensor([0, 1], device=DEVICE)
    A = 2
    inp = make_inputs(A, env_ids)
    obs = ob.build(*inp, coverage_map=cmap)
    assert obs.shape == (A, OB.OBS_DIM), obs.shape
    assert obs.shape[-1] == 283
    print("PASS test_build_shape")


def test_build_block_layout():
    """Confirm the concatenation order: ego(14) map(100) window(121) nbr(44) mask(4)."""
    ob = make_builder(num_envs=1, num_agents=1)
    cmap = make_cmap(num_envs=1)
    env_ids = torch.tensor([0], device=DEVICE)
    inp = make_inputs(1, env_ids)
    obs = ob.build(*inp, coverage_map=cmap)
    # ego is cols 0:14, map 14:114, window 114:235, nbr 235:279, mask 279:283
    assert obs.shape[-1] == 283
    # window block (114:235) should be all zero in Phase 0
    assert obs[0, 114:235].sum().item() == 0.0, "local window must be zeroed P0"
    # neighbor block (235:279) zero
    assert obs[0, 235:279].sum().item() == 0.0, "neighbor slots must be zeroed P0"
    # validity mask (279:283) zero
    assert obs[0, 279:283].sum().item() == 0.0, "validity mask must be zeroed P0"
    print("PASS test_build_block_layout")


# ----------------------------------------------------------------------
# gating
# ----------------------------------------------------------------------
def test_gating_active_no_change():
    """ACTIVE drone over a scanned map: full_map reflects real coverage, not sentinel."""
    ob = make_builder(num_envs=1, num_agents=1)
    cmap = make_cmap(num_envs=1)
    # mark a few cells so map is partially 1, partially 0
    cmap.mark_scanned(torch.tensor([0], device=DEVICE),
                      torch.tensor([[12.5, 12.5, 5.0]], device=DEVICE), scan_range=2.0)
    env_ids = torch.tensor([0], device=DEVICE)
    inp = make_inputs(1, env_ids, mode=MLC.MODE_ACTIVE)
    obs = ob.build(*inp, coverage_map=cmap)
    full_map = obs[0, 14:114]
    # active drone: map should NOT be all-ones (it's mostly unscanned)
    assert full_map.sum().item() < 100.0, "active map must reflect real (mostly unscanned) coverage"
    assert full_map.sum().item() > 0.0, "should have at least the one marked cell"
    print("PASS test_gating_active_no_change")


def test_gating_returning_sentinels_map_keeps_ego():
    """RETURNING: scan fields → 1.0, but ego stays live (position non-zero)."""
    ob = make_builder(num_envs=1, num_agents=1)
    cmap = make_cmap(num_envs=1)  # empty map (all zeros)
    env_ids = torch.tensor([0], device=DEVICE)
    inp = make_inputs(1, env_ids, mode=MLC.MODE_RETURNING)
    obs = ob.build(*inp, coverage_map=cmap)
    full_map = obs[0, 14:114]
    local_window = obs[0, 114:235]
    # scan fields sentinel-filled to 1.0 despite empty real map
    assert full_map.sum().item() == 100.0, "returning full_map must be all 1.0 sentinel"
    assert local_window.sum().item() == 121.0, "returning local_window must be all 1.0 sentinel"
    # ego still live — position (25,25,5) normalized, not zeroed
    ego = obs[0, 0:14]
    assert ego[0].item() > 0.0, "returning ego position must stay live for critic"
    print("PASS test_gating_returning_sentinels_map_keeps_ego")


def test_gating_dead_zeros_ego_except_mode():
    """DEAD: scan fields → 1.0, ego zeroed EXCEPT mode one-hot (cols 8:12)."""
    ob = make_builder(num_envs=1, num_agents=1)
    cmap = make_cmap(num_envs=1)
    env_ids = torch.tensor([0], device=DEVICE)
    inp = make_inputs(1, env_ids, mode=MLC.MODE_DEAD)
    obs = ob.build(*inp, coverage_map=cmap)
    ego = obs[0, 0:14]
    # pos/vel/heading (0:8) zeroed
    assert ego[0:8].sum().item() == 0.0, "dead ego pos/vel/heading must be zeroed"
    # budget/dist (12:14) zeroed
    assert ego[12:14].sum().item() == 0.0, "dead ego budget/dist must be zeroed"
    # mode one-hot (8:12) intact — DEAD is index 3 → col 11 should be 1
    assert ego[11].item() == 1.0, "dead mode one-hot must be preserved (col 11)"
    assert ego[8:12].sum().item() == 1.0, "exactly one mode bit set"
    # scan fields sentinel
    assert obs[0, 14:114].sum().item() == 100.0, "dead full_map sentinel"
    print("PASS test_gating_dead_zeros_ego_except_mode")


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running ObservationBuilder tests on device: {DEVICE}\n")

    tests = [
        test_mode_onehot,
        test_heading_sincos,
        test_arena_diagonal,
        test_ego_shape,
        test_ego_position_normalized,
        test_ego_budget_fields,
        test_ego_budget_can_exceed_one,
        test_neighbors_zeroed,
        test_build_shape,
        test_build_block_layout,
        test_gating_active_no_change,
        test_gating_returning_sentinels_map_keeps_ego,
        test_gating_dead_zeros_ego_except_mode,
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
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{failed}/{len(tests)} tests failed.")
