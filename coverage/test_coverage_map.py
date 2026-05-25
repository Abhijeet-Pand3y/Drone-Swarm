"""
Test suite for CoverageMap.

Run from project root:
    python test_coverage_map.py

Tests are plain asserts — no pytest dependency required, but pytest-compatible
(each test_* function can also be collected by pytest if you prefer).
"""

import torch
from coverage_map import CoverageMap


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def make_map(num_envs=2, arena_size=50.0, cell_size=5.0):
    return CoverageMap(num_envs=num_envs, arena_size=arena_size,
                       cell_size=cell_size, device=DEVICE)


def t(*vals):
    return torch.tensor(vals, device=DEVICE)


# ----------------------------------------------------------------------
# __init__ / grid construction
# ----------------------------------------------------------------------
def test_grid_dimensions():
    cm = make_map(num_envs=4, arena_size=50.0, cell_size=5.0)
    assert cm.grid_size == 10, f"expected grid_size 10, got {cm.grid_size}"
    assert cm.grid.shape == (4, 10, 10), f"bad grid shape {cm.grid.shape}"
    assert cm.grid.dtype == torch.bool
    assert cm.grid.sum().item() == 0, "grid should start empty"
    print("PASS test_grid_dimensions")


def test_non_divisible_arena():
    # arena not perfectly divisible by cell_size — int() floors
    cm = make_map(num_envs=1, arena_size=52.0, cell_size=5.0)
    assert cm.grid_size == 10, f"expected floor(52/5)=10, got {cm.grid_size}"
    print("PASS test_non_divisible_arena")


def test_cell_centers():
    cm = make_map(num_envs=1, arena_size=50.0, cell_size=5.0)
    assert cm.cell_centers.shape == (10, 10, 2)
    # cell (0,0) center should be (2.5, 2.5)
    assert torch.allclose(cm.cell_centers[0, 0], t(2.5, 2.5))
    # cell (9,9) center should be (47.5, 47.5)
    assert torch.allclose(cm.cell_centers[9, 9], t(47.5, 47.5))
    # cell (5,3) center should be (27.5, 17.5)
    assert torch.allclose(cm.cell_centers[5, 3], t(27.5, 17.5))
    print("PASS test_cell_centers")


# ----------------------------------------------------------------------
# pos_to_cell
# ----------------------------------------------------------------------
def test_pos_to_cell_basic():
    cm = make_map()
    pos = torch.tensor([
        [2.5, 2.5, 5.0],    # cell (0,0)
        [27.3, 18.0, 5.0],  # cell (5,3)
        [47.5, 47.5, 5.0],  # cell (9,9)
    ], device=DEVICE)
    cells = cm.pos_to_cell(pos)
    assert torch.equal(cells, torch.tensor([[0, 0], [5, 3], [9, 9]], device=DEVICE)), cells
    print("PASS test_pos_to_cell_basic")


def test_pos_to_cell_clamp():
    cm = make_map()
    # out of bounds positions should clamp to valid range
    pos = torch.tensor([
        [-10.0, -10.0, 5.0],   # below 0 → clamp to (0,0)
        [999.0, 999.0, 5.0],   # above arena → clamp to (9,9)
    ], device=DEVICE)
    cells = cm.pos_to_cell(pos)
    assert torch.equal(cells, torch.tensor([[0, 0], [9, 9]], device=DEVICE)), cells
    print("PASS test_pos_to_cell_clamp")


def test_pos_to_cell_accepts_2d_and_3d():
    cm = make_map()
    pos3 = torch.tensor([[27.0, 18.0, 5.0]], device=DEVICE)
    pos2 = torch.tensor([[27.0, 18.0]], device=DEVICE)
    assert torch.equal(cm.pos_to_cell(pos3), cm.pos_to_cell(pos2))
    print("PASS test_pos_to_cell_accepts_2d_and_3d")


# ----------------------------------------------------------------------
# mark_scanned
# ----------------------------------------------------------------------
def test_mark_single_agent_one_cell():
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.0, 18.0, 5.0]], device=DEVICE)
    # scan_range smaller than cell → marks only the cell whose center is within range
    cm.mark_scanned(env_ids, pos, scan_range=2.5)
    # drone at (27,18), nearest center (27.5,17.5) dist ~0.7 < 2.5 → marked
    assert cm.grid[0, 5, 3].item() is True
    assert cm.grid.sum().item() == 1, f"expected exactly 1 cell, got {cm.grid.sum().item()}"
    print("PASS test_mark_single_agent_one_cell")


def test_mark_larger_scan_range():
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)  # exactly on a cell center
    # scan_range 5m → should catch the center cell plus immediate neighbors
    cm.mark_scanned(env_ids, pos, scan_range=5.0)
    n = cm.grid.sum().item()
    assert n >= 5, f"larger scan range should mark multiple cells, got {n}"
    assert cm.grid[0, 5, 3].item() is True, "center cell must be marked"
    print(f"PASS test_mark_larger_scan_range (marked {n} cells)")


def test_mark_overlapping_agents_same_env():
    """Two agents in the same env with overlapping scans — accumulate path."""
    cm = make_map(num_envs=2)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    positions = torch.tensor([
        [12.0, 12.0, 5.0],  # env 0 agent A
        [13.0, 13.0, 5.0],  # env 0 agent B — overlaps A
        [40.0, 40.0, 5.0],  # env 1 agent C
    ], device=DEVICE)
    cm.mark_scanned(env_ids, positions, scan_range=5.0)
    # env 0 should have a merged (OR'd) region, env 1 separate
    assert cm.grid[0].sum().item() > 0, "env 0 should have scanned cells"
    assert cm.grid[1].sum().item() > 0, "env 1 should have scanned cells"
    # critical: no crash on duplicate env_ids in accumulate
    print(f"PASS test_mark_overlapping_agents_same_env "
          f"(env0={cm.grid[0].sum().item()}, env1={cm.grid[1].sum().item()})")


def test_mark_is_cumulative():
    """Marking again should not erase previous scans (OR behavior)."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    # use cell-center positions so a small scan_range reliably hits exactly one cell
    cm.mark_scanned(env_ids, torch.tensor([[12.5, 12.5, 5.0]], device=DEVICE), scan_range=2.0)
    first = cm.grid.sum().item()
    assert first == 1, f"first scan should mark exactly 1 cell, got {first}"
    cm.mark_scanned(env_ids, torch.tensor([[42.5, 42.5, 5.0]], device=DEVICE), scan_range=2.0)
    second = cm.grid.sum().item()
    assert second == first + 1, f"second scan should add a cell, got {first}->{second}"
    print("PASS test_mark_is_cumulative")


def test_mark_does_not_leak_across_envs():
    cm = make_map(num_envs=3)
    env_ids = t(1).long()
    cm.mark_scanned(env_ids, torch.tensor([[22.5, 22.5, 5.0]], device=DEVICE), scan_range=2.0)
    assert cm.grid[0].sum().item() == 0, "env 0 should be untouched"
    assert cm.grid[1].sum().item() > 0, "env 1 should be marked"
    assert cm.grid[2].sum().item() == 0, "env 2 should be untouched"
    print("PASS test_mark_does_not_leak_across_envs")


# ----------------------------------------------------------------------
# reset
# ----------------------------------------------------------------------
def test_reset_clears_only_given_envs():
    cm = make_map(num_envs=3)
    all_ids = torch.arange(3, device=DEVICE)
    pos = torch.tensor([
        [22.5, 22.5, 5.0],
        [22.5, 22.5, 5.0],
        [22.5, 22.5, 5.0],
    ], device=DEVICE)
    cm.mark_scanned(all_ids, pos, scan_range=2.0)
    assert (cm.grid.sum(dim=(1, 2)) > 0).all(), "all envs should have scans"

    cm.reset(torch.tensor([0, 2], device=DEVICE))
    assert cm.grid[0].sum().item() == 0, "env 0 cleared"
    assert cm.grid[1].sum().item() > 0, "env 1 preserved"
    assert cm.grid[2].sum().item() == 0, "env 2 cleared"
    print("PASS test_reset_clears_only_given_envs")


# ----------------------------------------------------------------------
# get_full_map
# ----------------------------------------------------------------------
def test_get_full_map_shape_and_expansion():
    cm = make_map(num_envs=2)
    env_ids = torch.tensor([0, 0, 0, 1, 1], device=DEVICE)  # 3 agents env0, 2 agents env1
    fm = cm.get_full_map(env_ids)
    assert fm.shape == (5, 100), f"expected (5,100), got {fm.shape}"
    print("PASS test_get_full_map_shape_and_expansion")


def test_get_full_map_agents_share_env_map():
    cm = make_map(num_envs=2)
    cm.mark_scanned(t(0).long(), torch.tensor([[22.5, 22.5, 5.0]], device=DEVICE), scan_range=2.0)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    fm = cm.get_full_map(env_ids)
    # agents 0 and 1 are both in env 0 → identical maps
    assert torch.equal(fm[0], fm[1]), "agents in same env must see identical map"
    # agent 2 in env 1 (empty) → different
    assert not torch.equal(fm[0], fm[2]), "different envs should differ"
    assert fm[2].sum().item() == 0, "env 1 map should be empty"
    print("PASS test_get_full_map_agents_share_env_map")


# ----------------------------------------------------------------------
# get_coverage_pct
# ----------------------------------------------------------------------
def test_coverage_pct_per_env():
    cm = make_map(num_envs=2)
    # mark 1 cell in env 0
    cm.mark_scanned(t(0).long(), torch.tensor([[27.0, 18.0, 5.0]], device=DEVICE), scan_range=2.5)
    pct = cm.get_coverage_pct()  # None → per env
    assert pct.shape == (2,), f"expected (2,), got {pct.shape}"
    assert abs(pct[0].item() - 0.01) < 1e-6, f"1/100 cells = 0.01, got {pct[0].item()}"
    assert pct[1].item() == 0.0
    print("PASS test_coverage_pct_per_env")


def test_coverage_pct_per_agent():
    cm = make_map(num_envs=2)
    cm.mark_scanned(t(0).long(), torch.tensor([[27.0, 18.0, 5.0]], device=DEVICE), scan_range=2.5)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    pct = cm.get_coverage_pct(env_ids)
    assert pct.shape == (3,), f"expected (3,), got {pct.shape}"
    assert abs(pct[0].item() - 0.01) < 1e-6
    assert abs(pct[1].item() - 0.01) < 1e-6  # same env as agent 0
    assert pct[2].item() == 0.0              # env 1
    print("PASS test_coverage_pct_per_agent")


# ----------------------------------------------------------------------
# is_fully_covered
# ----------------------------------------------------------------------
def test_is_fully_covered():
    cm = make_map(num_envs=2)
    # fully cover env 0 by marking everything with a huge scan range
    all_centers = cm.cell_centers.reshape(-1, 2)  # (100, 2)
    # mark every cell in env 0 by scanning each center with tiny range
    env_ids = torch.zeros(all_centers.shape[0], dtype=torch.long, device=DEVICE)
    pos3 = torch.cat([all_centers, torch.full((all_centers.shape[0], 1), 5.0, device=DEVICE)], dim=1)
    cm.mark_scanned(env_ids, pos3, scan_range=1.0)

    covered = cm.is_fully_covered(threshold=0.95)
    assert covered[0].item() is True, "env 0 should be fully covered"
    assert covered[1].item() is False, "env 1 should not be"
    print("PASS test_is_fully_covered")


# ----------------------------------------------------------------------
# get_local_window
# ----------------------------------------------------------------------
def test_local_window_shape():
    cm = make_map(num_envs=2)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    positions = torch.tensor([
        [25.0, 25.0, 5.0],
        [10.0, 10.0, 5.0],
        [40.0, 40.0, 5.0],
    ], device=DEVICE)
    win = cm.get_local_window(env_ids, positions, local_range=25.0, output_size=11)
    assert win.shape == (3, 121), f"expected (3,121), got {win.shape}"
    print("PASS test_local_window_shape")


def test_local_window_edge_padding():
    """Drone in the corner — window should be padded with 1.0 (scanned) on out-of-bounds."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    # drone at corner (0,0) cell, grid is empty
    positions = torch.tensor([[2.5, 2.5, 5.0]], device=DEVICE)
    win = cm.get_local_window(env_ids, positions, local_range=25.0, output_size=11)
    # grid is empty (all 0), but out-of-bounds is padded 1.0
    # so the window must contain some 1.0 values from the padding
    assert win.max().item() > 0.0, "corner window should contain padded 1.0 values"
    print("PASS test_local_window_edge_padding")


def test_local_window_fixed_size_across_cell_size():
    """Different cell_size, same local_range → same output shape (the scaling guarantee)."""
    cm_coarse = make_map(num_envs=1, arena_size=50.0, cell_size=5.0)
    cm_fine = make_map(num_envs=1, arena_size=50.0, cell_size=2.5)
    env_ids = t(0).long()
    pos = torch.tensor([[25.0, 25.0, 5.0]], device=DEVICE)
    w1 = cm_coarse.get_local_window(env_ids, pos, local_range=25.0, output_size=11)
    w2 = cm_fine.get_local_window(env_ids, pos, local_range=25.0, output_size=11)
    assert w1.shape == w2.shape == (1, 121), "output size must be fixed regardless of cell_size"
    print("PASS test_local_window_fixed_size_across_cell_size")


def test_batch_mask_buffer_no_leak_between_calls():
    """The reused _batch_mask must be cleared each call — a previous call's
    scans must not bleed into a later call for a different env."""
    cm = make_map(num_envs=2)
    # first call marks env 0 only
    cm.mark_scanned(t(0).long(), torch.tensor([[22.5, 22.5, 5.0]], device=DEVICE), scan_range=2.0)
    env0_after_first = cm.grid[0].sum().item()
    # second call marks env 1 only — must NOT re-apply env 0's stale mask
    cm.mark_scanned(t(1).long(), torch.tensor([[2.5, 2.5, 5.0]], device=DEVICE), scan_range=2.0)
    # env 0 grid unchanged by the second call
    assert cm.grid[0].sum().item() == env0_after_first, "env 0 must be untouched by second call"
    # env 1 got exactly its own scan, not env 0's leftover
    assert cm.grid[1].sum().item() == 1, f"env 1 should have exactly 1 cell, got {cm.grid[1].sum().item()}"
    # the buffer itself should be zeroed at the START of each call, so after the
    # second call it reflects only env 1's mark
    assert cm._batch_mask[0].sum().item() == 0, "buffer slot for env 0 should be zero after env-1 call"
    print("PASS test_batch_mask_buffer_no_leak_between_calls")

# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running CoverageMap tests on device: {DEVICE}\n")

    tests = [
        test_grid_dimensions,
        test_non_divisible_arena,
        test_cell_centers,
        test_pos_to_cell_basic,
        test_pos_to_cell_clamp,
        test_pos_to_cell_accepts_2d_and_3d,
        test_mark_single_agent_one_cell,
        test_mark_larger_scan_range,
        test_mark_overlapping_agents_same_env,
        test_mark_is_cumulative,
        test_mark_does_not_leak_across_envs,
        test_reset_clears_only_given_envs,
        test_get_full_map_shape_and_expansion,
        test_get_full_map_agents_share_env_map,
        test_coverage_pct_per_env,
        test_coverage_pct_per_agent,
        test_is_fully_covered,
        test_local_window_shape,
        test_local_window_edge_padding,
        test_local_window_fixed_size_across_cell_size,
        test_mark_does_not_leak_across_envs,
        test_batch_mask_buffer_no_leak_between_calls,   
        test_reset_clears_only_given_envs,
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

    print(f"\n{'='*50}")
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{failed}/{len(tests)} tests failed.")