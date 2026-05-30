"""
Test suite for CoverageMap (float scan-progress version).

    python test_coverage_map.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from coverage_map import CoverageMap


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_map(num_envs=2, arena_size=50.0, cell_size=5.0):
    return CoverageMap(num_envs=num_envs, arena_size=arena_size,
                       cell_size=cell_size, device=DEVICE)


def t(*vals):
    return torch.tensor(vals, device=DEVICE)


# ----------------------------------------------------------------------
# init / construction
# ----------------------------------------------------------------------
def test_grid_dimensions():
    cm = make_map(num_envs=4)
    assert cm.grid_size == 10, cm.grid_size
    assert cm.grid.shape == (4, 10, 10), cm.grid.shape
    assert cm.grid.dtype == torch.float, cm.grid.dtype
    assert cm.grid.sum().item() == 0.0, "grid starts empty"
    print("PASS test_grid_dimensions")


def test_cell_centers():
    cm = make_map(num_envs=1)
    assert cm.cell_centers.shape == (10, 10, 2)
    assert torch.allclose(cm.cell_centers[0, 0], t(2.5, 2.5))
    assert torch.allclose(cm.cell_centers[5, 3], t(27.5, 17.5))
    print("PASS test_cell_centers")


# ----------------------------------------------------------------------
# pos_to_cell
# ----------------------------------------------------------------------
def test_pos_to_cell_basic():
    cm = make_map()
    pos = torch.tensor([[2.5, 2.5, 5.0], [27.3, 18.0, 5.0], [47.5, 47.5, 5.0]], device=DEVICE)
    cells = cm.pos_to_cell(pos)
    assert torch.equal(cells, torch.tensor([[0, 0], [5, 3], [9, 9]], device=DEVICE)), cells
    print("PASS test_pos_to_cell_basic")


def test_pos_to_cell_clamp():
    cm = make_map()
    pos = torch.tensor([[-10.0, -10.0, 5.0], [999.0, 999.0, 5.0]], device=DEVICE)
    cells = cm.pos_to_cell(pos)
    assert torch.equal(cells, torch.tensor([[0, 0], [9, 9]], device=DEVICE)), cells
    print("PASS test_pos_to_cell_clamp")


# ----------------------------------------------------------------------
# mark_scanned — float progress accumulation
# ----------------------------------------------------------------------
def test_mark_single_step_increment():
    """One step of presence adds exactly scan_rate to the cell."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)  # on cell (5,3) center
    cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    assert abs(cm.grid[0, 5, 3].item() - 0.2) < 1e-6, cm.grid[0, 5, 3].item()
    print("PASS test_mark_single_step_increment")


def test_mark_accumulates_over_steps():
    """Repeated presence accumulates toward 1.0."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)
    for _ in range(5):
        cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    # 5 * 0.2 = 1.0
    assert abs(cm.grid[0, 5, 3].item() - 1.0) < 1e-6, cm.grid[0, 5, 3].item()
    print("PASS test_mark_accumulates_over_steps")


def test_mark_clamps_at_one():
    """Progress never exceeds 1.0 even with extra steps."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)
    for _ in range(10):  # would be 2.0 unclamped
        cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    assert abs(cm.grid[0, 5, 3].item() - 1.0) < 1e-6, cm.grid[0, 5, 3].item()
    print("PASS test_mark_clamps_at_one")


def test_mark_multi_agent_same_cell_is_max_not_sum():
    """Two agents over the same cell in one step add scan_rate ONCE (max), not 2x."""
    cm = make_map(num_envs=1)
    env_ids = torch.tensor([0, 0], device=DEVICE)  # both in env 0
    pos = torch.tensor([
        [27.5, 17.5, 5.0],  # agent A on cell (5,3)
        [27.6, 17.6, 5.0],  # agent B on same cell (5,3)
    ], device=DEVICE)
    cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    # MAX semantics: cell gains 0.2, not 0.4
    assert abs(cm.grid[0, 5, 3].item() - 0.2) < 1e-6, \
        f"expected 0.2 (max), got {cm.grid[0, 5, 3].item()} (sum bug?)"
    print("PASS test_mark_multi_agent_same_cell_is_max_not_sum")


def test_mark_no_leak_across_envs():
    cm = make_map(num_envs=3)
    env_ids = t(1).long()
    cm.mark_scanned(env_ids, torch.tensor([[22.5, 22.5, 5.0]], device=DEVICE),
                    scan_range=2.0, scan_rate=0.2)
    assert cm.grid[0].sum().item() == 0.0, "env 0 untouched"
    assert cm.grid[1].sum().item() > 0.0, "env 1 marked"
    assert cm.grid[2].sum().item() == 0.0, "env 2 untouched"
    print("PASS test_mark_no_leak_across_envs")


def test_mark_separate_envs_independent_progress():
    """Two envs, different agents, progress accumulates independently."""
    cm = make_map(num_envs=2)
    env_ids = torch.tensor([0, 1], device=DEVICE)
    pos = torch.tensor([
        [27.5, 17.5, 5.0],   # env 0 cell (5,3)
        [12.5, 12.5, 5.0],   # env 1 cell (2,2)
    ], device=DEVICE)
    cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    assert abs(cm.grid[0, 5, 3].item() - 0.4) < 1e-6
    assert abs(cm.grid[1, 2, 2].item() - 0.4) < 1e-6
    print("PASS test_mark_separate_envs_independent_progress")


# ----------------------------------------------------------------------
# reset
# ----------------------------------------------------------------------
def test_reset_clears_progress():
    cm = make_map(num_envs=3)
    all_ids = torch.arange(3, device=DEVICE)
    pos = torch.tensor([[22.5, 22.5, 5.0]] * 3, device=DEVICE)
    cm.mark_scanned(all_ids, pos, scan_range=2.0, scan_rate=0.2)
    assert (cm.grid.sum(dim=(1, 2)) > 0).all()
    cm.reset(torch.tensor([0, 2], device=DEVICE))
    assert cm.grid[0].sum().item() == 0.0, "env 0 cleared"
    assert cm.grid[1].sum().item() > 0.0, "env 1 preserved"
    assert cm.grid[2].sum().item() == 0.0, "env 2 cleared"
    print("PASS test_reset_clears_progress")


# ----------------------------------------------------------------------
# get_full_map
# ----------------------------------------------------------------------
def test_full_map_shape_and_progress_values():
    cm = make_map(num_envs=2)
    cm.mark_scanned(t(0).long(), torch.tensor([[22.5, 22.5, 5.0]], device=DEVICE),
                    scan_range=2.0, scan_rate=0.2)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    fm = cm.get_full_map(env_ids)
    assert fm.shape == (3, 100), fm.shape
    # the marked cell shows 0.2 progress (not binary 1.0)
    assert abs(fm[0].max().item() - 0.2) < 1e-6, "full map carries float progress"
    # agents in same env identical, different env differs
    assert torch.equal(fm[0], fm[1])
    assert not torch.equal(fm[0], fm[2])
    assert fm[2].sum().item() == 0.0
    print("PASS test_full_map_shape_and_progress_values")


# ----------------------------------------------------------------------
# get_coverage_pct — threshold based
# ----------------------------------------------------------------------
def test_coverage_pct_threshold():
    """A cell only counts as covered once progress >= threshold."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)
    # 3 steps → 0.6 progress, below default 0.95 threshold → NOT covered
    for _ in range(3):
        cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    pct_partial = cm.get_coverage_pct()  # per-env
    assert pct_partial[0].item() == 0.0, f"0.6 progress should not count as covered, got {pct_partial[0].item()}"
    # 2 more steps → 1.0 progress → covered (1 cell / 100)
    for _ in range(2):
        cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    pct_full = cm.get_coverage_pct()
    assert abs(pct_full[0].item() - 0.01) < 1e-6, f"1 covered cell = 0.01, got {pct_full[0].item()}"
    print("PASS test_coverage_pct_threshold")


def test_coverage_pct_per_agent():
    cm = make_map(num_envs=2)
    env_ids_mark = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)
    for _ in range(5):  # fully cover the cell
        cm.mark_scanned(env_ids_mark, pos, scan_range=2.0, scan_rate=0.2)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    pct = cm.get_coverage_pct(env_ids)
    assert pct.shape == (3,)
    assert abs(pct[0].item() - 0.01) < 1e-6
    assert abs(pct[1].item() - 0.01) < 1e-6  # same env
    assert pct[2].item() == 0.0
    print("PASS test_coverage_pct_per_agent")


def test_coverage_pct_custom_threshold():
    """Lower threshold counts partially-scanned cells as covered."""
    cm = make_map(num_envs=1)
    env_ids = t(0).long()
    pos = torch.tensor([[27.5, 17.5, 5.0]], device=DEVICE)
    for _ in range(3):  # 0.6 progress
        cm.mark_scanned(env_ids, pos, scan_range=2.0, scan_rate=0.2)
    # threshold 0.5 → the 0.6 cell counts
    pct = cm.get_coverage_pct(threshold=0.5)
    assert abs(pct[0].item() - 0.01) < 1e-6, pct[0].item()
    # threshold 0.7 → does not count
    pct2 = cm.get_coverage_pct(threshold=0.7)
    assert pct2[0].item() == 0.0
    print("PASS test_coverage_pct_custom_threshold")


# ----------------------------------------------------------------------
# is_fully_covered
# ----------------------------------------------------------------------
def test_is_fully_covered():
    cm = make_map(num_envs=2)
    # fully cover env 0: mark every cell center for enough steps
    centers = cm.cell_centers.reshape(-1, 2)              # (100,2)
    env_ids = torch.zeros(centers.shape[0], dtype=torch.long, device=DEVICE)
    pos3 = torch.cat([centers, torch.full((centers.shape[0], 1), 5.0, device=DEVICE)], dim=1)
    for _ in range(5):  # 5*0.2 = 1.0 each cell
        cm.mark_scanned(env_ids, pos3, scan_range=1.0, scan_rate=0.2)
    covered = cm.is_fully_covered(threshold=0.95)
    assert covered[0].item() is True, "env 0 fully covered"
    assert covered[1].item() is False, "env 1 not"
    print("PASS test_is_fully_covered")


# ----------------------------------------------------------------------
# get_local_window (unchanged pooled version, float grid)
# ----------------------------------------------------------------------
def test_local_window_shape():
    cm = make_map(num_envs=2)
    env_ids = torch.tensor([0, 0, 1], device=DEVICE)
    positions = torch.tensor([
        [25.0, 25.0, 5.0], [10.0, 10.0, 5.0], [40.0, 40.0, 5.0],
    ], device=DEVICE)
    win = cm.get_local_window(env_ids, positions, local_range=25.0, output_size=11)
    assert win.shape == (3, 121), win.shape
    print("PASS test_local_window_shape")


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running CoverageMap (float) tests on device: {DEVICE}\n")
    tests = [
        test_grid_dimensions,
        test_cell_centers,
        test_pos_to_cell_basic,
        test_pos_to_cell_clamp,
        test_mark_single_step_increment,
        test_mark_accumulates_over_steps,
        test_mark_clamps_at_one,
        test_mark_multi_agent_same_cell_is_max_not_sum,
        test_mark_no_leak_across_envs,
        test_mark_separate_envs_independent_progress,
        test_reset_clears_progress,
        test_full_map_shape_and_progress_values,
        test_coverage_pct_threshold,
        test_coverage_pct_per_agent,
        test_coverage_pct_custom_threshold,
        test_is_fully_covered,
        test_local_window_shape,
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