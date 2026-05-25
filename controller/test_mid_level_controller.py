"""
Test suite for MidLevelController.

Run from project root:
    python test_mid_level_controller.py
"""

import torch
from mid_level_controller import MidLevelController as MLC


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_ctrl(num_envs=2, num_agents=1, **kw):
    return MLC(num_envs=num_envs, num_agents=num_agents, device=DEVICE, **kw)


def set_start(ctrl, pos):
    """Helper: emulate what the env must do after reset — seed prev_pos
    so the first step doesn't accumulate a bogus jump from origin."""
    ctrl.prev_pos = pos.clone()


# ----------------------------------------------------------------------
# init / reset
# ----------------------------------------------------------------------
def test_init_shapes():
    c = make_ctrl(num_envs=4, num_agents=3)
    assert c.mode.shape == (4, 3)
    assert c.odometer.shape == (4, 3)
    assert c.target.shape == (4, 3, 3)
    assert c.prev_pos.shape == (4, 3, 3)
    assert (c.mode == MLC.MODE_ACTIVE).all()
    assert c.odometer.sum().item() == 0.0
    print("PASS test_init_shapes")


def test_base_xyz():
    c = make_ctrl(base_position=(0.0, 0.0))
    assert torch.allclose(c.base_xyz, torch.tensor([0.0, 0.0, MLC.RETURN_HEIGHT], device=DEVICE))
    print("PASS test_base_xyz")


def test_reset_subset():
    c = make_ctrl(num_envs=3, num_agents=1)
    c.mode[:] = MLC.MODE_DOCKED
    c.odometer[:] = 99.0
    c.reset(torch.tensor([0, 2], device=DEVICE))
    assert c.mode[0].item() == MLC.MODE_ACTIVE
    assert c.mode[1].item() == MLC.MODE_DOCKED   # untouched
    assert c.mode[2].item() == MLC.MODE_ACTIVE
    assert c.odometer[0].item() == 0.0
    assert c.odometer[1].item() == 99.0          # untouched
    print("PASS test_reset_subset")


def test_mark_dead():
    c = make_ctrl(num_envs=2, num_agents=2)
    dead = torch.zeros((2, 2), dtype=torch.bool, device=DEVICE)
    dead[0, 1] = True  # kill env0 agent1 only
    c.mark_dead(dead)
    assert c.mode[0, 1].item() == MLC.MODE_DEAD
    assert c.mode[0, 0].item() == MLC.MODE_ACTIVE  # others untouched
    assert c.mode[1, 0].item() == MLC.MODE_ACTIVE
    print("PASS test_mark_dead")


# ----------------------------------------------------------------------
# odometer
# ----------------------------------------------------------------------
def test_odometer_accumulates():
    c = make_ctrl(num_envs=1, num_agents=1)
    start = torch.tensor([[[0.0, 0.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.zeros((1, 1, 2), device=DEVICE)
    p1 = torch.tensor([[[3.0, 0.0, 5.0]]], device=DEVICE)
    c.step(action, p1)
    assert abs(c.odometer[0, 0].item() - 3.0) < 1e-4, c.odometer
    p2 = torch.tensor([[[3.0, 4.0, 5.0]]], device=DEVICE)  # +4m in y
    c.step(action, p2)
    assert abs(c.odometer[0, 0].item() - 7.0) < 1e-4, c.odometer
    print("PASS test_odometer_accumulates")


def test_odometer_reset_jump_bug():
    """EXPOSES Issue 1: without seeding prev_pos, first step adds bogus distance."""
    c = make_ctrl(num_envs=1, num_agents=1)
    action = torch.zeros((1, 1, 2), device=DEVICE)
    real_start = torch.tensor([[[25.0, 25.0, 5.0]]], device=DEVICE)
    c.step(action, real_start)
    jump = c.odometer[0, 0].item()
    if jump > 1.0:
        print(f"KNOWN ISSUE test_odometer_reset_jump_bug: first step added "
              f"{jump:.1f}m bogus distance (env must seed prev_pos after reset)")
    else:
        print("PASS test_odometer_reset_jump_bug (no bogus jump)")


def test_seed_positions_prevents_jump():
    """Confirms the fix: seeding prev_pos eliminates the reset jump."""
    c = make_ctrl(num_envs=1, num_agents=1)
    real_start = torch.tensor([[[25.0, 25.0, 5.0]]], device=DEVICE)
    c.seed_positions(real_start)
    c.step(torch.zeros((1, 1, 2), device=DEVICE), real_start)
    jump = c.odometer[0, 0].item()
    assert jump < 1e-4, f"seeded start should not jump, got {jump}"
    print("PASS test_seed_positions_prevents_jump")


# ----------------------------------------------------------------------
# return trigger
# ----------------------------------------------------------------------
def test_return_trigger():
    c = make_ctrl(num_envs=1, num_agents=1, max_range=10.0, safety_factor=0.8)
    start = torch.tensor([[[0.0, 0.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.zeros((1, 1, 2), device=DEVICE)
    c.step(action, torch.tensor([[[5.0, 0.0, 5.0]]], device=DEVICE))
    assert c.mode[0, 0].item() == MLC.MODE_ACTIVE, "should still be active at 5m"
    c.step(action, torch.tensor([[[9.0, 0.0, 5.0]]], device=DEVICE))
    assert c.mode[0, 0].item() == MLC.MODE_RETURNING, "should be returning past 8m"
    print("PASS test_return_trigger")


def test_returning_odometer_frozen():
    """After Edit 1: RETURNING drones no longer accumulate odometer."""
    c = make_ctrl(num_envs=1, num_agents=1, max_range=10.0, safety_factor=0.8)
    start = torch.tensor([[[0.0, 0.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.zeros((1, 1, 2), device=DEVICE)
    c.step(action, torch.tensor([[[9.0, 0.0, 5.0]]], device=DEVICE))
    assert c.mode[0, 0].item() == MLC.MODE_RETURNING
    odo_at_return = c.odometer[0, 0].item()
    c.step(action, torch.tensor([[[15.0, 0.0, 5.0]]], device=DEVICE))
    assert abs(c.odometer[0, 0].item() - odo_at_return) < 1e-4, \
        f"returning odometer should freeze at {odo_at_return}, got {c.odometer[0,0].item()}"
    print("PASS test_returning_odometer_frozen")


# ----------------------------------------------------------------------
# target computation
# ----------------------------------------------------------------------
def test_active_target():
    c = make_ctrl(num_envs=1, num_agents=1, radius_cap=5.0)
    start = torch.tensor([[[10.0, 10.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.tensor([[[1.0, 0.0]]], device=DEVICE)
    c.step(action, start)
    expected = torch.tensor([15.0, 10.0, MLC.SCAN_HEIGHT], device=DEVICE)
    assert torch.allclose(c.target[0, 0], expected), c.target[0, 0]
    print("PASS test_active_target")


def test_returning_target_is_base():
    c = make_ctrl(num_envs=1, num_agents=1)
    c.mode[0, 0] = MLC.MODE_RETURNING
    start = torch.tensor([[[30.0, 30.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.tensor([[[1.0, 1.0]]], device=DEVICE)  # should be ignored
    c.step(action, start)
    assert torch.allclose(c.target[0, 0], c.base_xyz), c.target[0, 0]
    print("PASS test_returning_target_is_base")


def test_docked_stays_put():
    c = make_ctrl(num_envs=1, num_agents=1)
    c.mode[0, 0] = MLC.MODE_DOCKED
    start = torch.tensor([[[7.0, 7.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.tensor([[[1.0, 1.0]]], device=DEVICE)
    vel = c.step(action, start)
    assert torch.allclose(vel[0, 0], torch.zeros(3, device=DEVICE)), vel[0, 0]
    print("PASS test_docked_stays_put")


def test_dead_drone_frozen():
    """DEAD drones: zero velocity, frozen odometer, stays dead even if position drifts."""
    c = make_ctrl(num_envs=1, num_agents=1)
    start = torch.tensor([[[10.0, 10.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    c.mark_dead(torch.tensor([[True]], device=DEVICE))
    action = torch.tensor([[[1.0, 1.0]]], device=DEVICE)
    drifted = torch.tensor([[[20.0, 20.0, 5.0]]], device=DEVICE)
    vel = c.step(action, drifted)
    assert c.mode[0, 0].item() == MLC.MODE_DEAD, "should stay dead"
    assert c.odometer[0, 0].item() == 0.0, f"dead odometer must stay 0, got {c.odometer[0,0].item()}"
    assert torch.allclose(vel[0, 0], torch.zeros(3, device=DEVICE)), "dead drone zero velocity"
    print("PASS test_dead_drone_frozen")


# ----------------------------------------------------------------------
# arrival / docking
# ----------------------------------------------------------------------
def test_arrival_docks():
    c = make_ctrl(num_envs=1, num_agents=1, arrival_threshold=0.5)
    c.mode[0, 0] = MLC.MODE_RETURNING
    at_base = c.base_xyz.view(1, 1, 3).clone()
    set_start(c, at_base)
    action = torch.zeros((1, 1, 2), device=DEVICE)
    c.step(action, at_base)
    assert c.mode[0, 0].item() == MLC.MODE_DOCKED, "should dock when at base"
    print("PASS test_arrival_docks")


def test_returning_does_not_dock_when_far():
    c = make_ctrl(num_envs=1, num_agents=1, arrival_threshold=0.5)
    c.mode[0, 0] = MLC.MODE_RETURNING
    far = torch.tensor([[[30.0, 30.0, 5.0]]], device=DEVICE)
    set_start(c, far)
    action = torch.zeros((1, 1, 2), device=DEVICE)
    c.step(action, far)
    assert c.mode[0, 0].item() == MLC.MODE_RETURNING, "should not dock when far"
    print("PASS test_returning_does_not_dock_when_far")


# ----------------------------------------------------------------------
# velocity direction
# ----------------------------------------------------------------------
def test_velocity_points_toward_target():
    c = make_ctrl(num_envs=1, num_agents=1, radius_cap=5.0, scan_speed=1.5)
    start = torch.tensor([[[10.0, 10.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.tensor([[[1.0, 0.0]]], device=DEVICE)
    vel = c.step(action, start)
    v = vel[0, 0]
    assert v[0].item() > 0, "should move +x"
    assert abs(v[1].item()) < 1e-4, "no y motion expected"
    speed = torch.norm(v).item()
    assert abs(speed - 1.5) < 1e-3, f"speed should equal scan_speed, got {speed}"
    print("PASS test_velocity_points_toward_target")


def test_velocity_overshoot_behavior():
    """DOCUMENTS Issue 3: velocity is constant scan_speed even very near target."""
    c = make_ctrl(num_envs=1, num_agents=1, radius_cap=5.0, scan_speed=1.5)
    start = torch.tensor([[[10.0, 10.0, 5.0]]], device=DEVICE)
    set_start(c, start)
    action = torch.tensor([[[0.01, 0.0]]], device=DEVICE)
    vel = c.step(action, start)
    speed = torch.norm(vel[0, 0]).item()
    print(f"KNOWN ISSUE test_velocity_overshoot_behavior: target 0.05m away but "
          f"speed={speed:.2f} (no distance-based clamping)")


# ----------------------------------------------------------------------
# multi-agent / multi-env integrity
# ----------------------------------------------------------------------
def test_multi_agent_independent():
    c = make_ctrl(num_envs=2, num_agents=2, max_range=10.0, safety_factor=0.8)
    start = torch.zeros((2, 2, 3), device=DEVICE)
    start[..., 2] = 5.0
    set_start(c, start)
    action = torch.zeros((2, 2, 2), device=DEVICE)
    new_pos = start.clone()
    new_pos[0, 0, 0] = 9.0  # 9m > 8m threshold
    c.step(action, new_pos)
    assert c.mode[0, 0].item() == MLC.MODE_RETURNING, "env0 agent0 should return"
    assert c.mode[0, 1].item() == MLC.MODE_ACTIVE, "env0 agent1 should stay active"
    assert c.mode[1, 0].item() == MLC.MODE_ACTIVE, "env1 untouched"
    print("PASS test_multi_agent_independent")


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running MidLevelController tests on device: {DEVICE}\n")

    tests = [
        test_init_shapes,
        test_base_xyz,
        test_reset_subset,
        test_mark_dead,
        test_odometer_accumulates,
        test_odometer_reset_jump_bug,
        test_seed_positions_prevents_jump,
        test_return_trigger,
        test_returning_odometer_frozen,
        test_active_target,
        test_returning_target_is_base,
        test_docked_stays_put,
        test_dead_drone_frozen,
        test_arrival_docks,
        test_returning_does_not_dock_when_far,
        test_velocity_points_toward_target,
        test_velocity_overshoot_behavior,
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
    if failed == 0:
        print(f"All {len(tests)} tests passed "
              f"(note: 2 KNOWN ISSUE messages are informational, not failures).")
    else:
        print(f"{failed}/{len(tests)} tests failed.")
        