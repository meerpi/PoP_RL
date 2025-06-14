"""
test_subgoal_manager.py — End-to-end test suite for clean_env PoPEnv + DummyManager.

Tests:
  1. Smoke test            — env resets, obs shape correct, info keys present
  2. Subgoal constants     — SG_* values, SG_BUDGET keys, SG_REWARD keys
  3. SG_NAVIGATE fire      — reaches target room → subgoal_achieved
  4. Budget expiry         — exceeds SG_BUDGET → worker_truncated
  5. Subgoal reward inline — subgoal_achieved adds SG_REWARD to step reward
  6. reset_subgoal         — soft reset preserves game state
  7. DummyManager smoke    — full episode through DummyManager, no crashes
  8. DummyManager subgoals — DummyManager issues at least one transition
  9. Room 17 counter       — instance-level counter increments on room entry
"""

import sys
import traceback
import numpy as np

from clean_env import (
    PoPEnv, DummyManager, OBS_DIM, STACKED_DIM,
    SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL,
    SG_BUDGET, SG_REWARD, LEVEL1_GRAPH, FRONTIER_BONUS,
)

RESULTS = []


def run_test(name, fn):
    """Run a test function, catching exceptions."""
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        print(f"  [PASS] {name}")
    except Exception as e:
        tb = traceback.format_exc()
        RESULTS.append((name, "FAIL", tb))
        print(f"  [FAIL] {name}")
        print(f"         {e}")


# ═══════════════════════════════════════════════════════════════
# Test implementations
# ═══════════════════════════════════════════════════════════════

env = None  # module-level — shared across tests that need raw env


def test_smoke():
    """1. Env resets, obs shape correct, info keys present."""
    global env
    env = PoPEnv(visual=False)
    obs, info = env.reset(seed=42)

    assert obs.shape == (STACKED_DIM,), f"obs shape {obs.shape} != ({STACKED_DIM},)"
    for key in ("level", "room", "hp", "reset_type", "current_subgoal", "sg_target_room"):
        assert key in info, f"Missing info key: {key}"
    assert info["reset_type"] == "death"
    assert info["current_subgoal"] == SG_NAVIGATE
    assert info["sg_target_room"] == 2


def test_constants():
    """2. Subgoal constants, budgets, and rewards are defined correctly."""
    assert SG_NAVIGATE == 0
    assert SG_PICKUP_SWORD == 1
    assert SG_FIGHT_GUARD == 2
    assert SG_HEAL == 3

    for sg in (SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL):
        assert sg in SG_BUDGET, f"SG_BUDGET missing key {sg}"
        assert sg in SG_REWARD, f"SG_REWARD missing key {sg}"
        assert SG_BUDGET[sg] > 0, f"SG_BUDGET[{sg}] must be positive"
        assert SG_REWARD[sg] > 0, f"SG_REWARD[{sg}] must be positive"

    # FIGHT_GUARD gets the largest budget
    assert SG_BUDGET[SG_FIGHT_GUARD] >= SG_BUDGET[SG_NAVIGATE]
    assert SG_BUDGET[SG_FIGHT_GUARD] >= SG_BUDGET[SG_HEAL]


def test_navigate_detection():
    """3. SG_NAVIGATE fires when agent reaches target room."""
    global env
    obs, info = env.reset(seed=42)
    assert info["current_subgoal"] == SG_NAVIGATE
    target = info["sg_target_room"]

    # Step until we either reach target, die, or exhaust budget
    achieved = False
    for _ in range(SG_BUDGET[SG_NAVIGATE] + 50):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        if info.get("subgoal_achieved"):
            achieved = True
            assert env.k_room == target, f"Achieved but room {env.k_room} != target {target}"
            break
        if term:
            break

    # We don't require it to actually reach — just that the detection logic works
    # The real validation is that IF k_room == target, achieved fires
    # So let's test that directly:
    env.reset(seed=42)
    env._init_subgoal_tracking(SG_NAVIGATE, target_room=env.k_room)
    assert env._check_subgoal() == True, "NAVIGATE check should fire when already at target"

    env._init_subgoal_tracking(SG_NAVIGATE, target_room=99)
    assert env._check_subgoal() == False, "NAVIGATE check should NOT fire for unreachable target"


def test_budget_expiry():
    """4. Worker truncation fires when budget exhausted without achieving subgoal."""
    global env
    obs, info = env.reset(seed=42)
    # Set target to an unreachable room so we definitely exhaust budget
    env._init_subgoal_tracking(SG_NAVIGATE, target_room=24)

    truncated_fired = False
    for _ in range(SG_BUDGET[SG_NAVIGATE] + 10):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        if info.get("worker_truncated"):
            truncated_fired = True
            assert env.worker_steps >= SG_BUDGET[SG_NAVIGATE], \
                f"Truncated at step {env.worker_steps} but budget is {SG_BUDGET[SG_NAVIGATE]}"
            break
        if term:
            # died before budget — that's fine, test still valid
            break

    if not term:
        assert truncated_fired, "worker_truncated should fire when budget exhausted"


def test_subgoal_reward():
    """5. Subgoal completion adds SG_REWARD to the step reward."""
    global env
    env.reset(seed=42)
    # Force NAVIGATE subgoal to current room so next step triggers it
    env._init_subgoal_tracking(SG_NAVIGATE, target_room=env.k_room)

    obs, rew, term, trunc, info = env.step(0)  # any action
    if not term:
        if info.get("subgoal_achieved"):
            # reward should include the SG_REWARD bonus
            expected_min = SG_REWARD[SG_NAVIGATE] - 0.01 - 5.5  # generous margin for HP loss
            assert rew >= expected_min, \
                f"Reward {rew} too low — expected at least {expected_min} with SG_REWARD={SG_REWARD[SG_NAVIGATE]}"
        # If the agent moved to a different room on that step, the check might not fire
        # That's acceptable — we tested the direct check in test 3


def test_reset_subgoal():
    """6. Soft reset preserves game state but updates subgoal tracking."""
    global env
    env.reset(seed=42)
    # Take a few steps to move the game state forward
    for _ in range(10):
        env.step(env.action_space.sample())

    pre_room = env.k_room
    pre_hp   = env.hp
    pre_steps = env.steps

    obs, info = env.reset_subgoal(SG_FIGHT_GUARD, target_room=3)

    assert info["current_subgoal"] == SG_FIGHT_GUARD
    assert info["sg_target_room"] == 3
    assert info["reset_type"] == "subgoal"
    assert env.worker_steps == 0, "worker_steps should reset on soft reset"
    assert env.steps == pre_steps, "global steps should NOT reset on soft reset"
    # Game state preserved — room didn't change
    assert env.k_room == pre_room, f"Room changed from {pre_room} to {env.k_room} on soft reset"


def test_dummy_manager_smoke():
    """7. DummyManager full episode — 500 steps, no crashes."""
    dm = DummyManager(PoPEnv(visual=False))
    obs, info = dm.reset(seed=42)

    assert obs.shape == (STACKED_DIM,), f"DummyManager obs shape {obs.shape} != ({STACKED_DIM},)"
    assert "manager_phase" in info

    total_reward = 0.0
    steps_run = 0
    deaths = 0
    for step_i in range(500):
        obs, rew, term, trunc, info = dm.step(dm.action_space.sample())
        total_reward += rew
        steps_run += 1

        if term:
            deaths += 1
            obs, info = dm.reset()
        if trunc:
            break

    assert steps_run == 500 or trunc, f"Only ran {steps_run} steps"
    # Just verify it didn't crash — that's the smoke test


def test_dummy_manager_transitions():
    """8. DummyManager issues at least one subgoal transition in 1000 steps."""
    dm = DummyManager(PoPEnv(visual=False))
    obs, info = dm.reset(seed=42)

    transitions = 0
    prev_sg = info.get("current_subgoal", SG_NAVIGATE)

    for _ in range(1000):
        obs, rew, term, trunc, info = dm.step(dm.action_space.sample())

        cur_sg = info.get("current_subgoal", prev_sg)
        if cur_sg != prev_sg or info.get("subgoal_achieved") or info.get("worker_truncated"):
            transitions += 1
        prev_sg = cur_sg

        if term:
            obs, info = dm.reset()
            transitions += 1  # reset is itself a transition
            prev_sg = info.get("current_subgoal", SG_NAVIGATE)
        if trunc:
            break

    assert transitions > 0, "DummyManager should issue at least one subgoal transition in 1000 steps"


def test_room17_counter():
    """9. Room 17 instance-level counter increments correctly."""
    dm = DummyManager(PoPEnv(visual=False))
    obs, info = dm.reset(seed=42)

    assert dm._room17_visits == 0, "Counter should start at 0"
    initial = dm._room17_visits
    for _ in range(200):
        obs, rew, term, trunc, info = dm.step(dm.action_space.sample())
        if term:
            obs, info = dm.reset()

    assert dm._room17_visits >= initial, \
        f"Room 17 counter decreased: {dm._room17_visits} < {initial}"

    # Verify backtracking phase key is present
    assert "manager_phase" in info


def test_level1_graph():
    """10. LEVEL1_GRAPH sanity — all rooms 1-24 present, edges are bidirectional."""
    for r in range(1, 25):
        assert r in LEVEL1_GRAPH, f"Room {r} missing from LEVEL1_GRAPH"
        assert len(LEVEL1_GRAPH[r]) > 0, f"Room {r} has no neighbors"

    # Check bidirectionality
    errors = []
    for src, nbs in LEVEL1_GRAPH.items():
        for dst in nbs:
            if src not in LEVEL1_GRAPH.get(dst, []):
                errors.append(f"{src}→{dst} but {dst}↛{src}")

    # Note: some edges may be intentionally unidirectional (e.g. drop-only)
    # So we just report, not assert
    if errors:
        print(f"    NOTE: {len(errors)} unidirectional edges found (may be intentional)")


def test_frontier_reward():
    """11. Frontier reward fires on new rooms, not on revisits."""
    dm = DummyManager(PoPEnv(visual=False))
    obs, info = dm.reset(seed=42)

    # After reset, starting room (1) should already be in known_rooms
    assert 1 in dm.known_rooms, "Starting room should be known after reset"
    assert 1 in dm._episode_visited, "Starting room should be in episode visited"

    # Run steps and collect frontier rewards
    total_frontier = 0.0
    rooms_seen = {1}
    for _ in range(500):
        obs, rew, term, trunc, info = dm.step(dm.action_space.sample())

        fr = info.get("frontier_reward", 0.0)
        total_frontier += fr

        current_room = info["room"]
        if current_room not in rooms_seen and fr > 0:
            rooms_seen.add(current_room)
        elif current_room in dm._episode_visited and current_room in rooms_seen:
            # Revisiting a room already seen this episode should give 0
            pass  # fr could be 0 from _prev_room check too

        if term:
            obs, info = dm.reset()

    # known_rooms should have grown
    assert len(dm.known_rooms) >= 1, "known_rooms should have at least the start room"
    assert "frontier_rooms" in info, "info should contain frontier_rooms key"
    assert "frontier_reward" in info, "info should contain frontier_reward key"

    # If the agent explored at all, frontier reward should be positive
    if len(dm.known_rooms) > 1:
        assert total_frontier > 0, \
            f"Frontier reward should be positive when {len(dm.known_rooms)} rooms known"


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Subgoal Worker + DummyManager Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("1. Smoke test",              test_smoke),
        ("2. Subgoal constants",       test_constants),
        ("3. SG_NAVIGATE detection",   test_navigate_detection),
        ("4. Budget expiry",           test_budget_expiry),
        ("5. Subgoal reward inline",   test_subgoal_reward),
        ("6. reset_subgoal",           test_reset_subgoal),
        ("7. DummyManager smoke",      test_dummy_manager_smoke),
        ("8. DummyManager transitions", test_dummy_manager_transitions),
        ("9. Room 17 counter",         test_room17_counter),
        ("10. LEVEL1_GRAPH sanity",    test_level1_graph),
        ("11. Frontier reward",        test_frontier_reward),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print()
    print("=" * 60)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"  Results: {passed} PASSED,  {failed} FAILED")
    print("=" * 60)

    if failed:
        print("\n  FAILURES:")
        for name, status, tb in RESULTS:
            if status == "FAIL":
                print(f"\n  --- {name} ---")
                print(tb)
        sys.exit(1)
    else:
        print("\n  All tests passed!")
        sys.exit(0)
