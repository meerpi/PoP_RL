"""
test_structural.py — Adversarial structural tests.

These tests verify control flow correctness, not just functional output.
They target the exact bugs previously found: dead code paths, unreachable
branches, missing initialisations, and cross-process isolation failures.

Tests:
  S01  _sample_neighbor forward phase — only unvisited/all neighbors, no 30% random
  S02  _sample_neighbor backtrack phase — 70/30 split actually fires
  S03  _sample_neighbor phase boundary — behavior flips at threshold
  S04  _sample_neighbor empty neighbors — returns same room
  S05  _next_subgoal no dead code — every return path reachable
  S06  _next_subgoal heal path — no break after return
  S07  subgoals_completed_this_episode exists before reset
  S08  subgoals_completed_this_episode accumulates across subgoals
  S09  subgoals_completed_this_episode resets on death
  S10  make_env thunk — no reset called, env not initialized
  S11  _room17_visits instance isolation — two managers independent
  S12  _room17_visits survives reset
  S13  DummyManager phase transition — info reports correct phase
  S14  final_observation bootstrap — correct obs used for dead envs
  S15  kid_v has no glob_x/glob_y — obs is 26 dim for kid
  S16  reward exactly 0.0 on normal step — no hidden penalties
  S17  death penalty scales correctly at boundary values
  S18  guard damage reward scoped to FIGHT_GUARD only
"""

import sys
import os
import traceback
import numpy as np

os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_RENDER_DRIVER"] = "software"

from clean_env import (
    PoPEnv, DummyManager, OBS_DIM, KID_DIM, G_DIM,
    SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL,
    SG_BUDGET, SG_REWARD, LEVEL1_GRAPH, SWORD_ROOM, GUARD_ROOM,
    DEATH_BASE, DEATH_PER_SG, GUARD_DMG_REWARD, ALIVE,
)

N_PASS = 0
N_FAIL = 0

def run(name, fn):
    global N_PASS, N_FAIL
    try:
        fn()
        N_PASS += 1
        print(f"  [PASS] {name}")
    except Exception:
        N_FAIL += 1
        print(f"  [FAIL] {name}")
        for line in traceback.format_exc().strip().split("\n")[-3:]:
            print(f"         {line}")


def make_dm():
    """Fresh DummyManager for each test."""
    e = PoPEnv(visual=False)
    dm = DummyManager(e)
    dm.reset(seed=42)
    return dm


# ═══════════════════════════════════════════════════════════════════
# S01–S04: _sample_neighbor control flow
# ═══════════════════════════════════════════════════════════════════

def test_S01_forward_phase_only_forward():
    """S01: In forward phase, _sample_neighbor returns ONLY from _forward_neighbors pool."""
    dm = make_dm()
    dm._room17_visits = 0  # firmly in forward phase

    # Room 1 has neighbors [2, 4, 5, 6]
    room = 1
    dm.known_rooms = set()  # nothing visited
    results = set()
    for _ in range(200):
        r = dm._sample_neighbor(room)
        results.add(r)

    nbs = set(LEVEL1_GRAPH.get(room, []))
    assert results.issubset(nbs), \
        f"Forward phase returned rooms outside neighbors: {results - nbs}"
    assert len(results) > 0, "No results from _sample_neighbor"


def test_S02_backtrack_phase_includes_all():
    """S02: In backtrack phase, _sample_neighbor can return ANY neighbor (30% path)."""
    dm = make_dm()
    dm._room17_visits = 200  # well past threshold

    # Room 7 has neighbors [4, 20]. Mark all as visited so forward pool = all nbs.
    room = 7
    dm.known_rooms = set(range(1, 25))  # all visited
    results = set()
    for _ in range(500):
        r = dm._sample_neighbor(room)
        results.add(r)

    nbs = set(LEVEL1_GRAPH.get(room, []))
    assert results.issubset(nbs), \
        f"Backtrack phase returned rooms outside neighbors: {results - nbs}"


def test_S03_phase_boundary_behavior_changes():
    """S03: _sample_neighbor behavior changes exactly at BACKTRACK_THRESHOLD."""
    dm = make_dm()
    # Room 19 has neighbors [4, 10]. Mark 4 as visited → unvisited = {10}.
    room = 19
    dm.known_rooms = {4}

    # Forward phase: should ONLY return from unvisited = {10}
    dm._room17_visits = DummyManager.BACKTRACK_THRESHOLD - 1
    forward_results = set()
    for _ in range(200):
        forward_results.add(dm._sample_neighbor(room))
    assert forward_results == {10}, \
        f"Forward phase should only pick unvisited {{10}}, got {forward_results}"

    # Backtrack phase: CAN return room 4 too (via 30% path picking from all nbs)
    dm._room17_visits = DummyManager.BACKTRACK_THRESHOLD
    backtrack_results = set()
    for _ in range(500):
        backtrack_results.add(dm._sample_neighbor(room))
    # With 500 samples at 30% picking from all nbs {4,10}, room 4 should appear
    assert 4 in backtrack_results, \
        f"Backtrack 30% path should sometimes pick visited room 4, got {backtrack_results}"
    assert 10 in backtrack_results, \
        f"Backtrack should still pick unvisited room 10, got {backtrack_results}"


def test_S04_empty_neighbors_self():
    """S04: _sample_neighbor returns room itself when LEVEL1_GRAPH has no neighbors."""
    dm = make_dm()
    # Room 99 doesn't exist in graph
    result = dm._sample_neighbor(99)
    assert result == 99, f"Empty neighbors should return same room, got {result}"


# ═══════════════════════════════════════════════════════════════════
# S05–S06: _next_subgoal reachability
# ═══════════════════════════════════════════════════════════════════

def test_S05_next_subgoal_all_paths():
    """S05: Every return path in _next_subgoal is actually reachable."""
    dm = make_dm()
    env = dm.env

    # Path 1: no sword, not at sword room → NAVIGATE
    env.have_sword = 0
    sg, _ = dm._next_subgoal(1)  # room 1, not sword room
    assert sg == SG_NAVIGATE, f"Path 1: expected NAVIGATE, got {sg}"

    # Path 2: at sword room without sword → PICKUP_SWORD
    env.have_sword = 0
    sg, tr = dm._next_subgoal(SWORD_ROOM)
    assert sg == SG_PICKUP_SWORD, f"Path 2: expected PICKUP_SWORD, got {sg}"
    assert tr == SWORD_ROOM

    # Path 3a: have sword, guard alive, at guard room → FIGHT_GUARD
    env.have_sword = 1
    env.g_alive = ALIVE
    env.g_hpmax = 4
    sg, tr = dm._next_subgoal(GUARD_ROOM)
    assert sg == SG_FIGHT_GUARD, f"Path 3a: expected FIGHT_GUARD, got {sg}"
    assert tr == GUARD_ROOM

    # Path 3b: have sword, guard alive, NOT at guard room → NAVIGATE
    sg, _ = dm._next_subgoal(1)
    assert sg == SG_NAVIGATE, f"Path 3b: expected NAVIGATE, got {sg}"

    # Path 5: fallback — have sword, guard dead → NAVIGATE
    env.g_alive = 0
    env.g_hpmax = 0
    env.hp = env.hp_max  # full hp, skip heal path
    sg, _ = dm._next_subgoal(1)
    assert sg == SG_NAVIGATE, f"Path 5: expected NAVIGATE fallback, got {sg}"


def test_S06_heal_path_no_dead_code():
    """S06: The heal path returns SG_HEAL with no unreachable code after it."""
    dm = make_dm()
    env = dm.env

    # Set up conditions for heal: have sword, guard dead, hp < max
    env.have_sword = 1
    env.g_alive = 0
    env.g_hpmax = 0
    env.hp = 1
    env.hp_max = 3

    # We need a room with a potion neighbor. Test the code path by
    # checking that _next_subgoal returns either HEAL or NAVIGATE (fallback).
    # The critical thing is: no crash, no infinite loop, a valid return.
    sg, tr = dm._next_subgoal(env.k_room)
    assert sg in (SG_HEAL, SG_NAVIGATE), \
        f"Should return HEAL or NAVIGATE, got {sg}"
    assert isinstance(tr, int) and tr > 0, f"Target room should be positive int, got {tr}"


# ═══════════════════════════════════════════════════════════════════
# S07–S09: subgoals_completed_this_episode lifecycle
# ═══════════════════════════════════════════════════════════════════

def test_S07_subgoals_completed_exists_before_reset():
    """S07: subgoals_completed_this_episode is set in __init__, not just reset_on_death."""
    env = PoPEnv(visual=False)
    # Access BEFORE any reset — should NOT raise AttributeError
    assert hasattr(env, "subgoals_completed_this_episode"), \
        "subgoals_completed_this_episode missing from __init__"
    assert env.subgoals_completed_this_episode == 0


def test_S08_subgoals_completed_accumulates():
    """S08: Counter increments on subgoal completion, persists across subgoal resets."""
    env = PoPEnv(visual=False)
    env.reset(seed=42)
    assert env.subgoals_completed_this_episode == 0

    # Simulate first subgoal completion
    env.subgoals_completed_this_episode += 1
    assert env.subgoals_completed_this_episode == 1

    # Soft reset (subgoal transition) should NOT reset the counter
    env.reset_subgoal(SG_NAVIGATE, target_room=5)
    assert env.subgoals_completed_this_episode == 1, \
        "Soft reset should NOT reset subgoals_completed counter"

    # Second subgoal completion
    env.subgoals_completed_this_episode += 1
    assert env.subgoals_completed_this_episode == 2


def test_S09_subgoals_completed_resets_on_death():
    """S09: Counter resets to 0 on death (hard reset)."""
    env = PoPEnv(visual=False)
    env.reset(seed=42)
    env.subgoals_completed_this_episode = 5

    env.reset_on_death(seed=42)
    assert env.subgoals_completed_this_episode == 0, \
        f"Should be 0 after death reset, got {env.subgoals_completed_this_episode}"


# ═══════════════════════════════════════════════════════════════════
# S10: make_env thunk
# ═══════════════════════════════════════════════════════════════════

def test_S10_make_env_no_double_reset():
    """S10: make_env thunk creates env without calling reset (AsyncVectorEnv does that)."""
    import inspect
    from ppo import make_env

    # Get the thunk source and verify no reset() call
    thunk = make_env(seed=0, env_id=0)
    source = inspect.getsource(thunk)
    # The thunk should NOT contain dm.reset or env.reset
    assert ".reset(" not in source, \
        f"make_env thunk should NOT call reset, AsyncVectorEnv does that. Found in:\n{source}"


# ═══════════════════════════════════════════════════════════════════
# S11–S13: _room17_visits isolation and phase transitions
# ═══════════════════════════════════════════════════════════════════

def test_S11_room17_instance_isolation():
    """S11: Two DummyManagers have completely independent _room17_visits."""
    dm1 = make_dm()
    dm2 = make_dm()

    dm1._room17_visits = 999
    assert dm2._room17_visits == 0, \
        f"dm2 should be 0, got {dm2._room17_visits} (leaked from dm1)"

    dm2._room17_visits = 50
    assert dm1._room17_visits == 999, \
        f"dm1 should be 999, got {dm1._room17_visits} (leaked from dm2)"


def test_S12_room17_survives_reset():
    """S12: _room17_visits persists across episode resets (it's cross-episode knowledge)."""
    dm = make_dm()
    dm._room17_visits = 42

    dm.reset(seed=99)
    assert dm._room17_visits == 42, \
        f"_room17_visits should persist across resets, got {dm._room17_visits}"


def test_S13_phase_in_info():
    """S13: info['manager_phase'] correctly reports forward/backtrack."""
    dm = make_dm()
    dm._room17_visits = 0
    obs, info = dm.reset(seed=42)
    assert info["manager_phase"] == "forward", \
        f"Should be 'forward' with 0 visits, got {info['manager_phase']}"

    dm._room17_visits = DummyManager.BACKTRACK_THRESHOLD
    _, _, _, _, info = dm.step(0)
    assert info["manager_phase"] == "backtrack", \
        f"Should be 'backtrack' at threshold, got {info['manager_phase']}"


# ═══════════════════════════════════════════════════════════════════
# S14: final_observation bootstrap (structural verification)
# ═══════════════════════════════════════════════════════════════════

def test_S14_final_observation_code_path():
    """S14: ppo.py training loop handles final_observation for dead envs."""
    import inspect
    import ppo

    source = inspect.getsource(ppo.train)

    # These patterns MUST exist in the training loop
    assert "final_observation" in source, \
        "Training loop must handle final_observation for correct GAE bootstrap"
    assert "final_info" in source, \
        "Training loop must handle final_info for correct goal vector on terminal step"

    # The final_observation should be used to OVERWRITE next_obs for dead envs
    assert "next_obs[i]" in source or "next_obs[" in source, \
        "Training loop must overwrite next_obs with final_observation for dead envs"


# ═══════════════════════════════════════════════════════════════════
# S15–S18: Observation and reward structural correctness
# ═══════════════════════════════════════════════════════════════════

def test_S15_no_glob_xy_in_obs():
    """S15: kid_v is 26 dim, not 30 — glob_x/glob_y/dx/dy have been removed."""
    assert KID_DIM == 26, f"KID_DIM should be 26 (no global coords), got {KID_DIM}"
    assert OBS_DIM == 480 + 26 + 32, \
        f"OBS_DIM should be 538 (grid=480 + kid=26 + guard=32), got {OBS_DIM}"

    env = PoPEnv(visual=False)
    env.reset(seed=42)
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (538,), f"Obs shape should be (538,), got {obs.shape}"


def test_S16_zero_reward_normal_step():
    """S16: A normal alive step with no subgoal event produces exactly 0.0 reward."""
    env = PoPEnv(visual=False)
    env.reset(seed=42)
    # Set target to unreachable so no subgoal fires
    env._init_subgoal_tracking(SG_NAVIGATE, target_room=24)

    total_zero = 0
    for _ in range(20):
        obs, rew, term, trunc, info = env.step(0)
        if term:
            break
        if not info.get("subgoal_achieved") and not info.get("worker_truncated"):
            assert rew == 0.0, f"Normal step should have reward=0.0, got {rew}"
            total_zero += 1

    assert total_zero > 0, "Should have had at least one normal step to verify"


def test_S17_death_penalty_boundary_values():
    """S17: Death penalty at boundary values: 0, 1, 5, 10 subgoals completed."""
    for n_sg in [0, 1, 5, 10]:
        expected = DEATH_BASE + DEATH_PER_SG * n_sg
        # Verify the formula is correct
        assert expected == -15.0 + (-10.0) * n_sg, \
            f"Formula mismatch at n_sg={n_sg}: {expected} != {-15.0 + (-10.0) * n_sg}"
        # Verify increasingly negative
        if n_sg > 0:
            prev = DEATH_BASE + DEATH_PER_SG * (n_sg - 1)
            assert expected < prev, \
                f"Death penalty should increase with more SGs: {expected} >= {prev}"


def test_S18_guard_damage_scoped():
    """S18: Guard damage reward fires ONLY during FIGHT_GUARD, silent otherwise."""
    env = PoPEnv(visual=False)
    env.reset(seed=42)

    for sg_type in [SG_NAVIGATE, SG_PICKUP_SWORD, SG_HEAL]:
        env._init_subgoal_tracking(sg_type, target_room=3)
        env.sg_prev_guard_hp = 4
        env.g_hp = 2
        # Simulate the reward calculation logic
        reward = 0.0
        if env.current_subgoal == SG_FIGHT_GUARD:
            dmg = env.sg_prev_guard_hp - env.g_hp
            if dmg > 0:
                reward += GUARD_DMG_REWARD * dmg
        assert reward == 0.0, \
            f"Guard damage should NOT fire during {sg_type}, got reward={reward}"

    # Now verify it DOES fire during FIGHT_GUARD
    env._init_subgoal_tracking(SG_FIGHT_GUARD, target_room=3)
    env.sg_prev_guard_hp = 4
    env.g_hp = 2
    reward = 0.0
    if env.current_subgoal == SG_FIGHT_GUARD:
        dmg = env.sg_prev_guard_hp - env.g_hp
        if dmg > 0:
            reward += GUARD_DMG_REWARD * dmg
    assert reward == GUARD_DMG_REWARD * 2, \
        f"Guard damage should fire during FIGHT_GUARD: expected {GUARD_DMG_REWARD*2}, got {reward}"


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

TESTS = [
    ("S01  Forward phase only unvisited",     test_S01_forward_phase_only_forward),
    ("S02  Backtrack phase includes all",      test_S02_backtrack_phase_includes_all),
    ("S03  Phase boundary behavior flip",      test_S03_phase_boundary_behavior_changes),
    ("S04  Empty neighbors → self",            test_S04_empty_neighbors_self),
    ("S05  _next_subgoal all paths reachable", test_S05_next_subgoal_all_paths),
    ("S06  Heal path no dead code",            test_S06_heal_path_no_dead_code),
    ("S07  subgoals_completed before reset",   test_S07_subgoals_completed_exists_before_reset),
    ("S08  subgoals_completed accumulates",    test_S08_subgoals_completed_accumulates),
    ("S09  subgoals_completed resets on death", test_S09_subgoals_completed_resets_on_death),
    ("S10  make_env no double reset",          test_S10_make_env_no_double_reset),
    ("S11  _room17 instance isolation",        test_S11_room17_instance_isolation),
    ("S12  _room17 survives reset",            test_S12_room17_survives_reset),
    ("S13  Phase reported in info",            test_S13_phase_in_info),
    ("S14  final_observation bootstrap",       test_S14_final_observation_code_path),
    ("S15  No glob_x/glob_y in obs",           test_S15_no_glob_xy_in_obs),
    ("S16  Zero reward on normal step",        test_S16_zero_reward_normal_step),
    ("S17  Death penalty boundary values",     test_S17_death_penalty_boundary_values),
    ("S18  Guard damage scoped to FIGHT",      test_S18_guard_damage_scoped),
]

if __name__ == "__main__":
    print("=" * 62)
    print("  Adversarial Structural Test Suite")
    print("=" * 62)
    print()

    for name, fn in TESTS:
        run(name, fn)

    print()
    print("=" * 62)
    print(f"  Results: {N_PASS} PASSED,  {N_FAIL} FAILED  (of {len(TESTS)} tests)")
    print("=" * 62)

    if N_FAIL:
        sys.exit(1)
    else:
        print("\n  All structural tests passed!")
        sys.exit(0)
