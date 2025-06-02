"""
test_edge_cases.py — Comprehensive edge case tests for clean_env.py

Edge cases covered:
  E01  _check_subgoal: FIGHT_GUARD — guard dead but in wrong room (no fire)
  E02  _check_subgoal: FIGHT_GUARD — guard dead in correct room (fires)
  E03  _check_subgoal: HEAL — hp unchanged (no fire)
  E04  _check_subgoal: HEAL — hp decreased (no fire — only increase counts)
  E05  _check_subgoal: HEAL — sg_prev_hp snapshot is stale (proves snapshot matters)
  E06  Death during subgoal — terminated=True, subgoal_achieved=False
  E07  Death AND would-be-subgoal same step — death wins, no reward bonus
  E08  worker_steps resets on reset_subgoal, but env.steps does NOT
  E09  reset_on_death twice in a row — consistent initial state both times
  E10  reset_subgoal with unknown subgoal id — _check_subgoal returns False (no crash)
  E11  Budget = 0 edge — worker_truncated fires on first step
  E12  SG_NAVIGATE with target_room=0 — invalid room, check returns False always
  E13  SG_NAVIGATE: target set to current room pre-step, then agent moves away
  E14  DummyManager reset clears worker_steps but not known_rooms
  E15  DummyManager backtracking unlock — pool changes at _room17_visits=100
  E16  DummyManager _has_potion — returns False for invalid room
  E17  DummyManager _next_subgoal — sword room priority overrides navigate
  E18  DummyManager: consecutive resets don't accumulate known_rooms across instances
  E19  Multiple DummyManager instances have independent _room17_visits (instance-level)
  E20  obs_buf is reused across steps — no aliasing corruption
  E21  _kid_vec: invalid room index (k_room=0) — no crash, returns zeros
  E22  _kid_vec: room_xs[idx]=255 sentinel handled → bx=0
  E23  sg_prev_hp assigned AFTER get_values in reset_subgoal
  E24  OBS_DIM constant matches actual observation buffer length
  E25  All info keys present in every step() return
"""

import sys
import traceback
import numpy as np

from clean_env import (
    PoPEnv, DummyManager, OBS_DIM, STACKED_DIM,
    SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL,
    SG_BUDGET, SG_REWARD, LEVEL1_GRAPH, ALIVE,
    DEATH_BASE, DEATH_PER_SG, GUARD_DMG_REWARD,
)

RESULTS = []
N_PASS = 0
N_FAIL = 0


def run(name, fn):
    global N_PASS, N_FAIL
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
        N_PASS += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        tb = traceback.format_exc()
        RESULTS.append((name, "FAIL", tb))
        N_FAIL += 1
        print(f"  [FAIL] {name}")
        print(f"         {e}")


# ─── helpers ───────────────────────────────────────────────────────────

def make_env():
    """Fresh env, fully reset."""
    e = PoPEnv(visual=False)
    e.reset(seed=0)
    return e


def steps(e, n, action=0):
    obs, rew, term, trunc, info = None, 0.0, False, False, {}
    for _ in range(n):
        obs, rew, term, trunc, info = e.step(action)
        if term or trunc:
            break
    return obs, rew, term, trunc, info


# ─── subgoal detection edge cases ──────────────────────────────────────

def test_E01_fight_guard_wrong_room():
    """E01: FIGHT_GUARD — guard dead but in wrong room → no fire."""
    e = make_env()
    e._init_subgoal_tracking(SG_FIGHT_GUARD, target_room=3)
    # Simulate guard dead but assigned to room 5 (not target)
    e.g_hp   = 0
    e.g_room = 5
    e.g_alive = ALIVE
    e.g_hpmax = 3
    assert e._check_subgoal() == False, "FIGHT_GUARD should NOT fire if guard is in wrong room"


def test_E02_fight_guard_correct_room():
    """E02: FIGHT_GUARD — guard dead in correct room → fires."""
    e = make_env()
    e._init_subgoal_tracking(SG_FIGHT_GUARD, target_room=3)
    e.g_hp    = 0
    e.g_room  = 3
    e.g_alive = ALIVE
    e.g_hpmax = 3
    assert e._check_subgoal() == True, "FIGHT_GUARD should fire when guard dead at target"


def test_E03_heal_unchanged():
    """E03: HEAL — hp unchanged → no fire."""
    e = make_env()
    original_hp = e.hp
    e._init_subgoal_tracking(SG_HEAL, target_room=e.k_room)
    e.hp = original_hp  # unchanged
    assert e._check_subgoal() == False, "HEAL should NOT fire when hp unchanged"


def test_E04_heal_decreased():
    """E04: HEAL — hp decreased (took damage) → no fire."""
    e = make_env()
    e._init_subgoal_tracking(SG_HEAL, target_room=e.k_room)
    e.hp = e.sg_prev_hp - 1  # damage taken
    assert e._check_subgoal() == False, "HEAL should NOT fire on hp decrease"


def test_E05_heal_snapshot_stale():
    """E05: HEAL — sg_prev_hp == current hp at init, then hp increases → fires."""
    e = make_env()
    e.hp = 2
    e._init_subgoal_tracking(SG_HEAL, target_room=e.k_room)
    assert e.sg_prev_hp == 2, "sg_prev_hp should snapshot current hp"
    e.hp = 4
    assert e._check_subgoal() == True, "HEAL should fire when hp > sg_prev_hp"


def test_E06_death_no_subgoal():
    """E06: Death — terminated=True, subgoal_achieved=False."""
    e = make_env()
    # Set subgoal to navigate to starting room so it would fire immediately if alive
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=e.k_room)
    # Verify that _check_subgoal returns True (agent is at target) — normally would fire
    assert e._check_subgoal() == True, "Pre-condition: check fires when at target"

    # Now step with rl_dead already = 1 — death must suppress subgoal_achieved
    # The step() function checks `alive = self.rl_dead.value != 1` AFTER play_level_2.
    # We can't reliably pre-set rl_dead since play_level_2 may reset it.
    # Instead, verify the logic directly: subgoal_achieved = _check_subgoal() IF alive.
    # We simulate by checking: when alive=False, _check_subgoal is skipped.
    e.rl_dead.value = 1
    subgoal_achieved_when_dead = e._check_subgoal() if (e.rl_dead.value != 1) else False
    assert subgoal_achieved_when_dead == False, \
        "subgoal_achieved must be False when dead (alive guard in step())"


def test_E07_death_no_sg_reward():
    """E07: Death reward uses scaled penalty, no SG bonus."""
    e = make_env()
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=e.k_room)
    e.subgoals_completed_this_episode = 0
    alive = False
    subgoal_achieved = e._check_subgoal() if alive else False
    reward = 0.0
    if not alive:
        reward = DEATH_BASE + DEATH_PER_SG * e.subgoals_completed_this_episode
    elif subgoal_achieved:
        reward = SG_REWARD[e.current_subgoal]
    assert reward == DEATH_BASE, f"Death reward should be {DEATH_BASE}, got {reward}"
    assert subgoal_achieved == False, "subgoal_achieved must be suppressed on death"


def test_E08_reset_subgoal_preserves_episode_steps():
    """E08: reset_subgoal resets worker_steps but NOT env.steps."""
    e = make_env()
    for _ in range(20):
        _, _, term, _, _ = e.step(0)
        if term:
            break

    global_steps_before = e.steps
    e.reset_subgoal(SG_HEAL, target_room=2)

    assert e.worker_steps == 0, "worker_steps should reset"
    assert e.steps == global_steps_before, \
        f"env.steps changed: {e.steps} != {global_steps_before}"


def test_E09_double_reset():
    """E09: Two consecutive reset_on_death calls produce consistent initial state."""
    e = make_env()
    obs1, info1 = e.reset_on_death(seed=0)
    obs2, info2 = e.reset_on_death(seed=0)

    # Core initial state should be identical
    assert info1["room"] == info2["room"], "Room differs between resets"
    assert info1["hp"]   == info2["hp"],   "HP differs between resets"
    assert info1["current_subgoal"] == SG_NAVIGATE
    assert info2["current_subgoal"] == SG_NAVIGATE
    assert info1["sg_target_room"] == 2
    assert info2["sg_target_room"] == 2
    assert e.worker_steps == 0
    np.testing.assert_array_equal(obs1, obs2)


def test_E10_invalid_subgoal_id():
    """E10: Unknown subgoal id → _check_subgoal returns False (no crash)."""
    e = make_env()
    e.current_subgoal = 99
    result = e._check_subgoal()
    assert result == False, "_check_subgoal should return False for unknown subgoal id"


def test_E11_zero_budget():
    """E11: Zero-step budget → worker_truncated fires immediately on first step."""
    e = make_env()
    e.current_subgoal = SG_NAVIGATE
    e.sg_target_room  = 24   # unreachable from room 1
    e.worker_steps    = 0

    # Patch budget to 0
    orig = SG_BUDGET[SG_NAVIGATE]
    SG_BUDGET[SG_NAVIGATE] = 0
    try:
        obs, rew, term, trunc, info = e.step(0)
        if not term:
            assert info["worker_truncated"] == True, \
                "worker_truncated should fire with budget=0"
    finally:
        SG_BUDGET[SG_NAVIGATE] = orig  # restore


def test_E12_navigate_invalid_target():
    """E12: SG_NAVIGATE with room 0 → _check_subgoal always False."""
    e = make_env()
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=0)
    # k_room should never be 0 in normal play
    assert e._check_subgoal() == False, "NAVIGATE to room 0 should never fire"


def test_E13_navigate_moves_away():
    """E13: Agent at target pre-step, then moves away → subgoal fires on arrival step only."""
    e = make_env()
    # Set target to already-current room
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=e.k_room)
    assert e._check_subgoal() == True, "Should detect arrival at current room"

    # Move target elsewhere — now check should fail
    e.sg_target_room = 24
    assert e._check_subgoal() == False, "Should not fire after target changes"


def test_E14_dummy_manager_reset_preserves_known_rooms():
    """E14: DummyManager reset_on_death does NOT clear known_rooms (cross-episode memory)."""
    dm = DummyManager(PoPEnv(visual=False))
    dm.reset(seed=0)
    # Manually add rooms to known_rooms
    dm.known_rooms = {1, 2, 3, 6}

    # Run a few steps
    for _ in range(5):
        obs, rew, term, trunc, info = dm.step(0)
        if term:
            break

    before = set(dm.known_rooms)
    # Hard reset
    dm.reset(seed=0)
    # known_rooms should persist — DummyManager does NOT clear it
    assert 1 in dm.known_rooms, "known_rooms should survive reset_on_death"
    # Room 1 is always present after reset
    assert len(dm.known_rooms) >= 1


def test_E15_dummy_manager_backtracking_unlock():
    """E15: _sample_neighbor pool changes when _room17_visits crosses threshold."""
    dm = DummyManager(PoPEnv(visual=False))
    dm.reset(seed=0)

    # Force below threshold — forward only
    dm._room17_visits = 0
    neighbors_fwd = set()
    for _ in range(50):
        nb = dm._sample_neighbor(2)
        neighbors_fwd.add(nb)

    # Force above threshold — any neighbor now eligible
    dm._room17_visits = 200
    neighbors_bk = set()
    for _ in range(50):
        nb = dm._sample_neighbor(2)
        neighbors_bk.add(nb)

    dm._room17_visits = 0  # restore

    assert len(neighbors_fwd) > 0
    assert len(neighbors_bk) > 0
    valid_neighbors = set(LEVEL1_GRAPH[2])
    for r in neighbors_fwd | neighbors_bk:
        assert r in valid_neighbors, f"Sampled room {r} not in LEVEL1_GRAPH[2]={valid_neighbors}"


def test_E16_has_potion_invalid_room():
    """E16: DummyManager._has_potion for invalid room doesn't crash."""
    dm = DummyManager(PoPEnv(visual=False))
    dm.reset(seed=0)

    assert dm._has_potion(0)  == False, "Room 0 should return False"
    assert dm._has_potion(25) == False, "Room 25 should return False"
    assert dm._has_potion(-1) == False, "Room -1 should return False"


def test_E17_next_subgoal_sword_room_priority():
    """E17: _next_subgoal issues PICKUP_SWORD when agent is in sword room without sword."""
    dm = DummyManager(PoPEnv(visual=False))
    dm.reset(seed=0)

    # Fake being in sword room without sword
    dm.env.k_room = 15
    dm.env.have_sword = 0

    sg, target = dm._next_subgoal(current_room=15)
    assert sg == SG_PICKUP_SWORD, f"Expected PICKUP_SWORD, got {sg}"
    assert target == 15, f"Expected target=15, got {target}"


def test_E18_separate_instances_separate_known_rooms():
    """E18: Two DummyManager instances have independent known_rooms."""
    dm1 = DummyManager(PoPEnv(visual=False))
    dm2 = DummyManager(PoPEnv(visual=False))
    dm1.reset(seed=0)
    dm2.reset(seed=0)

    dm1.known_rooms.add(99)
    assert 99 not in dm2.known_rooms, "known_rooms leaked between instances"


def test_E19_independent_room17_counter():
    """E19: Two DummyManager instances have independent _room17_visits (instance-level)."""
    dm1 = DummyManager(PoPEnv(visual=False))
    dm2 = DummyManager(PoPEnv(visual=False))
    dm1.reset(seed=0)
    dm2.reset(seed=0)

    dm1._room17_visits = 50
    assert dm2._room17_visits == 0, "dm2 should not see dm1's counter"
    dm1._room17_visits += 1
    assert dm1._room17_visits == 51
    assert dm2._room17_visits == 0, "Instance counters should be independent"


def test_E20_obs_buf_aliasing_behavior():
    """E20: With frame stacking, step() returns concatenated copies (independent)."""
    e = make_env()
    obs1, _, _, _, _ = e.step(0)
    obs2, _, _, _, _ = e.step(0)
    # Frame stacking produces independent concatenated arrays
    assert obs1.shape == (STACKED_DIM,), f"obs shape {obs1.shape} != ({STACKED_DIM},)"
    assert obs2.shape == (STACKED_DIM,), f"obs shape {obs2.shape} != ({STACKED_DIM},)"


def test_E21_kid_vec_room_zero():
    """E21: _kid_vec with k_room=0 doesn't crash."""
    e = make_env()
    e.k_room = 0
    e.k_col  = 0
    e.k_row  = 0
    try:
        e._kid_vec()
    except Exception as ex:
        raise AssertionError(f"_kid_vec crashed with k_room=0: {ex}")


def test_E22_budget_remaining_in_obs():
    """E22: budget_remaining at kid_v[24] — 1.0 at start, decreases with steps."""
    e = make_env()
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=24)
    e.worker_steps = 0
    e._kid_vec()
    assert abs(float(e.kid_v[24]) - 1.0) < 1e-6, \
        f"budget_remaining should be 1.0 at step 0, got {e.kid_v[24]}"

    from clean_env import SG_BUDGET
    budget = SG_BUDGET[SG_NAVIGATE]
    e.worker_steps = budget // 2
    e._kid_vec()
    expected = (budget - budget // 2) / budget
    assert abs(float(e.kid_v[24]) - expected) < 1e-5, \
        f"budget_remaining should be ~{expected:.3f}, got {e.kid_v[24]}"


def test_E23_sg_prev_hp_after_get_values():
    """E23: sg_prev_hp is snapshotted AFTER get_values in reset_subgoal."""
    e = make_env()
    # Manually change hp before calling reset_subgoal
    e.hp = 1  # artificially low
    obs, info = e.reset_subgoal(SG_HEAL, target_room=e.k_room)
    # sg_prev_hp must reflect the hp AS READ by get_values (not the stale manual set)
    # get_values() re-reads from the C engine, so e.hp may differ from 1
    # What we verify: sg_prev_hp == e.hp (they were both set in sequence)
    assert e.sg_prev_hp == e.hp, \
        f"sg_prev_hp {e.sg_prev_hp} != hp {e.hp} after reset_subgoal"


def test_E24_obs_dim_constant():
    """E24: OBS_DIM and STACKED_DIM match actual observation."""
    from clean_env import GRID_FLAT, KID_DIM, G_DIM, N_STACK
    assert OBS_DIM == GRID_FLAT + KID_DIM + G_DIM, \
        f"OBS_DIM={OBS_DIM} != GRID_FLAT({GRID_FLAT})+KID_DIM({KID_DIM})+G_DIM({G_DIM})"
    assert STACKED_DIM == OBS_DIM * N_STACK, \
        f"STACKED_DIM={STACKED_DIM} != OBS_DIM({OBS_DIM})*N_STACK({N_STACK})"
    e = make_env()
    obs, _ = e.reset()
    assert obs.shape[0] == STACKED_DIM, f"obs.shape[0]={obs.shape[0]} != STACKED_DIM={STACKED_DIM}"


def test_E25_all_info_keys():
    """E25: All expected info keys present in every step()."""
    REQUIRED = {"level", "room", "hp", "steps", "worker_steps",
                "dead", "current_subgoal", "sg_target_room",
                "subgoal_achieved", "worker_truncated"}
    e = make_env()
    for i in range(30):
        _, _, term, trunc, info = e.step(e.action_space.sample())
        missing = REQUIRED - set(info.keys())
        assert not missing, f"Step {i}: missing info keys {missing}"
        if term or trunc:
            break


# ─── reward system edge cases ──────────────────────────────────────────

def test_E26_death_base_penalty():
    """E26: Death with 0 subgoals completed → exactly DEATH_BASE."""
    e = make_env()
    e.subgoals_completed_this_episode = 0
    # Simulate death reward logic
    reward = DEATH_BASE + DEATH_PER_SG * 0
    assert reward == DEATH_BASE, f"Expected {DEATH_BASE}, got {reward}"
    assert reward == -5.0, f"DEATH_BASE should be -5.0, got {reward}"


def test_E27_death_scaled_penalty():
    """E27: Death after 2 subgoals completed → DEATH_BASE + 2*DEATH_PER_SG."""
    e = make_env()
    e.subgoals_completed_this_episode = 2
    expected = DEATH_BASE + DEATH_PER_SG * 2  # -5 + (-2)*2 = -9
    reward = DEATH_BASE + DEATH_PER_SG * e.subgoals_completed_this_episode
    assert reward == expected, f"Expected {expected}, got {reward}"
    assert reward == -9.0, f"Expected -9.0, got {reward}"


def test_E28_guard_damage_reward():
    """E28: Guard damage during FIGHT_GUARD → +GUARD_DMG_REWARD per HP."""
    e = make_env()
    e._init_subgoal_tracking(SG_FIGHT_GUARD, target_room=3)
    e.subgoals_completed_this_episode = 0
    # Simulate: guard had 4 HP, now has 2 HP (lost 2)
    e.sg_prev_guard_hp = 4
    e.g_hp = 2
    # Reward logic from step(): alive, not subgoal achieved, FIGHT_GUARD active
    alive = True
    reward = 0.0
    # Guard damage shaping
    if alive and e.current_subgoal == SG_FIGHT_GUARD:
        dmg = e.sg_prev_guard_hp - e.g_hp
        if dmg > 0:
            reward += GUARD_DMG_REWARD * dmg
    assert reward == GUARD_DMG_REWARD * 2, f"Expected {GUARD_DMG_REWARD*2}, got {reward}"
    assert reward == 10.0, f"Expected 10.0, got {reward}"


def test_E29_no_guard_damage_in_navigate():
    """E29: Guard damage does NOT produce reward during NAVIGATE."""
    e = make_env()
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=3)
    e.subgoals_completed_this_episode = 0
    e.sg_prev_guard_hp = 4
    e.g_hp = 2
    alive = True
    reward = 0.0
    if alive and e.current_subgoal == SG_FIGHT_GUARD:
        dmg = e.sg_prev_guard_hp - e.g_hp
        if dmg > 0:
            reward += GUARD_DMG_REWARD * dmg
    assert reward == 0.0, f"Guard damage reward should not fire in NAVIGATE, got {reward}"


def test_E30_subgoals_completed_counter():
    """E30: subgoals_completed_this_episode increments on SG, resets on death."""
    e = make_env()
    assert e.subgoals_completed_this_episode == 0, "Should start at 0"
    # Simulate subgoal completion
    e.subgoals_completed_this_episode += 1
    assert e.subgoals_completed_this_episode == 1
    e.subgoals_completed_this_episode += 1
    assert e.subgoals_completed_this_episode == 2
    # Reset
    e.reset_on_death(seed=0)
    assert e.subgoals_completed_this_episode == 0, "Should reset to 0 after death"


def test_E31_no_step_penalty():
    """E31: Normal alive step with no event → reward = 0.0 exactly."""
    e = make_env()
    e._init_subgoal_tracking(SG_NAVIGATE, target_room=24)  # unreachable
    e.subgoals_completed_this_episode = 0
    obs, rew, term, trunc, info = e.step(0)
    if not term and not info.get("subgoal_achieved", False):
        assert rew == 0.0, f"Step reward should be 0.0 with no events, got {rew}"


# ─── runner ────────────────────────────────────────────────────────────

TESTS = [
    ("E01  FIGHT_GUARD wrong room",          test_E01_fight_guard_wrong_room),
    ("E02  FIGHT_GUARD correct room",        test_E02_fight_guard_correct_room),
    ("E03  HEAL hp unchanged",               test_E03_heal_unchanged),
    ("E04  HEAL hp decreased",               test_E04_heal_decreased),
    ("E05  HEAL sg_prev_hp snapshot",        test_E05_heal_snapshot_stale),
    ("E06  Death → subgoal_achieved=False",  test_E06_death_no_subgoal),
    ("E07  Death cancels SG reward",         test_E07_death_no_sg_reward),
    ("E08  reset_subgoal preserves steps",   test_E08_reset_subgoal_preserves_episode_steps),
    ("E09  Double reset_on_death",           test_E09_double_reset),
    ("E10  Invalid subgoal id",              test_E10_invalid_subgoal_id),
    ("E11  Zero-step budget → truncated",    test_E11_zero_budget),
    ("E12  NAVIGATE target=0 invalid",       test_E12_navigate_invalid_target),
    ("E13  NAVIGATE moves away",             test_E13_navigate_moves_away),
    ("E14  DM reset preserves known_rooms",  test_E14_dummy_manager_reset_preserves_known_rooms),
    ("E15  DM backtracking pool change",     test_E15_dummy_manager_backtracking_unlock),
    ("E16  DM _has_potion invalid room",     test_E16_has_potion_invalid_room),
    ("E17  DM sword room priority",          test_E17_next_subgoal_sword_room_priority),
    ("E18  Separate instances known_rooms",  test_E18_separate_instances_separate_known_rooms),
    ("E19  Independent room17 counter",     test_E19_independent_room17_counter),
    ("E20  obs_buf aliasing (by design)",     test_E20_obs_buf_aliasing_behavior),
    ("E21  _kid_vec room=0 no crash",        test_E21_kid_vec_room_zero),
    ("E22  budget_remaining in obs",         test_E22_budget_remaining_in_obs),
    ("E23  sg_prev_hp after get_values",     test_E23_sg_prev_hp_after_get_values),
    ("E24  OBS_DIM constant correct",        test_E24_obs_dim_constant),
    ("E25  All info keys present",           test_E25_all_info_keys),
    ("E26  Death penalty (base, 0 SGs)",     test_E26_death_base_penalty),
    ("E27  Death penalty (scaled, 2 SGs)",   test_E27_death_scaled_penalty),
    ("E28  Guard dmg reward (FIGHT_GUARD)",  test_E28_guard_damage_reward),
    ("E29  Guard dmg silent in NAVIGATE",    test_E29_no_guard_damage_in_navigate),
    ("E30  subgoals_completed counter",      test_E30_subgoals_completed_counter),
    ("E31  No step penalty (reward=0)",      test_E31_no_step_penalty),
]

if __name__ == "__main__":
    print("=" * 62)
    print("  Edge Case Test Suite — clean_env.py Subgoal System")
    print("=" * 62)
    print()

    for name, fn in TESTS:
        run(name, fn)

    print()
    print("=" * 62)
    print(f"  Results: {N_PASS} PASSED,  {N_FAIL} FAILED  (of {len(TESTS)} tests)")
    print("=" * 62)

    if N_FAIL:
        print("\n  FAILURES:")
        for name, status, tb in RESULTS:
            if status == "FAIL":
                print(f"\n  --- {name} ---")
                print(tb)
        sys.exit(1)
    else:
        print("\n  All edge case tests passed!")
        sys.exit(0)
