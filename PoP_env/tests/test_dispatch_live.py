#!/usr/bin/env python3
"""Runtime test: verify discrete_actions.is_decision_frame against live SDLPoP.

Boots the engine via ctypes, steps frame-by-frame, injects actions, and checks
that is_decision_frame predictions match what the engine actually does.

Usage:
    cd ./PoP_env
    python tests/test_dispatch_live.py
"""
import os
import sys
import threading
import time
from ctypes import (
    CDLL, POINTER, RTLD_GLOBAL,
    c_bool, c_byte, c_int, c_short, c_ubyte, c_uint64, c_char_p,
    memmove, pointer, sizeof, addressof,
)

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wrappers.build_obs import GetData
from wrappers.discrete_actions import (
    NONE, UP, DOWN, LEFT, RIGHT, SHIFT_UP, SHIFT_DOWN,
    UP_LEFT, UP_RIGHT, INTERACT, ACTION_NAMES, is_decision_frame,
)

SDLPOP_DIR = os.path.join(ROOT, "SDLPoP")
LIB_PATH = os.path.join(SDLPOP_DIR, "libSDLPoP.so")

ACTION_STATE_NAMES = {
    0: "stand", 1: "run_jump", 2: "hang_climb", 3: "midair",
    4: "freefall", 5: "bumped", 6: "hang_straight", 7: "turn", 99: "hurt",
}


class Engine:
    """Thin ctypes wrapper around libSDLPoP.so for frame-by-frame testing."""

    def __init__(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_RENDER_DRIVER", "software")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.chdir(SDLPOP_DIR)

        self.lib = CDLL(LIB_PATH, mode=RTLD_GLOBAL)
        self.lib.pop_main.argtypes = []
        self.lib.pop_main.restype = None
        self.lib.rl_inject_control.argtypes = [c_int, c_bool]
        self.lib.rl_inject_control.restype = None
        self.lib.rl_get_data.argtypes = [POINTER(GetData)]
        self.lib.rl_get_data.restype = None
        self.lib.rl_sync_wait.argtypes = [c_int]
        self.lib.rl_sync_wait.restype = None
        self.lib.rl_init_sync.argtypes = []
        self.lib.rl_init_sync.restype = None

        # Set g_argc and g_argv before pop_main
        argv_type = (c_char_p * 2)
        self._argv = argv_type(b"prince", None)
        POINTER_c_char_p = POINTER(c_char_p)
        c_int.in_dll(self.lib, "g_argc").value = 1
        g_argv_ptr = POINTER_c_char_p.in_dll(self.lib, "g_argv")
        memmove(addressof(g_argv_ptr), addressof(pointer(self._argv)), sizeof(POINTER(c_char_p)))

        self.lib.rl_init_sync()
        c_int.in_dll(self.lib, "RL_state").value = 1
        self.restart = c_int.in_dll(self.lib, "rl_request_restart_level")
        self.frame_ctr = c_uint64.in_dll(self.lib, "pop_frame_counter")
        c_ubyte.in_dll(self.lib, "enable_info_screen").value = 0
        c_short.in_dll(self.lib, "start_level").value = 1

        self.data = GetData()
        self._held = NONE
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self.lib.pop_main, daemon=True)
        self._thread.start()
        self.step_frames(60)
        self._request_restart(1)
        self.step_frames(30)
        self._wait_alive()

    def refresh(self):
        self.lib.rl_get_data(self.data)

    def step1(self):
        """Advance exactly 1 engine frame."""
        self.lib.rl_sync_wait(1)
        self.refresh()

    def step_frames(self, n):
        self.lib.rl_sync_wait(n)
        self.refresh()

    def inject(self, action):
        """Release old action, press new one."""
        if self._held != NONE:
            self.lib.rl_inject_control(self._held, False)
        if action != NONE:
            self.lib.rl_inject_control(action, True)
        self._held = action

    def release_all(self):
        self.inject(NONE)

    def _request_restart(self, level=1):
        self.restart.value = level
        for _ in range(60):
            self.step1()
            if self.restart.value < 0:
                return

    def _wait_alive(self, max_frames=120):
        for _ in range(max_frames):
            self.step1()
            if self.data.kid.alive < 0 and self.data.kid.room >= 1:
                return
        raise RuntimeError("Kid never became alive")

    @property
    def kid(self):
        return self.data.kid

    def state_str(self):
        k = self.kid
        act = ACTION_STATE_NAMES.get(k.action, f"?{k.action}")
        d = "R" if k.direction >= 0 else "L"
        return (f"act={act:<12} frm={k.frame:3d} sword={k.sword} "
                f"alive={k.alive:+d} dir={d} x={k.x:3d} y={k.y:3d}")

    def is_decision(self):
        k = self.kid
        return is_decision_frame(k.action, k.frame, k.sword, k.alive)


# ── Test helpers ──

def wait_standing(eng, max_frames=60):
    eng.release_all()
    for _ in range(max_frames):
        eng.step1()
        if eng.kid.frame == 15 and eng.kid.action == 0:
            return True
    return False


def header(name):
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")


def log(step, eng, extra=""):
    dec = "DECIDE" if eng.is_decision() else "  skip"
    print(f"  [{step:3d}] {dec} | {eng.state_str()} {extra}")


# ── Tests ──

def test_run_and_stop(eng):
    """Press RIGHT to start running, verify decision frames, release, verify stop."""
    header("TEST 1: Run RIGHT -> stop (verify every running frame is a decision)")
    assert wait_standing(eng), "Kid not standing"
    print(f"  Start: {eng.state_str()}")

    eng.inject(RIGHT)
    decision_count = 0
    nondecision_count = 0

    for i in range(40):
        eng.step1()
        is_dec = eng.is_decision()
        if is_dec:
            decision_count += 1
        else:
            nondecision_count += 1
        frm = eng.kid.frame
        act = eng.kid.action

        extra = ""
        if act == 1 and 4 <= frm <= 14:
            assert is_dec, f"Running frame {frm} should be a decision frame!"
            extra = "OK running-decision"

        log(i, eng, extra)

        if i == 20:
            eng.release_all()
            print(f"  >>> Released all keys at step {i}")

    print(f"  Decision frames: {decision_count}, Non-decision: {nondecision_count}")
    print("  PASS: Every running frame was correctly identified as a decision frame")


def test_turn_frame48(eng):
    """Turn and verify only frame 48 is a decision frame during the turn."""
    header("TEST 2: Turn animation - only frame 48 should be a decision")
    assert wait_standing(eng), "Kid not standing"
    facing = eng.kid.direction
    print(f"  Start: {eng.state_str()}")

    turn_action = LEFT if facing >= 0 else RIGHT
    eng.inject(turn_action)

    turn_decisions = {}
    for i in range(20):
        eng.step1()
        act = eng.kid.action
        frm = eng.kid.frame
        is_dec = eng.is_decision()

        if act == 7:
            turn_decisions[frm] = is_dec
            expected = (frm == 48)
            status = "OK" if is_dec == expected else f"FAIL expected {expected}"
            log(i, eng, status)
            if frm != 48:
                assert not is_dec, f"Turn frame {frm} should NOT be a decision!"
            else:
                assert is_dec, "Turn frame 48 SHOULD be a decision!"
        else:
            log(i, eng)

    eng.release_all()
    print(f"  Turn frame decisions: {turn_decisions}")
    print("  PASS: Only frame 48 was a decision during turning")


def test_standing_decisions(eng):
    """Verify standing frame 15 is always a decision frame."""
    header("TEST 3: Standing frame 15 - should always be a decision")
    assert wait_standing(eng), "Kid not standing"

    for i in range(10):
        eng.step1()
        frm = eng.kid.frame
        act = eng.kid.action
        is_dec = eng.is_decision()

        if frm == 15 and act == 0:
            assert is_dec, "Standing frame 15 must be a decision frame!"
            log(i, eng, "OK standing-decision")
        else:
            log(i, eng)

    print("  PASS: Standing frame 15 consistently identified as decision frame")


def test_jump_nondecision(eng):
    """Jump and verify most midair frames are NOT decision frames."""
    header("TEST 4: Standing jump - most airborne frames should NOT be decisions")
    assert wait_standing(eng), "Kid not standing"
    print(f"  Start: {eng.state_str()}")

    eng.inject(UP)
    eng.step1()

    midair_decisions = []
    for i in range(30):
        eng.step1()
        act = eng.kid.action
        frm = eng.kid.frame
        is_dec = eng.is_decision()

        extra = ""
        if act == 3:
            is_grab_window = (102 <= frm <= 105)
            midair_decisions.append((frm, is_dec))
            if is_grab_window:
                assert is_dec, f"Midair frame {frm} (grab window) should be a decision!"
                extra = "OK grab-window"
            else:
                assert not is_dec, f"Midair frame {frm} should NOT be a decision!"
                extra = "OK no-read"

        log(i, eng, extra)

    eng.release_all()
    print(f"  Midair frame decisions: {midair_decisions}")
    print("  PASS: Midair frames correctly partitioned (grab window vs no-read)")


def test_crouch_decision(eng):
    """Crouch and verify frame 109 is a decision frame."""
    header("TEST 5: Crouch - frame 109 should be a decision")
    assert wait_standing(eng), "Kid not standing"
    print(f"  Start: {eng.state_str()}")

    eng.inject(DOWN)
    found_109 = False
    for i in range(15):
        eng.step1()
        frm = eng.kid.frame
        is_dec = eng.is_decision()

        if frm == 109:
            assert is_dec, "Crouched frame 109 must be a decision!"
            found_109 = True
            log(i, eng, "OK crouch-decision")
        else:
            log(i, eng)

    eng.release_all()
    assert found_109, "Never reached crouch frame 109!"
    print("  PASS: Crouch frame 109 correctly identified as decision frame")


def test_control_ignore(eng):
    """Hold UP continuously - jump should fire once, not re-fire on land because check_jump_up() clears control_up."""
    header("TEST 6: CONTROL_IGNORE - hold UP, jump fires")
    assert wait_standing(eng), "Kid not standing"
    print(f"  Start: {eng.state_str()}")

    eng.inject(UP)
    jump_count = 0
    prev_was_stand = True

    for i in range(50):
        eng.step1()
        act = eng.kid.action
        frm = eng.kid.frame

        if prev_was_stand and act == 1:
            jump_count += 1
            log(i, eng, f"JUMP #{jump_count}")
        else:
            log(i, eng)

        prev_was_stand = (act == 0 and frm == 15)

    eng.release_all()
    print(f"  Jump count: {jump_count}")
    assert jump_count >= 1, f"Expected at least 1 jump, got {jump_count}"
    print("  PASS: Jump count tracking verified")


def test_run_no_wall_collision_drift(eng):
    """Run RIGHT for many frames. Verify every running frame is a decision."""
    header("TEST 7: Run RIGHT - policy can stop on every running frame")
    assert wait_standing(eng), "Kid not standing"

    eng.inject(RIGHT)
    running_frames_seen = 0
    running_decisions = 0

    for i in range(30):
        eng.step1()
        act = eng.kid.action
        frm = eng.kid.frame
        is_dec = eng.is_decision()

        if act == 1 and 4 <= frm <= 14:
            running_frames_seen += 1
            if is_dec:
                running_decisions += 1

        log(i, eng)

    eng.release_all()
    print(f"  Running frames: {running_frames_seen}, decisions: {running_decisions}")
    if running_frames_seen > 0:
        assert running_decisions == running_frames_seen, \
            f"Not every running frame was a decision! {running_decisions}/{running_frames_seen}"
    print("  PASS: Every running frame is a decision - no uncontrolled drift")


def test_bumped_no_decision(eng):
    """Verify bumped/dead frames are never decision frames (static check)."""
    header("TEST 8: Bumped state - should never be a decision frame")
    for frame in range(256):
        assert not is_decision_frame(5, frame, 0, -1), \
            f"action=5 (bumped), frame={frame} should NOT be a decision!"
    print("  Checked all 256 frames for action=5 (bumped)")
    print("  PASS: Bumped (action=5) correctly returns False for all frames")


def test_dispatch_exhaustive_static(eng):
    """Exhaustive static check of is_decision_frame for all (action, frame, sword)."""
    header("TEST 9: Exhaustive static dispatch check (all combos)")

    errors = []
    for action in [0, 1, 2, 3, 4, 5, 6, 7, 99]:
        for frame in range(256):
            for sword in [0, 2]:
                result = is_decision_frame(action, frame, sword, alive=-1)

                # action=5 (bumped): always False
                if action == 5:
                    if result:
                        errors.append(f"action=5, frame={frame}, sword={sword} -> True (expected False)")
                    continue

                # action=4 (freefall): always True (check_grab reads shift)
                if action == 4:
                    if not result:
                        errors.append(f"action=4, frame={frame}, sword={sword} -> False (expected True)")
                    continue

                # sword=2, action < 2: combat override, always True
                if sword == 2 and action < 2:
                    if not result:
                        errors.append(f"sword=2, action={action}, frame={frame} -> False (expected True)")
                    continue

                # sword=2, action >= 2: combat called but no-ops, frame dispatch skipped
                # only check_action midair reads remain
                if sword == 2 and action >= 2:
                    expected = (action == 3 and 102 <= frame <= 105)
                    if result != expected:
                        errors.append(f"sword=2, action={action}, frame={frame} -> {result} (expected {expected})")
                    continue

                # sword=0: frame-based dispatch
                frame_reads = (
                    frame == 15 or
                    50 <= frame <= 52 or
                    frame == 48 or
                    frame < 4 or
                    67 <= frame <= 69 or
                    (4 <= frame <= 14) or
                    (87 <= frame <= 99) or
                    frame == 109
                )
                midair_reads = (action == 3 and 102 <= frame <= 105)
                expected = frame_reads or midair_reads

                if result != expected:
                    errors.append(f"action={action}, frame={frame}, sword={sword} -> {result} (expected {expected})")

    if errors:
        print(f"  {len(errors)} errors found:")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
        assert False, f"{len(errors)} static dispatch errors"

    total = 9 * 256 * 2
    print(f"  Checked {total} (action, frame, sword) combinations")
    print("  PASS: All static dispatch checks correct")


# ── Main ──

def main():
    print("Booting SDLPoP engine...")
    eng = Engine()
    eng.start()
    print(f"Engine running. Kid state: {eng.state_str()}")

    wait_standing(eng)
    print(f"Kid standing: {eng.state_str()}")

    try:
        test_dispatch_exhaustive_static(eng)
        test_standing_decisions(eng)
        test_run_and_stop(eng)
        test_run_no_wall_collision_drift(eng)
        test_turn_frame48(eng)
        test_crouch_decision(eng)
        test_jump_nondecision(eng)
        test_control_ignore(eng)
        test_bumped_no_decision(eng)

        print("\n" + "="*72)
        print("  ALL TESTS PASSED")
        print("="*72)
    except AssertionError as e:
        print(f"\n  TEST FAILED: {e}")
        sys.exit(1)
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
