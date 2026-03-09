"""Action constants and per-frame dispatch lookup for SDLPoP RL environment.

Transcribed from control_dispatch_reference.md, verified against:
  seg005.c:L252-310  control() dispatch — the main if/else chain
  seg005.c:L964-995  control_with_sword() — combat override
  seg006.c:L908-948  check_action() — midair/freefall grab reads
"""

# ── RL Action IDs (matches rl_bridge.c:L128-222) ──

NONE        = 0
UP          = 1
DOWN        = 2
LEFT        = 3
RIGHT       = 4
SHIFT_UP    = 5
SHIFT_DOWN  = 6
SHIFT_LEFT  = 7
SHIFT_RIGHT = 8
UP_LEFT     = 9
UP_RIGHT    = 10
DOWN_LEFT   = 11
DOWN_RIGHT  = 12
INTERACT    = 13  # shift key alone

NUM_ACTIONS = 14

ACTION_NAMES = (
    "NONE", "UP", "DOWN", "LEFT", "RIGHT",
    "SHIFT+UP", "SHIFT+DOWN", "SHIFT+LEFT", "SHIFT+RIGHT",
    "UP+LEFT", "UP+RIGHT", "DOWN+LEFT", "DOWN+RIGHT",
    "INTERACT",
)


def is_decision_frame(action, frame, sword, alive=-1):
    """True when the engine reads at least one control_* flag this frame.

    Mirrors the exact if/else chain in control() (seg005.c:L252-310) plus
    the independent check_action() reads (seg006.c:L908-948).
    """
    # Dead/dying: control() triggers dying sequence, no input read
    # seg005.c:L254-261
    if alive >= 0:
        return False

    # ── First branch in control(): action-based guards ──

    # Bumped: release_arrows() only, no input dispatch
    # seg005.c:L264-267
    if action == 5:
        return False

    # Freefall: control() calls release_arrows() (clears directional flags,
    # but NOT control_shift — seg005.c:L264-265). Then check_action() calls
    # do_fall() → check_grab() which reads control_shift (seg006.c:L938-939).
    if action == 4:
        return True

    # ── Second branch: sword override ──
    # seg005.c:L268-269: else if (Char.sword == sword_2_drawn)
    # This preempts ALL frame-based dispatch below (it's an else-if chain).

    if sword == 2:
        # control_with_sword() only reads input when action < 2
        # seg005.c:L965: if (Char.action < actions_2_hang_climb)
        if action < 2:
            return True
        # sword==2 but action >= 2: control_with_sword() is called but
        # does nothing. Frame-based dispatch is skipped. Fall through to
        # check_action() reads below.

    else:
        # ── Frame-based dispatch (seg005.c:L272-288) ──
        # Only reached when sword != 2.

        # Standing: frame 15, or end-of-turn frames 50..52
        # seg005.c:L272-274
        if frame == 15 or 50 <= frame <= 52:
            return True

        # Turning: only frame 48 reads input
        # seg005.c:L276
        if frame == 48:
            return True

        # Start-run: frames 0..3
        # seg005.c:L278
        if frame < 4:
            return True

        # Start-jump-up: frames 67..69
        # seg005.c:L280
        if 67 <= frame <= 69:
            return True

        # Running: frames 4..14
        # seg005.c:L282
        if 4 <= frame <= 14:
            return True

        # Hanging: frames 87..99
        # seg005.c:L284
        if 87 <= frame <= 99:
            return True

        # Crouched: frame 109
        # seg005.c:L286
        if frame == 109:
            return True

    # ── check_action() extra reads (independent of control()) ──

    # Midair grab window: frames 102..105 read control_shift
    # seg006.c:L940-944
    if action == 3 and 102 <= frame <= 105:
        return True

    return False
