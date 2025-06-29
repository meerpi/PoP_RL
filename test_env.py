"""
Grid validation test — runs the game with random actions and checks every
channel of the [20, 3, 10] grid each frame.

Reports:
  • Per-frame ASCII grid showing tile types, player, guard, items
  • Continuous validation of channel invariants
  • Summary statistics at the end (channel ranges, unique values, errors)
"""
import ctypes
from ctypes import c_int, c_short
import os
import sys
import random
import time
import numpy as np

# ── Import PoPEnv ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_env import PoPEnv

# ── Channel names for pretty-printing ─────────────────────────────────────────
CH_NAMES = [
    "TILE_TYPE", "WALKABLE", "WALL", "FLOOR_EDGE", "LOOSE_STATE",
    "GATE", "BUTTON", "CHOMPER", "PLAYER_POS", "PLAYER_ACT",
    "ENEMY_POS", "ENEMY_DIR", "ITEM",
    "PLAYER_DIR", "PLAYER_SWORD", "PLAYER_FRAME",
    "ENEMY_ACT", "ENEMY_SWORD", "ENEMY_TYPE", "ENEMY_FRAME",
]

# Tile type enum → short label for ASCII display
TILE_LABELS = {
    0: "  .", 1: " FL", 2: " SK", 3: " PL", 4: " GA",
    5: " SB", 6: " DB", 7: " TA", 8: " Bb", 9: " Bt",
    10:" PO", 11:" LO", 12:" DT", 13:" MI", 14:" DE",
    15:" RB", 16:" EL", 17:" ER", 18:" CH", 19:" TO",
    20:" WL", 21:" SK", 22:" SW", 30:" TD",
}

# Action table (matches C-side switch)
ACTION_TABLE = {
    0:  ( 0,  0,  0),   # nothing
    1:  ( 1,  0,  0),   # forward
    2:  (-1,  0,  0),   # backward
    3:  ( 0, -1,  0),   # up / jump
    4:  ( 0,  1,  0),   # down / crouch
    5:  ( 0,  0, -1),   # shift / grab
    6:  ( 1, -1,  0),   # forward + up (running jump)
    7:  ( 1,  1,  0),   # forward + down
    8:  (-1, -1,  0),   # backward + up
    9:  (-1,  1,  0),   # backward + down
    10: ( 1,  0, -1),   # forward + shift
    11: ( 1, -1, -1),   # forward + up + shift
    12: ( 1,  1, -1),   # forward + down + shift
    13: (-1,  0, -1),   # backward + shift
    14: (-1, -1, -1),   # backward + up + shift
    15: (-1,  1, -1),   # backward + down + shift
    16: ( 0, -1, -1),   # up + shift
    17: ( 0,  1, -1),   # down + shift
}

ACTION_NAMES = [
    "stand", "fwd", "back", "jump", "crouch", "shift",
    "run-jump", "fwd+dn", "back+up", "back+dn",
    "care-step", "fwd+up+sh", "fwd+dn+sh", "back+sh",
    "back+up+sh", "back+dn+sh", "up+sh", "dn+sh",
]

PLAYER_ACT_NAMES = [
    "standing", "running", "hang-climb", "midair",
    "freefall", "bumped", "hanging", "turning",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def validate_grid(grid, env, step_i, errors):
    """Run invariant checks on every channel. Appends to errors list."""
    g = grid  # shorthand

    # Shape
    if g.shape != (20, 3, 10):
        errors.append(f"step {step_i}: BAD SHAPE {g.shape}")
        return

    # Generic normalization check: all values must be in [0.0, 1.0]
    for ch in range(20):
        out_of_bounds = (g[ch] < 0.0) | (g[ch] > 1.0)
        if np.any(out_of_bounds):
            bad_locs = np.argwhere(out_of_bounds)
            for r, c in bad_locs:
                errors.append(f"step {step_i} [{r},{c}]: CH_{CH_NAMES[ch]} = {g[ch, r, c]:.3f} (must be 0.0–1.0)")

    # Player position: exactly one cell should have PLAYER_POS=1
    player_count = int(np.sum(g[8]))
    if player_count != 1:
        # Player might be in a different room → channel all zeros is valid
        if player_count > 1:
            errors.append(f"step {step_i}: PLAYER_POS sum is {player_count} (expected 0 or 1)")

    # Enemy position: at most one cell
    enemy_count = int(np.sum(g[10] > 0))
    if enemy_count > 1:
        errors.append(f"step {step_i}: ENEMY_POS non-zero in {enemy_count} cells (expected 0 or 1)")


def print_ascii_grid(grid, env, step_i, act):
    """Print a compact ASCII representation of the room."""
    print(f"\n{'═'*72}")
    print(f"  Step {step_i:5d}  |  Action: {act:2d} ({ACTION_NAMES[act]:>12s})  |"
          f"  Room {env.kid_room}  |  HP {env.hitp_curr}/{env.hitp_max}  |"
          f"  Lvl {env.current_level}")
    print(f"  Kid: col={env.kid_curr_col} row={env.kid_curr_row} "
          f"act={env.kid_action} dir={env.kid_direction} alive={env.kid_alive}")
    if env.guard_room == env.kid_room:
        print(f"  Guard: col={env.guard_curr_col} row={env.guard_curr_row} "
              f"hp={env.guardhp_curr} dir={env.guard_direction} alive={env.guard_alive}")
    print(f"{'─'*72}")

    # Tile type row with player/guard/item overlay
    print("  Tile map (player=@, guard=G, items=*):")
    for r in range(3):
        line = "    "
        for c in range(10):
            t = int(round(grid[0, r, c] * 30.0))
            label = TILE_LABELS.get(t, f"{t:3d}")

            # Overlay markers
            if grid[8, r, c] == 1.0:           # player
                label = " @" + label[2]
            elif grid[10, r, c] > 0:           # guard
                label = " G" + str(int(round(grid[10, r, c] * 4.0)))
            elif grid[12, r, c] > 0:           # item
                item_v = int(round(grid[12, r, c] * 6.0))
                item_chars = {1: "♥", 2: "♥+", 3: "☠", 4: "~", 5: "†", 6: "⇒"}
                label = f" {item_chars.get(item_v, '?'):>2s}"

            line += label
        print(line)

    # Channel summary (non-zero channels only)
    active_chs = []
    for ch in range(20):
        nz = np.count_nonzero(grid[ch])
        if nz > 0:
            vals = np.unique(grid[ch][grid[ch] != 0])
            active_chs.append(f"{CH_NAMES[ch]}({nz}nz, vals={vals})")
    if active_chs:
        print(f"  Active: {', '.join(active_chs)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main test loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Creating PoPEnv...")
    env = PoPEnv()

    # Set up function signatures
    env.lib.pop_main.argtypes  = []
    env.lib.pop_main.restype   = None
    env.lib.init_game.argtypes = [c_int]
    env.lib.init_game.restype  = None
    env.lib.play_level_2.argtypes = []
    env.lib.play_level_2.restype  = c_int

    # Globals for step injection
    action_g      = c_int.in_dll(env.lib, "action")
    kid_death  = c_int.in_dll(env.lib, "kid_death")

    print("Calling pop_main() to initialize...")
    env.lib.pop_main()
    env.get_values()
    print("Game initialized.\n")

    # ═══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: Scan all 24 rooms to see what tiles exist in the level
    # ═══════════════════════════════════════════════════════════════════════════
    SPECIAL_TILES = {
        4: "GATE", 5: "STUCK_BTN", 6: "DROP_BTN", 10: "POTION",
        11: "LOOSE", 15: "RAISE_BTN", 16: "EXIT_L", 17: "EXIT_R",
        18: "CHOMPER", 2: "SPIKES", 22: "SWORD",
    }
    print("── Level scan: special tiles per room ──")
    rooms_with_guard = []
    for room in range(1, 25):
        offset = (room - 1) * 30
        fg_raw = env.fg[offset : offset + 30]
        bg_raw = env.bg[offset : offset + 30]
        specials = {}
        for i in range(30):
            t = int(fg_raw[i]) & 0x1F
            if t in SPECIAL_TILES:
                name = SPECIAL_TILES[t]
                specials[name] = specials.get(name, 0) + 1
        # Check if this room has a guard
        guard_tile = int(env.guards_tile[room - 1])
        has_guard = guard_tile != 0 and guard_tile != 255
        if has_guard:
            rooms_with_guard.append(room)
        if specials or has_guard:
            parts = [f"{v}×{k}" for k, v in specials.items()]
            guard_str = f" [GUARD at tile {guard_tile}]" if has_guard else ""
            print(f"  Room {room:2d}: {', '.join(parts)}{guard_str}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: Test grid for each room by faking kid_room
    #  This validates that the grid correctly encodes tiles for rooms
    #  the random agent can't easily reach
    # ═══════════════════════════════════════════════════════════════════════════
    print("── Room-by-room grid tile check ──")
    room_errors = []
    for room in range(1, 25):
        # Temporarily set kid_room to scan this room's grid
        saved_room = env.kid_room
        env.kid_room = room
        grid = env.create_grid()
        env.kid_room = saved_room

        # Check tile types are all valid (0–30)
        tt = grid[0]
        max_tt = tt.max()
        if max_tt > 30:
            room_errors.append(f"Room {room}: TILE_TYPE max={max_tt}")

        # Report active channels
        active = []
        for ch in range(13):
            nz = np.count_nonzero(grid[ch])
            if nz > 0:
                uvals = sorted(set(grid[ch].flatten()))
                uvals = [v for v in uvals if v != 0]
                active.append(f"{CH_NAMES[ch]}={uvals}")
        if len(active) > 1:  # more than just TILE_TYPE
            print(f"  Room {room:2d}: {', '.join(active)}")

    if room_errors:
        print(f"\n  ⚠ Room scan errors: {room_errors}")
    else:
        print(f"\n  ✓ All 24 rooms have valid tile types (0–30)")
    print()


    # ── Config ─────────────────────────────────────────────────────────────────
    N_STEPS       = 10000      # total frames to run
    PRINT_EVERY   = 100        # print ASCII grid every N steps
    VERBOSE_PRINT = 500        # print full channel dump every N steps

    # Cycle spawn rooms to test gates, potions, spikes, guard
    SPAWN_ROOMS   = [1, 5, 6, 3]   # rooms to test
    spawn_idx     = 0

    # Level data offsets for start_room / start_pos
    START_ROOM_OFFSET = 2112
    START_POS_OFFSET  = 2113

    # ── Statistics accumulators ────────────────────────────────────────────────
    errors          = []
    ch_min          = np.full(20, np.inf)
    ch_max          = np.full(20, -np.inf)
    ch_unique       = [set() for _ in range(20)]
    player_pos_seen = set()
    enemy_pos_seen  = set()
    rooms_seen      = set()
    deaths          = 0
    episodes        = 1

    # Helper to set spawn room
    raw_level = (ctypes.c_uint8 * 2305).in_dll(env.lib, "level")
    def set_spawn_room(room):
        raw_level[START_ROOM_OFFSET] = room
        raw_level[START_POS_OFFSET]  = 3   # mid-ish tile

    print(f"Running {N_STEPS} steps, cycling spawn rooms {SPAWN_ROOMS}...\n")

    for step_i in range(N_STEPS):
        # Pick a random action
        act = random.randint(0, 17)

        # Inject action and advance frame
        action_g.value = act
        env.lib.play_level_2()

        # Read everything
        env.get_values()
        grid = env.create_grid()
        state = env.get_obs_state()  # Verify it runs without error

        # Validate
        validate_grid(grid, env, step_i, errors)

        # Accumulate stats
        rooms_seen.add(env.kid_room)
        for ch in range(20):
            layer = grid[ch]
            ch_min[ch] = min(ch_min[ch], layer.min())
            ch_max[ch] = max(ch_max[ch], layer.max())
            for v in np.unique(layer):
                ch_unique[ch].add(float(v))

        # Track player and enemy positions
        pp = np.argwhere(grid[8] == 1.0)
        if len(pp) == 1:
            player_pos_seen.add((env.kid_room, int(pp[0][0]), int(pp[0][1])))
        ep = np.argwhere(grid[10] > 0)
        if len(ep) >= 1:
            enemy_pos_seen.add((env.guard_room, int(ep[0][0]), int(ep[0][1])))

        # Print ASCII grid periodically
        if step_i % PRINT_EVERY == 0:
            print_ascii_grid(grid, env, step_i, act)

        # Full channel dump less frequently
        if step_i % VERBOSE_PRINT == 0:
            print(f"\n  ── Full channel dump (step {step_i}) ──")
            for ch in range(20):
                print(f"  CH {ch:2d} {CH_NAMES[ch]:>12s}: {grid[ch].tolist()}")

        # Handle death → reset with next spawn room
        if kid_death.value == 1:
            deaths += 1
            episodes += 1
            # Cycle to next spawn room
            spawn_idx = (spawn_idx + 1) % len(SPAWN_ROOMS)
            spawn_room = SPAWN_ROOMS[spawn_idx]
            print(f"\n  ★ DIED at step {step_i} (death #{deaths}). "
                  f"Resetting → spawn room {spawn_room}...")
            kid_death.value = 0
            env.lib.init_game(1)
            # Patch start_room/start_pos in the already-loaded level data
            raw_level[START_ROOM_OFFSET] = spawn_room
            raw_level[START_POS_OFFSET]  = 3   # mid tile
            # Re-run the engine's own do_startpos to place kid
            env.lib.do_startpos()

            # Verify grid right after reset
            env.get_values()
            grid = env.create_grid()
            validate_grid(grid, env, step_i, errors)
            pp = np.argwhere(grid[8] == 1.0)
            if len(pp) == 1:
                print(f"    After reset: player at row={pp[0][0]} col={pp[0][1]} "
                      f"room={env.kid_room}")
            print(f"    ═══ Episode {episodes} ═══")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Final report
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  FINAL REPORT — {N_STEPS} steps, {episodes} episodes, {deaths} deaths")
    print(f"{'═'*72}")

    print(f"\n  Rooms visited: {sorted(rooms_seen)}")
    print(f"  Unique player positions: {len(player_pos_seen)}")
    print(f"  Unique enemy  positions: {len(enemy_pos_seen)}")

    print(f"\n  Channel statistics:")
    print(f"  {'CH':>3s}  {'Name':>12s}  {'Min':>6s}  {'Max':>6s}  Unique values")
    print("  ---          ----     ---     ---  -------------")
    for ch in range(20):
        if ch_unique[ch]:
            # limit printing values if there are too many
            vals = sorted(list(ch_unique[ch]))
            ustr = str(vals) if len(vals) <= 12 else f"{vals[:6]}...({len(vals)} total)"
        else:
            ustr = "[]" # No unique values if set is empty
        print(f"  {ch:3d}  {CH_NAMES[ch]:>12s}  {ch_min[ch]:6.1f}  {ch_max[ch]:6.1f}  {ustr}")

    if errors:
        print(f"\n  ⚠ VALIDATION ERRORS ({len(errors)} total):")
        # Show first 30 unique errors
        seen = set()
        count = 0
        for e in errors:
            if e not in seen:
                seen.add(e)
                print(f"    • {e}")
                count += 1
                if count >= 30:
                    print(f"    ... and {len(errors) - 30} more")
                    break
    else:
        print(f"\n  ✓ NO VALIDATION ERRORS — all {N_STEPS} frames passed all checks!")

    print(f"\n{'═'*72}")
    print("Done.")


if __name__ == "__main__":
    main()