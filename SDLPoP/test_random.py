"""
SDLPoP RL tester - skip intro animation, then truly random in visual mode.
Waits until Kid.frame == 15 (standing idle) before counting steps.
"""

import ctypes
import random
import time

KID_FRAME  = 0
KID_X      = 1
KID_ROOM   = 9
KID_ALIVE  = 13

FRAME_STAND = 15  # frame_15_stand: idle
# Frames that signal the intro/reset animation is done and we can act
READY_FRAMES = {15, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}  # stand or running

def load_lib():
    lib = ctypes.CDLL("./src/libSDLPoP.so")
    argv = (ctypes.c_char_p * 3)()
    argv[0] = b"prince"
    argv[1] = b"-nosound"
    argv[2] = b"-megahit"
    ctypes.c_int.in_dll(lib, "g_argc").value = 3
    ctypes.c_void_p.in_dll(lib, "g_argv").value = ctypes.cast(argv, ctypes.c_void_p).value
    ctypes.c_int.in_dll(lib, "rl_mode").value = 1
    ctypes.c_int.in_dll(lib, "rl_visual_mode").value = 1
    ctypes.c_short.in_dll(lib, "start_level").value = 1
    return lib

def get_kid(lib):
    raw = (ctypes.c_uint8 * 16).in_dll(lib, "Kid")
    alive = raw[KID_ALIVE] if raw[KID_ALIVE] < 128 else raw[KID_ALIVE] - 256
    return raw[KID_FRAME], raw[KID_ROOM], raw[KID_X], alive

def skip_intro(lib, rl_action):
    """Idle-step until Kid reaches the standing frame (animation done)."""
    rl_action.value = 0
    lib.pop_main()
    skipped = 0
    while True:
        rl_action.value = 0
        lib.play_level_2()
        skipped += 1
        frame, room, x, alive = get_kid(lib)
        # x > 100 means Prince has walked away from spawn point (x=77)
        # so the turn animation is done regardless of which frame we're on
        if x > 100:
            print(f"  Intro done after {skipped} frames (x={x}, frame={frame})")
            break
        if skipped > 800:
            print(f"  Gave up waiting after {skipped} frames (x={x}, frame={frame})")
            break

def main():
    lib = load_lib()
    rl_action   = ctypes.c_int.in_dll(lib, "rl_action")
    rl_kid_dead = ctypes.c_int.in_dll(lib, "rl_kid_dead")

    print("Booting and skipping intro animation...")
    skip_intro(lib, rl_action)
    print("Intro skipped! Starting truly random actions (visual).\n")

    total_steps = 0
    episode     = 0
    ep_steps    = 0
    room2_count = 0
    prev_room   = 1
    ep_start    = time.time()

    while room2_count < 10:
        rl_action.value = random.randint(0, 17)
        lib.play_level_2()
        total_steps += 1
        ep_steps    += 1

        frame, room, x, alive = get_kid(lib)

        if rl_kid_dead.value:
            # After death, also wait for standing frame before continuing
            while True:
                rl_action.value = 0
                lib.play_level_2()
                f, r, _, _ = get_kid(lib)
                if f == FRAME_STAND:
                    break
            episode  += 1
            ep_steps  = 0
            prev_room = 1
            ep_start  = time.time()
            print(f"  [ep {episode:>3}] Death -> reset, intro skipped")

        if room != prev_room:
            elapsed = time.time() - ep_start
            print(f"  [ep {episode:>3} step {ep_steps:>4}] {prev_room} -> {room}  x={x}  t={elapsed:.2f}s")
            if room == 2:
                room2_count += 1
                print(f"  *** Room 2! ({room2_count}/10) ***")
            prev_room = room

        if total_steps > 500_000:
            print("500k step cap hit.")
            break

    print(f"\nDone. Room 2: {room2_count}/10 in {total_steps} steps, {episode+1} episodes.")

if __name__ == "__main__":
    main()
