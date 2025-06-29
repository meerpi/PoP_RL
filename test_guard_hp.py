"""Test: watch loose tile at Room 1 (2,6) + guardhp_curr when reaching guard"""
import ctypes
from ctypes import CDLL, c_int, c_short, c_char_p, c_void_p, cast, create_string_buffer, c_ushort
import os, random, time
import numpy as np

SDLPoP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP")
LIB_PATH   = os.path.join(SDLPoP_DIR, "src", "libSDLPoP.so")
os.chdir(SDLPoP_DIR)
os.environ["SDL_AUDIODRIVER"] = "dummy"
lib = CDLL(LIB_PATH)

def _set_g_argv(lib, argv_list):
    argv_buffers = []
    argv = (c_char_p * len(argv_list))()
    for i, s in enumerate(argv_list):
        buf = create_string_buffer(s.encode("utf-8"))
        argv_buffers.append(buf)
        argv[i] = cast(buf, c_char_p)
    c_int.in_dll(lib, "g_argc").value = len(argv_list)
    c_void_p.in_dll(lib, "g_argv").value = cast(argv, c_void_p).value
    return argv_buffers, argv

_refs = _set_g_argv(lib, ["prince", "megahit"])
lib.pop_main.argtypes, lib.pop_main.restype = [], None
lib.init_game.argtypes, lib.init_game.restype = [c_int], None
lib.play_level_2.argtypes, lib.play_level_2.restype = [], c_int

c_int.in_dll(lib, "RL_Mode").value = 1
c_int.in_dll(lib, "RL_Visual").value = 1
c_short.in_dll(lib, "start_level").value = 1
lib.pop_main()

raw_level = (ctypes.c_uint8 * 2305).in_dll(lib, "level")
kid_raw   = (ctypes.c_uint8 * 16).in_dll(lib, "Kid")
ghp       = c_ushort.in_dll(lib, "guardhp_curr")
prince_death = c_int.in_dll(lib, "prince_death")

# Track the loose tile in Room 1 at (row=2, col=6) = index 26
LOOSE_IDX = (1-1)*30 + 2*10 + 6  # room 1, row 2, col 6

print("Watching loose tile Room1(2,6) and guardhp_curr...")
print(f"Initial: fg[{LOOSE_IDX}]={raw_level[LOOSE_IDX]}  bg[{LOOSE_IDX+720}]={raw_level[LOOSE_IDX+720]}")

prev_fg = int(raw_level[LOOSE_IDX])
prev_bg = int(raw_level[LOOSE_IDX + 720])

for step in range(5000):
    act = random.randint(0, 17)
    c_int.in_dll(lib, "action").value = act
    time.sleep(0.005)
    lib.play_level_2()

    level_np = np.frombuffer(raw_level, dtype=np.uint8)
    fg_val = int(level_np[LOOSE_IDX])
    bg_val = int(level_np[LOOSE_IDX + 720])
    kid_np = np.frombuffer(kid_raw, dtype=np.uint8)
    kid_room = int(kid_np[9])

    # Print when loose tile changes
    if fg_val != prev_fg or bg_val != prev_bg:
        print(f"  step={step:4d} LOOSE CHANGED: fg={prev_fg}->{fg_val}  bg={prev_bg}->{bg_val}  kid_room={kid_room}")
        prev_fg, prev_bg = fg_val, bg_val

    # Print when guard HP changes or guard is in same room
    guard_np = np.frombuffer((ctypes.c_uint8 * 16).in_dll(lib, "Guard"), dtype=np.uint8)
    guard_room = int(guard_np[9])
    if guard_room == kid_room and guard_room != 0:
        print(f"  step={step:4d} GUARD IN ROOM! kid_room={kid_room} guardhp_curr={ghp.value}")

    if prince_death.value == 1:
        print(f"  step={step:4d} DIED. Resetting.")
        prince_death.value = 0
        lib.init_game(1)
        prev_fg = int(raw_level[LOOSE_IDX])
        prev_bg = int(raw_level[LOOSE_IDX + 720])

print(f"\nFinal: fg[{LOOSE_IDX}]={raw_level[LOOSE_IDX]}  bg[{LOOSE_IDX+720}]={raw_level[LOOSE_IDX+720]}")
