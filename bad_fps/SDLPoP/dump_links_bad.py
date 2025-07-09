import os
import ctypes

lib = ctypes.CDLL("./src/libSDLPoP.so")
argv = (ctypes.c_char_p * 2)()
argv[0] = b"prince"
argv[1] = b"megahit"
ctypes.c_int.in_dll(lib, "g_argc").value = 2
ctypes.c_void_p.in_dll(lib, "g_argv").value = ctypes.cast(argv, ctypes.c_void_p).value
ctypes.c_int.in_dll(lib, "RL_Mode").value = 1
ctypes.c_short.in_dll(lib, "start_level").value = 1

lib.pop_main()
for _ in range(25):
    lib.play_level_2()

raw_level = (ctypes.c_uint8 * 2305).in_dll(lib, "level")

print("ROOM CONNECTIONS:")
for r in range(24):
    b = 1952 + r * 4
    left = raw_level[b]
    right = raw_level[b+1]
    up = raw_level[b+2]
    down = raw_level[b+3]
    if left or right or up or down:
        print(f"Room {r+1:2d}: Left={left:2d}, Right={right:2d}, Up={up:2d}, Down={down:2d}")
