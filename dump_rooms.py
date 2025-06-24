#!/usr/bin/env python3
"""Dump tile data for rooms directly from res2000.bin (Level 1, no SDL)."""
import os

TILE_NAMES = {0:'EMPTY',1:'FLOOR',2:'SPIKE',3:'PILLAR',4:'GATE',5:'STUCK',
              6:'DROP_BTN',7:'TAPST',8:'BIGP_B',9:'BIGP_T',10:'POTION',
              11:'LOOSE',12:'DOORTOP',13:'MIRROR',14:'DEBRIS',15:'RAISE',
              16:'EXIT_L',17:'EXIT_R',18:'CHOMPER',19:'TORCH',20:'WALL',
              21:'SKEL',22:'SWORD'}

dat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "SDLPoP", "data", "LEVELS", "res2000.bin")
with open(dat_path, "rb") as f:
    data = f.read()

print(f"File size: {len(data)} bytes")
fg = data[0:720]
bg = data[720:1440]
rl_off = 1440 + 256 + 256  # roomlinks at offset 1952

print("\n=== ROOM LINKS (Level 1) ===")
for r in range(1, 25):
    b = rl_off + (r-1)*4
    L,R,U,D = data[b], data[b+1], data[b+2], data[b+3]
    if any([L,R,U,D]):
        print(f"  Room {r:2d}: L={L:2d} R={R:2d} U={U:2d} D={D:2d}")

for room in [5, 6, 7, 8]:
    off = (room - 1) * 30
    print(f"\n=== ROOM {room} ===")
    for row in range(3):
        line = "  "
        for col in range(10):
            idx = off + row * 10 + col
            t = fg[idx] & 0x1F
            m = bg[idx]
            nm = TILE_NAMES.get(t, f"T{t}")
            line += f"{nm:>8s}({m:3d})"
        print(line)
    for i in range(30):
        t = fg[off+i] & 0x1F
        m = bg[off+i]
        if t in (4,5,6,10,15,18,22):
            print(f"  SPECIAL: pos={i:2d} r={i//10} c={i%10}: {TILE_NAMES[t]} mod={m}")
