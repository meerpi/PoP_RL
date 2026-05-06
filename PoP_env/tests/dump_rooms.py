#!/usr/bin/env python3
"""Dump level 1 room data using PoPEnv (handles threading correctly)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from envs.PoP_env import PoPEnv

TILE_NAMES = {0:'empty',1:'floor',2:'spike',3:'pillar',4:'gate',5:'stuck_btn',
              6:'drop_btn',7:'tapestry',8:'bot_pillar',9:'top_pillar',
              10:'potion',11:'loose',12:'tapestry_top',13:'mirror',14:'rubble',
              15:'raise_btn',16:'exit_L',17:'exit_R',18:'chopper',19:'torch',
              20:'wall',21:'skeleton',22:'sword',23:'balc_L',24:'balc_R',
              25:'latt_pillar',26:'latt_down',27:'latt_small',28:'latt_L',29:'latt_R',30:'torch_rubble'}

env = PoPEnv(headless=True)
env.reset()
d = env.obs_builder.data

print("LEVEL 1 ROOM MAP")
print(f"{'Room':>4} | {'L':>2} {'R':>2} {'U':>2} {'D':>2} | {'Flags':<30} | Key Tiles")
print("-"*90)
for room in range(24):
    start = room * 30
    tiles = [int(d.level.fg[start + i]) & 0x1f for i in range(30)]
    unique = set(tiles)
    if not any(t != 0 for t in unique):
        continue
    flags = []
    if 4 in unique: flags.append('GATE')
    if any(t in (5,6,15) for t in unique): flags.append('BUTTON')
    if 22 in unique: flags.append('SWORD')
    if 10 in unique: flags.append('POTION')
    if 16 in unique or 17 in unique: flags.append('EXIT')
    if 18 in unique: flags.append('CHOPPER')
    if 2 in unique: flags.append('SPIKE')
    if 11 in unique: flags.append('LOOSE')
    if int(d.level.guards_tile[room]) > 0: flags.append('GUARD')
    link = d.level.roomlinks[room]
    tile_list = sorted(set(TILE_NAMES.get(t, f'?{t}') for t in unique if t not in (0,1,20,19)))
    print(f"  {room+1:2d}  | {link.left:2d} {link.right:2d} {link.up:2d} {link.down:2d} | "
          f"{' '.join(flags):<30} | {', '.join(tile_list)}")

env.close()
