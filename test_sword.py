"""
Sword detection test 2 — spawns kid in room 12, explicitly moves kid to the sword tile,
and checks if the item is picked up on the next tick.
"""
import ctypes
from ctypes import c_uint8
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_env import PoPEnv


def main():
    import time
    print("Creating PoPEnv in VISUAL mode...")
    env = PoPEnv(visual=True)

    # We need to manually spawn in room 15 like the wrapper does
    env.reset()
    raw_level = (c_uint8 * 2305).in_dll(env.lib, "level")
    raw_level[2112] = 15
    raw_level[2113] = 3
    env.lib.do_startpos()
    env.get_values()

    print(f"Spawned in room {env.kid_room}. kid_sword={env.kid_sword}")

    # Find the sword tile
    offset = (15 - 1) * 30
    fg_room15 = env.fg[offset:offset+30]
    sword_col, sword_row = -1, -1
    for i in range(30):
        t = int(fg_room15[i]) & 0x1F
        if t == 22:  # T_SWORD
            sword_row, sword_col = divmod(i, 10)
            break
            
    if sword_col == -1:
        print("Could not find sword tile in room 15!")
        return

    print(f"Sword tile found at col={sword_col}, row={sword_row}")
    
    # Don't teleport, let's see where the sword is
    env.get_values()
    print(f"Kid starts at col={env.kid_curr_col}, row={env.kid_curr_row}, action={env.kid_action}")
    
    # Let's perform a sequence of actions to naturally walk to the sword
    # Kid starts at col=3, row=0. Sword is at col=2, row=2.
    # 2 is backwards (left if facing right). 1 is forwards (right).
    # We will just print the grid and let's try pushing him left then dropping down.
    
    print("Turning and moving left...")
    # Give the user a moment to see the start
    time.sleep(1.0)
    
    # 2 is backward (which turns him left)
    print("Moving until col=2 and row=2...")
    max_steps = 100
    for i in range(max_steps):
        env.get_values()
        
        # Are we at the sword tile?
        if env.kid_curr_col == 2 and env.kid_curr_row == 2:
            print("Reached sword tile!")
            break
            
        # If we are row=0, we need to drop down or walk left then drop
        # Actually in room 15, kid spawns col 3, row 0. He needs to drop to row 2.
        # If we just keep pressing 2 (move backward), he will turn left, step off the ledge, 
        # drop to row 1, drop to row 2, and eventually land on col 2, row 2.
        env.step(2)
        time.sleep(0.1)
    else:
        print("Failed to reach sword tile in 100 steps.")
        
    env.get_values()
    print(f"Col={env.kid_curr_col}, Row={env.kid_curr_row}, Dir={env.kid_direction}")
    
    # Give him a moment to settle his animation
    for _ in range(10):
        env.step(0) # stand still
        time.sleep(0.05)
    
    print("Pressing shift to grab...")
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(5)
        if env.sword_found:
            print("  ✓ Sword Picked Up!")
            break
        time.sleep(0.5)
            
    # Give user time to see final state before closing
    time.sleep(3.0)
    print(f"\nFinal State:")
    print(f"  kid_sword={env.kid_sword}")
    print(f"  sword_found={env.sword_found}")


if __name__ == "__main__":
    main()
