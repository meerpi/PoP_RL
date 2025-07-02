import ctypes
from ctypes import c_uint8, c_uint16
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_env import PoPEnv

def main():
    env = PoPEnv(visual=True)
    env.reset()
    
    # Enable RL mode so script works
    ctypes.c_int.in_dll(env.lib, "RL_Mode").value = 1
    
    raw_level = (c_uint8 * 2305).in_dll(env.lib, "level")
    raw_level[2112] = 15  # Spawn Room
    raw_level[2113] = 3   # Spawn Pos
    env.lib.do_startpos()
    
    print("Moving Kid left to sword...")
    
    # 1. Kid starts at row 0, col 3. Turn left (backward = 2)
    for _ in range(30): env.step(2)
    
    # 2. He'll drop off ledge to row 2, col 2.
    # We poll until he's on row 2, col 2 and is standing (action 0).
    max_steps = 150
    for _ in range(max_steps):
        env.step(0)
        env.get_values()
        have_sword = ctypes.c_uint16.in_dll(env.lib, "have_sword").value
        if env.kid_curr_col <= 1 and env.kid_curr_row == 2 and env.kid_action == 0:
            print(f"Reached Col=1! X={env.kid_x}, Dir={env.kid_direction}, have_sword={have_sword}, Kid.sword={env.kid_sword}")
            
            if env.kid_direction == -1:
                print("Turning right to face sword...")
                for _ in range(10): env.step(0)
                env.step(4)  # 4 is turn right
                for _ in range(15): env.step(0)
                
            print("Taking a small step forward if needed, and pressing shift...")
            for _ in range(10): env.step(0)
            break
            
    # 3. Press shift (5) to grab it!
    for _ in range(30):
        env.step(5)
        env.get_values()
        have_sword = ctypes.c_uint16.in_dll(env.lib, "have_sword").value
        print(f"Action 5 (Grab) - Kid.frame={env.kid_frame}, Kid.action={env.kid_action}, Kid.x={env.kid_x}, have_sword={have_sword}")

    print("Checking final state...")
    time.sleep(2)
    env.get_values()
    have_sword = ctypes.c_uint16.in_dll(env.lib, "have_sword").value
    print(f"Final have_sword = {have_sword}, Kid.sword = {env.kid_sword}")

if __name__ == "__main__":
    main()
