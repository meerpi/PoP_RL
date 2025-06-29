"""
Interactive Play Test
Spawns the Kid in Room 15 (Sword Room) and disables RL_Mode
so you can manually play using the keyboard to test sword pickup.

Controls:
  Arrow keys to move/crouch/jump.
  Shift to grab/pickup.
"""
import ctypes
from ctypes import c_uint8
import os, sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_env import PoPEnv

def monitor_state(env):
    print("\n" + "="*50)
    print("READY! The game window should be open.")
    print("Use ARROW KEYS to move and SHIFT to grab the sword.")
    print("="*50 + "\n")
    import collections
    history = collections.deque(maxlen=60)
    sword_detected = False
    while True:
        env.get_values()
        state_str = f"Room={env.kid_room} | Col={env.kid_curr_col}, Row={env.kid_curr_row} | X={env.kid_x} | Dir={env.kid_direction} | Action={env.kid_action} | Frame={env.kid_frame}"
        history.append(state_str)
        # print(state_str)
        
        if (env.kid_sword > 0 or getattr(env, "have_sword", 0) != 0) and not sword_detected:
            sword_detected = True
            print("\n  ★ SWORD PICKUP DETECTED IN PYTHON STATE! (have_sword != 0)")
            print("  The environment logic works!")
            print("\n--- LAST 60 FRAMES OF STATE ---")
            for i, s in enumerate(history):
                print(f"Frame -{len(history)-i}: {s}")
            print("-------------------------------")
            
        time.sleep(0.05)

def main():
    print("Creating PoPEnv in VISUAL mode...")
    env = PoPEnv(visual=True)

    # Reset and patch level to spawn in room 15
    env.reset()
    raw_level = (c_uint8 * 2305).in_dll(env.lib, "level")
    raw_level[2112] = 15  # Spawn Room
    raw_level[2113] = 3   # Spawn Pos
    env.lib.do_startpos()
    env.get_values()

    # CRITICAL: Disable RL_Mode so SDL ignores our injected actions 
    # and listens to your real keyboard inputs!
    ctypes.c_int.in_dll(env.lib, "RL_Mode").value = 0
    
    # Start monitor thread
    t = threading.Thread(target=monitor_state, args=(env,), daemon=True)
    t.start()
    
    # Now call play_level_2. With RL_Mode=0, this will block and run the game normally.
    try:
        env.lib.play_level_2()
    except KeyboardInterrupt:
        print("\nExiting interactive test.")

if __name__ == "__main__":
    main()

