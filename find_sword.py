import ctypes
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_env import PoPEnv

def main():
    env = PoPEnv()
    env.reset()
    for room in range(1, 25):
        offset = (room - 1) * 30
        fg = env.fg[offset:offset+30]
        for i in range(30):
            t = int(fg[i]) & 0x1F
            if t == 22: # T_SWORD
                print(f"Sword tile found in room {room}")

if __name__ == "__main__":
    main()
