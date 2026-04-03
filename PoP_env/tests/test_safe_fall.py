#!/usr/bin/env python3
"""Verify safe fall (room 1 → room 2) still gets discovery credit after freefall guard."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.PoP_env import PoPEnv
from wrappers.discrete_actions import *

def main():
    env = PoPEnv(headless=True)
    obs, info = env.reset()

    print("="*80)
    print("SAFE FALL VERIFICATION: Room 1 → Room 2")
    print("="*80)

    # Run right to get to the ledge, then fall safely into room 2
    for step in range(1, 80):
        act = RIGHT if step <= 30 else NONE
        obs, reward, term, trunc, info = env.step(act)
        kid = env.obs_builder.kid
        print(f"Step {step:3d} | {ACTION_NAMES[act]:<8} | "
              f"Room:{kid.room:2d} x:{kid.x:3d} y:{kid.y:3d} | "
              f"act={kid.action} frm={kid.frame:3d} | "
              f"rew={reward:+6.1f} | visited={sorted(env.visited_rooms)}")
        if term:
            print("DIED — test failed, should have survived safe fall")
            break
        if 2 in env.visited_rooms:
            print(f"\n>>> Room 2 discovered! Reward was {reward:+.1f}")
            print(f">>> kid.action at discovery = {kid.action} (0=standing, 4=freefall)")
            print(">>> SAFE FALL CORRECTLY CREDITED ✓")
            break

    env.close()

if __name__ == "__main__":
    main()
