#!/usr/bin/env python3
"""Test script verifying exact (x, y) pixel coordinates and tile grid changes of Kid in PoPEnv."""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from envs.PoP_env import PoPEnv
from wrappers.discrete_actions import RIGHT, NONE, LEFT

def main():
    env = PoPEnv(headless=True)
    obs, info = env.reset()
    
    print("=" * 70)
    print("VERIFYING EXACT (x, y) PIXEL COORDINATES AND TILE COL/ROW MOVEMENT")
    print("=" * 70)
    print(f"Initial State | Room: {info['room']} | col: {info['grid_x']}, row: {info['grid_y']} | "
          f"x: {env.obs_builder.kid.x}, y: {env.obs_builder.kid.y}")

    print("\nExecuting RIGHT (running right) for 15 steps:")
    for step in range(1, 16):
        obs, reward, term, trunc, info = env.step(RIGHT)
        kid = env.obs_builder.kid
        print(f"Step {step:2d} | Room: {kid.room:2d} | col: {kid.curr_col:2d}, row: {kid.curr_row:2d} | "
              f"x: {kid.x:3d}, y: {kid.y:3d} | frame: {kid.frame:3d} | action_state: {kid.action}")

    print("\nExecuting NONE (stopping) for 5 steps:")
    for step in range(1, 6):
        obs, reward, term, trunc, info = env.step(NONE)
        kid = env.obs_builder.kid
        print(f"Step {step:2d} | Room: {kid.room:2d} | col: {kid.curr_col:2d}, row: {kid.curr_row:2d} | "
              f"x: {kid.x:3d}, y: {kid.y:3d} | frame: {kid.frame:3d} | action_state: {kid.action}")

    print("=" * 70)
    env.close()

if __name__ == "__main__":
    main()
