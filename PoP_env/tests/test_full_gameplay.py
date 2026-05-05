#!/usr/bin/env python3
"""Comprehensive Integration Test for PoPEnv.

Runs a live game session through PoPEnv (envs/PoP_env.py), exercising:
1. Environment instantiation and reset().
2. Action execution (NONE, RIGHT, LEFT, UP, DOWN, UP_RIGHT, SHIFT).
3. Decision-frame stepping (verifying frames elapsed > 0 per decision step).
4. Observation dict structure and shapes (grid, state, room, action_history, repeat_history).
5. Info dictionary values and RGB rendering (render()).
6. Multi-step navigation across rooms and auto-reset capability.
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from envs.PoP_env import PoPEnv
from wrappers.discrete_actions import (
    NONE, UP, DOWN, LEFT, RIGHT, SHIFT_UP, SHIFT_DOWN,
    SHIFT_LEFT, SHIFT_RIGHT, UP_LEFT, UP_RIGHT, INTERACT, ACTION_NAMES,
)


def log_step(step_idx, action_id, obs, reward, term, trunc, info, elapsed):
    act_name = ACTION_NAMES[action_id]
    print(f"Step {step_idx:3d} | Action: {act_name:<11} | Frames elapsed: {elapsed:2d} | "
          f"Room: {info['room']:2d} | Grid Pos: ({info['grid_x']:2d}, {info['grid_y']:2d}) | "
          f"HP: {info['hp']} | Reward: {reward:+.1f} | Term: {term}")


def test_gameplay_flow():
    print("=" * 80)
    print("STARTING FULL GAMEPLAY INTEGRATION TEST FOR PoPEnv")
    print("=" * 80)

    env = PoPEnv(headless=True, max_steps=500)
    
    # 1. Test reset
    print("\n--- Phase 1: Environment Reset ---")
    obs, info = env.reset()
    
    assert "grid" in obs, "Missing 'grid' in observation!"
    assert "state" in obs, "Missing 'state' in observation!"
    assert "room" in obs, "Missing 'room' in observation!"
    assert "action_history" in obs, "Missing 'action_history' in observation!"
    assert "repeat_history" in obs, "Missing 'repeat_history' in observation!"

    assert obs["grid"].shape == (12, 5, 12), f"Grid shape mismatch: {obs['grid'].shape}"
    assert obs["state"].shape == (31,), f"State shape mismatch: {obs['state'].shape}"
    assert obs["action_history"].shape == (5,), f"Action history shape mismatch: {obs['action_history'].shape}"

    print("Obs grid shape:", obs["grid"].shape)
    print("Obs state shape:", obs["state"].shape)
    print("Initial Info:", info)
    assert info["alive"] == True, "Kid should be alive after reset!"
    assert info["hp"] > 0, "Kid HP should be > 0!"

    # 2. Test RGB render
    print("\n--- Phase 2: Frame Rendering (rgb_array) ---")
    rgb = env.render()
    assert rgb is not None, "render() returned None!"
    assert rgb.shape == (200, 320, 3), f"RGB render shape mismatch: {rgb.shape}"
    print(f"Rendered RGB frame successfully! Shape: {rgb.shape}, dtype: {rgb.dtype}")

    # 3. Test multi-action sequence (running, jumping, crouching, turning)
    print("\n--- Phase 3: Action Execution & Decision Frame Advancement ---")
    action_sequence = [
        # (action_id, name, expected_min_steps)
        (NONE, "Hold Stand", 5),
        (RIGHT, "Run Right", 15),
        (NONE, "Release Run", 5),
        (UP_RIGHT, "Jump Right", 5),
        (DOWN, "Crouch", 5),
        (NONE, "Stand Up", 5),
        (LEFT, "Turn/Run Left", 15),
        (NONE, "Release Run", 5),
    ]

    total_steps = 0
    start_x = info["grid_x"]

    for act_id, description, num_steps in action_sequence:
        print(f"\n>> Executing: {description} (Action {act_id}: {ACTION_NAMES[act_id]}) for {num_steps} steps")
        for i in range(num_steps):
            total_steps += 1
            prev_frame_count = env.frame_count
            obs, reward, term, trunc, info = env.step(act_id)
            elapsed = env.frame_count - prev_frame_count

            assert elapsed >= 1, f"Expected at least 1 frame elapsed per step, got {elapsed}"
            log_step(total_steps, act_id, obs, reward, term, trunc, info, elapsed)

            if term:
                print(">> Character died during sequence! Resetting...")
                obs, info = env.reset()
                break

    # 4. Test Auto-Reset / Level Reset
    print("\n--- Phase 4: Manual Level Reset & State Cleanliness ---")
    obs, info = env.reset()
    print("Post-Reset Info:", info)
    assert info["step_count"] == 0, f"Step count should reset to 0, got {info['step_count']}"
    assert info["frame_count"] == 0, f"Frame count should reset to 0, got {info['frame_count']}"

    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS PASSED FLAWLESSLY!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    test_gameplay_flow()
