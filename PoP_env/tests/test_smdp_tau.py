#!/usr/bin/env python3
"""Verify SMDP τ (frames_elapsed) for all key actions via live engine."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.PoP_env import PoPEnv
from wrappers.discrete_actions import *

def run_sequence(env, label, actions):
    """Run a list of (action_id, count) pairs, print τ for each step."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    for act_id, count in actions:
        for i in range(count):
            obs, reward, term, trunc, info = env.step(act_id)
            kid = env.obs_builder.kid
            tau = info["frames_elapsed"]
            print(f"  {ACTION_NAMES[act_id]:<12} step {i+1}/{count} │ "
                  f"τ={tau:2d} │ action={kid.action} frame={kid.frame:3d} │ "
                  f"room={info['room']} pos=({info['grid_x']},{info['grid_y']})")
            if term:
                print("  !! DIED — resetting")
                env.reset()
                return

def main():
    env = PoPEnv(headless=True)
    env.reset()

    print("="*60)
    print("SMDP τ VERIFICATION — ALL KEY ACTIONS")
    print("="*60)

    # 1. Standing NONE (should be τ=1)
    run_sequence(env, "NONE while standing (expect τ=1)", [(NONE, 3)])

    # 2. RIGHT — start-run then running strides (τ=1 each for frames 0-14)
    run_sequence(env, "RIGHT — start-run + running (expect τ=1)", [(RIGHT, 12)])

    # 3. LEFT while facing RIGHT — should turn (τ≈4, frames skip to 48)
    run_sequence(env, "LEFT while facing right — TURN (expect τ>1)", [(LEFT, 5)])

    # 4. LEFT continued — now running left (expect τ=1)
    run_sequence(env, "LEFT running left (expect τ=1)", [(LEFT, 8)])

    # 5. NONE to stop
    run_sequence(env, "NONE — stop running (expect τ varies)", [(NONE, 5)])

    # 6. DOWN — crouch (expect large τ)
    run_sequence(env, "DOWN — crouch (expect τ≈18)", [(DOWN, 3)])

    # 7. NONE — stand up from crouch (expect τ≈6)
    run_sequence(env, "NONE — stand up from crouch (expect τ≈6)", [(NONE, 5)])

    # 8. UP+RIGHT — running jump
    env.reset()
    run_sequence(env, "RIGHT then UP+RIGHT — jump (expect τ=11-19)",
                 [(RIGHT, 6), (UP_RIGHT, 5), (NONE, 3)])

    # 9. UP — jump up
    env.reset()
    run_sequence(env, "UP — jump up (expect τ varies)", [(UP, 5), (NONE, 5)])

    print("\n" + "="*60)
    print("SMDP τ VERIFICATION COMPLETE")
    print("="*60)
    env.close()

if __name__ == "__main__":
    main()
