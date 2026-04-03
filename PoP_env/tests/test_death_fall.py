#!/usr/bin/env python3
"""Test death fall by teleporting kid to rooms with tall shafts and stepping off.

PoP Level 1 room layout (from roomlinks):
We'll read the actual roomlinks to find rooms stacked vertically,
then teleport to the top and walk off the edge to trigger a death fall.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.PoP_env import PoPEnv
from wrappers.discrete_actions import *

def log(step, act_id, env, reward, term, info):
    kid = env.obs_builder.kid
    dm = " *** DEAD ***" if term else ""
    print(f"Step {step:3d} | {ACTION_NAMES[act_id]:<12} | "
          f"Room:{kid.room:2d} x:{kid.x:3d} y:{kid.y:3d} | "
          f"act={kid.action} frm={kid.frame:3d} alive={kid.alive:2d} | "
          f"τ={info['frames_elapsed']:2d} | "
          f"rew={reward:+6.1f} | visited={sorted(env.visited_rooms)}{dm}")
    return term

def try_death_fall(env, start_room, start_col, label):
    """Teleport to a room and try to fall to death."""
    print(f"\n{'='*90}")
    print(f"  ATTEMPT: {label} — Start room {start_room}, col {start_col}")
    print(f"{'='*90}")

    obs, info = env.reset()
    # Teleport
    env.obs_builder.set_start_room(start_room, start_col, 0)
    env.obs_builder.sync_wait(3)
    env.obs_builder.refresh()

    kid = env.obs_builder.kid
    print(f"Teleported to Room {kid.room}, x={kid.x}, y={kid.y}")
    env.visited_rooms = {int(kid.room)}

    step = 0
    # Try walking right off ledge, then just wait
    actions = [(RIGHT, 15), (NONE, 10), (RIGHT, 10), (NONE, 30)]
    for act_id, count in actions:
        for _ in range(count):
            step += 1
            obs, reward, term, trunc, info = env.step(act_id)
            died = log(step, act_id, env, reward, term, info)
            if died:
                kid = env.obs_builder.kid
                print(f"\n  >>> DIED in room {kid.room}")
                print(f"  >>> Room {kid.room} in visited_rooms? "
                      f"{'YES — BUG!' if kid.room in env.visited_rooms else 'NO — correct'}")
                print(f"  >>> visited_rooms: {sorted(env.visited_rooms)}")
                return True
    return False

def main():
    env = PoPEnv(headless=True, max_steps=5000)

    # First, let's read the roomlinks to understand level 1 layout
    obs, info = env.reset()
    print("LEVEL 1 ROOM CONNECTIVITY (roomlinks):")
    print(f"{'Room':>4} | {'Left':>4} | {'Right':>5} | {'Up':>4} | {'Down':>4}")
    print("-"*35)
    for room in range(1, 25):
        link = env.obs_builder.data.level.roomlinks[room - 1]
        left  = link.left
        right = link.right
        up    = link.up
        down  = link.down
        if left > 0 or right > 0 or up > 0 or down > 0:
            print(f"{room:4d} | {left:4d} | {right:5d} | {up:4d} | {down:4d}")

    # Try different rooms to find death falls
    # In PoP level 1, rooms with "down" links that chain multiple levels create death falls
    attempts = [
        (3, 5, "Room 3, walk right off ledge"),
        (4, 2, "Room 4, walk right off ledge"),
        (5, 2, "Room 5, walk right off ledge"),
        (7, 5, "Room 7, walk right off ledge"),
        (8, 5, "Room 8, walk right off ledge"),
        (12, 5, "Room 12, walk right off ledge"),
    ]

    for start_room, start_col, label in attempts:
        died = try_death_fall(env, start_room, start_col, label)
        if died:
            break

    env.close()

if __name__ == "__main__":
    main()
