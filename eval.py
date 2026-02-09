#!/usr/bin/env python3
"""Eval mode: load a checkpoint, run the agent, record video to mp4."""
import os
import sys
import subprocess
import argparse
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PoP_env"))
from envs.PoP_env import PoPEnv
from ppo import Agent, obs_to_torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to .pt checkpoint")
    p.add_argument("-o", "--output", default="eval.mp4", help="Output video path")
    p.add_argument("-n", "--episodes", type=int, default=1, help="Number of episodes")
    p.add_argument("--max-steps", type=int, default=20000, help="Max steps per episode")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--no-video", action="store_true", help="Stats only, no video")
    p.add_argument("--visual", action="store_true", help="Show SDL window (watch live)")
    args = p.parse_args()

    device = torch.device("cpu")
    agent = Agent().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Handle torch.compile _orig_mod. prefix
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    agent.load_state_dict(state, assign=True)
    agent.eval()
    print(f"Loaded checkpoint: iter={ckpt.get('iteration')}, gs={ckpt.get('global_step')}")

    env = PoPEnv(headless=not args.visual, max_steps=args.max_steps)
    frame_delay = 1.0 / args.fps

    ep = 0
    stem = args.output.rsplit(".", 1)[0]
    while True:

        vpath = f"{stem}_ep{ep:04d}.mp4"
        proc = None
        if not args.no_video:
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                 "-s", "320x200", "-pix_fmt", "rgb24", "-r", str(args.fps),
                 "-i", "pipe:", "-c:v", "libx264", "-pix_fmt", "yuv420p", vpath],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )

        raw_obs, _ = env.reset(seed=42 + ep)
        # Discard stale framebuffer frames from the fade-in sequence after reset.
        # onscreen_surface_ is only blitted during specific game events; without
        # this drain the first N rendered frames are the previous episode's death screen.
        for _ in range(30):
            env.render()
        obs = obs_to_torch(raw_obs, device)
        # Add batch dim for single env
        obs = {k: v.unsqueeze(0) if v.dim() == len(v.shape) else v for k, v in obs.items()}
        # Ensure batch dim
        for k, v in obs.items():
            if k == "grid" and v.dim() == 3:
                obs[k] = v.unsqueeze(0)
            elif k == "state" and v.dim() == 1:
                obs[k] = v.unsqueeze(0)
            elif k == "room_table" and v.dim() == 2:
                obs[k] = v.unsqueeze(0)
            elif v.dim() == 1 and k not in ("state",):
                obs[k] = v.unsqueeze(0)

        total_reward = 0.0
        steps = 0
        rooms_visited = set()

        while True:
            if proc:
                frame = env.render()
                proc.stdin.write(frame.tobytes())
            if frame_delay:
                time.sleep(frame_delay)

            with torch.no_grad():
                feat = agent._encode(obs)
                act_logits = agent.actor(feat)
                rep_logits = agent.repeat_head(feat)
                
                # Temperature scaling: 1.0 = normal sampling, 0.0 = deterministic (argmax)
                temperature = 1.0  # Slightly deterministic to avoid loops
                
                if temperature > 0:
                    act_dist = torch.distributions.Categorical(logits=act_logits / temperature)
                    rep_dist = torch.distributions.Categorical(logits=rep_logits / temperature)
                    if steps == 0:
                        print(f"[debug step0] act_max_prob={act_dist.probs.max().item():.4f}  rep_max_prob={rep_dist.probs.max().item():.4f}  act_argmax={act_logits.argmax().item()}  rep_argmax={rep_logits.argmax().item()}")
                    act = act_dist.sample()
                    rep = rep_dist.sample()
                else:
                    act = act_logits.argmax(dim=-1)
                    rep = rep_logits.argmax(dim=-1)
                    
                action = torch.stack([act, rep], dim=1)

            raw_obs, reward, terminated, truncated, info = env.step(action[0].numpy())
            total_reward += reward
            steps += 1
            if "room" in info:
                rooms_visited.add(int(info["room"]))

            if terminated or truncated:
                break


            obs = obs_to_torch(raw_obs, device)
            for k, v in obs.items():
                if k == "grid" and v.dim() == 3:
                    obs[k] = v.unsqueeze(0)
                elif k == "state" and v.dim() == 1:
                    obs[k] = v.unsqueeze(0)
                elif k == "room_table" and v.dim() == 2:
                    obs[k] = v.unsqueeze(0)
                elif v.dim() == 1 and k not in ("state",):
                    obs[k] = v.unsqueeze(0)

        level_up = info.get("level", 1) > 1
        if level_up:
            env.obs_builder.destroy_window()

        if proc:
            proc.stdin.close()
            proc.wait()

        if level_up:
            status = f"LEVEL_UP(Level {info['level']})"
        elif terminated:
            status = "DEAD"
        else:
            status = "TRUNCATED"

        print(f"  ep={ep}  ret={total_reward:.2f}  steps={steps}  rooms={sorted(rooms_visited)}  {status}")
        if proc:
            print(f"  → {vpath}")

        if level_up:
            print("[LEVEL UP DETECTED] Exiting evaluation immediately as level 1 is completed.")
            break

        ep += 1

    # env.close() only reached on Ctrl+C (KeyboardInterrupt caught by Python)
    env.close()

if __name__ == "__main__":
    main()
