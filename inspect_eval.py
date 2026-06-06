import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PoP_env"))
import torch
import numpy as np
from envs.PoP_env import PoPEnv
from ppo import Agent, obs_to_torch

device = torch.device("cpu")
agent = Agent().to(device)
ckpt = torch.load("./runs/pop__pop_ppo__1__1785871831/ckpt_2000.pt", map_location=device, weights_only=False)
state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
agent.load_state_dict(state, assign=True)
agent.eval()

env = PoPEnv(headless=True)
raw_obs, _ = env.reset(seed=42)
obs = obs_to_torch(raw_obs, device)
obs = {k: v.unsqueeze(0) if v.dim() == len(v.shape) else v for k, v in obs.items()}

for step in range(10):
    with torch.no_grad():
        feat = agent._encode(obs)
        act_logits = agent.actor(feat)
        rep_logits = agent.repeat_head(feat)
        act = act_logits.argmax(dim=-1)
        rep = rep_logits.argmax(dim=-1)
        action = torch.stack([act, rep], dim=1)
    
    print(f"Step {step}: act={act.item()} (logits max {act_logits.max().item():.2f}, min {act_logits.min().item():.2f}), rep={rep.item()}")
    raw_obs, _, _, _, _ = env.step(action[0].numpy())
    obs = obs_to_torch(raw_obs, device)
    for k, v in obs.items():
        if k == "grid" and v.dim() == 3: obs[k] = v.unsqueeze(0)
        elif k == "state" and v.dim() == 1: obs[k] = v.unsqueeze(0)
        elif k == "room_table" and v.dim() == 2: obs[k] = v.unsqueeze(0)
        elif v.dim() == 1 and k not in ("state",): obs[k] = v.unsqueeze(0)
