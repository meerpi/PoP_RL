#!/usr/bin/env python3
"""Profile PPO training to find SPS bottlenecks."""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PoP_env"))
from envs.PoP_env import PoPEnv
from ppo import Agent, obs_to_torch, stack_obs
from wrappers.discrete_actions import NUM_ACTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

env = PoPEnv(headless=True)
agent = Agent().to(device)

raw_obs, _ = env.reset(seed=1)

# ── Benchmark individual components ──────────────────────────────────────────

N = 200

# 1. Env step timing
t0 = time.perf_counter()
for _ in range(N):
    raw_obs, rew, term, trunc, info = env.step(np.random.randint(NUM_ACTIONS))
    if term:
        raw_obs, _ = env.reset()
env_time = (time.perf_counter() - t0) / N
print(f"\n1. env.step():         {env_time*1000:.2f} ms/step  ({1/env_time:.0f} steps/s)")

# 2. obs_to_torch timing
t0 = time.perf_counter()
for _ in range(N):
    t_obs = obs_to_torch(raw_obs, device)
obs_torch_time = (time.perf_counter() - t0) / N
print(f"2. obs_to_torch():     {obs_torch_time*1000:.2f} ms/step  ({1/obs_torch_time:.0f} steps/s)")

# 3. Agent forward pass timing (inference, no grad)
t_obs = obs_to_torch(raw_obs, device)
# warmup
for _ in range(10):
    with torch.no_grad():
        agent.get_action_and_value(t_obs)
if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        action, logprob, entropy, value = agent.get_action_and_value(t_obs)
if device.type == "cuda":
    torch.cuda.synchronize()
fwd_time = (time.perf_counter() - t0) / N
print(f"3. agent forward:      {fwd_time*1000:.2f} ms/step  ({1/fwd_time:.0f} steps/s)")

# 4. Full rollout step (env + obs_to_torch + forward)
t0 = time.perf_counter()
for _ in range(N):
    t_obs = obs_to_torch(raw_obs, device)
    with torch.no_grad():
        action, logprob, entropy, value = agent.get_action_and_value(t_obs)
    raw_obs, rew, term, trunc, info = env.step(action.item())
    if term:
        raw_obs, _ = env.reset()
if device.type == "cuda":
    torch.cuda.synchronize()
rollout_time = (time.perf_counter() - t0) / N
print(f"4. full rollout step:  {rollout_time*1000:.2f} ms/step  ({1/rollout_time:.0f} steps/s)")

# 5. stack_obs timing (512 steps)
obs_list = []
for _ in range(512):
    t_obs = obs_to_torch(raw_obs, device)
    obs_list.append(t_obs)
    raw_obs, rew, term, trunc, info = env.step(np.random.randint(NUM_ACTIONS))
    if term:
        raw_obs, _ = env.reset()

t0 = time.perf_counter()
for _ in range(5):
    b_obs = stack_obs(obs_list, device)
stack_time = (time.perf_counter() - t0) / 5
print(f"5. stack_obs(512):     {stack_time*1000:.2f} ms")

# 6. PPO update timing (1 epoch, all minibatches)
b_obs = stack_obs(obs_list, device)
b_actions = torch.randint(0, NUM_ACTIONS, (512,), device=device)
b_logprobs = torch.randn(512, device=device)
b_returns = torch.randn(512, device=device)
b_advs = torch.randn(512, device=device)
optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

# warmup
for start in range(0, 512, 128):
    mb = slice(start, start + 128)
    mb_obs = {k: v[mb] for k, v in b_obs.items()}
    _, nl, ent, nv = agent.get_action_and_value(mb_obs, b_actions[mb])
    loss = (-nl).mean() + 0.5 * ((nv.view(-1) - b_returns[mb])**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t0 = time.perf_counter()
for _ in range(4):  # 4 epochs
    for start in range(0, 512, 128):
        mb = slice(start, start + 128)
        mb_obs = {k: v[mb] for k, v in b_obs.items()}
        _, nl, ent, nv = agent.get_action_and_value(mb_obs, b_actions[mb])
        loss = (-nl).mean() + 0.5 * ((nv.view(-1) - b_returns[mb])**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()
update_time = time.perf_counter() - t0
print(f"6. PPO update (4ep):   {update_time*1000:.1f} ms  ({update_time/512*1000:.2f} ms/step amortized)")

# 7. Break down env.step internals
raw_obs, _ = env.reset()
# Time the engine loop vs reward/obs building
import time as _time

engine_times = []
obs_build_times = []
room_table_times = []
for _ in range(N):
    action = np.random.randint(NUM_ACTIONS)
    
    # Engine loop
    t_eng = _time.perf_counter()
    env.obs_builder.press_action(action)
    frames_elapsed = 0
    while True:
        env.obs_builder.sync_wait(1)
        frames_elapsed += 1
        env.frame_count += 1
        env.obs_builder.refresh()
        kid = env.obs_builder.kid
        is_dead = env.obs_builder.is_dead
        from wrappers.discrete_actions import is_decision_frame
        is_dec = is_decision_frame(kid.action, kid.frame, kid.sword, kid.alive)
        if is_dead or is_dec:
            break
    engine_times.append(_time.perf_counter() - t_eng)
    
    # Reward computation + room table update
    t_rew = _time.perf_counter()
    env.step_count += 1
    env.last_tau = frames_elapsed
    # simulate reward logic
    room = int(kid.room)
    env._update_room_table_dynamic(room)
    room_table_times.append(_time.perf_counter() - t_rew)
    
    # Obs building
    t_obs = _time.perf_counter()
    obs = env._build_obs()
    obs_build_times.append(_time.perf_counter() - t_obs)
    
    if is_dead:
        env.obs_builder.release_held_action()
        raw_obs, _ = env.reset()

print(f"\n── env.step() breakdown ──")
print(f"  Engine loop:         {np.mean(engine_times)*1000:.2f} ms  ({np.mean(engine_times)/env_time*100:.0f}%)")
print(f"  Room table update:   {np.mean(room_table_times)*1000:.2f} ms  ({np.mean(room_table_times)/env_time*100:.0f}%)")
print(f"  Obs building:        {np.mean(obs_build_times)*1000:.2f} ms  ({np.mean(obs_build_times)/env_time*100:.0f}%)")

# Summary
print(f"\n── Summary ──")
total = env_time + obs_torch_time + fwd_time
print(f"  env.step:     {env_time/total*100:.0f}%")
print(f"  obs_to_torch: {obs_torch_time/total*100:.0f}%")
print(f"  agent fwd:    {fwd_time/total*100:.0f}%")
print(f"  Theoretical max SPS (rollout only): {1/total:.0f}")
print(f"  Measured full rollout SPS:           {1/rollout_time:.0f}")
print(f"  PPO update amortized overhead:       {update_time/512/total*100:.0f}%")

env.close()
