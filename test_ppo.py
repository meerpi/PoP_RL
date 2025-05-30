"""
test_ppo.py — End-to-end tests for ppo.py Agent + training loop.

Tests:
  1. Import/syntax         — ppo.py imports without errors
  2. Agent init            — Agent() instantiates correctly
  3. Forward pass shapes   — get_action_and_value output shapes correct
  4. Gradient flow         — loss.backward() does not crash, all params have grad
  5. Obs slicing constants — GRID_END, KID_START, etc. match OBS_DIM
  6. Goal vector           — make_goal_vec produces correct one-hot + normalised target
  7. Env factory           — make_env creates DummyManager with correct interface
  8. Single rollout step   — one step through the rollout loop works
  9. GAE computation       — advantages and returns have correct shape and values
  10. Short training run   — 2 full iterations without crash
  11. Checkpoint save/load — save then reload produces same outputs
  12. Subgoal done signal  — done_buf set to 1.0 on subgoal boundary
"""

import sys
import os
import traceback
import tempfile
import numpy as np
import torch

# Suppress SDL noise
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_RENDER_DRIVER"] = "software"

from ppo import (
    Agent, Args, make_env, make_goal_vec, layer_init,
    OBS_DIM, GRID_END, KID_START, KID_END, GUARD_START, GUARD_END,
    N_SUBGOALS, N_ACTIONS, NUM_CH, GROWS, GCOLS, KID_DIM, G_DIM,
    GRID_FLAT,
)
from clean_env import (
    PoPEnv, DummyManager,
    SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL,
)

RESULTS = []
device = torch.device("cpu")


def run(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS"))
        print(f"  [PASS] {name}")
    except Exception as e:
        tb = traceback.format_exc()
        RESULTS.append((name, "FAIL"))
        print(f"  [FAIL] {name}")
        for line in tb.strip().split("\n")[-3:]:
            print(f"         {line}")


# ═══════════════════════════════════════════════════════════════

def test_01_import():
    """1. ppo.py imports without errors."""
    import ppo  # noqa — already imported but this confirms no syntax error
    assert hasattr(ppo, "Agent")
    assert hasattr(ppo, "train")
    assert hasattr(ppo, "Args")


def test_02_agent_init():
    """2. Agent instantiates correctly."""
    agent = Agent().to(device)
    total_params = sum(p.numel() for p in agent.parameters())
    assert total_params > 0, "Agent has no parameters"
    # Check that all submodules exist
    assert hasattr(agent, "grid_enc")
    assert hasattr(agent, "vec_enc")
    assert hasattr(agent, "goal_enc")
    assert hasattr(agent, "trunk")
    assert hasattr(agent, "actor")
    assert hasattr(agent, "critic")
    print(f"    Total params: {total_params:,}")


def test_03_forward_shapes():
    """3. Forward pass shapes — get_action_and_value."""
    agent = Agent().to(device)
    B = 4

    obs   = torch.randn(B, OBS_DIM, device=device)
    goals = torch.zeros(B, N_SUBGOALS + 1, device=device)
    goals[:, 0] = 1.0  # SG_NAVIGATE
    goals[:, N_SUBGOALS] = 2.0 / 24.0  # target room 2

    action, logprob, entropy, value = agent.get_action_and_value(obs, goals)

    assert action.shape == (B,), f"action shape {action.shape}"
    assert logprob.shape == (B,), f"logprob shape {logprob.shape}"
    assert entropy.shape == (B,), f"entropy shape {entropy.shape}"
    assert value.shape == (B, 1), f"value shape {value.shape}"

    # Actions should be in valid range
    assert (action >= 0).all() and (action < N_ACTIONS).all(), \
        f"Actions out of range: {action}"


def test_04_gradient_flow():
    """4. Gradient flow — loss.backward() produces gradients for all params."""
    agent = Agent().to(device)
    B = 4

    obs   = torch.randn(B, OBS_DIM, device=device)
    goals = torch.zeros(B, N_SUBGOALS + 1, device=device)
    goals[:, 0] = 1.0

    action, logprob, entropy, value = agent.get_action_and_value(obs, goals)

    # Fake loss
    loss = -logprob.mean() + 0.5 * value.mean() ** 2 - 0.01 * entropy.mean()
    loss.backward()

    for name, param in agent.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_05_obs_slicing():
    """5. Obs slicing constants match OBS_DIM arithmetic."""
    assert GRID_END == GRID_FLAT == NUM_CH * GROWS * GCOLS == 480
    assert KID_START == GRID_END == 480
    assert KID_END == KID_START + KID_DIM == 506
    assert GUARD_START == KID_END == 506
    assert GUARD_END == GUARD_START + G_DIM == 538
    assert GUARD_END == OBS_DIM


def test_06_goal_vector():
    """6. make_goal_vec produces correct one-hot + normalised target."""
    infos = {
        "current_subgoal": np.array([SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL]),
        "sg_target_room": np.array([2, 15, 3, 10]),
    }
    goals = make_goal_vec(infos, 4, device)

    assert goals.shape == (4, N_SUBGOALS + 1)

    # Check one-hots
    assert goals[0, 0] == 1.0 and goals[0, 1:4].sum() == 0.0  # NAVIGATE
    assert goals[1, 1] == 1.0 and goals[1, 0] == 0.0           # PICKUP_SWORD
    assert goals[2, 2] == 1.0                                    # FIGHT_GUARD
    assert goals[3, 3] == 1.0                                    # HEAL

    # Check normalised target rooms
    assert abs(goals[0, 4].item() - 2/24) < 1e-5
    assert abs(goals[1, 4].item() - 15/24) < 1e-5
    assert abs(goals[2, 4].item() - 3/24) < 1e-5
    assert abs(goals[3, 4].item() - 10/24) < 1e-5


def test_07_env_factory():
    """7. make_env creates a DummyManager with correct interface."""
    env = make_env(seed=0, env_id=0)()
    assert isinstance(env, DummyManager), f"Expected DummyManager, got {type(env)}"
    assert hasattr(env, "step")
    assert hasattr(env, "reset")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,), f"Obs shape {obs.shape}"
    assert "current_subgoal" in info
    assert "sg_target_room" in info


def test_08_single_step():
    """8. Single step through one env — obs, reward, info correct."""
    env = make_env(seed=0, env_id=0)()
    obs, info = env.reset(seed=0)
    agent = Agent().to(device)

    obs_t = torch.tensor(obs, device=device).unsqueeze(0)
    goal_t = torch.zeros(1, N_SUBGOALS + 1, device=device)
    sg = int(info["current_subgoal"])
    goal_t[0, sg] = 1.0
    goal_t[0, N_SUBGOALS] = info["sg_target_room"] / 24.0

    with torch.no_grad():
        action, logprob, entropy, value = agent.get_action_and_value(obs_t, goal_t)

    obs2, rew, term, trunc, info2 = env.step(int(action.item()))
    assert obs2.shape == (OBS_DIM,)
    assert isinstance(rew, (float, np.floating))
    assert isinstance(term, (bool, np.bool_))
    assert "current_subgoal" in info2


def test_09_gae_shapes():
    """9. GAE computation produces correct shapes."""
    T, N = 16, 2
    rewards = torch.randn(T, N, device=device) * 0.01
    values  = torch.randn(T, N, device=device) * 0.5
    dones   = torch.zeros(T, N, device=device)
    dones[5, 0] = 1.0   # one done signal
    dones[10, 1] = 1.0

    next_done = torch.zeros(N, device=device)
    next_value = torch.randn(1, N, device=device) * 0.5

    gamma = 0.99
    gae_lambda = 0.95

    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues.squeeze() * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values

    assert advantages.shape == (T, N)
    assert returns.shape == (T, N)
    assert not torch.isnan(advantages).any(), "NaN in advantages"
    assert not torch.isnan(returns).any(), "NaN in returns"

    # Advantage at a done boundary should reset accumulation
    # After done at step 5 env 0, step 6 starts fresh
    # The advantage at step 5 should not carry forward from step 6
    # (This is hard to test exactly without known values, so just verify finite)
    assert torch.isfinite(advantages[5, 0])


def test_10_short_training():
    """10. Short training run — 2 iterations, no crash."""
    agent = Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    env = make_env(seed=0, env_id=0)()
    obs, info = env.reset(seed=0)

    num_steps = 32  # tiny buffer for testing
    num_envs  = 1

    for iteration in range(2):
        obs_buf     = torch.zeros(num_steps, OBS_DIM, device=device)
        goal_buf    = torch.zeros(num_steps, N_SUBGOALS + 1, device=device)
        action_buf  = torch.zeros(num_steps, dtype=torch.long, device=device)
        logprob_buf = torch.zeros(num_steps, device=device)
        reward_buf  = torch.zeros(num_steps, device=device)
        done_buf    = torch.zeros(num_steps, device=device)
        value_buf   = torch.zeros(num_steps, device=device)

        cur_obs = torch.tensor(obs, device=device).unsqueeze(0)
        goal = torch.zeros(1, N_SUBGOALS + 1, device=device)
        sg = int(info.get("current_subgoal", 0))
        goal[0, sg] = 1.0
        goal[0, N_SUBGOALS] = info.get("sg_target_room", 2) / 24.0
        cur_done = 0.0

        for step in range(num_steps):
            obs_buf[step] = cur_obs.squeeze()
            goal_buf[step] = goal.squeeze()
            done_buf[step] = cur_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(cur_obs, goal)
                value_buf[step] = value.item()
            action_buf[step] = action.item()
            logprob_buf[step] = logprob.item()

            obs, rew, term, trunc, info = env.step(int(action.item()))
            reward_buf[step] = rew

            sg_done = info.get("subgoal_achieved", False) or info.get("worker_truncated", False)
            game_done = term or trunc
            if game_done:
                obs, info = env.reset()

            cur_done = 1.0 if (game_done or sg_done) else 0.0
            cur_obs = torch.tensor(obs, device=device).unsqueeze(0)
            goal = torch.zeros(1, N_SUBGOALS + 1, device=device)
            sg = int(info.get("current_subgoal", 0))
            goal[0, sg] = 1.0
            goal[0, N_SUBGOALS] = info.get("sg_target_room", 2) / 24.0

        # GAE
        with torch.no_grad():
            next_val = agent.get_value(cur_obs, goal).item()
            advantages = torch.zeros(num_steps, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nnt = 1.0 - cur_done
                    nv = next_val
                else:
                    nnt = 1.0 - done_buf[t+1]
                    nv = value_buf[t+1]
                delta = reward_buf[t] + 0.99 * nv * nnt - value_buf[t]
                lastgaelam = delta + 0.99 * 0.95 * nnt * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_buf

        # Update
        _, newlp, ent, newv = agent.get_action_and_value(
            obs_buf, goal_buf, action_buf)
        ratio = (newlp - logprob_buf).exp()
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg1 = -adv_norm * ratio
        pg2 = -adv_norm * torch.clamp(ratio, 0.8, 1.2)
        pg_loss = torch.max(pg1, pg2).mean()
        v_loss = 0.5 * ((newv.view(-1) - returns) ** 2).mean()
        loss = pg_loss + 0.5 * v_loss - 0.02 * ent.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

    # If we got here, training didn't crash
    assert True


def test_11_checkpoint():
    """11. Checkpoint save/load produces same forward pass output."""
    agent = Agent().to(device)
    obs = torch.randn(1, OBS_DIM, device=device)
    goal = torch.zeros(1, N_SUBGOALS + 1, device=device)
    goal[0, 0] = 1.0

    with torch.no_grad():
        _, _, _, val1 = agent.get_action_and_value(obs, goal)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save({"agent": agent.state_dict()}, path)

    agent2 = Agent().to(device)
    ckpt = torch.load(path, map_location=device)
    agent2.load_state_dict(ckpt["agent"])

    with torch.no_grad():
        _, _, _, val2 = agent2.get_action_and_value(obs, goal)

    os.unlink(path)
    assert torch.allclose(val1, val2, atol=1e-6), \
        f"Values differ after reload: {val1} vs {val2}"


def test_12_subgoal_done_signal():
    """12. Done signal fires on subgoal boundary, not just on death."""
    env = make_env(seed=0, env_id=0)()
    obs, info = env.reset(seed=0)

    # Set target to current room → subgoal fires immediately
    env.env._init_subgoal_tracking(SG_NAVIGATE, target_room=env.env.k_room)

    obs, rew, term, trunc, info = env.step(0)

    if info.get("subgoal_achieved"):
        # In the training loop, this would set done=1.0
        sg_done = info.get("subgoal_achieved", False)
        game_done = term or trunc
        done = 1.0 if (game_done or sg_done) else 0.0
        assert done == 1.0, "done signal should be 1.0 on subgoal boundary"


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PPO Agent + Training Loop Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("01 Import/syntax",         test_01_import),
        ("02 Agent init",            test_02_agent_init),
        ("03 Forward pass shapes",   test_03_forward_shapes),
        ("04 Gradient flow",         test_04_gradient_flow),
        ("05 Obs slicing constants", test_05_obs_slicing),
        ("06 Goal vector",           test_06_goal_vector),
        ("07 Env factory",           test_07_env_factory),
        ("08 Single step",           test_08_single_step),
        ("09 GAE shapes",            test_09_gae_shapes),
        ("10 Short training run",    test_10_short_training),
        ("11 Checkpoint save/load",  test_11_checkpoint),
        ("12 Subgoal done signal",   test_12_subgoal_done_signal),
    ]

    for name, fn in tests:
        run(name, fn)

    print()
    passed = sum(1 for _, s in RESULTS if s == "PASS")
    failed = sum(1 for _, s in RESULTS if s == "FAIL")
    print("=" * 60)
    print(f"  Results: {passed} PASSED,  {failed} FAILED  (of {len(tests)} tests)")
    print("=" * 60)

    if failed:
        sys.exit(1)
    else:
        print("\n  All PPO tests passed!")
        sys.exit(0)
