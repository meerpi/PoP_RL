"""
test_ppo_deep.py — Deep architecture tests for ppo.py Agent.

Focuses on correctness of the multi-stream architecture, gradient flow
isolation, initialization, and numerical properties.

Tests:
  D01  Grid CNN receptive field — conv layers handle 8×5×12 input correctly
  D02  Grid reshape — flat obs[:480] → (B,8,5,12) round-trips correctly
  D03  Vector encoding — kid+guard slice shapes through MLP
  D04  Goal conditioning — different goals produce different outputs
  D05  Goal conditioning — same state+goal produces identical output (deterministic critic)
  D06  Orthogonal init verification — hidden weight matrices are orthogonal
  D07  Actor init scale — near-uniform initial action distribution (std=0.01)
  D08  Critic init scale — output scale ~1.0 for random input
  D09  Gradient isolation — actor-only loss has zero grad on critic head
  D10  Gradient isolation — critic-only loss has zero grad on actor head
  D11  Shared trunk — shared trunk receives gradients from both heads
  D12  Real env observation — actual PoPEnv obs through full pipeline without NaN/Inf
  D13  Batch consistency — identical inputs in a batch produce identical outputs
  D14  Entropy range — entropy within [0, ln(18)] for 18 discrete actions
  D15  Large batch — 256-sample batch doesn't OOM or produce NaN
  D16  Action distribution — initial policy near-uniform over 18 actions
  D17  Value function — values are finite and reasonable scale after init
  D18  Backward through full PPO loss — combined policy+value+entropy loss
"""

import sys
import traceback
import math
import numpy as np
import torch
import torch.nn as nn

from ppo import (
    Agent, OBS_DIM, N_ACTIONS, N_SUBGOALS,
    GRID_END, KID_START, KID_END, GUARD_START, GUARD_END,
    NUM_CH, GROWS, GCOLS, KID_DIM, G_DIM, GRID_FLAT,
    layer_init,
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


def make_agent():
    return Agent().to(device)


def make_obs(B=4):
    return torch.randn(B, OBS_DIM, device=device)


def make_goal(B=4, sg=0, tr=2):
    g = torch.zeros(B, N_SUBGOALS + 1, device=device)
    g[:, sg] = 1.0
    g[:, N_SUBGOALS] = tr / 24.0
    return g


# ═══════════════════════════════════════════════════════════════

def test_D01_grid_cnn_shapes():
    """D01: Conv layers handle 8×5×12 input → correct intermediate shapes."""
    agent = make_agent()
    B = 4
    grid = torch.randn(B, NUM_CH, GROWS, GCOLS, device=device)

    # Step through grid_enc manually
    x = grid
    for i, layer in enumerate(agent.grid_enc):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            # padding=1 + kernel=3 preserves spatial dims
            assert x.shape[2] == GROWS, f"Conv {i}: height {x.shape[2]} != {GROWS}"
            assert x.shape[3] == GCOLS, f"Conv {i}: width {x.shape[3]} != {GCOLS}"
        elif isinstance(layer, nn.Flatten):
            assert x.shape == (B, 32 * GROWS * GCOLS), \
                f"Flatten: {x.shape} != ({B}, {32 * GROWS * GCOLS})"


def test_D02_grid_reshape():
    """D02: Flat obs[:480] reshapes to (B,8,5,12) correctly."""
    B = 2
    # Create a known pattern: channel c, row r, col c2 → value = c*100 + r*10 + c2
    grid_3d = torch.zeros(B, NUM_CH, GROWS, GCOLS)
    for c in range(NUM_CH):
        for r in range(GROWS):
            for c2 in range(GCOLS):
                grid_3d[:, c, r, c2] = c * 100 + r * 10 + c2

    flat = grid_3d.reshape(B, -1)  # (B, 480)
    assert flat.shape == (B, GRID_FLAT)

    # Round-trip
    recovered = flat.reshape(B, NUM_CH, GROWS, GCOLS)
    assert torch.allclose(recovered, grid_3d), "Grid reshape round-trip failed"


def test_D03_vec_encoding():
    """D03: Vector encoder processes kid+guard concat correctly."""
    agent = make_agent()
    B = 4
    vec = torch.randn(B, KID_DIM + G_DIM, device=device)
    out = agent.vec_enc(vec)
    assert out.shape == (B, 64), f"vec_enc output shape {out.shape} != (4, 64)"
    assert torch.isfinite(out).all(), "Non-finite values in vec_enc output"


def test_D04_goal_changes_output():
    """D04: Different goals produce different critic values."""
    agent = make_agent()
    obs = make_obs(1)

    goal_nav = make_goal(1, sg=0, tr=2)
    goal_fight = make_goal(1, sg=2, tr=3)

    with torch.no_grad():
        v1 = agent.get_value(obs, goal_nav)
        v2 = agent.get_value(obs, goal_fight)

    # With orthogonal init and different goal inputs, outputs should differ
    assert not torch.allclose(v1, v2, atol=1e-6), \
        f"Same critic value for different goals: {v1.item():.6f} vs {v2.item():.6f}"


def test_D05_deterministic_critic():
    """D05: Same state+goal always produces identical critic output."""
    agent = make_agent()
    agent.eval()
    obs = make_obs(1)
    goal = make_goal(1)

    with torch.no_grad():
        v1 = agent.get_value(obs, goal)
        v2 = agent.get_value(obs, goal)

    assert torch.allclose(v1, v2, atol=1e-7), \
        f"Critic not deterministic: {v1.item()} vs {v2.item()}"


def test_D06_orthogonal_init():
    """D06: Hidden layer weights are approximately orthogonal after init."""
    agent = make_agent()
    # Check the first linear layer in vec_enc
    W = agent.vec_enc[0].weight.data  # shape (128, 62)
    # Since W is 128x62 (taller than it is wide), torch.nn.init.orthogonal_
    # makes columns orthogonal. Thus W.T @ W ≈ scale² * I.
    # W @ W.T would not be proportional to I.
    if W.shape[0] >= W.shape[1]:
        ortho_mat = W.T @ W
    else:
        ortho_mat = W @ W.T
        
    diag_mean = ortho_mat.diag().mean().item()
    # scale = sqrt(2), so scale² = 2.0
    assert 1.5 < diag_mean < 2.5, \
        f"Diagonal mean {diag_mean} doesn't match orthogonal init scale sqrt(2)^2 = 2.0"



def test_D07_actor_near_uniform():
    """D07: Actor output layer initialized with std=0.01 → near-uniform policy."""
    agent = make_agent()
    obs = make_obs(8)
    goal = make_goal(8)

    with torch.no_grad():
        logits = agent.actor(agent._encode(obs, goal))
        probs = torch.softmax(logits, dim=-1)

    # Near-uniform: each action ≈ 1/18 ≈ 0.0556
    uniform = 1.0 / N_ACTIONS
    max_deviation = (probs - uniform).abs().max().item()
    # With std=0.01 init, deviation should be tiny
    assert max_deviation < 0.05, \
        f"Initial policy too far from uniform: max deviation {max_deviation:.4f}"


def test_D08_critic_output_scale():
    """D08: Critic output scale reasonable at init (std=1.0 for output layer)."""
    agent = make_agent()
    obs = make_obs(64)
    goal = make_goal(64)

    with torch.no_grad():
        values = agent.get_value(obs, goal)

    # Values should be finite and not enormous
    assert torch.isfinite(values).all(), "Non-finite critic values at init"
    val_std = values.std().item()
    val_mean = values.mean().item()
    # Reasonable: mean near 0, std < 10
    assert abs(val_mean) < 5.0, f"Critic mean {val_mean} too large at init"
    assert val_std < 10.0, f"Critic std {val_std} too large at init"


def test_D09_actor_grad_isolation():
    """D09: Actor-only loss produces zero grad on critic head parameters."""
    agent = make_agent()
    agent.zero_grad()

    obs = make_obs(4)
    goal = make_goal(4)
    action, logprob, entropy, value = agent.get_action_and_value(obs, goal)

    # Actor-only loss (detach value)
    actor_loss = -logprob.mean()
    actor_loss.backward()

    # Critic output layer should have NO gradient from actor loss
    critic_last = agent.critic[-1]  # last Linear layer
    # Since value was computed but not included in loss, its grad should be None or zero
    # Actually, value IS computed from shared trunk, so critic head params might get grad
    # BUT: the value wasn't included in the loss, so critic head params should have grad=None
    has_grad = critic_last.weight.grad is not None and critic_last.weight.grad.abs().sum() > 0
    # The critic head should NOT receive gradient from the actor loss
    # However, the trunk IS shared, so trunk params get gradients from the actor
    # The key: critic HEAD (the separate MLP after trunk) should have zero grad
    # from actor-only loss because value was not used in loss
    # BUT: get_action_and_value calls self.critic(feat), creating a computation graph
    # Since we didn't use the value output in the loss, the critic head won't get grad
    assert not has_grad, \
        "Critic head received gradient from actor-only loss — heads not properly separate"


def test_D10_critic_grad_isolation():
    """D10: Critic-only loss produces zero grad on actor head parameters."""
    agent = make_agent()
    agent.zero_grad()

    obs = make_obs(4)
    goal = make_goal(4)

    # Use only get_value (no actor computation)
    value = agent.get_value(obs, goal)
    critic_loss = (value ** 2).mean()
    critic_loss.backward()

    # Actor output layer should have NO gradient
    actor_last = agent.actor[-1]
    has_grad = actor_last.weight.grad is not None and actor_last.weight.grad.abs().sum() > 0
    assert not has_grad, \
        "Actor head received gradient from critic-only loss"


def test_D11_shared_trunk_gradients():
    """D11: Shared trunk receives gradients from both actor and critic losses."""
    agent = make_agent()

    obs = make_obs(4)
    goal = make_goal(4)

    # Actor gradient
    agent.zero_grad()
    _, logprob, _, _ = agent.get_action_and_value(obs, goal)
    (-logprob.mean()).backward()
    trunk_grad_actor = agent.trunk[0].weight.grad.clone()

    # Critic gradient
    agent.zero_grad()
    value = agent.get_value(obs, goal)
    (value ** 2).mean().backward()
    trunk_grad_critic = agent.trunk[0].weight.grad.clone()

    # Both should be non-zero
    assert trunk_grad_actor.abs().sum() > 0, "Trunk got zero grad from actor"
    assert trunk_grad_critic.abs().sum() > 0, "Trunk got zero grad from critic"
    # They should be different (different loss functions)
    assert not torch.allclose(trunk_grad_actor, trunk_grad_critic, atol=1e-8), \
        "Trunk gradients from actor and critic are identical — suspicious"


def test_D12_real_env_obs():
    """D12: Real PoPEnv observation through full pipeline — no NaN/Inf."""
    import os
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    from clean_env import PoPEnv, DummyManager

    env = DummyManager(PoPEnv(visual=False))
    obs, info = env.reset(seed=42)

    agent = make_agent()

    obs_t = torch.tensor(obs, device=device).unsqueeze(0)
    goal_t = torch.zeros(1, N_SUBGOALS + 1, device=device)
    sg = int(info["current_subgoal"])
    goal_t[0, sg] = 1.0
    goal_t[0, N_SUBGOALS] = info["sg_target_room"] / 24.0

    with torch.no_grad():
        action, logprob, entropy, value = agent.get_action_and_value(obs_t, goal_t)

    assert torch.isfinite(logprob), f"logprob not finite: {logprob}"
    assert torch.isfinite(entropy), f"entropy not finite: {entropy}"
    assert torch.isfinite(value), f"value not finite: {value}"
    assert 0 <= action.item() < N_ACTIONS, f"action {action.item()} out of range"

    # Step 10 times through env
    for _ in range(10):
        obs, rew, term, trunc, info = env.step(int(action.item()))
        if term or trunc:
            obs, info = env.reset()
        obs_t = torch.tensor(obs, device=device).unsqueeze(0)
        goal_t = torch.zeros(1, N_SUBGOALS + 1, device=device)
        sg = int(info["current_subgoal"])
        goal_t[0, sg] = 1.0
        goal_t[0, N_SUBGOALS] = info["sg_target_room"] / 24.0
        with torch.no_grad():
            action, logprob, entropy, value = agent.get_action_and_value(obs_t, goal_t)
        assert torch.isfinite(value), f"Non-finite value on real obs at step"


def test_D13_batch_consistency():
    """D13: Identical inputs in a batch produce identical outputs."""
    agent = make_agent()
    agent.eval()

    single_obs = torch.randn(1, OBS_DIM, device=device)
    single_goal = make_goal(1)

    # Duplicate into a batch of 4
    batch_obs = single_obs.repeat(4, 1)
    batch_goal = single_goal.repeat(4, 1)

    with torch.no_grad():
        values = agent.get_value(batch_obs, batch_goal)

    # All 4 values should be identical
    for i in range(1, 4):
        assert torch.allclose(values[0], values[i], atol=1e-6), \
            f"Value[0]={values[0].item():.6f} != Value[{i}]={values[i].item():.6f}"


def test_D14_entropy_range():
    """D14: Entropy within valid range [0, ln(18)] for 18 discrete actions."""
    agent = make_agent()
    obs = make_obs(32)
    goal = make_goal(32)

    with torch.no_grad():
        _, _, entropy, _ = agent.get_action_and_value(obs, goal)

    max_entropy = math.log(N_ACTIONS)  # ln(18) ≈ 2.89
    assert (entropy >= 0).all(), f"Negative entropy: {entropy.min()}"
    assert (entropy <= max_entropy + 0.01).all(), \
        f"Entropy exceeds ln(18)={max_entropy:.3f}: max={entropy.max():.3f}"
    # At init with std=0.01, entropy should be near max (near-uniform)
    mean_ent = entropy.mean().item()
    assert mean_ent > max_entropy * 0.9, \
        f"Initial entropy {mean_ent:.3f} too low — policy not near-uniform at init"


def test_D15_large_batch():
    """D15: 256-sample batch forward+backward without OOM or NaN."""
    agent = make_agent()
    B = 256
    obs = make_obs(B)
    goal = make_goal(B)

    action, logprob, entropy, value = agent.get_action_and_value(obs, goal)

    assert torch.isfinite(logprob).all(), "NaN/Inf in logprob (B=256)"
    assert torch.isfinite(value).all(), "NaN/Inf in value (B=256)"

    # Full PPO-style loss
    advantages = torch.randn(B, device=device)
    returns = torch.randn(B, device=device)
    old_logprob = logprob.detach() + torch.randn(B, device=device) * 0.01

    ratio = (logprob - old_logprob).exp()
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 0.8, 1.2)
    pg_loss = torch.max(pg1, pg2).mean()
    v_loss = 0.5 * ((value.squeeze() - returns) ** 2).mean()
    loss = pg_loss + 0.5 * v_loss - 0.02 * entropy.mean()

    agent.zero_grad()
    loss.backward()

    # All gradients should be finite
    for name, param in agent.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"


def test_D16_action_distribution():
    """D16: Initial policy produces near-uniform action distribution."""
    agent = make_agent()
    agent.eval()
    obs = make_obs(100)
    goal = make_goal(100)

    with torch.no_grad():
        actions, _, _, _ = agent.get_action_and_value(obs, goal)

    counts = torch.zeros(N_ACTIONS)
    for a in actions:
        counts[a.item()] += 1

    # With 100 samples from near-uniform over 18 actions:
    # expected ≈ 5.56 per action
    # At least some actions should appear (no dead actions)
    n_seen = (counts > 0).sum().item()
    assert n_seen >= N_ACTIONS * 0.5, \
        f"Only {n_seen}/{N_ACTIONS} actions sampled — distribution too concentrated"


def test_D17_value_finite():
    """D17: Values are finite with diverse inputs."""
    agent = make_agent()

    # Test with various input patterns
    patterns = [
        torch.zeros(1, OBS_DIM),       # all zeros
        torch.ones(1, OBS_DIM),        # all ones
        torch.randn(1, OBS_DIM) * 10,  # large values
        torch.randn(1, OBS_DIM) * 0.001,  # tiny values
    ]
    goal = make_goal(1)

    for i, obs in enumerate(patterns):
        with torch.no_grad():
            v = agent.get_value(obs.to(device), goal)
        assert torch.isfinite(v).all(), f"Non-finite value for pattern {i}: {v}"


def test_D18_full_ppo_loss_backward():
    """D18: Full PPO loss (policy + value + entropy) backward pass."""
    agent = make_agent()
    B = 32
    obs = make_obs(B)
    goal = make_goal(B)

    # Forward
    action, logprob, entropy, value = agent.get_action_and_value(obs, goal)

    # Fake old values
    old_logprob = logprob.detach()
    old_values = value.detach().squeeze()
    returns = torch.randn(B, device=device)
    advantages = torch.randn(B, device=device)

    # Normalise advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    ratio = (logprob - old_logprob).exp()
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 0.8, 1.2)
    pg_loss = torch.max(pg1, pg2).mean()

    # Clipped value loss
    newvalue = value.squeeze()
    v_unclipped = (newvalue - returns) ** 2
    v_clipped_val = old_values + torch.clamp(newvalue - old_values, -0.2, 0.2)
    v_clipped = (v_clipped_val - returns) ** 2
    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

    # Entropy
    ent_loss = entropy.mean()

    # Combined
    loss = pg_loss + 0.5 * v_loss - 0.02 * ent_loss

    agent.zero_grad()
    loss.backward()

    # Verify gradient exists and is finite for every parameter
    grads_ok = 0
    total_params = 0
    for name, param in agent.named_parameters():
        total_params += 1
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
            if param.grad.abs().sum() > 0:
                grads_ok += 1

    # Most parameters should have non-zero gradients
    assert grads_ok >= total_params * 0.8, \
        f"Only {grads_ok}/{total_params} params have non-zero gradients"

    # Clipping should work
    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    total_norm = sum(p.grad.norm().item() ** 2 for p in agent.parameters() if p.grad is not None) ** 0.5
    assert total_norm <= 0.5 + 1e-5, f"Grad norm {total_norm} exceeds clip threshold 0.5"


# ═══════════════════════════════════════════════════════════════

TESTS = [
    ("D01 Grid CNN shapes",          test_D01_grid_cnn_shapes),
    ("D02 Grid reshape round-trip",  test_D02_grid_reshape),
    ("D03 Vector encoding",          test_D03_vec_encoding),
    ("D04 Goal changes output",      test_D04_goal_changes_output),
    ("D05 Deterministic critic",     test_D05_deterministic_critic),
    ("D06 Orthogonal init",          test_D06_orthogonal_init),
    ("D07 Actor near-uniform init",  test_D07_actor_near_uniform),
    ("D08 Critic output scale",      test_D08_critic_output_scale),
    ("D09 Actor grad → no critic",   test_D09_actor_grad_isolation),
    ("D10 Critic grad → no actor",   test_D10_critic_grad_isolation),
    ("D11 Shared trunk both grads",  test_D11_shared_trunk_gradients),
    ("D12 Real env observation",     test_D12_real_env_obs),
    ("D13 Batch consistency",        test_D13_batch_consistency),
    ("D14 Entropy range",            test_D14_entropy_range),
    ("D15 Large batch (256)",        test_D15_large_batch),
    ("D16 Action distribution",      test_D16_action_distribution),
    ("D17 Value finite patterns",    test_D17_value_finite),
    ("D18 Full PPO loss backward",   test_D18_full_ppo_loss_backward),
]

if __name__ == "__main__":
    print("=" * 62)
    print("  Deep PPO Architecture Test Suite")
    print("=" * 62)
    print()

    for name, fn in TESTS:
        run(name, fn)

    passed = sum(1 for _, s in RESULTS if s == "PASS")
    failed = sum(1 for _, s in RESULTS if s == "FAIL")
    print()
    print("=" * 62)
    print(f"  Results: {passed} PASSED,  {failed} FAILED  (of {len(TESTS)} tests)")
    print("=" * 62)

    if failed:
        sys.exit(1)
    else:
        print("\n  All deep architecture tests passed!")
        sys.exit(0)
