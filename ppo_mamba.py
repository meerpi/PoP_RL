import argparse
import os
import random
import time
from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
# NOTE: Mamba2 CUDA kernel segfaults on variable sequence lengths.
# Leaving Mamba policy for future revision.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from mamba_ssm import Mamba2

from clean_env import (
    PoPEnv, NUM_CH, GROWS, GCOLS, GRID_FLAT, KID_DIM, G_DIM, OBS_DIM, N_ACTIONS,
)


# ── Utilities ──

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_masked_gae(rewards, values, dones, next_value, next_done,
                       gamma, gae_lambda, num_steps, mask):
    """GAE where only timesteps with mask==True contribute."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        lastgaelam = lastgaelam * mask[t].float()
        advantages[t] = lastgaelam
    return advantages, advantages + values


def make_pop_env(seed, rank):
    def thunk():
        env = PoPEnv(visual=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + rank)
        return env
    return thunk


# ── Args ──

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=459916)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=2048)
    p.add_argument("--total-timesteps", type=int, default=900_000_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--mamba-lr", type=float, default=1.5e-4)
    p.add_argument("--p2-lr", type=float, default=2.5e-4, help="LR for P2 heads")
    p.add_argument("--p2-mamba-lr", type=float, default=1.5e-4, help="Mamba LR for P2")
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=5)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.04)
    p.add_argument("--p2-ent-coef", type=float, default=0.10)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    # mamba
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--headdim", type=int, default=64)
    # infra
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--exp-name", type=str, default="pop_mamba_shared")
    p.add_argument("--track", action="store_true", default=False)
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint .pt to resume from")
    args = p.parse_args()
    args.num_minibatches = min(args.num_minibatches, args.num_envs)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


# ── Agent ──

FLAT_DIM = KID_DIM + G_DIM  # 25 + 32 = 57


class PolicyHead(nn.Module):
    """Per-policy Mamba core + actor/critic heads."""

    def __init__(self, H, args, layer_idx):
        super().__init__()
        self.mamba = Mamba2(
            d_model=H, d_state=args.d_state, d_conv=args.d_conv,
            expand=args.expand, headdim=args.headdim,
        )
        self.mamba.layer_idx = layer_idx
        self.norm = nn.LayerNorm(H)
        self.post_mamba_mlp = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, H),
        )
        self.actor = layer_init(nn.Linear(H, N_ACTIONS), std=0.01)
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(H, H // 2)), nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(H // 2, 1), std=1.0)


class DualPolicyAgent(nn.Module):
    """Shared encoder (CNN+MLP) with two separate policy heads (Mamba+actor+critic)."""

    def __init__(self, args):
        super().__init__()
        H = args.hidden_dim

        # ── Shared encoder: spatial features usable by both policies ──
        self.grid_encoder = nn.Sequential(
            layer_init(nn.Conv2d(NUM_CH, 32, kernel_size=(1, 3), padding=(0, 1))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))),
            nn.ReLU(),
            nn.Flatten(),                        # 64 * 5 * 12 = 3840
            layer_init(nn.Linear(64 * GROWS * GCOLS, H // 2)),
            nn.ReLU(),
        )
        self.flat_encoder = nn.Sequential(
            layer_init(nn.Linear(FLAT_DIM, H // 4)),
            nn.ReLU(),
            layer_init(nn.Linear(H // 4, H // 2)),
            nn.ReLU(),
        )

        # ── Per-policy heads ──
        self.p1 = PolicyHead(H, args, layer_idx=0)  # pre-sword
        self.p2 = PolicyHead(H, args, layer_idx=1)  # post-sword

    def _head(self, pid):
        return self.p1 if pid == 0 else self.p2

    def encode(self, x):
        """x: (B, OBS_DIM) → (B, H). Shared across both policies."""
        grid = x[:, :GRID_FLAT].reshape(-1, NUM_CH, GROWS, GCOLS)
        flat = x[:, GRID_FLAT:]
        return torch.cat([self.grid_encoder(grid), self.flat_encoder(flat)], dim=-1)

    def _mamba_step(self, encoded, mamba_state, pid):
        """Single-token step through a policy's Mamba (inference/rollout)."""
        head = self._head(pid)
        if pid == 1:
            encoded = encoded.detach()  # freeze encoder for P2
        cur = encoded.unsqueeze(1)  # (B, 1, H)
        out, conv_s, ssm_s = head.mamba.step(cur, mamba_state[0], mamba_state[1])
        out = head.post_mamba_mlp(out.squeeze(1)) + encoded
        out = head.norm(out)
        return out, (conv_s, ssm_s)

    def forward_sequence(self, x, init_state, pid, dones=None):
        """Full-sequence forward (training). x: (T, B, OBS_DIM)."""
        head = self._head(pid)
        T, B = x.shape[:2]
        feat = self.encode(x.reshape(-1, *x.shape[2:])).reshape(T, B, -1)
        if pid == 1:
            feat = feat.detach()  # freeze encoder for P2 — use features, don't modify them
        feat = feat.transpose(0, 1)  # (B, T, H)

        seq_idx = None
        if dones is not None:
            seq_idx = dones.cumsum(0).to(torch.int32).transpose(0, 1).contiguous()

        if init_state is not None:
            ip = SimpleNamespace(
                key_value_memory_dict={head.mamba.layer_idx: init_state},
                seqlen_offset=0,
            )
        else:
            ip = None

        out = head.mamba(feat, seq_idx=seq_idx, inference_params=ip)
        out = head.post_mamba_mlp(out) + feat
        out = head.norm(out)
        return out.transpose(0, 1)  # (T, B, H)

    def get_value(self, x, mamba_state, pid):
        head = self._head(pid)
        h, new_state = self._mamba_step(self.encode(x), mamba_state, pid)
        return head.critic(head.critic_trunk(h)).flatten(), new_state

    def get_action_and_value(self, x, mamba_state, pid, action=None):
        head = self._head(pid)
        h, new_state = self._mamba_step(self.encode(x), mamba_state, pid)
        logits = head.actor(h)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (action, probs.log_prob(action), probs.entropy(),
                head.critic(head.critic_trunk(h)).flatten(), new_state)


def ppo_update(agent, optimizer, obs, actions, logprobs, advantages, returns,
               values, dones, init_mamba, policy_mask, pid, args, device):
    """Run PPO update for policy `pid` on timesteps where policy_mask is True."""
    num_steps, num_envs = obs.shape[:2]
    total_valid = policy_mask.sum().item()
    if total_valid < 32:
        return None

    head = agent._head(pid)
    envsperbatch = num_envs // args.num_minibatches
    envinds = np.arange(num_envs)
    flatinds = np.arange(num_steps * num_envs).reshape(num_steps, num_envs)

    b_obs = obs.reshape(-1, OBS_DIM)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_adv = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_mask = policy_mask.reshape(-1)

    clipfracs, losses_pg, losses_v, losses_ent = [], [], [], []
    losses_total, kl_list, grad_norms, mamba_grad_norms = [], [], [], []

    for epoch in range(args.update_epochs):
        np.random.shuffle(envinds)
        for start in range(0, num_envs, envsperbatch):
            mb_env = envinds[start : start + envsperbatch]
            mb_inds = flatinds[:, mb_env].ravel()

            mb_active = b_mask[mb_inds]
            if mb_active.sum().item() < 1:
                continue

            mb_init = (init_mamba[0][mb_env].clone(), init_mamba[1][mb_env].clone())
            mb_obs = obs[:, mb_env]

            # Forward through shared encoder + this policy's Mamba
            seq_out = agent.forward_sequence(mb_obs, mb_init, pid, dones=dones[:, mb_env])
            T, B, H = seq_out.shape
            flat_h = seq_out.reshape(-1, H)

            logits = head.actor(flat_h)
            probs = Categorical(logits=logits)
            new_lp = probs.log_prob(b_actions[mb_inds].long())
            new_ent = probs.entropy()
            new_val = head.critic(head.critic_trunk(flat_h)).reshape(-1)

            active = mb_active.bool()

            logratio = new_lp - b_logprobs[mb_inds]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio[active] - 1) - logratio[active]).mean()
                clipfracs.append(((ratio[active] - 1.0).abs() > args.clip_coef).float().mean().item())

            mb_adv = b_adv[mb_inds]
            if args.norm_adv:
                active_adv = mb_adv[active]
                if active_adv.numel() > 1:
                    mb_adv = mb_adv.clone()
                    mb_adv[active] = (active_adv - active_adv.mean()) / (active_adv.std() + 1e-8)

            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg1, pg2)[active].mean()

            if args.clip_vloss:
                v_unclip = (new_val - b_returns[mb_inds]) ** 2
                v_clip = b_values[mb_inds] + torch.clamp(
                    new_val - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss = 0.5 * torch.max(v_unclip, (v_clip - b_returns[mb_inds]) ** 2)[active].mean()
            else:
                v_loss = 0.5 * ((new_val - b_returns[mb_inds]) ** 2)[active].mean()

            ent_loss = new_ent[active].mean()
            ent_c = args.p2_ent_coef if pid == 1 else args.ent_coef
            loss = pg_loss - ent_c * ent_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()

            total_gn = sum(
                p.grad.data.norm(2).item() ** 2 for p in agent.parameters() if p.grad is not None
            ) ** 0.5
            mamba_gn = sum(
                p.grad.data.norm(2).item() ** 2 for p in head.mamba.parameters() if p.grad is not None
            ) ** 0.5
            grad_norms.append(total_gn)
            mamba_grad_norms.append(mamba_gn)

            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            losses_total.append(loss.item())
            losses_pg.append(pg_loss.item())
            losses_v.append(v_loss.item())
            losses_ent.append(ent_loss.item())
            kl_list.append(approx_kl.item())

        if args.target_kl is not None and kl_list and kl_list[-1] > args.target_kl:
            break

    if not losses_total:
        return None

    return {
        "total": np.mean(losses_total),
        "policy": np.mean(losses_pg),
        "value": np.mean(losses_v),
        "entropy": np.mean(losses_ent),
        "approx_kl": np.mean(kl_list),
        "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
        "grad_norm": np.mean(grad_norms),
        "mamba_grad_norm": np.mean(mamba_grad_norms),
    }


# ── Main ──

if __name__ == "__main__":
    args = parse_args()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    if args.resume:
        run_dir = os.path.dirname(os.path.abspath(args.resume))
        run_name = os.path.basename(run_dir)
    else:
        run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
        run_dir = os.path.join(SCRIPT_DIR, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    if args.track:
        import wandb
        wandb.init(project="pop-mamba-shared", name=run_name, config=vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    envs = gym.vector.AsyncVectorEnv(
        [make_pop_env(args.seed, i) for i in range(args.num_envs)],
        context="spawn",
    )

    # ═══ SINGLE AGENT, SHARED ENCODER ═══
    agent = DualPolicyAgent(args).to(device)

    # Optimizer: shared encoder + P1 heads + P2 heads (different LRs)
    # NOTE: Encoder is frozen for P2 (detached), so only P1 updates it.
    optimizer = optim.Adam([
        # Shared encoder — only trained by P1 (P2 features are detached)
        {"params": list(agent.grid_encoder.parameters()) + list(agent.flat_encoder.parameters())},
        # P1 heads
        {"params": agent.p1.norm.parameters()},
        {"params": agent.p1.post_mamba_mlp.parameters()},
        {"params": list(agent.p1.critic_trunk.parameters()) + list(agent.p1.critic.parameters())},
        {"params": agent.p1.actor.parameters()},
        {"params": agent.p1.mamba.parameters(), "lr": args.mamba_lr},
        # P2 heads (separate LR)
        {"params": agent.p2.norm.parameters(), "lr": args.p2_lr},
        {"params": agent.p2.post_mamba_mlp.parameters(), "lr": args.p2_lr},
        {"params": list(agent.p2.critic_trunk.parameters()) + list(agent.p2.critic.parameters()), "lr": args.p2_lr},
        {"params": agent.p2.actor.parameters(), "lr": args.p2_lr},
        {"params": agent.p2.mamba.parameters(), "lr": args.p2_mamba_lr},
    ], lr=args.learning_rate, eps=1e-5)

    # Resume checkpoint
    resume_update = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "update" in ckpt:
            resume_update = ckpt["update"]
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        print(f"Resumed from {args.resume}  (update={resume_update})")
    resume_global_step = global_step

    shared_params = sum(p.numel() for p in agent.grid_encoder.parameters()) + \
                    sum(p.numel() for p in agent.flat_encoder.parameters())
    p1_params = sum(p.numel() for p in agent.p1.parameters())
    p2_params = sum(p.numel() for p in agent.p2.parameters())
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Shared encoder: {shared_params/1e6:.2f}M | P1: {p1_params/1e6:.2f}M | "
          f"P2: {p2_params/1e6:.2f}M | Total: {total_params/1e6:.2f}M")

    # Rollout storage
    obs           = torch.zeros((args.num_steps, args.num_envs, OBS_DIM), device=device)
    actions       = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs      = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards       = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones         = torch.zeros((args.num_steps, args.num_envs), device=device)
    values        = torch.zeros((args.num_steps, args.num_envs), device=device)
    active_policy = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int8, device=device)

    start_time = time.time()
    ep_infos = deque(maxlen=100)

    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Per-env sword state
    env_has_sword = [False] * args.num_envs

    # Mamba states for both policy heads
    p1_conv, p1_ssm = agent.p1.mamba.allocate_inference_cache(args.num_envs, max_seqlen=1)
    p2_conv, p2_ssm = agent.p2.mamba.allocate_inference_cache(args.num_envs, max_seqlen=1)
    p1_mamba = (p1_conv, p1_ssm)
    p2_mamba = (p2_conv, p2_ssm)

    num_updates = args.total_timesteps // args.batch_size

    start_update = resume_update + 1
    for update in range(start_update, num_updates + 1):
        t0 = time.time()
        init_p1_mamba = (p1_mamba[0].clone(), p1_mamba[1].clone())
        init_p2_mamba = (p2_mamba[0].clone(), p2_mamba[1].clone())

        # Dashboard counters
        update_rooms = set()
        update_p2_rooms = set()
        update_deaths = 0
        update_sword_found = 0
        update_guard_kills = 0

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            # Shared encoder + P1 heads (param groups 0-4)
            for pg in optimizer.param_groups[:5]:
                pg["lr"] = frac * args.learning_rate
            optimizer.param_groups[5]["lr"] = frac * args.mamba_lr     # P1 mamba
            # P2 heads (param groups 6-9)
            for pg in optimizer.param_groups[6:10]:
                pg["lr"] = frac * args.p2_lr
            optimizer.param_groups[10]["lr"] = frac * args.p2_mamba_lr  # P2 mamba

        # ═══ ROLLOUT ═══
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            p1_envs = []
            p2_envs = []
            for eid in range(args.num_envs):
                if env_has_sword[eid]:
                    p2_envs.append(eid)
                    active_policy[step, eid] = 1
                else:
                    p1_envs.append(eid)
                    active_policy[step, eid] = 0

            combined_act = torch.zeros(args.num_envs, device=device)
            combined_lp = torch.zeros(args.num_envs, device=device)
            combined_val = torch.zeros(args.num_envs, device=device)

            with torch.no_grad():
                if p1_envs:
                    p1_idx = torch.tensor(p1_envs, device=device)
                    p1_obs = next_obs[p1_idx]
                    p1_ms = (p1_mamba[0][p1_idx], p1_mamba[1][p1_idx])
                    act1, lp1, _, val1, new_p1_ms = agent.get_action_and_value(p1_obs, p1_ms, pid=0)
                    combined_act[p1_idx] = act1.float()
                    combined_lp[p1_idx] = lp1
                    combined_val[p1_idx] = val1
                    p1_mamba[0][p1_idx] = new_p1_ms[0]
                    p1_mamba[1][p1_idx] = new_p1_ms[1]

                if p2_envs:
                    p2_idx = torch.tensor(p2_envs, device=device)
                    p2_obs = next_obs[p2_idx]
                    p2_ms = (p2_mamba[0][p2_idx], p2_mamba[1][p2_idx])
                    act2, lp2, _, val2, new_p2_ms = agent.get_action_and_value(p2_obs, p2_ms, pid=1)
                    combined_act[p2_idx] = act2.float()
                    combined_lp[p2_idx] = lp2
                    combined_val[p2_idx] = val2
                    p2_mamba[0][p2_idx] = new_p2_ms[0]
                    p2_mamba[1][p2_idx] = new_p2_ms[1]

            values[step] = combined_val
            actions[step] = combined_act
            logprobs[step] = combined_lp

            next_obs, rew, terminated, truncated, info = envs.step(combined_act.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(rew, dtype=torch.float32).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Reset Mamba states on episode boundaries
            for eid in range(args.num_envs):
                if done[eid]:
                    p1_mamba[0][eid].zero_()
                    p1_mamba[1][eid].zero_()
                    p2_mamba[0][eid].zero_()
                    p2_mamba[1][eid].zero_()
                    env_has_sword[eid] = False

            # Detect P1→P2 transition (sword pickup)
            # Mark as done boundary so GAE doesn't leak values across P1↔P2
            sf_arr = info.get("sword_found")
            if sf_arr is not None:
                for eid in range(args.num_envs):
                    if not done[eid]:
                        if int(sf_arr[eid]) == 1 and not env_has_sword[eid]:
                            env_has_sword[eid] = True
                            dones[step, eid] = 1.0  # treat as episode boundary for GAE
                            p2_mamba[0][eid].zero_()
                            p2_mamba[1][eid].zero_()
                            update_sword_found += 1

            # Dashboard metrics
            room_arr = info.get("room")
            if room_arr is not None:
                for eid, r in enumerate(room_arr):
                    if r and int(r) > 0:
                        update_rooms.add(int(r))
                        if env_has_sword[eid]:
                            update_p2_rooms.add(int(r))

            dead_arr = info.get("dead")
            if dead_arr is not None:
                update_deaths += int(np.sum(dead_arr))

            gk_arr = info.get("guard_killed")
            if gk_arr is not None:
                update_guard_kills += int(np.sum(np.array(gk_arr) > 0))

            fi = info.get("final_info")
            if fi is not None:
                for i, entry in enumerate(fi):
                    if entry is None:
                        continue
                    ep = entry.get("episode")
                    if ep is None and "episode" in info and isinstance(info["episode"], dict):
                        ep = {k: info["episode"][k][i] for k in info["episode"]}
                    if ep is not None:
                        ep_infos.append(ep)
                        writer.add_scalar("charts/ep_return", ep["r"], global_step)
                        writer.add_scalar("charts/ep_length", ep["l"], global_step)

        # ═══ MASKS ═══
        p1_mask = (active_policy == 0)
        p2_mask = (active_policy == 1)
        p1_steps = p1_mask.sum().item()
        p2_steps = p2_mask.sum().item()

        # ═══ GAE ═══
        with torch.no_grad():
            nv_p1, _ = agent.get_value(next_obs, p1_mamba, pid=0)
            nv_p2, _ = agent.get_value(next_obs, p2_mamba, pid=1)

        nv_combined = torch.where(
            torch.tensor(env_has_sword, device=device), nv_p2, nv_p1,
        )

        adv_p1, ret_p1 = compute_masked_gae(
            rewards, values, dones, nv_combined, next_done,
            args.gamma, args.gae_lambda, args.num_steps, p1_mask,
        )
        adv_p2, ret_p2 = compute_masked_gae(
            rewards, values, dones, nv_combined, next_done,
            args.gamma, args.gae_lambda, args.num_steps, p2_mask,
        )

        # ═══ PPO UPDATES ═══
        metrics_p1 = ppo_update(
            agent, optimizer, obs, actions, logprobs, adv_p1, ret_p1,
            values, dones, init_p1_mamba, p1_mask, pid=0, args=args, device=device,
        )

        metrics_p2 = ppo_update(
            agent, optimizer, obs, actions, logprobs, adv_p2, ret_p2,
            values, dones, init_p2_mamba, p2_mask, pid=1, args=args, device=device,
        )

        # ═══ LOGGING ═══
        sps = int((global_step - resume_global_step) / max(time.time() - start_time, 1.0))
        avg_ret = np.mean([e["r"] for e in ep_infos]) if ep_infos else 0.0

        b_values = values.reshape(-1).cpu().numpy()
        b_returns_p1 = ret_p1.reshape(-1).cpu().numpy()
        b_returns_p2 = ret_p2.reshape(-1).cpu().numpy()
        b_mask_flat = p1_mask.reshape(-1).cpu().numpy()
        b_returns = np.where(b_mask_flat, b_returns_p1, b_returns_p2)
        var_y = np.var(b_returns)
        ev = np.nan if var_y == 0 else 1 - np.var(b_returns - b_values) / var_y

        p1_pi = metrics_p1["policy"] if metrics_p1 else 0.0
        p2_pi = metrics_p2["policy"] if metrics_p2 else 0.0
        p1_ent = metrics_p1["entropy"] if metrics_p1 else 0.0
        p2_ent = metrics_p2["entropy"] if metrics_p2 else 0.0

        print(f"upd {update:4d} | SPS {sps:5d} | ret {avg_ret:8.2f} | "
              f"P1 pi {p1_pi:.4f} ent {p1_ent:.4f} steps {p1_steps} | "
              f"P2 pi {p2_pi:.4f} ent {p2_ent:.4f} steps {p2_steps} | ev {ev:.4f}")

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/avg_return", avg_ret, global_step)
        writer.add_scalar("charts/explained_var", ev, global_step)
        writer.add_scalar("charts/p1_timesteps", p1_steps, global_step)
        writer.add_scalar("charts/p2_timesteps", p2_steps, global_step)
        writer.add_scalar("charts/encoder_lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/p1_lr", optimizer.param_groups[1]["lr"], global_step)

        writer.add_scalar("charts/p2_lr", optimizer.param_groups[6]["lr"], global_step)

        if metrics_p1:
            for k, v in metrics_p1.items():
                writer.add_scalar(f"p1/{k}", v, global_step)
        if metrics_p2:
            for k, v in metrics_p2.items():
                writer.add_scalar(f"p2/{k}", v, global_step)

        # PoP dashboard
        writer.add_scalar("dashboard/rooms_visited", len(update_rooms), global_step)
        writer.add_scalar("dashboard/deaths", update_deaths, global_step)
        writer.add_scalar("dashboard/sword_found", update_sword_found, global_step)
        writer.add_scalar("dashboard/guard_kills", update_guard_kills, global_step)
        print(f"       Rooms({len(update_rooms)}): {sorted(update_rooms)} | "
              f"P2 rooms: {sorted(update_p2_rooms)} | "
              f"deaths {update_deaths} | swords {update_sword_found} | kills {update_guard_kills}")

        if update % args.save_interval == 0:
            path = os.path.join(run_dir, f"ckpt_{update}.pt")
            torch.save({
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "update": update,
                "global_step": global_step,
            }, path)
            print(f"  saved → {path}")

    writer.close()
    envs.close()