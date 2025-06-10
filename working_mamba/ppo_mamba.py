import argparse
import os
import random
import time
from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from mamba_ssm import Mamba2

from clean_env import PoPEnv, NUM_CH, GROWS, GCOLS, GRID_FLAT, KID_DIM, G_DIM, OBS_DIM, N_ACTIONS


# ── Utilities (inlined to avoid external deps) ──

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda, num_steps):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
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
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=5)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.04)
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
    p.add_argument("--exp-name", type=str, default="pop_mamba")
    p.add_argument("--track", action="store_true", default=False)
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint .pt to resume from")
    args = p.parse_args()
    args.num_minibatches = min(args.num_minibatches, args.num_envs)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


# ── Agent ──

FLAT_DIM = KID_DIM + G_DIM  # 25 + 32 = 57

class Agent(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_dim

        # Dual encoder: CNN for spatial grid, MLP for flat kid/guard vectors
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

        # Mamba2 recurrent core
        self.mamba = Mamba2(
            d_model=H, d_state=args.d_state, d_conv=args.d_conv,
            expand=args.expand, headdim=args.headdim,
        )
        self.mamba.layer_idx = 0
        self.norm = nn.LayerNorm(H)
        self.post_mamba_mlp = nn.Sequential(
            nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, H),
        )

        # Actor / critic (separate critic trunk avoids gradient conflict)
        self.actor = layer_init(nn.Linear(H, N_ACTIONS), std=0.01)
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(H, H // 2)), nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(H // 2, 1), std=1.0)

    def encode(self, x):
        """x: (B, OBS_DIM) flat obs → (B, H) encoded."""
        grid = x[:, :GRID_FLAT].reshape(-1, NUM_CH, GROWS, GCOLS)
        flat = x[:, GRID_FLAT:]
        return torch.cat([self.grid_encoder(grid), self.flat_encoder(flat)], dim=-1)

    def _mamba_step(self, encoded, mamba_state):
        """Single-token step through Mamba (inference / rollout)."""
        cur = encoded.unsqueeze(1)  # (B, 1, H)
        out, conv_s, ssm_s = self.mamba.step(cur, mamba_state[0], mamba_state[1])
        out = self.post_mamba_mlp(out.squeeze(1)) + encoded
        out = self.norm(out)
        return out, (conv_s, ssm_s)

    def forward_sequence(self, x, init_state=None, dones=None):
        """Full-sequence forward (training). x: (T, B, OBS_DIM)."""
        T, B = x.shape[:2]
        feat = self.encode(x.reshape(-1, *x.shape[2:])).reshape(T, B, -1)
        feat = feat.transpose(0, 1)  # (B, T, H)

        # seq_idx lets Mamba2's Triton kernels reset SSM state at episode boundaries
        seq_idx = None
        if dones is not None:
            seq_idx = dones.cumsum(0).to(torch.int32).transpose(0, 1).contiguous()  # (B, T)

        if init_state is not None:
            ip = SimpleNamespace(
                key_value_memory_dict={self.mamba.layer_idx: init_state},
                seqlen_offset=0,
            )
        else:
            ip = None

        out = self.mamba(feat, seq_idx=seq_idx, inference_params=ip)
        out = self.post_mamba_mlp(out) + feat
        out = self.norm(out)
        return out.transpose(0, 1)  # (T, B, H)

    def get_value(self, x, mamba_state):
        h, new_state = self._mamba_step(self.encode(x), mamba_state)
        return self.critic(self.critic_trunk(h)).flatten(), new_state

    def get_action_and_value(self, x, mamba_state, action=None):
        h, new_state = self._mamba_step(self.encode(x), mamba_state)
        logits = self.actor(h)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (action, probs.log_prob(action), probs.entropy(),
                self.critic(self.critic_trunk(h)).flatten(), new_state)


# ── Main ──

if __name__ == "__main__":
    args = parse_args()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Run directory — reuse on resume
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
        wandb.init(project="pop-mamba", name=run_name, config=vars(args))

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
    agent = Agent(args).to(device)

    optimizer = optim.Adam([
        {"params": list(agent.grid_encoder.parameters()) + list(agent.flat_encoder.parameters())},
        {"params": agent.norm.parameters()},
        {"params": agent.post_mamba_mlp.parameters()},
        {"params": list(agent.critic_trunk.parameters()) + list(agent.critic.parameters())},
        {"params": agent.actor.parameters()},
        {"params": agent.mamba.parameters(), "lr": args.mamba_lr},
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

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")

    # Rollout storage
    obs      = torch.zeros((args.num_steps, args.num_envs, OBS_DIM), device=device)
    actions  = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards  = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones    = torch.zeros((args.num_steps, args.num_envs), device=device)
    values   = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_time = time.time()
    ep_infos   = deque(maxlen=100)

    next_obs, _ = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    next_obs  = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    conv_state, ssm_state = agent.mamba.allocate_inference_cache(args.num_envs, max_seqlen=1)
    next_mamba_state = (conv_state, ssm_state)
    num_updates = args.total_timesteps // args.batch_size

    start_update = resume_update + 1
    for update in range(start_update, num_updates + 1):
        t0 = time.time()
        init_mamba = (next_mamba_state[0].clone(), next_mamba_state[1].clone())

        # Dashboard counters per update
        update_rooms = set()
        update_deaths = 0
        update_sword_found = 0
        update_guard_kills = 0

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for pg in optimizer.param_groups[:-1]:
                pg["lr"] = frac * args.learning_rate
            optimizer.param_groups[-1]["lr"] = frac * args.mamba_lr

        # ═══ ROLLOUT ═══
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step]  = next_obs
            dones[step] = next_done

            with torch.no_grad():
                act, lp, _, val, next_mamba_state = agent.get_action_and_value(
                    next_obs, next_mamba_state
                )
            values[step]   = val
            actions[step]  = act
            logprobs[step] = lp

            next_obs, rew, terminated, truncated, info = envs.step(act.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(rew, dtype=torch.float32).to(device)
            next_obs  = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # Reset Mamba state on episode boundaries
            for eid, d in enumerate(done):
                if d:
                    next_mamba_state[0][eid].zero_()
                    next_mamba_state[1][eid].zero_()

            # Collect PoP dashboard metrics from info
            room_arr = info.get("room")
            if room_arr is not None:
                for r in room_arr:
                    if r and int(r) > 0:
                        update_rooms.add(int(r))

            dead_arr = info.get("dead")
            if dead_arr is not None:
                update_deaths += int(np.sum(dead_arr))

            sf_arr = info.get("sword_found")
            if sf_arr is not None:
                update_sword_found += int(np.sum(np.array(sf_arr) == 1))

            gk_arr = info.get("guard_killed")
            if gk_arr is not None:
                update_guard_kills += int(np.sum(np.array(gk_arr) > 0))

            # Episode stats
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

        # ═══ GAE ═══
        with torch.no_grad():
            nv, _ = agent.get_value(next_obs, next_mamba_state)
            advantages, returns = compute_gae(
                rewards, values, dones, nv, next_done,
                args.gamma, args.gae_lambda, args.num_steps,
            )

        b_obs      = obs.reshape(-1, OBS_DIM)
        b_logprobs = logprobs.reshape(-1)
        b_actions  = actions.reshape(-1)
        b_adv      = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)
        b_values   = values.reshape(-1)

        # ═══ PPO UPDATE ═══
        envsperbatch = args.num_envs // args.num_minibatches
        envinds  = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs, losses_pg, losses_v, losses_ent = [], [], [], []
        losses_total, kl_list, grad_norms, mamba_grad_norms = [], [], [], []

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                mb_env = envinds[start : start + envsperbatch]
                mb_inds = flatinds[:, mb_env].ravel()

                mb_init = (init_mamba[0][mb_env].clone(), init_mamba[1][mb_env].clone())
                mb_obs  = obs[:, mb_env]

                seq_out = agent.forward_sequence(mb_obs, mb_init, dones=dones[:, mb_env])
                T, B, H = seq_out.shape
                flat_h  = seq_out.reshape(-1, H)

                logits = agent.actor(flat_h)
                probs  = Categorical(logits=logits)
                new_lp = probs.log_prob(b_actions[mb_inds].long())
                new_ent = probs.entropy()
                new_val = agent.critic(agent.critic_trunk(flat_h)).reshape(-1)

                logratio = new_lp - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_adv[mb_inds]
                if args.norm_adv:
                    mb_adv = mb_adv.reshape(args.num_steps, -1)
                    mb_adv = (mb_adv - mb_adv.mean(0)) / (mb_adv.std(0) + 1e-8)
                    mb_adv = mb_adv.reshape(-1)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                if args.clip_vloss:
                    v_unclip = (new_val - b_returns[mb_inds]) ** 2
                    v_clip   = b_values[mb_inds] + torch.clamp(
                        new_val - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclip, (v_clip - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_val - b_returns[mb_inds]) ** 2).mean()

                ent_loss = new_ent.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()

                total_gn = sum(
                    p.grad.data.norm(2).item() ** 2 for p in agent.parameters() if p.grad is not None
                ) ** 0.5
                mamba_gn = sum(
                    p.grad.data.norm(2).item() ** 2 for p in agent.mamba.parameters() if p.grad is not None
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

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ═══ LOGGING ═══
        sps = int((global_step - resume_global_step) / max(time.time() - start_time, 1.0))
        avg_ret = np.mean([e["r"] for e in ep_infos]) if ep_infos else 0.0
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        ev = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"upd {update:4d} | SPS {sps:5d} | ret {avg_ret:8.2f} | "
              f"pi {np.mean(losses_pg):.4f} | vf {np.mean(losses_v):.4f} | "
              f"ent {np.mean(losses_ent):.4f} | ev {ev:.4f}")

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/avg_return", avg_ret, global_step)
        writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/mamba_lr", optimizer.param_groups[-1]["lr"], global_step)
        writer.add_scalar("losses/total", np.mean(losses_total), global_step)
        writer.add_scalar("losses/policy", np.mean(losses_pg), global_step)
        writer.add_scalar("losses/value", np.mean(losses_v), global_step)
        writer.add_scalar("losses/entropy", np.mean(losses_ent), global_step)
        writer.add_scalar("losses/approx_kl", np.mean(kl_list), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_var", ev, global_step)
        writer.add_scalar("losses/grad_norm", np.mean(grad_norms), global_step)
        writer.add_scalar("losses/mamba_grad_norm", np.mean(mamba_grad_norms), global_step)

        # PoP dashboard
        writer.add_scalar("dashboard/rooms_visited", len(update_rooms), global_step)
        writer.add_scalar("dashboard/deaths", update_deaths, global_step)
        writer.add_scalar("dashboard/sword_found", update_sword_found, global_step)
        writer.add_scalar("dashboard/guard_kills", update_guard_kills, global_step)
        print(f"       Rooms({len(update_rooms)}): {sorted(update_rooms)} | "
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