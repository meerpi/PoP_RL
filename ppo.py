"""
ppo.py — PPO training for Prince of Persia worker policy.

Agent: multi-stream (grid CNN + vector MLP + goal encoder) with separate
actor/critic heads. Subgoal boundaries injected as done signals in GAE.
AsyncVectorEnv (spawn) gives each env its own libSDLPoP.so process.
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from clean_env import (
    PoPEnv, DummyManager, OBS_DIM, STACKED_DIM, N_STACK,
    NUM_CH, GROWS, GCOLS, GRID_FLAT, KID_DIM, G_DIM,
    SG_NAVIGATE, SG_PICKUP_SWORD, SG_FIGHT_GUARD, SG_HEAL,
    N_ACTIONS,
)


# ── Hyperparameters ──────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = "pop_ppo"
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project: str = "pop_rl"

    # PPO core
    total_timesteps: int = 10_000_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 16
    num_steps: int = 2048          # per env per rollout
    gamma: float = 0.992
    gae_lambda: float = 0.95
    num_minibatches: int = 8       # batch=16*2048=32768, mb=4096
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.08         # bumped from 0.05 to fight KL collapse
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    target_kl: float = 0.02        # early-stop minibatch loop if KL exceeds this
    norm_adv: bool = True

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 50           # iterations between saves
    resume_from: str = ""          # path to checkpoint to resume

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ── Observation slicing (now handled dynamically in Agent._encode) ──

N_SUBGOALS = 4


# ── Weight initialisation ───────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ── Agent ────────────────────────────────────────────────────────────

class Agent(nn.Module):
    """Multi-stream PPO actor-critic for PoPEnv (with N_STACK frame stacking)."""

    def __init__(self):
        super().__init__()

        # Grid encoder: N_STACK * NUM_CH channels stacked
        self.grid_enc = nn.Sequential(
            layer_init(nn.Conv2d(NUM_CH * N_STACK, 48, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(48, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        grid_out = 64 * GROWS * GCOLS  # 3840

        # Vector encoder: kid+guard from all N_STACK frames
        vec_in = (KID_DIM + G_DIM) * N_STACK
        self.vec_enc = nn.Sequential(
            layer_init(nn.Linear(vec_in, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
        )
        vec_out = 128

        goal_in = N_SUBGOALS + 1
        self.goal_enc = nn.Sequential(
            layer_init(nn.Linear(goal_in, 16)),
            nn.Tanh(),
        )
        goal_out = 16

        trunk_in = grid_out + vec_out + goal_out
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(trunk_in, 512)),
            nn.Tanh(),
        )
        trunk_out = 512

        self.actor = nn.Sequential(
            layer_init(nn.Linear(trunk_out, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, N_ACTIONS), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(trunk_out, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def _encode(self, obs, goals):
        B = obs.shape[0]
        # Unpack N_STACK frames: each frame is OBS_DIM wide
        grids = []
        vecs  = []
        for f in range(N_STACK):
            base = f * OBS_DIM
            grids.append(obs[:, base:base + GRID_FLAT])
            vecs.append(obs[:, base + GRID_FLAT:base + OBS_DIM])
        # Stack grids as channels: (B, N_STACK*NUM_CH, GROWS, GCOLS)
        grid_cat = torch.cat(grids, dim=1).reshape(B, NUM_CH * N_STACK, GROWS, GCOLS)
        # Concat all frame vectors: (B, N_STACK*(KID_DIM+G_DIM))
        vec_cat = torch.cat(vecs, dim=1)

        g_feat = self.grid_enc(grid_cat)
        v_feat = self.vec_enc(vec_cat)
        o_feat = self.goal_enc(goals)
        combined = torch.cat([g_feat, v_feat, o_feat], dim=1)
        return self.trunk(combined)

    def get_value(self, obs, goals):
        return self.critic(self._encode(obs, goals))

    def get_action_and_value(self, obs, goals, action=None):
        feat = self._encode(obs, goals)
        logits = self.actor(feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(feat)


# ── Goal vector construction ────────────────────────────────────────

def make_goal_vec(infos, n_envs, device):
    """Build (n_envs, 5) goal tensor from vectorized info dict."""
    goals = torch.zeros(n_envs, N_SUBGOALS + 1, device=device)
    sgs = infos.get("current_subgoal", np.zeros(n_envs, dtype=int))
    trs = infos.get("sg_target_room", np.full(n_envs, 2, dtype=int))
    for i in range(n_envs):
        sg = int(sgs[i]) if hasattr(sgs, '__getitem__') else int(sgs)
        tr = int(trs[i]) if hasattr(trs, '__getitem__') else int(trs)
        if 0 <= sg < N_SUBGOALS:
            goals[i, sg] = 1.0
        goals[i, N_SUBGOALS] = float(tr) / 24.0
    return goals


# ── Environment factory ─────────────────────────────────────────────

def make_env(seed, env_id):
    """Create a DummyManager-wrapped PoPEnv thunk for AsyncVectorEnv."""
    def thunk():
        env = PoPEnv(visual=False)
        dm = DummyManager(env)
        return dm
    return thunk


# ── Training loop ────────────────────────────────────────────────────

def train():
    args = Args()

    # Computed
    args.batch_size = args.num_envs * args.num_steps        # 8192
    args.minibatch_size = args.batch_size // args.num_minibatches  # 256
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    run_dir = os.path.join(os.path.abspath("runs"), run_name)
    os.makedirs(run_dir, exist_ok=True)

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=run_name, sync_tensorboard=True, save_code=True)

    writer = SummaryWriter(run_dir)
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # AsyncVectorEnv: each env in its own subprocess (libSDLPoP.so isolation)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.seed, i) for i in range(args.num_envs)],
        context="spawn",  # fork inherits parent X11/SDL state → stack smashing
    )

    # ── Agent ──
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration = 1
    global_step = 0

    # Resume from checkpoint
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"Resumed from {args.resume_from}, iteration {start_iteration}, global_step {global_step}")

    # ── Rollout storage ──
    obs_buf      = torch.zeros((args.num_steps, args.num_envs, STACKED_DIM), device=device)
    goal_buf     = torch.zeros((args.num_steps, args.num_envs, N_SUBGOALS + 1), device=device)
    action_buf   = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprob_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    reward_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf     = torch.zeros((args.num_steps, args.num_envs), device=device)
    value_buf    = torch.zeros((args.num_steps, args.num_envs), device=device)

    # ── Initial observations ──
    next_obs_np, next_infos = envs.reset(seed=args.seed)

    next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    next_goal = make_goal_vec(next_infos, args.num_envs, device)

    start_time = time.time()

    for iteration in range(start_iteration, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ══════════════════════════════════════════════════
        # Rollout phase
        # ══════════════════════════════════════════════════
        sg_achieved_count = 0
        sg_truncated_count = 0
        deaths = 0
        rollout_rooms = set()
        max_known = 0

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]  = next_obs
            goal_buf[step] = next_goal
            done_buf[step] = next_done

            # Action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, next_goal)
                value_buf[step] = value.flatten()
            action_buf[step]  = action
            logprob_buf[step] = logprob

            # Step all envs simultaneously (each in its own subprocess)
            next_obs_np, rewards, terminations, truncations, infos = envs.step(
                action.cpu().numpy())

            reward_buf[step] = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Subgoal boundary signals from vectorized infos.
            # DummyManager handles reset_subgoal() internally; the env continues.
            # But for GAE, subgoal boundaries are episode boundaries.
            sg_achieved = np.array(infos.get(
                "subgoal_achieved", np.zeros(args.num_envs, dtype=bool)), dtype=bool)
            worker_truncated = np.array(infos.get(
                "worker_truncated", np.zeros(args.num_envs, dtype=bool)), dtype=bool)
            sg_done = np.logical_or(sg_achieved, worker_truncated)
            game_done = np.logical_or(terminations, truncations)

            # Logging counts
            sg_achieved_count += int(sg_achieved.sum())
            sg_truncated_count += int(worker_truncated.sum())
            deaths += int(terminations.sum())
            rooms = infos.get("room", None)
            if rooms is not None:
                rollout_rooms.update(int(r) for r in rooms)
            known = infos.get("known_rooms", None)
            if known is not None:
                max_known = max(max_known, int(np.max(known)))

            # done = game end OR subgoal boundary (for GAE)
            next_done = torch.tensor(
                np.logical_or(game_done, sg_done),
                dtype=torch.float32, device=device)

            # AsyncVectorEnv auto-resets terminated/truncated envs.
            # next_obs_np already has post-reset obs for those envs.
            # For correct GAE bootstrap, we need the TERMINAL state's value,
            # not the new episode's. Use final_observation/final_info.
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_goal = make_goal_vec(infos, args.num_envs, device)

            # Overwrite next_obs/next_goal for terminated envs with their
            # final (pre-reset) observation and info, so the critic bootstrap
            # evaluates V(s_terminal) instead of V(s_new_episode).
            if "final_observation" in infos:
                for i in range(args.num_envs):
                    if game_done[i] and infos["final_observation"][i] is not None:
                        next_obs[i] = torch.tensor(
                            infos["final_observation"][i],
                            dtype=torch.float32, device=device)
                    if game_done[i] and "final_info" in infos and infos["final_info"][i] is not None:
                        fi = infos["final_info"][i]
                        sg = int(fi.get("current_subgoal", 0))
                        tr = int(fi.get("sg_target_room", 2))
                        next_goal[i] = 0.0
                        if 0 <= sg < N_SUBGOALS:
                            next_goal[i, sg] = 1.0
                        next_goal[i, N_SUBGOALS] = float(tr) / 24.0

        # ══════════════════════════════════════════════════
        # GAE computation
        # ══════════════════════════════════════════════════
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_goal).reshape(1, -1)
            advantages = torch.zeros_like(reward_buf)
            lastgaelam = torch.zeros(args.num_envs, device=device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = value_buf[t + 1]
                delta = reward_buf[t] + args.gamma * nextvalues.squeeze() * nextnonterminal - value_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            returns = advantages + value_buf

        # ══════════════════════════════════════════════════
        # Training phase
        # ══════════════════════════════════════════════════
        b_obs      = obs_buf.reshape(-1, STACKED_DIM)
        b_goals    = goal_buf.reshape(-1, N_SUBGOALS + 1)
        b_logprobs = logprob_buf.reshape(-1)
        b_actions  = action_buf.reshape(-1)
        b_adv      = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)
        b_values   = value_buf.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_goals[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                # Advantage normalisation per minibatch
                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ══════════════════════════════════════════════════
        # Logging
        # ══════════════════════════════════════════════════
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/deaths", deaths, global_step)
        writer.add_scalar("charts/sg_achieved", sg_achieved_count, global_step)
        writer.add_scalar("charts/sg_truncated", sg_truncated_count, global_step)
        writer.add_scalar("charts/known_rooms", max_known, global_step)

        print(f"iter {iteration:4d} | step {global_step:>8d} | SPS {sps:>4d} | "
              f"pg {pg_loss.item():.4f} | vl {v_loss.item():.4f} | "
              f"ent {entropy_loss.item():.3f} | kl {approx_kl.item():.4f} | "
              f"ev {explained_var:.3f} | sg+ {sg_achieved_count} sg- {sg_truncated_count} "
              f"deaths {deaths} | rooms {sorted(rollout_rooms)} | known {max_known}")

        # ══════════════════════════════════════════════════
        # Checkpointing
        # ══════════════════════════════════════════════════
        if iteration % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{iteration}.pt")
            torch.save({
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "iteration": iteration,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "iteration": args.num_iterations,
        "args": vars(args),
    }, os.path.join(args.ckpt_dir, "ckpt_final.pt"))

    envs.close()
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()