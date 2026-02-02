
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import gymnasium as gym
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


class Welford:
    """Online Welford algorithm for running mean and variance of intrinsic returns."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return np.sqrt(self.var + 1e-8)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PoP_env"))
from envs.PoP_env import PoPEnv
from wrappers.discrete_actions import NUM_ACTIONS


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = "pop_ppo"
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    track: bool = False
    wandb_project_name: str = "pop_ppo"
    wandb_entity: str = None

    total_timesteps: int = 150_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 24  # 24 parallel AsyncVectorEnv workers 24            # parallel environments (matches 16 CPU cores)
    num_steps: int = 2048         # rollout length per env (decision steps, not engine frames)
    anneal_lr: bool = True
    gamma: float = 0.9999
    gamma_int: float = 0.999          # discount for intrinsic reward stream (per frame, SMDP)
    gae_lambda: float = 0.97
    num_minibatches: int = 8      # minibatch_size = (16*2048)/8 = 4096
    update_epochs: int = 5
    norm_adv: bool = True
    clip_coef: float = 0.15
    clip_vloss: bool = True
    ent_coef: float = 0.05
    ent_coef_end: float = 0.003  # linear entropy coefficient annealing 0.035
    ent_coef_end: float = 0.003       # final entropy coef (linear anneal)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    checkpoint_interval: int = 25   # save every N iterations (0 = off)
    checkpoint_path: str = ""       # path to load checkpoint from
    eval_on_checkpoint: bool = True # record an eval mp4 on each checkpoint



REPEAT_CHOICES = [1, 2, 3, 4, 8, 13, 18]
N_REPEATS = len(REPEAT_CHOICES)

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


# CoordConv: concat row/col normalized meshes; Conv2d with dilation=2
class Agent(nn.Module):
    """CNN over the tile grid + MLP over the state vector.

    CoordConv: two extra channels (normalised row, col) so conv filters
    can distinguish position. Dilated conv (dilation=2) on the middle
    layer widens the effective RF to 5×5 without merging spatial cells.
    """

    GRID_CH = 12    # raw tile channels
    GRID_H, GRID_W = 5, 12
    COORD_IN = GRID_CH + 2  # +row_coord, +col_coord
    ROOM_EMB_DIM = 8
    ACTION_EMB_DIM = 4
    HIST_LEN = 5
    # state(29) + room_emb(8) + ah_emb(5*4=20) + rh(5) = 62
    STATE_DIM = 29 + ROOM_EMB_DIM + HIST_LEN * ACTION_EMB_DIM + HIST_LEN

    def __init__(self, action_space=None):
        super().__init__()
        n_repeats = int(action_space.nvec[1]) if action_space is not None else N_REPEATS

        # pre-compute normalised coordinate grids (registered as buffers
        # so they follow .to(device) automatically)
        row_coord = torch.linspace(0, 1, self.GRID_H).view(1, 1, self.GRID_H, 1).expand(1, 1, self.GRID_H, self.GRID_W)
        col_coord = torch.linspace(0, 1, self.GRID_W).view(1, 1, 1, self.GRID_W).expand(1, 1, self.GRID_H, self.GRID_W)
        self.register_buffer("row_coord", row_coord)
        self.register_buffer("col_coord", col_coord)

        # grid encoder: (14, 5, 12) → flat 256
        # layer 1: standard 3×3, preserves spatial dims
        self.grid_conv1 = layer_init(nn.Conv2d(self.COORD_IN, 32, kernel_size=3, padding=1))
        # layer 2: dilated 3×3 (effective 5×5 RF), padding=dilation keeps dims
        self.grid_conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2))
        self.grid_fc = layer_init(nn.Linear(64 * self.GRID_H * self.GRID_W, 256))

        # Embeddings for categorical inputs
        self.room_emb = nn.Embedding(25, self.ROOM_EMB_DIM)     # room IDs 0-24
        self.action_emb = nn.Embedding(NUM_ACTIONS, self.ACTION_EMB_DIM)  # action IDs 0-13

        # state encoder
        self.state_enc = nn.Sequential(
            layer_init(nn.Linear(self.STATE_DIM, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )

        # room table encoder: (24*13=312) → 64
        self.room_enc = nn.Sequential(
            layer_init(nn.Linear(24 * 13, 64)),
            nn.ReLU(),
        )

        feat_dim = 256 + 128 + 64 + 1  # +1 for have_sword
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(feat_dim, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, NUM_ACTIONS), std=0.01)
        self.repeat_head = layer_init(nn.Linear(256, n_repeats), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)
        self.critic_int = layer_init(nn.Linear(256, 1), std=1.0) layer_init(nn.Linear(256, 1), std=1.0)
        self.critic_int = layer_init(nn.Linear(256, 1), std=1.0)

    def _encode(self, obs):
        grid = obs["grid"].float()                 # already 0/1 uint8
        B = grid.shape[0]
        # CoordConv: concat normalised row/col channels
        grid = torch.cat([grid,
                          self.row_coord.expand(B, -1, -1, -1),
                          self.col_coord.expand(B, -1, -1, -1)], dim=1)
        g = torch.relu(self.grid_conv1(grid))
        g = torch.relu(self.grid_conv2(g))
        g = self.grid_fc(g.reshape(B, -1))
        g = torch.relu(g)

        state = obs["state"]
        room = self.room_emb(obs["room"].long().squeeze(-1))       # (B, 8)
        ah = self.action_emb(obs["action_history"].long())          # (B, 5, 4)
        ah = ah.reshape(B, -1)                                      # (B, 20)
        rh = obs["repeat_history"].float() / (N_REPEATS - 1)          # (B, 5) normalised to [0,1]
        vec = torch.cat([state, room, ah, rh], dim=-1)              # (B, 62)

        rt = obs["room_table"].reshape(B, -1)  # (B, 24*13)
        sword = obs["have_sword"].float()        # (B, 1)
        return self.fusion(torch.cat([g, self.state_enc(vec), self.room_enc(rt), sword], dim=-1))

    def get_value(self, obs):
        feat = self._encode(obs)
        return self.critic(feat), self.critic_int(feat)

    def get_action_and_value(self, obs, action=None):
        """FiGAR joint policy: logprob = log π_action + log π_repeat."""
        feat = self._encode(obs)
        act_dist = Categorical(logits=self.actor(feat))
        rep_dist = Categorical(logits=self.repeat_head(feat))
        if action is None:
            act = act_dist.sample()
            rep = rep_dist.sample()
        else:
            act = action[:, 0].long()
            rep = action[:, 1].long()
        return (torch.stack([act, rep], dim=1),
                act_dist.log_prob(act) + rep_dist.log_prob(rep),
                act_dist.entropy() + rep_dist.entropy(),
                self.critic(feat), self.critic_int(feat))


# ── Obs helpers ───────────────────────────────────────────────────────────────

def obs_to_torch(obs, device):
    out = {}
    for k, v in obs.items():
        t = torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
        if k in ("room", "action_history", "repeat_history"):
            t = t.float()
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        out[k] = t
    return out


def make_obs_buffer(num_steps, num_envs, device):
    return {
        "grid":           torch.zeros(num_steps, num_envs, 12, 5, 12, device=device),
        "state":          torch.zeros(num_steps, num_envs, 29, device=device),
        "room":           torch.zeros(num_steps, num_envs, 1, device=device),
        "action_history": torch.zeros(num_steps, num_envs, 5, device=device),
        "repeat_history": torch.zeros(num_steps, num_envs, 5, device=device),
        "room_table":     torch.zeros(num_steps, num_envs, 24, 13, device=device),
        "have_sword":     torch.zeros(num_steps, num_envs, 1, device=device),
    }


# ── Auto-eval on checkpoint ───────────────────────────────────────────────────

def _run_eval_episode(ckpt_path, out_mp4, max_steps=5000):
    """Spawned process: load checkpoint, play one episode, record mp4."""
    import subprocess
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_RENDER_DRIVER"] = "software"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        dev = torch.device("cpu")
        eval_agent = Agent().to(dev)
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        eval_agent.load_state_dict(state, assign=True)
        eval_agent.eval()

        env = PoPEnv(headless=False, max_steps=max_steps)
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", "320x200", "-pix_fmt", "rgb24", "-r", "15",
             "-i", "pipe:", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw_obs, _ = env.reset(seed=0)
        for _ in range(30):
            env.render()
        obs = {k: torch.from_numpy(v).unsqueeze(0).float() if isinstance(v, np.ndarray)
               else torch.tensor(v).unsqueeze(0).float()
               for k, v in raw_obs.items()}
        while True:
            proc.stdin.write(env.render().tobytes())
            with torch.no_grad():
                feat = eval_agent._encode(obs)
                act = eval_agent.actor(feat).argmax(dim=-1)
                rep = eval_agent.repeat_head(feat).argmax(dim=-1)
                action = torch.stack([act, rep], dim=1)
            raw_obs, _, terminated, truncated, _ = env.step(action[0].numpy())
            if terminated or truncated:
                break
            obs = {k: torch.from_numpy(v).unsqueeze(0).float() if isinstance(v, np.ndarray)
                   else torch.tensor(v).unsqueeze(0).float()
                   for k, v in raw_obs.items()}
        proc.stdin.close()
        proc.wait()
        env.obs_builder.release_held_action()
        print(f"  eval → {out_mp4}")
    except Exception as e:
        print(f"  [eval error] {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"pop__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    _PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
    writer = SummaryWriter(os.path.join(_PROJ_DIR, "runs", run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Multi-env via gymnasium AsyncVectorEnv
    env_fns = [lambda: PoPEnv(headless=True) for _ in range(args.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns, context="fork",
                                     autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    agent = Agent(envs.single_action_space).to(device)
    if device.type == "cuda":
        agent = torch.compile(agent, mode="reduce-overhead")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration = 1
    global_step = 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Loaded checkpoint: {args.checkpoint_path}  (iter={start_iteration-1}, gs={global_step})")

    # Pre-allocated rollout storage: (num_steps, num_envs, ...)
    obs_buf   = make_obs_buffer(args.num_steps, args.num_envs, device)
    actions   = torch.zeros(args.num_steps, args.num_envs, 2, dtype=torch.long, device=device)
    logprobs  = torch.zeros(args.num_steps, args.num_envs, device=device)
    rewards   = torch.zeros(args.num_steps, args.num_envs, device=device)
    rewards_int = torch.zeros(args.num_steps, args.num_envs, device=device)
    dones     = torch.zeros(args.num_steps, args.num_envs, device=device)
    values    = torch.zeros(args.num_steps, args.num_envs, device=device)
    values_int = torch.zeros(args.num_steps, args.num_envs, device=device)
    durations = torch.ones(args.num_steps, args.num_envs, device=device)

    runs_dir = os.path.join(_PROJ_DIR, "runs", run_name)
    os.makedirs(runs_dir, exist_ok=True)

    start_time = time.time()
    raw_obs, infos = envs.reset(seed=args.seed)
    next_obs = obs_to_torch(raw_obs, device)
    next_done = torch.zeros(args.num_envs, device=device)

    # Reward normalization: running return variance (Welford)
    ret_mean = 0.0
    ret_var = 1.0
    ret_count = 1e-4
    running_return = np.zeros(args.num_envs)

    # Persistent tracking across iterations
    last_infos = [{} for _ in range(args.num_envs)]
    ep_ret_sums = np.zeros(args.num_envs)
    ep_len_sums = np.zeros(args.num_envs, dtype=int)
    ep_rooms = [set() for _ in range(args.num_envs)]
    ep_sword = np.zeros(args.num_envs, dtype=bool)  # sword found this episode
    seen_rooms_global = set()
    _eval_proc = [None]

    for iteration in range(start_iteration, args.num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        ent_coef = args.ent_coef * frac + args.ent_coef_end * (1 - frac)

        # Per-rollout tracking
        ro_rooms = [set() for _ in range(args.num_envs)]
        ro_post_sword_rooms = [set() for _ in range(args.num_envs)]
        ro_events = [[] for _ in range(args.num_envs)]
        episode_returns = []
        episode_lengths = []
        episode_room_counts = []
        episode_swords = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            for k in obs_buf:
                obs_buf[k][step] = next_obs[k]
            dones[step] = next_done

            with torch.inference_mode():
                action, logprob, _, val_ext, val_int = agent.get_action_and_value(next_obs)
                values[step] = val_ext.squeeze(-1)
                values_int[step] = val_int.squeeze(-1)
            actions[step] = action
            logprobs[step] = logprob

            raw_obs, reward_np, terminated, truncated, infos = envs.step(action.cpu().numpy())

            # Extract intrinsic reward from info
            # Under SAME_STEP autoreset, terminal info is returned directly in infos
            rew_int_np = np.zeros(args.num_envs, dtype=np.float32)
            if "reward_int" in infos:
                rew_int_np[:] = infos["reward_int"]
            rew_ext_np = (reward_np - rew_int_np).astype(np.float32)

            # Reward normalization: update running return variance (Welford), SMDP-consistent
            tau_np = infos["frames_elapsed"] if "frames_elapsed" in infos else np.ones(args.num_envs)
            running_return = running_return * (args.gamma ** tau_np) + reward_np
            bm, bv, bc = np.mean(running_return), np.var(running_return), len(running_return)
            delta = bm - ret_mean
            tot = ret_count + bc
            ret_mean += delta * bc / tot
            ret_var = (ret_var * ret_count + bv * bc + delta**2 * ret_count * bc / tot) / tot
            ret_count = tot
            ret_std = np.sqrt(ret_var + 1e-8)
            rewards[step] = torch.from_numpy(np.clip(rew_ext_np / ret_std, -10.0, 10.0)).to(device)
            rewards_int[step] = torch.from_numpy(np.clip(rew_int_np / ret_std, -10.0, 10.0)).to(device)
            if "frames_elapsed" in infos:
                durations[step] = torch.tensor(infos["frames_elapsed"], dtype=torch.float32, device=device)
            done_np = terminated | truncated
            next_done = torch.from_numpy(done_np.astype(np.float32)).to(device)

            # Event tracking
            for i in range(args.num_envs):
                ep_ret_sums[i] += reward_np[i]
                ep_len_sums[i] += 1

                if "room" in infos:
                    rid = int(infos["room"][i])
                    ro_rooms[i].add(rid)
                    ep_rooms[i].add(rid)
                    if "have_sword" in infos and infos["have_sword"][i] > 0:
                        ro_post_sword_rooms[i].add(rid)
                    if rid not in seen_rooms_global:
                        seen_rooms_global.add(rid)
                        print(f"gs={global_step} env{i} NEW room={rid} total={len(seen_rooms_global)}")

                if "have_sword" in infos:
                    if infos["have_sword"][i] == 1 and last_infos[i].get("have_sword", 0) == 0:
                        ro_events[i].append("Sword")
                        ep_sword[i] = True

                if "guard_hp" in infos and "kid_sword_drawn" in infos:
                    ghp = infos["guard_hp"][i]
                    prev_ghp = last_infos[i].get("guard_hp", -1)
                    if infos["kid_sword_drawn"][i] == 1 and last_infos[i].get("kid_sword_drawn", 0) == 0:
                        ro_events[i].append("SwordDrawn")
                    if prev_ghp >= 0 and ghp >= 0 and ghp < prev_ghp:
                        ro_events[i].append("GuardHit")
                    if prev_ghp > 0 and ghp == 0:
                        ro_events[i].append("GuardDead")

                is_level_up = False
                if done_np[i] and "final_info" in infos:
                    final_info = infos["final_info"]
                    fi = None
                    if isinstance(final_info, dict):
                        fi = final_info.get(i)
                    elif isinstance(final_info, (list, tuple, np.ndarray)) and i < len(final_info):
                        fi = final_info[i]

                    if isinstance(fi, dict) and "level" in fi:
                        term_lvl = int(fi["level"])
                        if term_lvl > 1:
                            is_level_up = True
                            ro_events[i].append(f"LevelUp({term_lvl})")

                if not is_level_up and "level" in infos:
                    lvl = int(infos["level"][i])
                    prev_lvl = last_infos[i].get("level", 1)
                    if lvl > prev_lvl:
                        is_level_up = True
                        ro_events[i].append(f"LevelUp({lvl})")

                # Update last_infos
                for key in ("room", "have_sword", "guard_hp", "kid_sword_drawn", "hp", "level"):
                    if key in infos:
                        last_infos[i][key] = infos[key][i]

                if done_np[i]:
                    if not is_level_up:
                        ro_events[i].append("Death" if terminated[i] else "Truncated")
                    episode_returns.append(float(ep_ret_sums[i]))
                    episode_lengths.append(int(ep_len_sums[i]))
                    episode_room_counts.append(len(ep_rooms[i]))
                    episode_swords.append(bool(ep_sword[i]))
                    writer.add_scalar("charts/episodic_return", ep_ret_sums[i], global_step)
                    writer.add_scalar("charts/episodic_length", ep_len_sums[i], global_step)
                    writer.add_scalar("charts/episodic_rooms", len(ep_rooms[i]), global_step)
                    writer.add_scalar("charts/episodic_sword", float(ep_sword[i]), global_step)
                    ep_ret_sums[i] = 0
                    ep_len_sums[i] = 0
                    ep_rooms[i] = set()
                    ep_sword[i] = False
                    last_infos[i] = {}
                    running_return[i] = 0.0

            next_obs = obs_to_torch(raw_obs, device)

        # Rollout summary
        print(f"\n[rollout] gs={global_step}")
        for i in range(args.num_envs):
            ev_str = ", ".join(ro_events[i])
            if ro_rooms[i] or ev_str:
                ps = sorted(ro_post_sword_rooms[i])
                print(f"  env{i}: rooms={sorted(ro_rooms[i])}  post_sword={ps}  [{ev_str}]")

        # Bootstrap — dual GAE
        with torch.inference_mode():
            nxt_val_ext, nxt_val_int = agent.get_value(next_obs)
            nxt_val_ext = nxt_val_ext.squeeze(-1)
            nxt_val_int = nxt_val_int.squeeze(-1)

            # SMDP Dual GAE: gamma^tau for ext and gamma_int^tau for int streams: uses gamma^tau (SMDP-correct)
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                tau = durations[t]
                gamma_tau = args.gamma ** tau
                if t == args.num_steps - 1:
                    nxt_nonterminal = 1.0 - next_done
                    nxt_val = nxt_val_ext
                else:
                    nxt_nonterminal = 1.0 - dones[t + 1]
                    nxt_val = values[t + 1]
                delta = rewards[t] + gamma_tau * nxt_val * nxt_nonterminal - values[t]
                lastgaelam = delta + gamma_tau * args.gae_lambda * nxt_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

            # Intrinsic GAE: uses gamma_int^tau (SMDP)
            advantages_int = torch.zeros_like(rewards_int)
            lastgaelam_int = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                tau = durations[t]
                gamma_tau_int = args.gamma_int ** tau
                if t == args.num_steps - 1:
                    nxt_nonterminal = 1.0 - next_done
                    nxt_val = nxt_val_int
                else:
                    nxt_nonterminal = 1.0 - dones[t + 1]
                    nxt_val = values_int[t + 1]
                delta = rewards_int[t] + gamma_tau_int * nxt_val * nxt_nonterminal - values_int[t]
                lastgaelam_int = delta + gamma_tau_int * args.gae_lambda * nxt_nonterminal * lastgaelam_int
                advantages_int[t] = lastgaelam_int
            returns_int = advantages_int + values_int

        # Flatten (num_steps, num_envs, ...) → (batch_size, ...)
        b_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs_buf.items()}
        b_actions  = actions.reshape(-1, 2)       # (batch, 2): [action_id, repeat_idx]
        b_logprobs = logprobs.reshape(-1)
        b_advs     = (advantages + advantages_int).reshape(-1)  # combined advantage
        b_returns  = returns.reshape(-1)
        b_returns_int = returns_int.reshape(-1)
        b_values   = values.reshape(-1)
        b_values_int = values_int.reshape(-1)

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                mb_obs = {k: v[mb] for k, v in b_obs.items()}

                _, newlogprob, entropy, newvalue_ext, newvalue_int = agent.get_action_and_value(mb_obs, b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advs[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                # Extrinsic value loss
                newvalue_ext = newvalue_ext.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue_ext - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(newvalue_ext - b_values[mb], -args.clip_coef, args.clip_coef)
                    v_loss_ext = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss_ext = 0.5 * ((newvalue_ext - b_returns[mb]) ** 2).mean()

                # Intrinsic value loss
                newvalue_int = newvalue_int.view(-1)
                v_loss_int = 0.5 * ((newvalue_int - b_returns_int[mb]) ** 2).mean()

                v_loss = v_loss_ext + v_loss_int
                loss = pg_loss - ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        elapsed = int(time.time() - start_time)

        # Tensorboard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/rooms_discovered", len(seen_rooms_global), global_step)
        if episode_swords:
            writer.add_scalar("charts/sword_rate", np.mean(episode_swords), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # Console summary
        print("-" * 50)
        print(f"iter={iteration}  gs={global_step}  SPS={sps}  elapsed={elapsed}s")
        ret_std = np.sqrt(ret_var + 1e-8)
        print(f"  pg={pg_loss.item():.4f}  vf={v_loss.item():.4f}  ent={entropy.mean().item():.4f}"
              f"  kl={approx_kl.item():.4f}  clip={np.mean(clipfracs):.3f}  ev={explained_var:.4f}"
              f"  ret_std={ret_std:.2f}")
        if episode_returns:
            sword_rate = np.mean(episode_swords) if episode_swords else 0.0
            print(f"  ep_ret={np.mean(episode_returns):.2f} (n={len(episode_returns)})"
                  f"  ep_len={np.mean(episode_lengths):.0f}"
                  f"  rooms_max={max(episode_room_counts)}"
                  f"  sword={sword_rate:.2f}")

        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(runs_dir, f"ckpt_{iteration}.pt")
            torch.save({"iteration": iteration, "global_step": global_step,
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
            print(f"  ckpt → {ckpt_path}")
            if args.eval_on_checkpoint:
                import multiprocessing as mp
                if _eval_proc[0] is not None and _eval_proc[0].is_alive():
                    print("  [eval skipped] previous eval still running")
                else:
                    out_mp4 = os.path.join(runs_dir, f"eval_iter{iteration:06d}.mp4")
                    p = mp.get_context("spawn").Process(
                        target=_run_eval_episode, args=(ckpt_path, out_mp4), daemon=True)
                    p.start()
                    _eval_proc[0] = p

    envs.close()
    writer.close()