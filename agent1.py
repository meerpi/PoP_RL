"""PPO Agent for Prince of Persia Grid Environment"""

class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM): x * (1 + gamma) + beta."""
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.film_gen = layer_init(nn.Linear(cond_dim, feature_dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_gen(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import tyro
from torch.distributions.categorical import Categorical

import env1
from env1 import REPEAT_CHOICES, N_REPEATS



class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean, batch_var, batch_count = np.mean(x), np.var(x), len(x)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a, m_b = self.var * self.count, batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.count = tot



import subprocess


def _eval_video_worker(seed, video_path, model_state, max_steps=2000, start_room=None, start_pos=0, warmup_steps=3, result_queue=None):
    """Isolated subprocess worker for evaluation video rendering."""
    thunk = make_env(visual_mode=False, max_steps=max_steps, warmup_steps=warmup_steps, start_room=start_room, start_pos=start_pos)
    e = thunk()

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", "320x200", "-pix_fmt", "rgb24", "-r", "15",
        "-i", "pipe:", "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    obs_raw, _ = e.reset(seed=seed)
    def _obs(r):
        return {
            "grid": torch.from_numpy(r["grid"]).float().unsqueeze(0),
            "state": torch.from_numpy(r["state"]).float().unsqueeze(0),
            "room": torch.from_numpy(r["room"]).unsqueeze(0),
            "action_history": torch.from_numpy(r["action_history"]).unsqueeze(0),
            "repeat_history": torch.from_numpy(r["repeat_history"]).unsqueeze(0),
            "graph": torch.from_numpy(r["graph"]).unsqueeze(0),
            "subgoal_hops": torch.from_numpy(r["subgoal_hops"]).float().unsqueeze(0),
        }

    agent = Agent(e)
    agent.load_state_dict(model_state)
    agent.eval()

    tot_rew, steps = 0.0, 0
    obs = _obs(obs_raw)
    done = False

    while not done and steps < max_steps:
        frame = e.render()
        proc.stdin.write(frame.tobytes())
        with torch.no_grad():
            act, *_ = agent.get_action_and_value(obs)
        raw, rew, term, trunc, _ = e.step(act[0].numpy())
        done = term or trunc
        tot_rew += rew
        steps += 1
        obs = _obs(raw)

    proc.stdin.close()
    proc.wait()
    e.close()
    if result_queue is not None:
        result_queue.put((tot_rew, steps))


def run_eval_video(agent, iteration, video_path, args):
    """Launch isolated subprocess to render an evaluation episode video."""
    raw_state = agent.state_dict()
    clean_state = {}
    for k, v in raw_state.items():
        clean_k = k.replace("_orig_mod.", "")
        clean_state[clean_k] = v.cpu()

    q = mp.Queue()
    p = mp.Process(
        target=_eval_video_worker,
        args=(args.seed + iteration, video_path, clean_state, args.max_episode_steps, args.start_room, args.start_pos, args.framestack_warmup, q)
    )
    p.start()
    p.join(timeout=180)
    if not q.empty():
        tot_rew, steps = q.get()
        print(f"  [EVAL VIDEO] Iter {iteration}: ret={tot_rew:.2f} steps={steps} → {video_path}")
    else:
        print(f"  [EVAL VIDEO] Iter {iteration}: worker finished → {video_path}")



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 32
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "principia"
    wandb_entity: str | None = None
    visual: bool = False

    env_id: str = "PoP_Grid"
    total_timesteps: int = 1_000_000_000
    num_envs: int = 16
    num_steps: int = 2048
    max_episode_steps: int = 10000
    framestack_warmup: int = 3
    start_room: int | None = None
    start_pos: int = 0

    learning_rate: float = 1.7e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 2
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.9
    clip_vloss: bool = True
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    log_interval: int = 1
    episode_log_interval: int = 10
    checkpoint_interval: int = 10
    eval_interval: int = 50
    checkpoint_path: str = ""
    eval_only: bool = False
    eval_episodes: int = 1

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(visual_mode, max_steps, warmup_steps, start_room=None, start_pos=0, speed_multiplier=1):
    """Factory that returns a thunk creating a wrapped PoPEnv."""
    def thunk():
        e = env1.PoPEnv(headless=not visual_mode, visual_mode=visual_mode,
                        max_steps=max_steps, start_room=start_room, start_pos=start_pos)
        if visual_mode and speed_multiplier > 1:
            e.set_speed(speed_multiplier)
        e = env1.FrameStackWrapper(e, n_frames=5, warmup_steps=warmup_steps)
        e = gym.wrappers.RecordEpisodeStatistics(e)
        return e
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight init with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO actor-critic: CNN on grid + MLP on state + action history embedding."""
    def __init__(self, envs):
        super().__init__()
        action_space = getattr(envs, "single_action_space", getattr(envs, "action_space", None))
        n_actions = int(action_space.nvec[0])  # action dim; nvec[1] is repeat dim

        self.grid_net = nn.Sequential(
            layer_init(nn.Conv2d(60, 32, kernel_size=3, stride=1, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),  # output is 3x10
            nn.Flatten(),
            layer_init(nn.Linear(64 * 3 * 10, 128)), nn.ReLU(),
        )

        self.state_net = nn.Sequential(
            layer_init(nn.Linear(28, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 64)), nn.ReLU(),
        )

        self.action_emb = nn.Embedding(n_actions, 8)
        self.action_net = nn.Sequential(layer_init(nn.Linear(5 * 8, 32)), nn.ReLU())

        n_repeats = int(action_space.nvec[1])
        self.repeat_emb = nn.Embedding(n_repeats, 8)
        self.repeat_net = nn.Sequential(layer_init(nn.Linear(5 * 8, 16)), nn.ReLU())

        self.room_emb = nn.Embedding(25, 8)  # rooms 0..24, dim 8
        self.room_net = nn.Sequential(layer_init(nn.Linear(8, 16)), nn.ReLU())

        self.graph_net = nn.Sequential(layer_init(nn.Linear(96 * 20, 32)), nn.ReLU())

        self.extra_layer = nn.Sequential(layer_init(nn.Linear(289, 192), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(192, 128)), nn.ReLU(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        )
        # FiGAR repeat head: same feature vector, |W|=N_REPEATS outputs.
        # Paper (Sharma 2017 Alg.1): fθx outputs |W|-dim distribution over repetition counts.
        # Entropy regularized with the same ent_coef as the action head (paper §5.1).
        self.repeat_head = layer_init(nn.Linear(192, N_REPEATS), std=0.01)
        self.critic = layer_init(nn.Linear(192, 1), std=0.01)

    def _features(self, x):
        """Concatenate grid, state, room embedding, action-history, graph, and subgoal_hops features."""
        g = self.grid_net(x["grid"].float())
        s = self.state_net(x["state"].float())
        r = self.room_net(self.room_emb(x["room"].long().squeeze(-1)))
        a = self.action_emb(x["action_history"].long()).view(x["action_history"].shape[0], -1)
        a = self.action_net(a)
        rp = self.repeat_emb(x["repeat_history"].long()).view(x["repeat_history"].shape[0], -1)
        rp = self.repeat_net(rp)

        gr = x["graph"].long()
        src_emb = self.room_emb(gr[:, 0, :])
        dst_emb = self.room_emb(gr[:, 1, :])
        flags = x["graph"][:, 2:6, :].float().permute(0, 2, 1)
        gr_in = torch.cat([src_emb, dst_emb, flags], dim=2).view(x["graph"].shape[0], -1)
        gr_feat = self.graph_net(gr_in)

        hops = x["subgoal_hops"].float()

        return self.extra_layer(torch.cat([g, s, r, a, rp, gr_feat, hops], dim=1))

    def get_value(self, x):
        """Critic forward pass."""
        return self.critic(self._features(x))

    def get_action_and_value(self, x, action=None):
        """FiGAR joint policy (Sharma 2017 eq.2):

        L(θa, θx) = (log πa(a|s) + log πx(x|s)) · A(s, a, x)

        Both heads share the same advantage, so the joint logprob is the sum.
        Entropy is summed over both heads (paper §5.1: same β for both).
        """
        f = self._features(x)
        action_dist = Categorical(logits=self.actor(f))
        repeat_dist = Categorical(logits=self.repeat_head(f))
        if action is None:
            act   = action_dist.sample()
            k_idx = repeat_dist.sample()
        else:
            act, k_idx = action[:, 0].long(), action[:, 1].long()
        # Joint logprob: sum of individual log-probs (factored policy, paper eq.2)
        logprob = action_dist.log_prob(act) + repeat_dist.log_prob(k_idx)
        # Entropy: sum over both heads (paper: same entropy coeff for both)
        entropy = action_dist.entropy() + repeat_dist.entropy()
        return torch.stack([act, k_idx], dim=1), logprob, entropy, self.critic(f)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs", run_name)
    os.makedirs(runs_dir, exist_ok=True)

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mp.set_start_method("spawn", force=True)
    env_fns = [make_env(args.visual, args.max_episode_steps, args.framestack_warmup,
                        args.start_room, args.start_pos,
                        speed_multiplier=2 if args.visual else 1) for _ in range(args.num_envs)]
    if args.num_envs == 1:
        envs = gym.vector.SyncVectorEnv(env_fns)
    else:
        envs = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

    agent = Agent(envs).to(device)
    if device.type == "cuda":
        try:
            agent = torch.compile(agent, mode="reduce-overhead")
        except Exception:
            agent = torch.compile(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        if not args.eval_only and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Storage
    grid_shape = envs.single_observation_space["grid"].shape
    state_shape = envs.single_observation_space["state"].shape
    act_hist_shape = envs.single_observation_space["action_history"].shape
    graph_shape = envs.single_observation_space["graph"].shape
    hops_shape = envs.single_observation_space["subgoal_hops"].shape

    obs_grid = torch.zeros((args.num_steps, args.num_envs) + grid_shape, dtype=torch.uint8, device=device)
    obs_state = torch.zeros((args.num_steps, args.num_envs) + state_shape, device=device)
    obs_room = torch.zeros((args.num_steps, args.num_envs, 1), dtype=torch.int32, device=device)
    obs_act_hist = torch.zeros((args.num_steps, args.num_envs) + act_hist_shape, dtype=torch.int32, device=device)
    obs_rep_hist = torch.zeros((args.num_steps, args.num_envs) + act_hist_shape, dtype=torch.int32, device=device)
    obs_graph = torch.zeros((args.num_steps, args.num_envs) + graph_shape, dtype=torch.int32, device=device)
    obs_hops = torch.zeros((args.num_steps, args.num_envs) + hops_shape, dtype=torch.float32, device=device)

    actions   = torch.zeros((args.num_steps, args.num_envs, 2), dtype=torch.long, device=device)  # [act_id, k_idx]
    logprobs  = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards   = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones     = torch.zeros((args.num_steps, args.num_envs), device=device)
    values    = torch.zeros((args.num_steps, args.num_envs), device=device)
    # SMDP: store actual tick counts per step for γ^k discount in GAE.
    # Default 4 (middle of REPEAT_CHOICES [1, 2, 4, 8, 12]) so Stage-1 fixed-k is trivially valid.
    durations = torch.full((args.num_steps, args.num_envs), 4.0, device=device)

    episode_returns, episode_lengths = deque(maxlen=100), deque(maxlen=100)
    ep_ret_sums = np.zeros(args.num_envs, dtype=np.float32)
    ep_len_sums = np.zeros(args.num_envs, dtype=np.int32)
    ep_count = 0
    ret_rms = RunningMeanStd()
    running_return = np.zeros(args.num_envs, dtype=np.float64)
    metrics_keys = ["visited_rooms_count", "visited_tiles_count", "level", "room", "have_sword", "guard_hp", "guard_hp_max", "kid_sword_drawn"]
    last_infos = [dict() for _ in range(args.num_envs)]
    seen_rooms = set()  # globally track all room IDs ever visited

    global_step = 0
    start_time = time.time()

    next_obs_raw, _ = envs.reset(seed=args.seed)
    def _obs(raw):
        return {
            "grid": torch.from_numpy(raw["grid"]).to(device, non_blocking=True),
            "state": torch.from_numpy(raw["state"]).to(device, non_blocking=True),
            "room": torch.from_numpy(raw["room"]).to(device, non_blocking=True),
            "action_history": torch.from_numpy(raw["action_history"]).to(device, non_blocking=True),
            "repeat_history": torch.from_numpy(raw["repeat_history"]).to(device, non_blocking=True),
            "graph": torch.from_numpy(raw["graph"]).to(device, non_blocking=True),
            "subgoal_hops": torch.from_numpy(raw["subgoal_hops"]).to(device, non_blocking=True),
        }

    next_obs = _obs(next_obs_raw)
    next_done = torch.zeros(args.num_envs, device=device)

    if args.eval_only:
        eval_returns = np.zeros(args.num_envs, dtype=np.float32)
        eval_lengths = np.zeros(args.num_envs, dtype=np.int32)
        done_count = 0
        while done_count < args.eval_episodes:
            with torch.no_grad():
                a, *_ = agent.get_action_and_value(next_obs)
            raw, rew, term, trunc, _ = envs.step(a.cpu().numpy())
            d = np.logical_or(term, trunc)
            eval_returns += rew
            eval_lengths += 1
            next_obs = _obs(raw)
            for idx in np.where(d)[0]:
                done_count += 1
                print(f"eval ep_return={eval_returns[idx]:.2f} ep_length={eval_lengths[idx]}")
                eval_returns[idx] = 0.0
                eval_lengths[idx] = 0
                if done_count >= args.eval_episodes:
                    break
        envs.close()
        raise SystemExit(0)

    _state = {"envs": envs, "next_obs": next_obs, "next_done": next_done}

    def rebuild_envs():
        """Tear down dead subprocesses and spin up fresh envs."""
        print("  [CRASH DETECTED] Rebuilding environments... (this may take a moment)")
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                _state["envs"].close()
            except Exception:
                pass
        
        env_fns_new = [make_env(args.visual, args.max_episode_steps, args.framestack_warmup,
                                args.start_room, args.start_pos) for _ in range(args.num_envs)]
        if args.num_envs == 1:
            _state["envs"] = gym.vector.SyncVectorEnv(env_fns_new)
        else:
            _state["envs"] = gym.vector.AsyncVectorEnv(env_fns_new, context="spawn")
        
        raw, _ = _state["envs"].reset(seed=args.seed)
        _state["next_obs"] = _obs(raw)
        _state["next_done"] = torch.zeros(args.num_envs, device=device)
        print("  [RECOVERY COMPLETE] Resuming training.")

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        envs = _state["envs"]
        ro_rooms = [set() for _ in range(args.num_envs)]
        ro_post_sword_rooms = [set() for _ in range(args.num_envs)]
        ro_events = [[] for _ in range(args.num_envs)]

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_grid[step] = next_obs["grid"].to(torch.uint8)
            obs_state[step] = next_obs["state"]
            obs_room[step] = next_obs["room"]
            obs_act_hist[step] = next_obs["action_history"]
            obs_rep_hist[step] = next_obs["repeat_history"]
            obs_graph[step] = next_obs["graph"]
            obs_hops[step] = next_obs["subgoal_hops"]
            dones[step] = next_done

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                values[step] = agent.get_value(next_obs).flatten()
                action, logprob, *_ = agent.get_action_and_value(next_obs)

            actions[step]  = action          # shape [num_envs, 2]
            logprobs[step] = logprob

            try:
                raw, rew, term, trunc, infos = envs.step(action.cpu().numpy())
            except (EOFError, ConnectionResetError, BrokenPipeError):
                print(f"  [CRASH] subprocess died at gs={global_step}, rebuilding...")
                rebuild_envs()
                envs = _state["envs"]
                next_obs = _state["next_obs"]
                next_done = _state["next_done"]
                rewards[step] = torch.zeros(args.num_envs, device=device)
                continue

            done_np = np.logical_or(term, trunc)
            ep_ret_sums += rew  # raw reward kept for logging
            running_return = running_return * args.gamma + rew
            ret_rms.update(running_return)
            norm_rew = np.clip(rew / np.sqrt(ret_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
            rewards[step].copy_(torch.from_numpy(norm_rew), non_blocking=True)
            running_return[done_np] = 0.0
            next_obs = _obs(raw)
            next_done = torch.from_numpy(done_np.astype(np.float32)).to(device, non_blocking=True)

            # Store actual tick counts for SMDP γ^k discount in GAE (FiGAR paper §4.2 critic target)
            if "frames_elapsed" in infos:
                fe = np.array(infos["frames_elapsed"], dtype=np.float32)
                durations[step].copy_(torch.from_numpy(fe), non_blocking=True)

            # Track room visits
            if "room" in infos:
                r_vals = infos["room"]
                s_vals = infos.get("have_sword", [0] * args.num_envs)
                for i in range(args.num_envs):
                    if r_vals[i] is not None:
                        rid = int(r_vals[i])
                        ro_rooms[i].add(rid)
                        if s_vals[i] > 0:
                            ro_post_sword_rooms[i].add(rid)
                            prev_r = last_infos[i].get("room")
                            prev_s = last_infos[i].get("have_sword", 0)
                            if rid == 3 and (prev_r != 3 or prev_s == 0):
                                ro_events[i].append("R3WithSword")

            if "have_sword" in infos:
                s_vals = infos["have_sword"]
                for i in range(args.num_envs):
                    # Check if sword status changed from 0 to 1
                    # Note: last_infos stores the state from the PREVIOUS step
                    prev_s = last_infos[i].get("have_sword", 0)
                    if s_vals[i] == 1 and prev_s == 0:
                        ro_events[i].append("Sword")

            if "level" in infos:
                l_vals = infos["level"]
                for i in range(args.num_envs):
                    prev_l = last_infos[i].get("level", 1)
                    if l_vals[i] > prev_l:
                        ro_events[i].append(f"Lvl {prev_l}->{l_vals[i]}")


            if "guard_hp" in infos and "kid_sword_drawn" in infos:
                for i in range(args.num_envs):
                    ghp = infos["guard_hp"][i]
                    prev_ghp = last_infos[i].get("guard_hp", -1)
                    drawn = infos["kid_sword_drawn"][i]
                    prev_drawn = last_infos[i].get("kid_sword_drawn", 0)
                    if drawn == 1 and prev_drawn == 0:
                        ro_events[i].append("SwordDrawn")
                    if prev_ghp >= 0 and ghp >= 0 and ghp < prev_ghp:
                        ro_events[i].append("GuardHit")
                    if prev_ghp > 0 and ghp == 0:
                        ro_events[i].append("GuardDead")

            if np.any(done_np):
                for i in np.where(done_np)[0]:
                    has_sword_or_level = any("Sword" in ev or "Lvl" in ev or "Guard" in ev for ev in ro_events[i])
                    if has_sword_or_level:
                        ro_events[i].append("Done")

            for key in metrics_keys:
                if key in infos:
                    vals = infos[key]
                    if vals is not None:
                        for idx in range(args.num_envs):
                            if vals[idx] is not None:
                                last_infos[idx][key] = vals[idx]

            for idx in range(args.num_envs):
                rid = last_infos[idx].get("room")
                if rid is not None and rid not in seen_rooms:
                    seen_rooms.add(rid)
                    tc = last_infos[idx].get("visited_tiles_count", 0)
                    print(f"gs={global_step} env{idx} NEW room_id={rid} tiles={tc} total_rooms={len(seen_rooms)}")

            # ep_ret_sums already updated above at reward normalization step
            ep_len_sums += 1
            if np.any(done_np):
                for idx in np.where(done_np)[0]:
                    er, el = float(ep_ret_sums[idx]), int(ep_len_sums[idx])
                    episode_returns.append(er)
                    episode_lengths.append(el)
                    ep_count += 1
                    if "final_info" in infos and infos["final_info"][idx] is not None:
                        for key in metrics_keys:
                            if key in infos["final_info"][idx]:
                                last_infos[idx][key] = infos["final_info"][idx][key]
                    ep_ret_sums[idx] = 0.0
                    ep_len_sums[idx] = 0

        print(f"\n[Rollout Log] gs={global_step}")
        for i in range(args.num_envs):
            r_list = sorted(list(ro_rooms[i]))
            ps_list = sorted(list(ro_post_sword_rooms[i]))
            ev_str = ", ".join(ro_events[i])
            if len(r_list) > 0 or len(ev_str) > 0:
                print(f"  Env {i}: Rooms {r_list} | PostSword: {ps_list} | Events: [{ev_str}]")

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # SMDP-corrected GAE (FiGAR paper §4.2, TempoRL eq.4):
                # V̂(sj) = Σ γ^{y_{k-j}} r_k + γ^{y_{n-j}} V(sn)
                # In GAE form: discount per step = γ^k where k = actual ticks elapsed.
                dk = args.gamma ** durations[t]   # shape [num_envs]
                delta = rewards[t] + dk * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + dk * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        b_grid     = obs_grid.reshape((-1,) + grid_shape)
        b_state    = obs_state.reshape((-1,) + state_shape)
        b_room     = obs_room.reshape(-1, 1)
        b_act_hist = obs_act_hist.reshape((-1,) + act_hist_shape)
        b_rep_hist = obs_rep_hist.reshape((-1,) + act_hist_shape)
        b_graph    = obs_graph.reshape((-1,) + graph_shape)
        b_hops     = obs_hops.reshape((-1,) + hops_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions  = actions.reshape(-1, 2)   # [batch, 2]: [:, 0]=act_id, [:, 1]=k_idx
        b_advantages = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)
        b_values   = values.reshape(-1)
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                mb_obs = {
                    "grid": b_grid[mb].float(),
                    "state": b_state[mb],
                    "room": b_room[mb],
                    "action_history": b_act_hist[mb],
                    "repeat_history": b_rep_hist[mb],
                    "graph": b_graph[mb],
                    "subgoal_hops": b_hops[mb],
                }
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, newlp, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb].long())
                    ratio = (newlp - b_logprobs[mb]).exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (newlp - b_logprobs[mb])).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    mb_adv = b_advantages[mb]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_unclip = (newvalue - b_returns[mb]) ** 2
                        v_clip = b_values[mb] + (newvalue - b_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                        v_loss = 0.5 * torch.max(v_unclip, (v_clip - b_returns[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        fps = int(global_step / (time.time() - start_time))

        if args.log_interval > 0 and (iteration % args.log_interval == 0 or iteration == 1):
            print("-" * 50)
            print(f"iter={iteration}  gs={global_step}  fps={fps}  elapsed={int(time.time()-start_time)}s")
            print(f"  pg_loss={pg_loss.item():.4f}  v_loss={v_loss.item():.4f}  ent={entropy.mean().item():.4f}")
            print(f"  kl={approx_kl.item():.4f}  clipfrac={np.mean(clipfracs):.4f}  expl_var={explained_var:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            if episode_returns:
                print(f"  ep_ret_mean={np.mean(episode_returns):.2f}  ep_len_mean={np.mean(episode_lengths):.0f}")

        if iteration % args.checkpoint_interval == 0:
            path = os.path.join(runs_dir, f"ckpt_{iteration}.pt")
            torch.save({"iteration": iteration, "global_step": global_step,
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, path)
            print(f"  ckpt → {path}")

        if args.eval_interval > 0 and (iteration % args.eval_interval == 0 or iteration == 1):
            videos_dir = os.path.join(runs_dir, "videos")
            os.makedirs(videos_dir, exist_ok=True)
            vpath = os.path.join(videos_dir, f"eval_iter_{iteration}.mp4")
            run_eval_video(agent, iteration, vpath, args)

    envs.close()
