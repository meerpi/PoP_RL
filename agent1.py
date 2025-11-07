"""PPO + FiGAR agent for Prince of Persia."""
import csv
import contextlib
import io
import os
import random
import subprocess
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


# --- reward normalisation ---------------------------------------------------

class RunningMeanStd:
    def dormant_fractions(model: nn.Module, threshold: float = 0.01) -> float:
    """Compute percentage of dormant neurons (activation variance < threshold)."""
    dormant_count = 0
    total_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and hasattr(module, "last_act"):
            var = module.last_act.var(dim=0)
            dormant_count += (var < threshold).sum().item()
            total_count += var.numel()
    return dormant_count / max(1, total_count)

def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = np.mean(x), np.var(x), len(x)
        delta      = bm - self.mean
        tot        = self.count + bc
        self.mean += delta * bc / tot
        self.var   = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot


# --- eval video (isolated subprocess so SDL doesn't conflict) ---------------

def _eval_video_worker(seed, video_path, model_state, max_steps=2000,
                       start_room=None, start_pos=0, warmup_steps=3,
                       result_queue=None):
    e = make_env(visual_mode=False, max_steps=max_steps, warmup_steps=warmup_steps,
                 start_room=start_room, start_pos=start_pos)()

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", "320x200", "-pix_fmt", "rgb24", "-r", "15",
         "-i", "pipe:", "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    def _t(r):
        return {k: torch.from_numpy(v).float().unsqueeze(0) if k != "room"
                else torch.from_numpy(v).unsqueeze(0)
                for k, v in r.items()}

    agent = Agent(e)
    agent.load_state_dict(model_state)
    agent.eval()

    obs, _ = e.reset(seed=seed)
    obs, done, tot_rew, steps = _t(obs), False, 0.0, 0

    while not done and steps < max_steps:
        proc.stdin.write(e.render().tobytes())
        with torch.no_grad():
            act, *_ = agent.get_action_and_value(obs)
        raw, rew, term, trunc, _ = e.step(act[0].numpy())
        done = term or trunc
        tot_rew += rew
        steps += 1
        obs = _t(raw)

    proc.stdin.close()
    proc.wait()
    e.close()
    if result_queue is not None:
        result_queue.put((tot_rew, steps))


def run_eval_video(agent, iteration, video_path, args):
    state = {k.replace("_orig_mod.", ""): v.cpu() for k, v in agent.state_dict().items()}
    q = mp.Queue()
    p = mp.Process(target=_eval_video_worker,
                   args=(args.seed + iteration, video_path, state,
                         args.max_episode_steps, args.start_room, args.start_pos,
                         args.framestack_warmup, q))
    p.start()
    p.join(timeout=180)
    if p.is_alive():
        p.terminate()
        p.join()
    if not q.empty():
        ret, steps = q.get()
        print(f"  [eval] iter={iteration}  ret={ret:.2f}  steps={steps}  → {video_path}")
    else:
        print(f"  [eval] iter={iteration}  → {video_path}")


# --- hyperparameters --------------------------------------------------------

@dataclass
class Args:
    exp_name:   str = os.path.basename(__file__)[:-len(".py")]
    seed:       int = 33
    torch_deterministic: bool = True
    cuda:       bool = True
    track:      bool = False
    wandb_project_name: str      = "principia"
    wandb_entity:       str|None = None
    visual:     bool = False

    env_id:             str = "PoP_Grid"
    total_timesteps:    int = 50_000_000
    num_envs:           int = 8
    num_steps:          int = 4096
    max_episode_steps:  int = 30000
    framestack_warmup:  int = 3
    start_room:     int|None = None
    start_pos:          int = 0

    learning_rate:  float    = 2.5e-4
    anneal_lr:      bool     = True
    gamma:          float    = 0.9994
    gae_lambda:     float    = 0.96
    num_minibatches:int      = 2
    update_epochs:  int      = 4
    norm_adv:       bool     = True
    clip_coef:      float    = 0.15
    vf_clip_coef:   float    = 10.0
    clip_vloss:     bool     = True
    ent_coef:       float    = 0.05
    ent_coef_final: float    = 0.003
    vf_coef:        float    = 0.5
    max_grad_norm:  float    = 0.5
    target_kl:  float|None   = 0.02

    log_interval:        int = 1
    episode_log_interval:int = 10
    checkpoint_interval: int = 10
    eval_interval:       int = 200
    checkpoint_path:     str = ""
    eval_only:          bool = False
    eval_episodes:       int = 1
    record_video:       bool = False
    video_path:          str = ""

    batch_size:     int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --- env factory ------------------------------------------------------------

def make_env(visual_mode, max_steps, warmup_steps, start_room=None,
             start_pos=0, speed_multiplier=1):
    def thunk():
        e = env1.PoPEnv(headless=not visual_mode, visual_mode=visual_mode,
                        max_steps=max_steps, start_room=start_room,
                        start_pos=start_pos)
        if visual_mode and speed_multiplier > 1:
            e.set_speed(speed_multiplier)
        e = env1.FrameStackWrapper(e, n_frames=5, warmup_steps=warmup_steps)
        e = gym.wrappers.RecordEpisodeStatistics(e)
        return e
    return thunk


# --- network ----------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """CNN-on-grid + MLP-on-state actor-critic with FiLM conditioning and FiGAR repeat head."""

    def __init__(self, envs):
        super().__init__()
        aspace = getattr(envs, "single_action_space", getattr(envs, "action_space", None))
        n_actions = int(aspace.nvec[0])
        n_repeats = int(aspace.nvec[1])

        self.grid_conv = nn.Sequential(
            layer_init(nn.Conv2d(60, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)), nn.ReLU(),
        )
        self.grid_head = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(64*3*10, 128)), nn.ReLU())

        # FiLM: conditioned on [have_sword, dir_dx, dir_dy]
        self.film_gen = nn.Sequential(
            layer_init(nn.Linear(3, 32)), nn.ReLU(),
            layer_init(nn.Linear(32, 64*2), std=0.01),
        )

        self.state_net = nn.Sequential(
            layer_init(nn.Linear(31, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 64)), nn.ReLU(),
        )

        self.action_emb = nn.Embedding(n_actions, 8)
        self.action_net = nn.Sequential(layer_init(nn.Linear(5*8, 32)), nn.ReLU())

        self.repeat_emb = nn.Embedding(n_repeats, 8)
        self.repeat_net = nn.Sequential(layer_init(nn.Linear(5*8, 16)), nn.ReLU())

        self.room_emb = nn.Embedding(25, 8)
        self.room_net = nn.Sequential(layer_init(nn.Linear(8, 16)), nn.ReLU())

        # 128 + 64 + 16 + 32 + 16 = 256 → 192
        self.trunk = nn.Sequential(layer_init(nn.Linear(256, 192), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(layer_init(nn.Linear(192, 128)), nn.ReLU(),
                                   layer_init(nn.Linear(128, n_actions), std=0.01))
        self.repeat_head = layer_init(nn.Linear(192, N_REPEATS), std=0.01)
        self.critic = layer_init(nn.Linear(192, 1), std=0.01)

    def _features(self, x):
        conv = self.grid_conv(x["grid"].float())
        ctx = torch.cat([x["state"][:, 4:5], x["state"][:, 28:30]], dim=1)
        g_raw, b_raw = self.film_gen(ctx).chunk(2, dim=-1)
        gamma = (1.0 + g_raw).unsqueeze(-1).unsqueeze(-1)
        beta = b_raw.unsqueeze(-1).unsqueeze(-1)
        g = self.grid_head(conv * gamma + beta)
        s = self.state_net(x["state"].float())
        r = self.room_net(self.room_emb(x["room"].long().squeeze(-1)))
        a = self.action_net(self.action_emb(x["action_history"].long()).view(x["action_history"].shape[0], -1))
        rp = self.repeat_net(self.repeat_emb(x["repeat_history"].long()).view(x["repeat_history"].shape[0], -1))
        return self.trunk(torch.cat([g, s, r, a, rp], dim=1))

    def get_value(self, x):
        return self.critic(self._features(x))

    def get_action_and_value(self, x, action=None):
        """FiGAR joint policy — logprob = log π_action + log π_repeat."""
        f = self._features(x)
        ad = Categorical(logits=self.actor(f))
        rd = Categorical(logits=self.repeat_head(f))
        if action is None:
            act, k_idx = ad.sample(), rd.sample()
        else:
            act, k_idx = action[:, 0].long(), action[:, 1].long()
        return (torch.stack([act, k_idx], dim=1),
                ad.log_prob(act) + rd.log_prob(k_idx),
                ad.entropy() + rd.entropy(),
                self.critic(f))


# --- main -------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir   = os.path.join(script_dir, "runs", run_name)
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
    device  = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    mp.set_start_method("spawn", force=True)
    env_fns = [make_env(args.visual, args.max_episode_steps, args.framestack_warmup,
                        args.start_room, args.start_pos,
                        speed_multiplier=2 if args.visual else 1)
               for _ in range(args.num_envs)]
    envs = (gym.vector.SyncVectorEnv(env_fns) if args.num_envs == 1
            else gym.vector.AsyncVectorEnv(env_fns, context="spawn"))

    agent = Agent(envs).to(device)
    if device.type == "cuda":
        try:    agent = torch.compile(agent, mode="reduce-overhead")
        except: agent = torch.compile(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration, global_step = 1, 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        if not args.eval_only and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0) + 1
        global_step     = ckpt.get("global_step", 0)

    # rollout buffers
    grid_shape     = envs.single_observation_space["grid"].shape
    state_shape    = envs.single_observation_space["state"].shape
    act_hist_shape = envs.single_observation_space["action_history"].shape

    obs_grid = torch.zeros((args.num_steps, args.num_envs) + grid_shape, dtype=torch.float16, device=device)
    obs_state = torch.zeros((args.num_steps, args.num_envs) + state_shape, device=device)
    obs_room = torch.zeros((args.num_steps, args.num_envs, 1), dtype=torch.int32, device=device)
    obs_act_hist = torch.zeros((args.num_steps, args.num_envs) + act_hist_shape, dtype=torch.int32, device=device)
    obs_rep_hist = torch.zeros((args.num_steps, args.num_envs) + act_hist_shape, dtype=torch.int32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, 2), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    # actual tick counts per step for SMDP γ^k discount (FiGAR §4.2)
    durations = torch.full((args.num_steps, args.num_envs), 9.0, device=device)

    episode_returns, episode_lengths = deque(maxlen=100), deque(maxlen=100)
    ep_ret_sums = np.zeros(args.num_envs, dtype=np.float32)
    ep_len_sums = np.zeros(args.num_envs, dtype=np.int32)
    ep_count = 0
    ret_rms = RunningMeanStd()
    running_return = np.zeros(args.num_envs, dtype=np.float64)

    metrics_keys = ["visited_rooms_count", "visited_tiles_count", "level", "room",
                    "have_sword", "guard_hp", "guard_hp_max", "kid_sword_drawn",
                    "episode_sword_found", "episode_guard_killed", "episode_level_up"]
    last_infos = [dict() for _ in range(args.num_envs)]
    seen_rooms = set()

    sword_deque   = deque(maxlen=50)
    guard_deque   = deque(maxlen=50)
    levelup_deque = deque(maxlen=50)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=75, threshold=0.01, threshold_mode="abs")

    start_time = time.time()

    next_obs_raw, _ = envs.reset(seed=args.seed)
    def _t(raw):
        return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in raw.items()}

    next_obs  = _t(next_obs_raw)
    next_done = torch.zeros(args.num_envs, device=device)

    if args.eval_only:
        if args.record_video or args.video_path:
            envs.close()
            base = args.video_path if args.video_path else "eval_recording.mp4"
            stem = base[:-4] if base.endswith(".mp4") else base
            ep = 0
            print(f"  [record] Recording episodes to {stem}_ep<N>.mp4 — press Ctrl+C to stop.")
            try:
                while True:
                    vpath = f"{stem}_ep{ep:04d}.mp4"
                    run_eval_video(agent, start_iteration + ep, vpath, args)
                    ep += 1
            except KeyboardInterrupt:
                print(f"\n  [record] Stopped after {ep} episode(s).")
            raise SystemExit(0)
        eval_ret = np.zeros(args.num_envs, dtype=np.float32)
        eval_len = np.zeros(args.num_envs, dtype=np.int32)
        done_count = 0
        while done_count < args.eval_episodes:
            with torch.no_grad():
                a, *_ = agent.get_action_and_value(next_obs)
            raw, rew, term, trunc, _ = envs.step(a.cpu().numpy())
            d = np.logical_or(term, trunc)
            eval_ret += rew; eval_len += 1
            next_obs = _t(raw)
            for idx in np.where(d)[0]:
                done_count += 1
                print(f"eval ep_return={eval_ret[idx]:.2f} ep_len={eval_len[idx]}")
                eval_ret[idx] = eval_len[idx] = 0
                if done_count >= args.eval_episodes:
                    break
        envs.close()
        raise SystemExit(0)

    _state = {"envs": envs, "next_obs": next_obs, "next_done": next_done}

    def rebuild_envs():
        print("  [crash] rebuilding envs...")
        with contextlib.redirect_stderr(io.StringIO()):
            try: _state["envs"].close()
            except: pass
        fns = [make_env(args.visual, args.max_episode_steps, args.framestack_warmup,
                        args.start_room, args.start_pos)
               for _ in range(args.num_envs)]
        _state["envs"] = (gym.vector.SyncVectorEnv(fns) if args.num_envs == 1
                          else gym.vector.AsyncVectorEnv(fns, context="spawn"))
        raw, _          = _state["envs"].reset(seed=args.seed)
        _state["next_obs"]  = _t(raw)
        _state["next_done"] = torch.zeros(args.num_envs, device=device)
        print("  [crash] recovered.")

    csv_path   = os.path.join(runs_dir, "metrics.csv")
    csv_is_new = not os.path.exists(csv_path)
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(["iteration", "global_step", "elapsed_sec", "fps",
            "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "vf_clipfrac",
            "explained_var", "lr", "ret_rms_std",
            "composite_metric", "sword_rate", "guard_kill_rate", "levelup_rate",
            "ep_ret_mean", "ep_len_mean"])
        csv_file.flush()

    for iteration in range(start_iteration, args.num_iterations + 1):
        envs = _state["envs"]
        ro_rooms  = [set() for _ in range(args.num_envs)]
        ro_events = [[]  for _ in range(args.num_envs)]

        frac = 1.0 - (iteration - 1) / args.num_iterations
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        cur_ent_coef = args.ent_coef_final + frac * (args.ent_coef - args.ent_coef_final)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_grid[step]     = next_obs["grid"].to(torch.float16)
            obs_state[step]    = next_obs["state"]
            obs_room[step]     = next_obs["room"]
            obs_act_hist[step] = next_obs["action_history"]
            obs_rep_hist[step] = next_obs["repeat_history"]
            dones[step]        = next_done

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                action, logprob, _, val = agent.get_action_and_value(next_obs)
                values[step] = val.flatten()

            actions[step]  = action
            logprobs[step] = logprob

            try:
                raw, rew, term, trunc, infos = envs.step(action.cpu().numpy())
            except (EOFError, ConnectionResetError, BrokenPipeError):
                print(f"  [crash] gs={global_step}")
                rebuild_envs()
                envs = _state["envs"]; next_obs = _state["next_obs"]; next_done = _state["next_done"]
                rewards[step] = torch.zeros(args.num_envs, device=device)
                continue

            done_np = np.logical_or(term, trunc)

            ep_ret_sums    += rew
            ret_rms.update(running_return)
            running_return  = running_return * args.gamma + rew
            norm_rew = np.clip(rew / np.sqrt(ret_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
            rewards[step].copy_(torch.from_numpy(norm_rew), non_blocking=True)
            running_return[done_np] = 0.0
            next_obs  = _t(raw)
            # GAE nonterminal mask uses term (not trunc) so timeouts bootstrap value
            next_done = torch.from_numpy(term.astype(np.float32)).to(device, non_blocking=True)

            if "frames_elapsed" in infos:
                durations[step].copy_(torch.from_numpy(
                    np.array(infos["frames_elapsed"], dtype=np.float32)), non_blocking=True)

            # event tracking for logging
            if "room" in infos:
                for i in range(args.num_envs):
                    if infos["room"][i] is not None:
                        ro_rooms[i].add(int(infos["room"][i]))

            if "have_sword" in infos:
                for i, sv in enumerate(infos["have_sword"]):
                    if sv == 1 and last_infos[i].get("have_sword", 0) == 0:
                        ro_events[i].append("Sword")

            if "level" in infos:
                for i in range(args.num_envs):
                    lv = infos["level"][i]
                    if lv is not None:
                        pl = last_infos[i].get("level", 1)
                        if int(lv) > pl:
                            ro_events[i].append(f"Lvl {pl}->{int(lv)}")

            if "guard_hp" in infos and "kid_sword_drawn" in infos:
                for i in range(args.num_envs):
                    ghp      = infos["guard_hp"][i]
                    prev_ghp = last_infos[i].get("guard_hp", -1)
                    drawn    = infos["kid_sword_drawn"][i]
                    if drawn == 1 and last_infos[i].get("kid_sword_drawn", 0) == 0:
                        ro_events[i].append("SwordDrawn")
                    if prev_ghp >= 0 and ghp >= 0 and ghp < prev_ghp:
                        ro_events[i].append("GuardHit")
                    if prev_ghp > 0 and ghp == 0:
                        ro_events[i].append("GuardDead")

            if np.any(done_np):
                for i in np.where(done_np)[0]:
                    if any("Sword" in ev or "Lvl" in ev or "Guard" in ev for ev in ro_events[i]):
                        ro_events[i].append("Done")

            for key in metrics_keys:
                if key in infos and infos[key] is not None:
                    for idx in range(args.num_envs):
                        if infos[key][idx] is not None:
                            last_infos[idx][key] = infos[key][idx]

            for idx in range(args.num_envs):
                rid = last_infos[idx].get("room")
                if rid is not None and rid not in seen_rooms:
                    seen_rooms.add(rid)
                    print(f"gs={global_step} env{idx} NEW room={rid} "
                          f"tiles={last_infos[idx].get('visited_tiles_count', 0)} "
                          f"total={len(seen_rooms)}")

            ep_len_sums += 1
            if np.any(done_np):
                for idx in np.where(done_np)[0]:
                    episode_returns.append(float(ep_ret_sums[idx]))
                    episode_lengths.append(int(ep_len_sums[idx]))
                    ep_count += 1
                    if "final_info" in infos and infos["final_info"][idx] is not None:
                        for key in metrics_keys:
                            if key in infos["final_info"][idx]:
                                last_infos[idx][key] = infos["final_info"][idx][key]
                    sword_deque.append(last_infos[idx].get("episode_sword_found",  0))
                    guard_deque.append(last_infos[idx].get("episode_guard_killed", 0))
                    levelup_deque.append(last_infos[idx].get("episode_level_up",   0))
                    ep_ret_sums[idx] = ep_len_sums[idx] = 0

        print(f"\n[rollout] gs={global_step}")
        for i in range(args.num_envs):
            ev_str = ", ".join(ro_events[i])
            if ro_rooms[i] or ev_str:
                print(f"  env{i}: rooms={sorted(ro_rooms[i])}  [{ev_str}]")

        # --- GAE with SMDP discount (γ^k per step) ---
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            next_val = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            gae        = 0.0
            for t in reversed(range(args.num_steps)):
                nnt = 1.0 - (next_done if t == args.num_steps - 1 else dones[t + 1])
                nv  = next_val if t == args.num_steps - 1 else values[t + 1]
                dk  = args.gamma ** durations[t]
                delta      = rewards[t] + dk * nv * nnt - values[t]
                advantages[t] = gae = delta + dk * args.gae_lambda * nnt * gae
            returns = advantages + values

        b_grid     = obs_grid.reshape((-1,) + grid_shape)
        b_state    = obs_state.reshape((-1,) + state_shape)
        b_room     = obs_room.reshape(-1, 1)
        b_act_hist = obs_act_hist.reshape((-1,) + act_hist_shape)
        b_rep_hist = obs_rep_hist.reshape((-1,) + act_hist_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions  = actions.reshape(-1, 2)
        b_adv      = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)
        b_values   = values.reshape(-1)
        b_inds     = np.arange(args.batch_size)

        clipfracs, vf_clipfracs = [], []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                mb_obs = {"grid": b_grid[mb].float(), "state": b_state[mb],
                          "room": b_room[mb], "action_history": b_act_hist[mb],
                          "repeat_history": b_rep_hist[mb]}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, newlp, entropy, newval = agent.get_action_and_value(
                        mb_obs, b_actions[mb].long())
                    ratio = (newlp - b_logprobs[mb]).exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (newlp - b_logprobs[mb])).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    mb_adv = b_adv[mb]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()

                    newval = newval.view(-1)
                    if args.clip_vloss:
                        v_unclip = (newval - b_returns[mb]) ** 2
                        v_clip   = b_values[mb] + (newval - b_values[mb]).clamp(-args.vf_clip_coef, args.vf_clip_coef)
                        v_loss   = 0.5 * torch.max(v_unclip, (v_clip - b_returns[mb]) ** 2).mean()
                    else:
                        v_loss   = 0.5 * ((newval - b_returns[mb]) ** 2).mean()

                    with torch.no_grad():
                        vf_clipfracs.append(((newval - b_values[mb]).abs() > args.vf_clip_coef).float().mean().item())

                    loss = pg_loss - cur_ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y        = np.var(y_true)
        explained_var= np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        fps          = int(global_step / (time.time() - start_time))

        warmed_up = len(sword_deque) >= 30 and len(guard_deque) >= 30 and len(levelup_deque) >= 30
        if warmed_up:
            composite = (sum(sword_deque)/len(sword_deque) + sum(guard_deque)/len(guard_deque)
                         + sum(levelup_deque)/len(levelup_deque))
            if not args.anneal_lr:
                scheduler.step(composite)

        if args.log_interval > 0 and (iteration % args.log_interval == 0 or iteration == 1):
            print("-" * 50)
            print(f"iter={iteration}  gs={global_step}  fps={fps}  elapsed={int(time.time()-start_time)}s")
            print(f"  pg={pg_loss.item():.4f}  vf={v_loss.item():.4f}  ent={entropy.mean().item():.4f}"
                  f"  kl={approx_kl.item():.4f}  clip={np.mean(clipfracs):.4f}"
                  f"  ev={explained_var:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  ec={cur_ent_coef:.4f}")
            if episode_returns:
                print(f"  ep_ret={np.mean(episode_returns):.2f}  ep_len={np.mean(episode_lengths):.0f}")

            csv_writer.writerow([iteration, global_step, int(time.time()-start_time), fps,
                pg_loss.item(), v_loss.item(), entropy.mean().item(),
                approx_kl.item(), float(np.mean(clipfracs)), float(np.mean(vf_clipfracs)),
                explained_var, optimizer.param_groups[0]["lr"],
                float(np.sqrt(ret_rms.var + 1e-8)),
                composite if warmed_up else "",
                (sum(sword_deque)/len(sword_deque))   if sword_deque   else "",
                (sum(guard_deque)/len(guard_deque))   if guard_deque   else "",
                (sum(levelup_deque)/len(levelup_deque)) if levelup_deque else "",
                float(np.mean(episode_returns)) if episode_returns else "",
                float(np.mean(episode_lengths)) if episode_lengths else ""])
            csv_file.flush()

        if iteration % args.checkpoint_interval == 0:
            path = os.path.join(runs_dir, f"ckpt_{iteration}.pt")
            torch.save({"iteration": iteration, "global_step": global_step,
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, path)
            print(f"  ckpt → {path}")

        if args.eval_interval > 0 and (iteration % args.eval_interval == 0 or iteration == 1):
            vdir  = os.path.join(runs_dir, "videos"); os.makedirs(vdir, exist_ok=True)
            vpath = os.path.join(vdir, f"eval_iter_{iteration}.mp4")
            run_eval_video(agent, iteration, vpath, args)

    envs.close()
    csv_file.close()
