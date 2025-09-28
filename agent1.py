"""PPO + FiGAR agent for Prince of Persia."""
REPEAT_CHOICES = [1, 2, 4, 8, 12]
import csv
import contextlib
import io
import json
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


# --- persistent experience memory -------------------------------------------

def save_checkpoint(agent, iteration, save_dir="checkpoints"):
    if iteration % 10 == 0:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(agent.state_dict(), f"{save_dir}/ckpt_{iteration}.pt")

def compute_smdp_gae(rewards, values, durations, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        gamma_tau = gamma ** durations[t]
        delta = rewards[t] + gamma_tau * values[t+1] - values[t]
        lastgaelam = delta + gamma_tau * gae_lambda * lastgaelam
        advantages[t] = lastgaelam
    return advantages

def update_edge_memory(mem, src, dst, direction, died: bool, alpha: float = 0.05):
    key = f"{src}:{dst}:{direction}"
    rec = mem["edges"].setdefault(key, {"death_ema": 0.0, "n": 0})
    rec["death_ema"] = (1.0 - alpha) * rec["death_ema"] + alpha * float(died)
    rec["n"] += 1


def update_gate_memory(mem, switch_event, gate_changes, threshold: int = 5):
    if switch_event is None or not gate_changes:
        return
    s_room, s_col, s_row, s_kind, s_action = switch_event
    sk = f"{s_room}:{s_col}:{s_row}"
    rec = mem["gates"].setdefault(sk, {"switch_kind": s_kind, "candidates": {}, "one_shot": False})
    if s_kind is not None:
        rec["switch_kind"] = s_kind
    for g_room, g_col, g_row, is_open in gate_changes:
        gk = f"{g_room}:{g_col}:{g_row}"
        c = rec["candidates"].setdefault(gk, {
            "press_opened_count": 0, "press_closed_count": 0,
            "release_opened_count": 0, "release_closed_count": 0,
        })
        if s_action == "press":
            if is_open:
                c["press_opened_count"] += 1
            else:
                c["press_closed_count"] += 1
        else:
            if is_open:
                c["release_opened_count"] += 1
            else:
                c["release_closed_count"] += 1
    # recompute every call so one_shot can revert when a release is later observed
    rec["one_shot"] = any(
        (c["press_opened_count"] + c["press_closed_count"]) >= threshold
        and (c["release_opened_count"] + c["release_closed_count"]) == 0
        for c in rec["candidates"].values()
    )


def update_poi_memory(mem, room, col, row, kind):
    key = f"{room}:{col}:{row}:{kind}"
    mem["poi"].setdefault(key, {"n_seen": 0})["n_seen"] += 1


def merge_pbrs_hints(memory_list):
    """Build PBRS hint from persistent memory for broadcast to all envs.

    Returns:
        {
          "paths_by_guard": {guard_room: [sword_room,...,guard_room], ...},
          "fallback":        [sword_room,...,start_room]   # reversed sword path
        }

    The env filters dead_guard_rooms at _build_pbrs_map time so we send all
    known paths and let each env pick the best active one for this episode.
    Shorter computed paths win over longer ones for each guard room, across
    all envs (shared level topology).
    """
    # Merge: for each guard room keep the shortest path across all envs
    paths_by_guard = {}
    fallback = None
    for mem in memory_list:
        paths = mem.get("paths", {})
        for gr, p in paths.get("computed_sword_to_guard_by_room", {}).items():
            if p and (gr not in paths_by_guard or len(p) < len(paths_by_guard[gr])):
                paths_by_guard[gr] = p
        p = paths.get("to_sword_reversed")
        if p and (fallback is None or len(p) < len(fallback)):
            fallback = p
    return {"paths_by_guard": paths_by_guard, "fallback": fallback or []}


def compute_sword_to_guard_path(path_to_sword, path_to_guard):
    """Derive path from sword room to guard room by finding the junction.

    Both paths start from room 1 (start). The sword-to-guard path is:
      reverse(path_to_sword from junction..sword_room) + path_to_guard from junction onward

    Example:
      path_to_sword = [1, 2, 3, 6, 8, 7, 20, 12, 15]
      path_to_guard = [1, 2, 6, 8, 7, 19, 22, 21]
      junction = room 7 (last room in path_to_sword that is also in path_to_guard)
      result   = [15, 12, 20, 7] + [19, 22, 21] = [15, 12, 20, 7, 19, 22, 21]

    Returns None if no common room is found (paths are disjoint).
    """
    if not path_to_sword or not path_to_guard:
        return None
    guard_idx = {r: i for i, r in enumerate(path_to_guard)}
    for j in range(len(path_to_sword) - 1, -1, -1):
        r = path_to_sword[j]
        if r in guard_idx:
            gi = guard_idx[r]
            return list(reversed(path_to_sword[j:])) + path_to_guard[gi + 1:]
    return None


# --- reward normalisation ---------------------------------------------------

class RunningMeanStd:
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

def _eval_video_worker(seed, video_path, model_state, initial_memory=None, max_steps=2000,
                        start_room=None, start_pos=0, warmup_steps=3,
                        result_queue=None, gamma=0.995):
    e = make_env(visual_mode=False, max_steps=max_steps, warmup_steps=warmup_steps,
                 start_room=start_room, start_pos=start_pos, gamma=gamma)()

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

    obs_raw, _ = e.reset(seed=seed)
    obs, done, tot_rew, steps = _t(obs_raw), False, 0.0, 0
    import copy
    eval_memory = copy.deepcopy(initial_memory) if initial_memory is not None else {"edges": {}, "gates": {}, "poi": {}}
    eval_pending_switch = None

    while not done and steps < max_steps:
        proc.stdin.write(e.render().tobytes())
        with torch.no_grad():
            mem_vec = agent.mem_encoder(eval_memory)
            act, *_ = agent.get_action_and_value(obs, mem_vec)
        raw, rew, term, trunc, info = e.step(act[0].numpy())
        done = term or trunc
        tot_rew += rew
        steps += 1
        obs = _t(raw)

        if "edge_resolved" in info and info["edge_resolved"] is not None:
            src, dst, direction, died = info["edge_resolved"]
            update_edge_memory(eval_memory, src, dst, direction, died=died)
        if "switch_event" in info and info["switch_event"] is not None:
            eval_pending_switch = info["switch_event"]
        if "gate_changes" in info and info["gate_changes"] is not None and eval_pending_switch is not None:
            update_gate_memory(eval_memory, eval_pending_switch, info["gate_changes"])
        if "sword_found_at" in info and info["sword_found_at"] is not None:
            sf = info["sword_found_at"]
            update_poi_memory(eval_memory, sf[0], sf[1], sf[2], "sword")
        if "potion_found_at" in info and info["potion_found_at"] is not None:
            pf = info["potion_found_at"]
            update_poi_memory(eval_memory, pf[0], pf[1], pf[2], f"potion_{pf[3]}")

    proc.stdin.close()
    proc.wait()
    e.close()
    if result_queue is not None:
        result_queue.put((tot_rew, steps))


def run_eval_video(agent, iteration, video_path, args, memory=None):
    state = {k.replace("_orig_mod.", ""): v.cpu() for k, v in agent.state_dict().items()}
    q = mp.Queue()
    init_mem = memory[0] if (isinstance(memory, list) and len(memory) > 0) else memory
    p = mp.Process(target=_eval_video_worker,
                   args=(args.seed + iteration, video_path, state, init_mem,
                         args.max_episode_steps, args.start_room, args.start_pos,
                         args.framestack_warmup, q, args.gamma))
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
    update_epochs:  int      = 5
    norm_adv:       bool     = True
    clip_coef:      float    = 0.15
    vf_clip_coef:   float    = 10.0
    clip_vloss:     bool     = True
    ent_coef:       float    = 0.05
    ent_coef_final: float    = 0.003  # annealed to this by end of training
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

    edge_ema_alpha:       float = 0.05
    gate_confirm_threshold: int = 5
    dormant_interval:       int = 25  # check dead-neuron fraction every N iterations (0=off)

    batch_size:     int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --- env factory ------------------------------------------------------------

def make_env(visual_mode, max_steps, warmup_steps, start_room=None,
             start_pos=0, speed_multiplier=1, gamma=0.995):
    def thunk():
        e = env1.PoPEnv(headless=not visual_mode, visual_mode=visual_mode,
                        max_steps=max_steps, start_room=start_room,
                        start_pos=start_pos, gamma=gamma)
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


_DIR_IDX = {"left": 0, "right": 1, "up": 2, "down": 3}
_POI_KIND_IDX = {"sword": 0, "potion_big": 1, "potion_small": 2}
_SW_KIND_IDX = {"opener": 0, "closer": 1}  # None → [0,0,1]
_POOL_DIM = 16  # per-item embedding dim; each category pools to 2*_POOL_DIM
_MEM_DIM = 64   # final memory vector width


class MemoryEncoder(nn.Module):
    """Nested Deep Sets encoder for (edges, gates, poi) persistent memory.

    edges — flat set of traversed directed edges.
    poi   — flat set of points of interest (sword, potions).
    gates — nested: a set of switches, each owning a set of candidate gate tiles.

    Empty categories pool to zeros. Gradients flow through all MLPs.
    """

    EDGE_FEAT = 8    # src/24, dst/24, dir[4], death_ema, log1p(n)
    POI_FEAT = 7     # room/24, col/10, row/3, kind[3], log1p(n_seen)
    CAND_FEAT = 8    # g_room/24, g_col/10, g_row/3, po, pc, ro, rc, log1p(total)
    SW_OWN_FEAT = 8  # sw_room/24, sw_col/10, sw_row/3, kind[3], one_shot, log1p(n_cands)

    def __init__(self):
        super().__init__()
        D = _POOL_DIM
        self.edge_mlp = nn.Sequential(
            layer_init(nn.Linear(self.EDGE_FEAT, D)), nn.ReLU(),
            layer_init(nn.Linear(D, D)), nn.ReLU(),
        )
        self.poi_mlp = nn.Sequential(
            layer_init(nn.Linear(self.POI_FEAT, D)), nn.ReLU(),
            layer_init(nn.Linear(D, D)), nn.ReLU(),
        )
        self.cand_mlp = nn.Sequential(
            layer_init(nn.Linear(self.CAND_FEAT, D)), nn.ReLU(),
            layer_init(nn.Linear(D, D)), nn.ReLU(),
        )
        # candidate pool (2D) concat with switch own features (SW_OWN_FEAT) → switch embedding
        self.switch_mlp = nn.Sequential(
            layer_init(nn.Linear(2 * D + self.SW_OWN_FEAT, D)), nn.ReLU(),
            layer_init(nn.Linear(D, D)), nn.ReLU(),
        )
        # three categories each produce 2D; project to _MEM_DIM
        self.proj = nn.Sequential(
            layer_init(nn.Linear(6 * D, _MEM_DIM)), nn.ReLU(),
        )

    def _meanmax(self, embs):
        # embs: (N, D) — returns (2D,) or zeros if empty
        if embs.shape[0] == 0:
            return torch.zeros(2 * _POOL_DIM, device=embs.device)
        return torch.cat([embs.mean(0), embs.max(0).values])

    def _encode_edges(self, edges: dict, device, cache_edges: dict = None):
        if cache_edges is None:
            cache_edges = {}

        new_keys = []
        for key, rec in edges.items():
            parts = key.split(":")
            if parts[2] not in _DIR_IDX:
                continue
            if key not in cache_edges:
                new_keys.append((key, rec))

        if new_keys:
            rows = []
            for key, rec in new_keys:
                parts = key.split(":")
                src, dst, direction = int(parts[0]), int(parts[1]), parts[2]
                d_hot = [0.0, 0.0, 0.0, 0.0]
                d_hot[_DIR_IDX[direction]] = 1.0
                row = [src / 24.0, dst / 24.0] + d_hot + [rec["death_ema"],
                                                            np.log1p(rec["n"]) / np.log1p(500)]
                rows.append(row)
            t = torch.tensor(rows, dtype=torch.float32, device=device)
            embedded = self.edge_mlp(t)
            for idx, (key, _) in enumerate(new_keys):
                cache_edges[key] = embedded[idx].float()

        active_embs = [cache_edges[k] for k in edges if k in cache_edges]
        if not active_embs:
            return torch.zeros(2 * _POOL_DIM, device=device)
        stacked = torch.stack(active_embs)
        return self._meanmax(stacked)

    def _encode_poi(self, poi: dict, device, cache_poi: dict = None):
        if cache_poi is None:
            cache_poi = {}

        new_keys = [(key, rec) for key, rec in poi.items() if key not in cache_poi]
        if new_keys:
            rows = []
            for key, rec in new_keys:
                parts = key.split(":")
                room, col, row, kind = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
                k_hot = [0.0, 0.0, 0.0]
                k_hot[_POI_KIND_IDX.get(kind, 0)] = 1.0
                rows.append([room / 24.0, col / 10.0, row / 3.0] + k_hot
                            + [np.log1p(rec["n_seen"]) / np.log1p(500)])
            t = torch.tensor(rows, dtype=torch.float32, device=device)
            embedded = self.poi_mlp(t)
            for idx, (key, _) in enumerate(new_keys):
                cache_poi[key] = embedded[idx].float()

        active_embs = [cache_poi[k] for k in poi if k in cache_poi]
        if not active_embs:
            return torch.zeros(2 * _POOL_DIM, device=device)
        stacked = torch.stack(active_embs)
        return self._meanmax(stacked)

    def _encode_gates(self, gates: dict, device, cache_gates: dict = None):
        if cache_gates is None:
            cache_gates = {}

        new_keys = [(sw_key, rec) for sw_key, rec in gates.items() if sw_key not in cache_gates]
        if new_keys:
            for sw_key, rec in new_keys:
                parts = sw_key.split(":")
                sw_room, sw_col, sw_row = int(parts[0]), int(parts[1]), int(parts[2])

                cand_rows = []
                for gk, c in rec["candidates"].items():
                    gp = gk.split(":")
                    g_room, g_col, g_row = int(gp[0]), int(gp[1]), int(gp[2])
                    total = (c["press_opened_count"] + c["press_closed_count"]
                             + c["release_opened_count"] + c["release_closed_count"])
                    total_p = max(c["press_opened_count"] + c["press_closed_count"], 1)
                    total_r = max(c["release_opened_count"] + c["release_closed_count"], 1)
                    cand_rows.append([
                        g_room / 24.0, g_col / 10.0, g_row / 3.0,
                        c["press_opened_count"] / total_p,
                        c["press_closed_count"] / total_p,
                        c["release_opened_count"] / total_r,
                        c["release_closed_count"] / total_r,
                        np.log1p(total) / np.log1p(200),
                    ])
                if cand_rows:
                    ct = torch.tensor(cand_rows, dtype=torch.float32, device=device)
                    cand_pool = self._meanmax(self.cand_mlp(ct)).float()
                else:
                    cand_pool = torch.zeros(2 * _POOL_DIM, device=device)

                sk = rec.get("switch_kind")
                sw_k = [0.0, 0.0, 0.0]
                if sk in _SW_KIND_IDX:
                    sw_k[_SW_KIND_IDX[sk]] = 1.0
                else:
                    sw_k[2] = 1.0  # None bucket

                sw_feat = torch.tensor(
                    [sw_room / 24.0, sw_col / 10.0, sw_row / 3.0] + sw_k
                    + [float(rec["one_shot"]), np.log1p(len(rec["candidates"])) / np.log1p(20)],
                    dtype=torch.float32, device=device,
                )
                sw_in = torch.cat([cand_pool, sw_feat])
                sw_emb = self.switch_mlp(sw_in.unsqueeze(0)).squeeze(0)
                cache_gates[sw_key] = sw_emb.float()

        active_embs = [cache_gates[k] for k in gates if k in cache_gates]
        if not active_embs:
            return torch.zeros(2 * _POOL_DIM, device=device)
        stacked = torch.stack(active_embs)
        return self._meanmax(stacked)

    def forward(self, memory: dict, cache: dict = None, device=None):
        if device is None:
            device = next(self.parameters()).device
        c_edges = cache.get("edges") if cache is not None else None
        c_poi = cache.get("poi") if cache is not None else None
        c_gates = cache.get("gates") if cache is not None else None

        e = self._encode_edges(memory["edges"], device, cache_edges=c_edges)
        p = self._encode_poi(memory["poi"], device, cache_poi=c_poi)
        g = self._encode_gates(memory["gates"], device, cache_gates=c_gates)
        cat_emb = torch.cat([e, p, g]).to(dtype=self.proj[0].weight.dtype)
        return self.proj(cat_emb)



class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM): x * (1 + gamma) + beta."""
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.film_gen = layer_init(nn.Linear(cond_dim, feature_dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_gen(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

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
            layer_init(nn.Conv2d(64, 64, 3)), nn.ReLU(),  # → (B,64,3,10)
        )
        self.grid_head = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(64*3*10, 128)), nn.ReLU())

        # FiLM: conditioned on [have_sword, dir_dx, dir_dy]
        self.film_gen = nn.Sequential(
            layer_init(nn.Linear(3, 32)), nn.ReLU(),
            layer_init(nn.Linear(32, 64*2), std=0.01),
        )

        self.state_net = nn.Sequential(
            layer_init(nn.Linear(30, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 64)), nn.ReLU(),
        )

        self.action_emb = nn.Embedding(n_actions, 8)
        self.action_net = nn.Sequential(layer_init(nn.Linear(5*8, 32)), nn.ReLU())

        self.repeat_emb = nn.Embedding(n_repeats, 8)
        self.repeat_net = nn.Sequential(layer_init(nn.Linear(5*8, 16)), nn.ReLU())

        self.room_emb = nn.Embedding(25, 8)
        self.room_net = nn.Sequential(layer_init(nn.Linear(8, 16)), nn.ReLU())

        self.mem_encoder = MemoryEncoder()

        # obs streams: 128+64+16+32+16 = 256 → trunk → 192
        # memory conditions trunk output via FiLM (not concat)
        self.trunk    = nn.Sequential(layer_init(nn.Linear(256, 192), std=0.1), nn.ReLU())
        self.film_mem = layer_init(nn.Linear(_MEM_DIM, 192 * 2), std=0.01)  # → (γ, β)
        self.trunk_ln = nn.LayerNorm(192)
        self.actor = nn.Sequential(layer_init(nn.Linear(192, 128)), nn.ReLU(),
                                   layer_init(nn.Linear(128, n_actions), std=0.01))
        self.repeat_head = layer_init(nn.Linear(192, N_REPEATS), std=0.01)
        self.critic = layer_init(nn.Linear(192, 1), std=0.01)

    def _features(self, x, mem_vec):
        # mem_vec: (B, 64) or (64,)
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
        h = self.trunk(torch.cat([g, s, r, a, rp], dim=1))
        B = h.shape[0]
        mv = mem_vec.unsqueeze(0).expand(B, -1) if mem_vec.dim() == 1 else mem_vec
        mv = mv.to(h.device)
        gm, bm = self.film_mem(mv).chunk(2, dim=-1)
        return self.trunk_ln(h * (1.0 + gm) + bm)

    def get_value(self, x, mem_vec):
        return self.critic(self._features(x, mem_vec))

    def get_action_and_value(self, x, mem_vec, action=None):
        """FiGAR joint policy — logprob = log π_action + log π_repeat (factored, Sharma 2017 eq.2)."""
        f = self._features(x, mem_vec)
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


# --- dormant neuron diagnostic ----------------------------------------------

@torch.no_grad()
def dormant_fractions(agent, sample_obs, sample_mem_vec, eps: float = 0.01) -> dict:
    """Fraction of neurons with max-abs activation <= eps over sample_obs.

    A neuron is dormant if it never activates meaningfully on the given batch —
    a steadily rising dormant fraction signals plasticity loss.
    Uses temporary forward hooks; bypasses torch.compile via _orig_mod.
    """
    m = getattr(agent, '_orig_mod', agent)
    acts = {}

    def make_hook(name):
        def hook(mod, inp, out):
            acts[name] = out.detach().float()
        return hook

    # (sequential, label_prefix, relu_indices)
    targets = [
        (m.grid_conv,  'grid_conv',  [1, 3, 5]),
        (m.grid_head,  'grid_head',  [2]),
        (m.state_net,  'state_net',  [1, 3]),
        (m.trunk,      'trunk',      [1]),
    ]
    hooks = []
    for seq, prefix, idxs in targets:
        for i in idxs:
            hooks.append(seq[i].register_forward_hook(make_hook(f'{prefix}[{i}]')))

    m._features(sample_obs, sample_mem_vec)

    for h in hooks:
        h.remove()

    result = {}
    for name, act in acts.items():
        if act.dim() == 4:          # conv: (B, C, H, W) — neuron = filter
            peak = act.abs().amax(dim=(0, 2, 3))   # (C,)
        else:                       # fc:   (B, D)       — neuron = unit
            peak = act.abs().amax(dim=0)            # (D,)
        result[name] = (peak <= eps).float().mean().item()
    return result


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
                        speed_multiplier=2 if args.visual else 1,
                        gamma=args.gamma) for _ in range(args.num_envs)]
    envs = (gym.vector.SyncVectorEnv(env_fns) if args.num_envs == 1
            else gym.vector.AsyncVectorEnv(env_fns, context="spawn"))

    agent = Agent(envs).to(device)
    if device.type == "cuda":
        # mem_encoder iterates dynamic Python dicts so torch.compile can't trace it
        # (hits recompile_limit immediately). Compile only the rest of the agent.
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
    # Snapshot of mem_vec actually used at each step — matched against logprobs[step]
    # so the PPO ratio is exactly 1 for an unchanged policy regardless of memory evolution.
    obs_mem_vec = torch.zeros((args.num_steps, args.num_envs, _MEM_DIM), device=device)
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

    memory = [{"edges": {}, "gates": {}, "poi": {}, "guard_rooms": [],
               "paths": {"to_sword": None, "to_sword_reversed": None,
                         "to_guard_by_room": {}, "computed_sword_to_guard_by_room": {}}}
              for _ in range(args.num_envs)]
    mem_cache = [{"edges": {}, "gates": {}, "poi": {}} for _ in range(args.num_envs)]
    dirty = [True] * args.num_envs
    mem_vec_cache = torch.zeros((args.num_envs, _MEM_DIM), device=device)
    pending_switch = [None] * args.num_envs

    if args.checkpoint_path:
        mem_path = os.path.join(os.path.dirname(args.checkpoint_path), "memory.json")
        if os.path.exists(mem_path):
            try:
                with open(mem_path) as _f:
                    loaded = json.load(_f)
                if isinstance(loaded, list) and loaded:
                    memory = [loaded[i % len(loaded)] for i in range(args.num_envs)]
                for m in memory:
                    p = m.setdefault("paths", {})
                    p.setdefault("to_sword", None)
                    p.setdefault("to_sword_reversed", None)
                    p.setdefault("to_guard_by_room", {})
                    p.setdefault("computed_sword_to_guard_by_room", {})
                dirty = [True] * args.num_envs
                print(f"  [memory] loaded {mem_path}")
            except Exception as e:
                print(f"  [memory] warning: {e}")

    sword_deque   = deque(maxlen=50)
    guard_deque   = deque(maxlen=50)
    levelup_deque = deque(maxlen=50)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=75, threshold=0.01, threshold_mode="abs")

    start_time  = time.time()

    next_obs_raw, _ = envs.reset(seed=args.seed)
    # Push initial (empty) PBRS hints into each env subprocess so _pbrs_hint is warm.
    envs.call("set_pbrs_hint", merge_pbrs_hints(memory))
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
                    run_eval_video(agent, start_iteration + ep, vpath, args, memory=memory)
                    ep += 1
            except KeyboardInterrupt:
                print(f"\n  [record] Stopped after {ep} episode(s).")
            raise SystemExit(0)
        eval_ret = np.zeros(args.num_envs, dtype=np.float32)
        eval_len = np.zeros(args.num_envs, dtype=np.int32)
        done_count = 0
        while done_count < args.eval_episodes:
            with torch.no_grad():
                mem_vec = agent.mem_encoder(memory[0])
                a, *_ = agent.get_action_and_value(next_obs, mem_vec)
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
                        args.start_room, args.start_pos, gamma=args.gamma)
               for _ in range(args.num_envs)]
        _state["envs"] = (gym.vector.SyncVectorEnv(fns) if args.num_envs == 1
                          else gym.vector.AsyncVectorEnv(fns, context="spawn"))
        raw, _          = _state["envs"].reset(seed=args.seed)
        _state["next_obs"]  = _t(raw)
        _state["next_done"] = torch.zeros(args.num_envs, device=device)
        print("  [crash] recovered.")

    def trunk_weight_norm(agent):
        prefixes = ("grid_conv", "grid_head", "film_gen", "state_net",
                    "action_emb", "action_net", "repeat_emb", "repeat_net",
                    "room_emb", "room_net", "trunk")
        sq = sum(float(p.detach().float().norm()**2)
                 for n, p in agent.named_parameters()
                 if n.replace("_orig_mod.", "").startswith(prefixes))
        return sq ** 0.5

    csv_path   = os.path.join(runs_dir, "metrics.csv")
    csv_is_new = not os.path.exists(csv_path)
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(["iteration", "global_step", "elapsed_sec", "fps",
            "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "vf_clipfrac",
            "explained_var", "lr", "ret_rms_std", "trunk_weight_l2norm",
            "composite_metric", "sword_rate", "guard_kill_rate", "levelup_rate",
            "ep_ret_mean", "ep_len_mean"])
        csv_file.flush()

    b_env_ids = np.tile(np.arange(args.num_envs), args.num_steps)

    for iteration in range(start_iteration, args.num_iterations + 1):
        envs = _state["envs"]
        ro_rooms           = [set() for _ in range(args.num_envs)]
        ro_post_sword_rooms= [set() for _ in range(args.num_envs)]
        ro_events          = [[]  for _ in range(args.num_envs)]

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
                for i in range(args.num_envs):
                    if dirty[i]:
                        mem_vec_cache[i] = agent.mem_encoder(memory[i], cache=mem_cache[i])
                        dirty[i] = False
                # Store the exact mem_vec used here — update phase will read it back
                # so newlp is conditioned on the same context that produced logprobs[step].
                obs_mem_vec[step] = mem_vec_cache
                action, logprob, _, val = agent.get_action_and_value(next_obs, mem_vec_cache)
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
            # Update RMS *before* folding in this step's reward so one-off spikes
            # (sword +100, guard-kill +300) don't permanently inflate ret_rms.var
            # and compress all subsequent normalised rewards.
            ret_rms.update(running_return)
            running_return  = running_return * args.gamma + rew
            norm_rew = np.clip(rew / np.sqrt(ret_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)
            rewards[step].copy_(torch.from_numpy(norm_rew), non_blocking=True)
            running_return[done_np] = 0.0
            next_obs  = _t(raw)
            # F-06: GAE nonterminal mask must use term (terminated) so timeout truncations
            # bootstrap value instead of treating timeouts as death resets.
            next_done = torch.from_numpy(term.astype(np.float32)).to(device, non_blocking=True)

            if "frames_elapsed" in infos:
                durations[step].copy_(torch.from_numpy(
                    np.array(infos["frames_elapsed"], dtype=np.float32)), non_blocking=True)

            if "room" in infos:
                r_vals = infos["room"]
                s_vals = infos.get("have_sword", [0] * args.num_envs)
                for i in range(args.num_envs):
                    if r_vals[i] is None: continue
                    rid = int(r_vals[i])
                    ro_rooms[i].add(rid)
                    if s_vals[i] > 0:
                        ro_post_sword_rooms[i].add(rid)
                        if rid == 3 and (last_infos[i].get("room") != 3 or last_infos[i].get("have_sword", 0) == 0):
                            ro_events[i].append("R3WithSword")
                    if rid == 21 and last_infos[i].get("room") != 21:
                        ro_events[i].append("R21Visited")

            if "have_sword" in infos:
                for i, sv in enumerate(infos["have_sword"]):
                    if sv == 1 and last_infos[i].get("have_sword", 0) == 0:
                        ro_events[i].append("Sword")

            if "level" in infos:
                l_vals = np.array(infos["level"], dtype=object)
                valid_l = np.where(l_vals != None)[0]
                for i in valid_l:
                    lv = int(l_vals[i])
                    pl = last_infos[i].get("level", 1)
                    if lv > pl:
                        ro_events[i].append(f"Lvl {pl}->{lv}")
                        memory[i]["edges"].clear()
                        memory[i]["gates"].clear()
                        memory[i]["poi"].clear()
                        memory[i]["guard_rooms"] = []
                        memory[i]["paths"] = {
                            "to_sword": None, "to_sword_reversed": None,
                            "to_guard_by_room": {}, "computed_sword_to_guard_by_room": {}}
                        mem_cache[i]["edges"].clear()
                        mem_cache[i]["gates"].clear()
                        mem_cache[i]["poi"].clear()
                        dirty[i] = True

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
                    # Detect guard rooms for PBRS memory
                    ghp_max = infos.get("guard_hp_max", [0] * args.num_envs)[i]
                    if ghp_max is not None and int(ghp_max) > 0:
                        room_i = infos.get("room", [0] * args.num_envs)[i]
                        if room_i is not None and int(room_i) > 0:
                            gr = memory[i].setdefault("guard_rooms", [])
                            ri = int(room_i)
                            if ri not in gr:
                                gr.append(ri)

            if np.any(done_np):
                for i in np.where(done_np)[0]:
                    if any("Sword" in ev or "Lvl" in ev or "Guard" in ev for ev in ro_events[i]):
                        ro_events[i].append("Done")

            if "edge_resolved" in infos and infos["edge_resolved"] is not None:
                for i, ev in enumerate(infos["edge_resolved"]):
                    if ev is not None:
                        src, dst, direction, died = ev
                        update_edge_memory(memory[i], src, dst, direction, died=died, alpha=args.edge_ema_alpha)
                        dirty[i] = True
                # Broadcast merged hint — union of all envs' safe edges and guard rooms.
                envs.call("set_pbrs_hint", merge_pbrs_hints(memory))

            if "switch_event" in infos and infos["switch_event"] is not None:
                for i, se in enumerate(infos["switch_event"]):
                    if se is not None:
                        pending_switch[i] = se

            if "gate_changes" in infos and infos["gate_changes"] is not None:
                for i, gc in enumerate(infos["gate_changes"]):
                    if gc is not None and pending_switch[i] is not None:
                        update_gate_memory(memory[i], pending_switch[i], gc, threshold=args.gate_confirm_threshold)
                        dirty[i] = True

            if "sword_found_at" in infos and infos["sword_found_at"] is not None:
                for i, sf in enumerate(infos["sword_found_at"]):
                    if sf is not None:
                        update_poi_memory(memory[i], sf[0], sf[1], sf[2], "sword")
                        dirty[i] = True

            if "path_to_sword" in infos and infos["path_to_sword"] is not None:
                needs_hint_refresh = False
                for i, p in enumerate(infos["path_to_sword"]):
                    if p:
                        cur = memory[i]["paths"].get("to_sword")
                        if cur is None or len(p) < len(cur):
                            memory[i]["paths"]["to_sword"] = p
                            memory[i]["paths"]["to_sword_reversed"] = list(reversed(p))
                            # Recompute sword->guard for every known guard room
                            computed = memory[i]["paths"].setdefault("computed_sword_to_guard_by_room", {})
                            for gr, gp in memory[i]["paths"].get("to_guard_by_room", {}).items():
                                computed[gr] = compute_sword_to_guard_path(p, gp)
                            dirty[i] = True
                            needs_hint_refresh = True
                if needs_hint_refresh:
                    envs.call("set_pbrs_hint", merge_pbrs_hints(memory))

            if "path_to_guard" in infos and infos["path_to_guard"] is not None:
                needs_hint_refresh = False
                for i, ev in enumerate(infos["path_to_guard"]):
                    if ev is not None:
                        guard_room, p = ev
                        by_room = memory[i]["paths"].setdefault("to_guard_by_room", {})
                        computed = memory[i]["paths"].setdefault("computed_sword_to_guard_by_room", {})
                        cur = by_room.get(guard_room)
                        if cur is None or len(p) < len(cur):
                            by_room[guard_room] = p
                            computed[guard_room] = compute_sword_to_guard_path(
                                memory[i]["paths"].get("to_sword"), p)
                            dirty[i] = True
                            needs_hint_refresh = True
                if needs_hint_refresh:
                    envs.call("set_pbrs_hint", merge_pbrs_hints(memory))

            if "potion_found_at" in infos and infos["potion_found_at"] is not None:
                for i, pf in enumerate(infos["potion_found_at"]):
                    if pf is not None:
                        update_poi_memory(memory[i], pf[0], pf[1], pf[2], f"potion_{pf[3]}")
                        dirty[i] = True

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
                print(f"  env{i}: rooms={sorted(ro_rooms[i])}  post_sword={sorted(ro_post_sword_rooms[i])}  [{ev_str}]")

        # --- GAE with SMDP discount (γ^k per step) ---
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for i in range(args.num_envs):
                if dirty[i]:
                    mem_vec_cache[i] = agent.mem_encoder(memory[i], cache=mem_cache[i])
                    dirty[i] = False
            next_val = agent.get_value(next_obs, mem_vec_cache).reshape(1, -1)
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

        # Use the per-step mem_vecs stored during rollout — these are the exact vectors
        # that produced logprobs[step], so ratio = exp(newlp - oldlp) = 1 for an
        # unchanged policy even when memory evolved mid-rollout.
        b_mem_vecs = obs_mem_vec.reshape(-1, _MEM_DIM)  # (batch, MEM_DIM)

        clipfracs, vf_clipfracs = [], []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                mb_mem_vec = b_mem_vecs[mb]
                mb_obs = {"grid": b_grid[mb].float(), "state": b_state[mb],
                          "room": b_room[mb], "action_history": b_act_hist[mb],
                          "repeat_history": b_rep_hist[mb]}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, newlp, entropy, newval = agent.get_action_and_value(
                        mb_obs, mb_mem_vec, b_actions[mb].long())
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

        if args.dormant_interval > 0 and iteration % args.dormant_interval == 0:
            # Sample 512 transitions from the stored rollout buffers — no env calls needed.
            n_sample = min(512, args.batch_size)
            idx = torch.randperm(args.batch_size, device=device)[:n_sample]
            s_obs = {"grid": b_grid[idx].float(), "state": b_state[idx],
                     "room": b_room[idx], "action_history": b_act_hist[idx],
                     "repeat_history": b_rep_hist[idx]}
            s_mem = b_mem_vecs[idx]
            df = dormant_fractions(agent, s_obs, s_mem)
            parts = "  ".join(f"{k}={v:.3f}" for k, v in df.items())
            print(f"  [dormant] {parts}")


            csv_writer.writerow([iteration, global_step, int(time.time()-start_time), fps,
                pg_loss.item(), v_loss.item(), entropy.mean().item(),
                approx_kl.item(), float(np.mean(clipfracs)), float(np.mean(vf_clipfracs)),
                explained_var, optimizer.param_groups[0]["lr"],
                float(np.sqrt(ret_rms.var + 1e-8)), trunk_weight_norm(agent),
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
            mem_path = os.path.join(runs_dir, "memory.json")
            try:
                with open(mem_path, "w") as f:
                    json.dump(memory, f, indent=2)
                print(f"  memory → {mem_path}")
            except Exception as e:
                print(f"  [memory] save failed: {e}")

        if args.eval_interval > 0 and (iteration % args.eval_interval == 0 or iteration == 1):
            vdir  = os.path.join(runs_dir, "videos"); os.makedirs(vdir, exist_ok=True)
            vpath = os.path.join(vdir, f"eval_iter_{iteration}.mp4")
            run_eval_video(agent, iteration, vpath, args, memory=memory)

    envs.close()
    csv_file.close()
