import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from final_env import PoPEnv

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "PrinceOfPersia"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 40
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    spawn_room: int = 1
    visual: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


STATIC_CHANNELS = 20
N_MIXTURES = 8
GAMMA_INT = 0.99

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames=5, warmup_steps=3):
        super().__init__(env)
        self.n_frames = n_frames
        self.warmup_steps = warmup_steps
        orig = env.observation_space["grid"].shape
        stacked = (orig[0] * n_frames, orig[1], orig[2])
        self.observation_space = gym.spaces.Dict({
            "grid": gym.spaces.Box(low=0, high=1, shape=stacked, dtype=np.float32),
            "state": env.observation_space["state"],
            "action_history": env.observation_space["action_history"],
        })
        self.frames = []
        self._ch = orig[0]
        self._s = STATIC_CHANNELS // n_frames
        self._stacked_grid = np.zeros(stacked, dtype=np.float32)

    def _stack(self, obs):
        n, s, d = self.n_frames, self._s, self._ch - 4
        for i, f in enumerate(self.frames):
            self._stacked_grid[i * s:(i + 1) * s] = f[:s]
            self._stacked_grid[n * s + i * d:n * s + (i + 1) * d] = f[s:]
        return {
            "grid": self._stacked_grid.copy(), 
            "state": obs["state"].copy(), 
            "action_history": obs["action_history"].copy()
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs["grid"].copy() for _ in range(self.n_frames)]
        for _ in range(self.warmup_steps):
            obs, _, _, _, info = self.env.step(0)
            self.frames.append(obs["grid"].copy())
            self.frames = self.frames[-self.n_frames:]
        return self._stack(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs["grid"].copy())
        self.frames = self.frames[-self.n_frames:]
        return self._stack(obs), reward, term, trunc, info


class SpawnRoomWrapper(gym.Wrapper):
    def __init__(self, env, spawn_room):
        super().__init__(env)
        self.spawn_room = spawn_room
        from ctypes import c_uint8
        self.unwrapped.lib.do_startpos.argtypes = []
        self.unwrapped.lib.do_startpos.restype = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        from ctypes import c_uint8
        raw_level = (c_uint8 * 2305).in_dll(self.unwrapped.lib, "level")
        raw_level[2112] = self.spawn_room
        raw_level[2113] = 3
        self.unwrapped.lib.do_startpos()
        self.unwrapped.get_values()
        
        obs = self.unwrapped._get_obs()
        info["room"] = self.unwrapped.kid_room
        return obs, info


def make_env(env_id, idx, run_name, args):
    def thunk():
        env = PoPEnv(visual=args.visual)
        if args.spawn_room is not None:
            env = SpawnRoomWrapper(env, args.spawn_room)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return FrameStackWrapper(env, n_frames=5, warmup_steps=5)
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GridEncoder(nn.Module):
    def __init__(self, total_ch, static_ch, H=3, W=10):
        super().__init__()
        self.split = static_ch
        
        def make_branch(in_c, out_c):
            mid = max(in_c // 4, 1)
            attn = nn.Sequential(layer_init(nn.Linear(in_c, mid)), nn.ReLU(),
                                 layer_init(nn.Linear(mid, in_c)), nn.Sigmoid())
            convs = nn.ModuleList([
                layer_init(nn.Conv2d(in_c, out_c, (1, 3), padding=(0, 1))),
                layer_init(nn.Conv2d(in_c, out_c, (1, 5), padding=(0, 2))),
                layer_init(nn.Conv2d(in_c, out_c, (3, 1), padding=(1, 0)))
            ])
            return attn, convs, out_c * H * W * 3

        self.s_attn, self.s_convs, s_dim = make_branch(static_ch, 16)
        self.d_attn, self.d_convs, d_dim = make_branch(total_ch - static_ch, 32)
        
        self.total_dim = s_dim + d_dim
        self.ln = nn.LayerNorm(self.total_dim)

    def forward(self, grid):
        def process(x, attn, convs):
            w = attn(x.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
            x = x * w
            return torch.cat([F.relu(c(x)).flatten(1) for c in convs], dim=1)
            
        s = process(grid[:, :self.split], self.s_attn, self.s_convs)
        d = process(grid[:, self.split:], self.d_attn, self.d_convs)
        return self.ln(torch.cat([s, d], dim=1))

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        grid_shape = envs.single_observation_space["grid"].shape
        state_dim = envs.single_observation_space["state"].shape[0]
        act_hist_dim = envs.single_observation_space["action_history"].shape[0]
        
        C, H, W = grid_shape
        self.grid_enc = GridEncoder(total_ch=C, static_ch=STATIC_CHANNELS, H=H, W=W)
        
        # We concatenate the environment explicit state with the action history vector
        # and we additionally pass the environment discrete mixture one-hot 
        total_state_dim = state_dim + act_hist_dim + N_MIXTURES
        
        self.state_enc = nn.Sequential(
            layer_init(nn.Linear(total_state_dim, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 64)), nn.ReLU(),
        )

        combined = self.grid_enc.total_dim + 64

        self.critic_ext = nn.Sequential(
            layer_init(nn.Linear(combined, 256)), nn.ReLU(),
            nn.LayerNorm(256),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.critic_int = nn.Sequential(
            layer_init(nn.Linear(combined, 256)), nn.ReLU(),
            nn.LayerNorm(256),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(combined, 256)), nn.ReLU(),
            nn.LayerNorm(256),
            layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01),
        )

    def _encode(self, grid, state, act_hist):
        state_repr = torch.cat([state, act_hist], dim=1)
        return torch.cat([self.grid_enc(grid), self.state_enc(state_repr)], dim=1)

    def get_value(self, grid, state, act_hist):
        feat = self._encode(grid, state, act_hist)
        return self.critic_ext(feat), self.critic_int(feat)

    def get_action_and_value(self, grid, state, act_hist, action=None):
        feat = self._encode(grid, state, act_hist)
        logits = self.actor(feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_ext(feat), self.critic_int(feat)


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    orig_cwd = os.getcwd()
    envs = gym.vector.AsyncVectorEnv([make_env(args.env_id, i, run_name, args) for i in range(args.num_envs)])
    os.chdir(orig_cwd)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    def compute_beta_schedule(N, beta_max=0.3):
        betas = torch.zeros(N)
        betas[0] = 0.0
        betas[N-1] = beta_max
        for i in range(1, N-1):
            betas[i] = beta_max * torch.sigmoid(torch.tensor(10.0 * (2*i - (N-2)) / (N-2)))
        return betas

    def compute_gamma_schedule(N, g_max=args.gamma, g_min=GAMMA_INT):
        gammas = torch.zeros(N)
        for i in range(N):
            val = ((N-1-i) * np.log(1-g_max) + i * np.log(1-g_min)) / (N-1)
            gammas[i] = 1 - torch.exp(torch.tensor(val))
        return gammas

    beta_schedule  = compute_beta_schedule(N_MIXTURES)
    gamma_schedule = compute_gamma_schedule(N_MIXTURES)
    
    env_mixture_idx = torch.tensor([i % N_MIXTURES for i in range(args.num_envs)], device=device, dtype=torch.long)
    env_betas  = torch.tensor([beta_schedule[i % N_MIXTURES]  for i in range(args.num_envs)], device=device)
    env_gammas = torch.tensor([gamma_schedule[i % N_MIXTURES] for i in range(args.num_envs)], device=device)

    env_betas_onehot = torch.zeros(args.num_envs, N_MIXTURES, device=device)
    env_betas_onehot.scatter_(1, env_mixture_idx.unsqueeze(1), 1.0)

    # ALGO Logic: Storage setup
    grid_shape = envs.single_observation_space["grid"].shape
    state_shape = envs.single_observation_space["state"].shape[0]
    act_hist_shape = envs.single_observation_space["action_history"].shape
    
    obs_grid = torch.zeros((args.num_steps, args.num_envs) + grid_shape).to(device)
    obs_state = torch.zeros((args.num_steps, args.num_envs, state_shape + N_MIXTURES)).to(device)
    obs_act_hist = torch.zeros((args.num_steps, args.num_envs) + act_hist_shape).to(device)
    
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_ext = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_int = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    
    next_grid = torch.Tensor(next_obs["grid"]).to(device)
    next_state = torch.Tensor(next_obs["state"]).to(device)
    next_act_hist = torch.Tensor(next_obs["action_history"]).to(device) / 17.0
    next_state = torch.cat([next_state, env_betas_onehot], dim=-1)
    next_done = torch.zeros(args.num_envs).to(device)

    episode_returns = deque(maxlen=100)
    episode_combined_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    last_infos = [dict() for _ in range(args.num_envs)]
    ro_rooms = [set() for _ in range(args.num_envs)]
    ro_events = [[] for _ in range(args.num_envs)]
    
    ep_ret_sums = np.zeros(args.num_envs, dtype=np.float32)
    ep_combined_sums = np.zeros(args.num_envs, dtype=np.float32)
    ep_len_sums = np.zeros(args.num_envs, dtype=np.int32)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        ro_events = [[] for _ in range(args.num_envs)]
        ro_rooms = [set() for _ in range(args.num_envs)]

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_grid[step] = next_grid
            obs_state[step] = next_state
            obs_act_hist[step] = next_act_hist
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, v_ext, v_int = agent.get_action_and_value(next_grid, next_state, next_act_hist)
                values_ext[step] = v_ext.flatten()
                values_int[step] = v_int.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            next_grid = torch.Tensor(next_obs["grid"]).to(device)
            next_state_raw = torch.Tensor(next_obs["state"]).to(device)
            next_state = torch.cat([next_state_raw, env_betas_onehot], dim=-1)
            next_act_hist = torch.Tensor(next_obs["action_history"]).to(device) / 17.0
            
            def get_info(k, i, def_val=None):
                v = infos.get(k, [def_val] * args.num_envs)[i]
                if next_done_np[i] and "final_info" in infos and infos["final_info"][i] is not None:
                    v = infos["final_info"][i].get(k, v)
                return v

            r_ep = [get_info("r_episodic", i, 0.0) or 0.0 for i in range(args.num_envs)]
            int_rewards[step] = torch.tensor(r_ep, dtype=torch.float32, device=device)

            for i in range(args.num_envs):
                if (r := get_info("room", i)) is not None: ro_rooms[i].add(int(r))
                if get_info("sword_found", i, False) and not last_infos[i].get("sword_found", False):
                    ro_events[i].append("Sword")
                
                lvl, prev_lvl = get_info("level", i, 1), last_infos[i].get("level", 1)
                if lvl > prev_lvl: ro_events[i].append(f"Lvl {prev_lvl}->{lvl}")

                for k in ("room", "level", "hp", "sword_found", "guard_hp"):
                    if (v := get_info(k, i)) is not None: last_infos[i][k] = v

            next_done = torch.Tensor(next_done_np).to(device)
            
            env_betas_np = env_betas.cpu().numpy()
            ep_ret_sums += reward
            ep_combined_sums += reward + env_betas_np * int_rewards[step].cpu().numpy()
            ep_len_sums += 1

            if np.any(next_done_np):
                for idx in np.where(next_done_np)[0]:
                    er, el = float(ep_ret_sums[idx]), int(ep_len_sums[idx])
                    ecr = float(ep_combined_sums[idx])
                    episode_returns.append(er)
                    episode_combined_returns.append(ecr)
                    episode_lengths.append(el)
                    rooms = len(ro_rooms[idx])
                    sword = last_infos[idx].get("sword_found", False)
                    hp = last_infos[idx].get("hp", 0)
                    lvl = last_infos[idx].get("level", 1)
                    guard_hp = last_infos[idx].get("guard_hp", 0)
                    sps_now = int(global_step / (time.time() - start_time))
                    
                    parts = [f"ep_r={er:.1f}", f"ep_l={el}", f"rooms={rooms}"]
                    if sword: parts.append("SWORD!")
                    if int(hp) > 3: parts.append(f"POTION(hp={hp})")
                    parts += [f"lvl={lvl}", f"ghp={guard_hp}", f"sps={sps_now}"]
                    # uncomment below to print every episode
                    # print("  " + "  ".join(parts))

                    writer.add_scalar("charts/episodic_return", er, global_step)
                    
                    mix_idx = env_mixture_idx[idx].item()
                    if mix_idx == 0:
                        writer.add_scalar("charts/episodic_return_exploit", er, global_step)
                    else:
                        writer.add_scalar("charts/episodic_return_explore", er, global_step)
                    
                    writer.add_scalar("charts/episodic_length", el, global_step)
                    writer.add_scalar("charts/hp", int(hp), global_step)
                    writer.add_scalar("charts/level", lvl, global_step)
                    writer.add_scalar("charts/sword_found", int(sword), global_step)
                    writer.add_scalar("charts/episodic_combined_return", ecr, global_step)
                    
                    last_infos[idx]["sword_found"] = False
                    ro_rooms[idx].clear()
                    ep_ret_sums[idx] = 0.0
                    ep_combined_sums[idx] = 0.0
                    ep_len_sums[idx] = 0

        # Rollout summary
        print(f"\n[Rollout Log] gs={global_step}")
        for i in range(args.num_envs):
            r_list = sorted(ro_rooms[i])
            ev_str = ", ".join(ro_events[i])
            if r_list or ev_str:
                print(f"  Env {i}: Rooms {r_list} | Events: [{ev_str}]")

        with torch.no_grad():
            next_v_ext, next_v_int = agent.get_value(next_grid, next_state, next_act_hist)
            next_v_ext = next_v_ext.reshape(1, -1)
            next_v_int = next_v_int.reshape(1, -1)
            
            adv_ext = torch.zeros_like(rewards).to(device)
            adv_int = torch.zeros_like(rewards).to(device)
            lastgaelam_ext = 0
            lastgaelam_int = 0
            
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextv_ext = next_v_ext
                    nextv_int = next_v_int
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextv_ext = values_ext[t + 1]
                    nextv_int = values_int[t + 1]
                    
                delta_ext = rewards[t] + args.gamma * nextv_ext * nextnonterminal - values_ext[t]
                adv_ext[t] = lastgaelam_ext = delta_ext + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_ext
                
                delta_int = int_rewards[t] + env_gammas * nextv_int * 1.0 - values_int[t]
                adv_int[t] = lastgaelam_int = delta_int + env_gammas * args.gae_lambda * 1.0 * lastgaelam_int
                
            returns_ext = adv_ext + values_ext
            returns_int = adv_int + values_int
            
            adv_ext_norm = (adv_ext - adv_ext.mean()) / (adv_ext.std() + 1e-8)
            adv_int_norm = (adv_int - adv_int.mean()) / (adv_int.std() + 1e-8)
            advantages = adv_ext_norm + env_betas.unsqueeze(0) * adv_int_norm

        b_grid = obs_grid.reshape((-1,) + grid_shape)
        b_state = obs_state.reshape((-1, state_shape + N_MIXTURES))
        b_act_hist = obs_act_hist.reshape((-1,) + act_hist_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns_ext = returns_ext.reshape(-1)
        b_returns_int = returns_int.reshape(-1)
        b_values_ext = values_ext.reshape(-1)
        b_values_int = values_int.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newv_ext, newv_int = agent.get_action_and_value(
                    b_grid[mb_inds], b_state[mb_inds], b_act_hist[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                def get_v_loss(newv, ret, oldv):
                    newv = newv.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newv - ret) ** 2
                        v_clipped = oldv + torch.clamp(newv - oldv, -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        return 0.5 * v_loss_max.mean()
                    else:
                        return 0.5 * ((newv - ret) ** 2).mean()

                v_loss = get_v_loss(newv_ext, b_returns_ext[mb_inds], b_values_ext[mb_inds]) + \
                         get_v_loss(newv_int, b_returns_int[mb_inds], b_values_int[mb_inds])

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

        y_pred, y_true = b_values_ext.cpu().numpy(), b_returns_ext.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/int_reward_mean", int_rewards.mean().item(), global_step)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps, "Step:", global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()