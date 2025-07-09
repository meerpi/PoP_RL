# Clean single-head PPO for Prince of Persia
# Based on CleanRL PPO — no intrinsic rewards, no dual critic
import collections
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from my_env import PoPEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "PrinceOfPersia"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 30
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.25
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(idx):
    def thunk():
        env = PoPEnv(visual=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeReward(env, gamma=0.995)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GridEncoder(nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 3, 10)
            self.output_dim = self.conv(dummy_input).shape[1]

    def forward(self, x):
        return self.conv(x)


class StateEncoder(nn.Module):
    def __init__(self, state_dim=9):
        super().__init__()
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.output_dim = 64

    def forward(self, x):
        return self.mlp(x)


class Agent(nn.Module):
    def __init__(self, action_dim=18):
        super().__init__()
        self.grid_encoder = GridEncoder()
        self.state_encoder = StateEncoder(state_dim=9)
        combined_dim = self.grid_encoder.output_dim + self.state_encoder.output_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_features(self, grid, state):
        g_feat = self.grid_encoder(grid)
        s_feat = self.state_encoder(state)
        return torch.cat([g_feat, s_feat], dim=1)

    def get_value(self, grid, state):
        x = self.get_features(grid, state)
        return self.critic(x)

    def get_action_and_value(self, grid, state, action=None):
        x = self.get_features(grid, state)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
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
    run_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i) for i in range(args.num_envs)],
    )

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs_grid = torch.zeros((args.num_steps, args.num_envs, 20, 3, 10)).to(device)
    obs_state = torch.zeros((args.num_steps, args.num_envs, 9)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    next_grid = torch.Tensor(next_obs["grid"]).to(device)
    next_state = torch.Tensor(next_obs["state"]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    ep_stats = collections.deque(maxlen=100)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_rooms = set()
        iter_start = time.time()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_grid[step] = next_grid
            obs_state[step] = next_state
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_grid, next_state)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            for i in range(args.num_envs):
                try:
                    rollout_rooms.add((int(infos["level"][i]), int(infos["room"][i])))
                except (KeyError, TypeError):
                    pass

            next_grid = torch.Tensor(next_obs["grid"]).to(device)
            next_state = torch.Tensor(next_obs["state"]).to(device)
            next_done = torch.Tensor(next_done).to(device)

            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)

            if "final_info" in infos:
                for idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        ep_stats.append({
                            "r": info["episode"]["r"], "l": info["episode"]["l"],
                            "room": info.get("room", 0), "hp": info.get("hp", 0),
                            "level": info.get("level", 1), "deaths": info.get("deaths", 0),
                            "frontier": info.get("frontier_connections", 0),
                            "ep_rooms": info.get("episode_rooms", 0),
                        })
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/hp", info.get("hp", 0), global_step)
                        writer.add_scalar("charts/room", info.get("room", 0), global_step)
                        writer.add_scalar("charts/deaths", info.get("deaths", 0), global_step)
                        writer.add_scalar("charts/frontier_connections", info.get("frontier_connections", 0), global_step)
                        writer.add_scalar("charts/episode_rooms", info.get("episode_rooms", 0), global_step)

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_grid, next_state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_grid = obs_grid.reshape((-1, 20, 3, 10))
        b_state = obs_state.reshape((-1, 9))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_grid[mb_inds], b_state[mb_inds], b_actions.long()[mb_inds]
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

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_clipped_loss = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        iter_time = time.time() - iter_start
        sps = int(args.batch_size / iter_time)  # per-iteration throughput
        cumulative_sps = int(global_step / (time.time() - start_time))

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/SPS_cumulative", cumulative_sps, global_step)

        # ── Rollout summary ───────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print("\n" + "-" * 50)
        if len(ep_stats) > 0:
            ep_r = np.mean([e["r"] for e in ep_stats])
            ep_l = np.mean([e["l"] for e in ep_stats])
            print(f"| rollout/                  |{'':>14}|")
            print(f"|   ep_rew_mean             | {ep_r:>12.2f} |")
            print(f"|   ep_len_mean             | {ep_l:>12.0f} |")
            print(f"|   avg_room                | {np.mean([e['room'] for e in ep_stats]):>12.1f} |")
            print(f"|   avg_hp                  | {np.mean([e['hp'] for e in ep_stats]):>12.1f} |")
            print(f"|   avg_level               | {np.mean([e['level'] for e in ep_stats]):>12.2f} |")
            print(f"|   deaths                  | {sum(e['deaths'] for e in ep_stats):>12d} |")
            print(f"|   avg_ep_rooms            | {np.mean([e['ep_rooms'] for e in ep_stats]):>12.1f} |")
            print(f"|   frontier_connections    | {sum(e['frontier'] for e in ep_stats):>12d} |")
        if rollout_rooms:
            by_level = {}
            for lv, rm in sorted(rollout_rooms):
                by_level.setdefault(lv, []).append(rm)
            parts = [f"L{lv}:{','.join(map(str, rms))}" for lv, rms in sorted(by_level.items())]
            print(f"|   rooms_visited          | {len(rollout_rooms):>12d} |")
            print(f"|   rooms: {' | '.join(parts)}")
        print(f"| time/                     |{'':>14}|")
        print(f"|   fps (this iter)        | {sps:>12d} |")
        print(f"|   fps (cumulative)       | {cumulative_sps:>12d} |")
        print(f"|   iterations             | {iteration:>12d} |")
        print(f"|   time_elapsed           | {elapsed:>12.0f} |")
        print(f"|   total_timesteps        | {global_step:>12d} |")
        print(f"| train/                    |{'':>14}|")
        print(f"|   approx_kl              | {approx_kl.item():>12.5f} |")
        print(f"|   clip_fraction          | {np.mean(clipfracs):>12.4f} |")
        print(f"|   entropy_loss           | {entropy_loss.item():>12.4f} |")
        print(f"|   explained_variance     | {explained_var:>12.4f} |")
        print(f"|   learning_rate          | {optimizer.param_groups[0]['lr']:>12.7f} |")
        print(f"|   policy_loss            | {pg_loss.item():>12.5f} |")
        print(f"|   value_loss             | {v_loss.item():>12.5f} |")
        print("-" * 50)

    envs.close()
    writer.close()