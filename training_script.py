"""
R2D2 + Phase Transition Value Flooding — Training Script for Prince of Persia

Custom training loop with rich Tensorboard + console logging matching ppo_mamba.py style.
Tracks PoP-specific metrics: rooms visited, sword pickups, guard kills, phase transitions,
Q-value distributions by capability state, and checkpoint curriculum stats.

Usage:
    .venv/bin/python training_script.py --num_actors 8 --num_iterations 10

Requires:
    - Rebuilt libSDLPoP.so with rl_save_checkpoint/rl_load_checkpoint functions
    - deep_rl_zoo package in PYTHONPATH
"""

from absl import app
from absl import flags
from absl import logging
import os
import sys
import time
import collections

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import queue
import numpy as np
import torch
import copy

# Add project root and deep_rl_zoo to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'deep_rl_zoo'))

from deep_rl_zoo.networks.value import R2d2DqnMlpNet, RnnDqnNetworkInputs
from deep_rl_zoo.r2d2 import agent
from deep_rl_zoo import main_loop
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib
from deep_rl_zoo import types as types_lib

from clean_env import PoPEnv, OBS_DIM, N_ACTIONS

# Tensorboard — optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

# ============================================================
# Flags — R2D2 hyperparameters tuned for Prince of Persia
# Laptop-friendly defaults (4 actors). Scale up with --num_actors 16
# for servers.  Total steps ≈ num_actors × num_train_steps × num_iterations.
# ============================================================
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_actors', 8, 'Number of actor processes.')
flags.DEFINE_integer('replay_capacity', 10000, 'Max replay size (sequences). ~180KB each, 10K≈1.8GB RAM.')
flags.DEFINE_integer('min_replay_size', 500, 'Min replay before learning starts.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm.')

flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for Adam.')
flags.DEFINE_float('adam_eps', 1e-3, 'Epsilon for Adam optimizer.')
flags.DEFINE_float('discount', 0.997, 'Discount factor gamma.')
flags.DEFINE_integer('unroll_length', 80, 'Sequence unroll length.')
flags.DEFINE_integer('burn_in', 40, 'Burn-in steps for LSTM hidden state warm-up.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for learner updates.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priority exponent for PER.')
flags.DEFINE_float('importance_sampling_exponent', 0.6, 'IS exponent for PER.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize IS weights.')
flags.DEFINE_float('priority_eta', 0.9, 'Eta for mixed max/mean TD error priorities.')
flags.DEFINE_float('rescale_epsilon', 0.001, 'Epsilon for value function rescaling.')
flags.DEFINE_integer('n_step', 5, 'N-step bootstrap horizon.')

flags.DEFINE_integer('num_iterations', 100, 'Number of training iterations.')
flags.DEFINE_integer('num_train_steps', int(5e5), 'Train steps per iteration per actor.')
flags.DEFINE_integer('num_eval_steps', int(1e4), 'Eval steps per iteration.')
flags.DEFINE_integer('target_net_update_interval', 400, 'Target net update interval.')
flags.DEFINE_integer('actor_update_interval', 400, 'Actor network sync interval.')
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Eval epsilon-greedy epsilon.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Enable Tensorboard logging.')
flags.DEFINE_bool('actors_on_gpu', False, 'Run actors on GPU.')

flags.DEFINE_float('checkpoint_start_prob', 0.5,
                   'Probability explorative actors start from post-sword checkpoint.')
flags.DEFINE_float('phase_transition_priority_multiplier', 5.0,
                   'Priority boost for sword-pickup sequences.')

flags.DEFINE_integer('log_interval', 100, 'Console log every N learner updates.')
flags.DEFINE_integer('save_interval', 1000, 'Save checkpoint every N learner updates.')
flags.DEFINE_string('tag', '', 'Tag for Tensorboard log.')
flags.DEFINE_string('results_csv_path', './logs/r2d2_pop_results.csv', 'CSV log path.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Checkpoint save directory.')
flags.DEFINE_string('run_name', '', 'Run name for logging. Auto-generated if empty.')


# ============================================================
# Gym-compatible wrapper for PoPEnv
# ============================================================
class PoPGymAdapter:
    """Wraps PoPEnv (gymnasium 5-tuple) to match deep_rl_zoo's old gym 4-tuple API.
    Implements checkpoint curriculum for post-sword exploration.
    """

    def __init__(self, checkpoint_start_prob=0.0, visual=False):
        self._env = PoPEnv(visual=visual)
        self._checkpoint_start_prob = checkpoint_start_prob
        self._rng = np.random.RandomState()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        obs, info = self._env.reset()
        # Checkpoint curriculum: sometimes start from post-sword state
        if (self._checkpoint_start_prob > 0.0
                and self._env.has_checkpoint()
                and self._rng.random() < self._checkpoint_start_prob):
            self._env.load_checkpoint()
            obs = self._env._get_obs()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        pass

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    @property
    def env(self):
        return self._env


# ============================================================
# Actor process — runs env loop and sends transitions to queue
# ============================================================
def run_actor_loop(
    rank: int,
    num_train_steps: int,
    data_queue: multiprocessing.Queue,
    shared_params: dict,
    stats_queue: multiprocessing.Queue,
    # Serializable config for building env + actor inside child process
    checkpoint_start_prob: float,
    seed: int,
    network_state_dict: dict,
    num_actors: int,
    action_dim: int,
    state_dim: int,
    unroll_length: int,
    burn_in: int,
    actor_update_interval: int,
    actor_device: str,
):
    """Run actor loop in a separate process. Creates env and actor locally
    to avoid pickling ctypes.CDLL and multiprocessing.Queue objects."""

    # ---- Build env inside child process ----
    env = PoPGymAdapter(checkpoint_start_prob=checkpoint_start_prob, visual=False)
    env.seed(seed + rank)

    # ---- Build actor inside child process ----
    device = torch.device(actor_device)
    net = R2d2DqnMlpNet(state_dim=state_dim, action_dim=action_dim)
    net.load_state_dict(network_state_dict)
    actor = agent.Actor(
        rank=rank, data_queue=data_queue, network=net,
        random_state=np.random.RandomState(seed + int(rank)),
        num_actors=num_actors, action_dim=action_dim,
        unroll_length=unroll_length, burn_in=burn_in,
        actor_update_interval=actor_update_interval,
        device=device, shared_params=shared_params,
    )

    # Override Ape-X epsilon schedule with PoP-specific higher exploration.
    # Standard Ape-X max is 0.4 — far too low for PoP's 18-action sequential navigation.
    # Explorative actors (first half): high epsilon for diverse trajectory generation.
    # Exploitative actors (second half): lower epsilon to exploit learned Q-values.
    if rank < num_actors // 2:
        # Explorative: ε from 1.0 (fully random) down to 0.5
        actor._exploration_epsilon = 1.0 - 0.5 * (rank / max(num_actors // 2 - 1, 1))
    else:
        # Exploitative: ε from 0.3 down to 0.01
        exploit_idx = rank - num_actors // 2
        exploit_n = num_actors - num_actors // 2
        actor._exploration_epsilon = 0.3 * (0.01 / 0.3) ** (exploit_idx / max(exploit_n - 1, 1))

    step_count = 0
    episode_count = 0
    episode_returns = []
    rooms_visited = set()
    sword_found_count = 0
    guard_kills = 0
    deaths = 0
    checkpoint_starts = 0
    phase_transitions_seen = 0

    while step_count < num_train_steps:
        actor.reset()
        obs = env.reset()

        # Track if this was a checkpoint start
        if env._env.sword_found:  # Started from post-sword checkpoint
            checkpoint_starts += 1

        episode_return = 0.0
        done = False
        first_step = True

        while not done and step_count < num_train_steps:
            if first_step:
                timestep = types_lib.TimeStep(
                    observation=obs, reward=0.0, done=False,
                    first=True, info={},
                )
            else:
                timestep = types_lib.TimeStep(
                    observation=obs,
                    reward=reward,
                    done=done,
                    first=False,
                    info=info,
                )

            a_t = actor.step(timestep)
            obs, reward, done, info = env.step(a_t)

            episode_return += reward
            step_count += 1
            first_step = False

            # Track PoP-specific metrics
            if 'room' in info:
                rooms_visited.add(info['room'])
            if info.get('sword_found', 0):
                sword_found_count += 1
            if info.get('phase_transition', False):
                phase_transitions_seen += 1
            if info.get('guard_killed', 0) > 0:
                guard_kills += info['guard_killed']
            if info.get('dead', False):
                deaths += 1

        # Final timestep
        if done:
            timestep = types_lib.TimeStep(
                observation=obs, reward=reward, done=True,
                first=False, info=info,
            )
            actor.step(timestep)

        episode_returns.append(episode_return)
        episode_count += 1

    # Send stats back to main process
    stats_queue.put({
        'rank': rank,
        'episodes': episode_count,
        'steps': step_count,
        'mean_return': np.mean(episode_returns) if episode_returns else 0.0,
        'rooms': list(rooms_visited),
        'sword_found': sword_found_count,
        'guard_kills': guard_kills,
        'deaths': deaths,
        'checkpoint_starts': checkpoint_starts,
        'phase_transitions': phase_transitions_seen,
    })

    # Signal done
    data_queue.put('PROCESS_DONE')


# ============================================================
# Main training function
# ============================================================
def main(argv):
    """Trains R2D2 agent on Prince of Persia with capability conditioning."""
    del argv

    # Ensure min_replay_size is at least batch_size to avoid sampling errors
    if FLAGS.min_replay_size < FLAGS.batch_size:
        print(f"  ⚠ min_replay_size ({FLAGS.min_replay_size}) < batch_size ({FLAGS.batch_size}), clamping to {FLAGS.batch_size}")
        FLAGS.min_replay_size = FLAGS.batch_size
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)

    # ---- Run name ----
    run_name = FLAGS.run_name or f"R2D2-PoP_{FLAGS.num_actors}act_{FLAGS.unroll_length}u_{time.strftime('%Y%m%d_%H%M%S')}"
    # Use absolute paths — PoPEnv.__init__ does os.chdir(SDLPoP_path) which breaks relative paths
    run_dir = os.path.abspath(os.path.join(FLAGS.checkpoint_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)
    results_csv_path = os.path.abspath(FLAGS.results_csv_path)
    os.makedirs(os.path.dirname(results_csv_path) or '.', exist_ok=True)

    # ---- Tensorboard ----
    writer = None
    if FLAGS.use_tensorboard and HAS_TB:
        tb_dir = os.path.abspath(os.path.join('runs', run_name))
        writer = SummaryWriter(tb_dir)
        print(f"Tensorboard: {tb_dir}")

    # ---- Environment builders ----
    def make_actor_env(rank):
        is_explorative = rank < FLAGS.num_actors // 2
        prob = FLAGS.checkpoint_start_prob if is_explorative else 0.0
        env = PoPGymAdapter(checkpoint_start_prob=prob, visual=False)
        env.seed(FLAGS.seed + rank)
        return env

    def make_eval_env():
        return PoPGymAdapter(checkpoint_start_prob=0.0, visual=False)

    eval_env = make_eval_env()
    state_dim = OBS_DIM   # 537
    action_dim = N_ACTIONS  # 18

    # ---- Print config ----
    print(f"\n{'='*70}")
    print(f"  R2D2 + Phase Transition Value Flooding — Prince of Persia")
    print(f"{'='*70}")
    print(f"  Device:       {runtime_device}")
    print(f"  Actors:       {FLAGS.num_actors} (explorative: {FLAGS.num_actors//2}, exploitative: {FLAGS.num_actors - FLAGS.num_actors//2})")
    print(f"  State dim:    {state_dim}  |  Action dim: {action_dim}")
    print(f"  Unroll:       {FLAGS.unroll_length}  |  Burn-in: {FLAGS.burn_in}  |  N-step: {FLAGS.n_step}")
    print(f"  Batch:        {FLAGS.batch_size}  |  Replay: {FLAGS.replay_capacity}")
    print(f"  LR:           {FLAGS.learning_rate}  |  Discount: {FLAGS.discount}")
    print(f"  Checkpoint:   start_prob={FLAGS.checkpoint_start_prob}")
    print(f"  Priority:     phase_transition_multiplier={FLAGS.phase_transition_priority_multiplier}")
    print(f"  Run:          {run_name}")
    print(f"{'='*70}\n")

    # ---- Network ----
    network = R2d2DqnMlpNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, eps=FLAGS.adam_eps)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Network: {total_params/1e3:.1f}K params | LSTM input: {network.lstm.input_size}")

    # Smoke test
    x = RnnDqnNetworkInputs(
        s_t=torch.from_numpy(eval_env.reset()[None, None, ...]).float(),
        a_tm1=torch.zeros(1, 1).long(),
        r_t=torch.zeros(1, 1).float(),
        hidden_s=network.get_initial_hidden_state(1),
        c_t=torch.zeros(1, 1).float(),
    )
    out = network(x)
    assert out.q_values.shape == (1, 1, action_dim)
    print(f"Network forward pass OK: {out.q_values.shape}\n")

    # ---- Replay ----
    importance_sampling_exponent = FLAGS.importance_sampling_exponent
    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=lambda x: importance_sampling_exponent,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
    )

    # ---- Shared state ----
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 2)
    stats_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)
    manager = multiprocessing.Manager()
    shared_params = manager.dict({'network': None})

    # ---- Learner ----
    learner = agent.Learner(
        network=network, optimizer=optimizer, replay=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_net_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discount, burn_in=FLAGS.burn_in,
        priority_eta=FLAGS.priority_eta, rescale_epsilon=FLAGS.rescale_epsilon,
        batch_size=FLAGS.batch_size, n_step=FLAGS.n_step,
        clip_grad=FLAGS.clip_grad, max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device, shared_params=shared_params,
    )
    learner._phase_transition_multiplier = FLAGS.phase_transition_priority_multiplier

    # ---- Actor config (no envs/actors created here — they're built in child processes) ----
    actor_device_strs = ['cpu'] * FLAGS.num_actors
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_device_strs = [f'cuda:{i % num_gpus}' for i in range(FLAGS.num_actors)]

    # Checkpoint start probs per actor (first half explorative)
    actor_ckpt_probs = [
        FLAGS.checkpoint_start_prob if i < FLAGS.num_actors // 2 else 0.0
        for i in range(FLAGS.num_actors)
    ]

    # ---- Eval Agent ----
    eval_agent = greedy_actors.R2d2EpsilonGreedyActor(
        network=network, exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state, device=runtime_device,
    )

    # ---- Training loop ----
    global_step = 0
    start_time = time.time()

    for iteration in range(1, FLAGS.num_iterations + 1):
        iter_start = time.time()
        print(f"\n{'─'*70}")
        print(f"  ITERATION {iteration}/{FLAGS.num_iterations}")
        print(f"{'─'*70}")

        # Snapshot network weights for actors (plain dict of CPU tensors = picklable)
        network_state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

        # Start actor processes — env and actor are created INSIDE each child
        actor_processes = []
        for i in range(FLAGS.num_actors):
            p = multiprocessing.Process(
                target=run_actor_loop,
                args=(i, FLAGS.num_train_steps,
                      data_queue, shared_params, stats_queue),
                kwargs=dict(
                    checkpoint_start_prob=actor_ckpt_probs[i],
                    seed=FLAGS.seed,
                    network_state_dict=network_state_dict,
                    num_actors=FLAGS.num_actors,
                    action_dim=action_dim,
                    state_dim=state_dim,
                    unroll_length=FLAGS.unroll_length,
                    burn_in=FLAGS.burn_in,
                    actor_update_interval=FLAGS.actor_update_interval,
                    actor_device=actor_device_strs[i],
                ),
            )
            p.start()
            actor_processes.append(p)

        # ---- Learner loop ----
        num_done_actors = 0
        iter_learner_updates = 0
        iter_items_received = 0
        iter_phase_transitions_in_replay = 0

        while num_done_actors < FLAGS.num_actors:
            # Pull items from queue
            try:
                item = data_queue.get(timeout=0.01)
                if item == 'PROCESS_DONE':
                    num_done_actors += 1
                    continue
                # Check for phase transition in this item
                if agent.Learner._single_has_phase_transition(item):
                    iter_phase_transitions_in_replay += 1
                learner.received_item_from_queue(item)
                iter_items_received += 1
            except queue.Empty:
                pass
            except EOFError:
                pass

            # Learner step
            stats_sequences = learner.step()
            if stats_sequences is not None:
                for stats in stats_sequences:
                    iter_learner_updates += 1
                    global_step += 1

                    # ── Per-update logging ──
                    if global_step % FLAGS.log_interval == 0:
                        elapsed = time.time() - start_time
                        sps = int(global_step / max(elapsed, 1.0))
                        loss = stats.get('loss', float('nan'))
                        updates = stats.get('updates', 0)
                        target_updates = stats.get('target_updates', 0)

                        print(f"  upd {global_step:6d} | SPS {sps:5d} | "
                              f"loss {loss:8.4f} | "
                              f"replay {replay.size:5d}/{FLAGS.replay_capacity} | "
                              f"actors_done {num_done_actors}/{FLAGS.num_actors} | "
                              f"target_upd {target_updates} | "
                              f"PT_seqs {iter_phase_transitions_in_replay}")

                        if writer:
                            writer.add_scalar("learner/loss", loss, global_step)
                            writer.add_scalar("learner/replay_size", replay.size, global_step)
                            writer.add_scalar("learner/target_updates", target_updates, global_step)
                            writer.add_scalar("learner/phase_transition_seqs", iter_phase_transitions_in_replay, global_step)
                            writer.add_scalar("charts/SPS", sps, global_step)

        # Wait for all actors to finish
        for p in actor_processes:
            p.join(timeout=30)

        # ── Collect actor stats ──
        iter_rooms = set()
        iter_sword = 0
        iter_kills = 0
        iter_deaths = 0
        iter_episodes = 0
        iter_returns = []
        iter_ckpt_starts = 0
        iter_actor_phase_transitions = 0

        while not stats_queue.empty():
            try:
                s = stats_queue.get_nowait()
                iter_rooms.update(s['rooms'])
                iter_sword += s['sword_found']
                iter_kills += s['guard_kills']
                iter_deaths += s['deaths']
                iter_episodes += s['episodes']
                iter_returns.append(s['mean_return'])
                iter_ckpt_starts += s['checkpoint_starts']
                iter_actor_phase_transitions += s['phase_transitions']
            except queue.Empty:
                break

        avg_return = np.mean(iter_returns) if iter_returns else 0.0
        iter_elapsed = time.time() - iter_start

        # ── Iteration summary console log (ppo_mamba style) ──
        print(f"\n  ═══ Iteration {iteration} Summary ═══")
        print(f"  upd {global_step:6d} | SPS {int(global_step / max(time.time() - start_time, 1)):5d} | "
              f"ret {avg_return:8.2f} | eps {iter_episodes:4d} | "
              f"loss {learner._loss_t:8.4f} | learner_upd {iter_learner_updates}")
        print(f"       Rooms({len(iter_rooms)}): {sorted(iter_rooms)} | "
              f"deaths {iter_deaths} | swords {iter_sword} | kills {iter_kills}")
        print(f"       PT_seqs_replay {iter_phase_transitions_in_replay} | "
              f"PT_actor {iter_actor_phase_transitions} | "
              f"ckpt_starts {iter_ckpt_starts} | "
              f"replay {replay.size}/{FLAGS.replay_capacity} | "
              f"time {iter_elapsed:.1f}s")

        # ── Tensorboard iteration logging ──
        if writer:
            writer.add_scalar("charts/avg_return", avg_return, global_step)
            writer.add_scalar("charts/episodes", iter_episodes, global_step)
            writer.add_scalar("dashboard/rooms_visited", len(iter_rooms), global_step)
            writer.add_scalar("dashboard/deaths", iter_deaths, global_step)
            writer.add_scalar("dashboard/sword_found", iter_sword, global_step)
            writer.add_scalar("dashboard/guard_kills", iter_kills, global_step)
            writer.add_scalar("dashboard/phase_transitions_replay", iter_phase_transitions_in_replay, global_step)
            writer.add_scalar("dashboard/phase_transitions_actor", iter_actor_phase_transitions, global_step)
            writer.add_scalar("dashboard/checkpoint_starts", iter_ckpt_starts, global_step)
            writer.add_scalar("learner/learning_rate", optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar("learner/max_priority", learner._max_seen_priority, global_step)
            writer.add_scalar("charts/iteration_time", iter_elapsed, global_step)
            writer.flush()

        # ── Eval ──
        if FLAGS.num_eval_steps > 0:
            print(f"\n  Evaluating ({FLAGS.num_eval_steps} steps)...")
            eval_agent.reset()
            eval_obs = eval_env.reset()
            eval_returns = []
            eval_ep_return = 0.0
            eval_rooms = set()
            eval_sword = 0
            eval_kills = 0

            for eval_step in range(FLAGS.num_eval_steps):
                ts = types_lib.TimeStep(
                    observation=eval_obs, reward=0.0 if eval_step == 0 else eval_reward,
                    done=False if eval_step == 0 else eval_done,
                    first=(eval_step == 0), info={} if eval_step == 0 else eval_info,
                )
                eval_a = eval_agent.step(ts)
                eval_obs, eval_reward, eval_done, eval_info = eval_env.step(eval_a)
                eval_ep_return += eval_reward

                if 'room' in eval_info:
                    eval_rooms.add(eval_info['room'])
                eval_sword += eval_info.get('sword_found', 0)
                eval_kills += eval_info.get('guard_killed', 0)

                if eval_done:
                    eval_returns.append(eval_ep_return)
                    eval_ep_return = 0.0
                    eval_agent.reset()
                    eval_obs = eval_env.reset()

            eval_avg = np.mean(eval_returns) if eval_returns else 0.0
            print(f"  Eval: ret {eval_avg:8.2f} | eps {len(eval_returns)} | "
                  f"rooms {len(eval_rooms)} | sword {eval_sword} | kills {eval_kills}")

            if writer:
                writer.add_scalar("eval/avg_return", eval_avg, global_step)
                writer.add_scalar("eval/episodes", len(eval_returns), global_step)
                writer.add_scalar("eval/rooms", len(eval_rooms), global_step)
                writer.add_scalar("eval/sword_found", eval_sword, global_step)
                writer.add_scalar("eval/guard_kills", eval_kills, global_step)

        # ── Save checkpoint ──
        if iteration % max(1, FLAGS.num_iterations // 5) == 0 or iteration == FLAGS.num_iterations:
            path = os.path.join(run_dir, f"ckpt_iter{iteration}.pt")
            torch.save({
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'global_step': global_step,
                'flags': {k: FLAGS[k].value for k in FLAGS},
            }, path)
            print(f"  saved → {path}")

    # ── Final summary ──
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Training complete! {FLAGS.num_iterations} iterations in {total_time:.1f}s")
    print(f"  Total learner updates: {global_step}")
    print(f"  Final replay size: {replay.size}")
    print(f"{'='*70}")

    if writer:
        writer.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app.run(main)
