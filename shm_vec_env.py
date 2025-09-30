"""
SharedMemoryVectorEnv — drop-in replacement for gym.vector.AsyncVectorEnv.

Each worker subprocess writes its obs dict directly into a pre-allocated
shared memory slot. The main process reads from the same physical RAM pages
with no pickle, no OS pipe copy, and no unpickling.

Architecture:
  Main process:
    - Allocates one SharedMemory block per obs key per env.
    - Sends actions to workers via multiprocessing.Queue.
    - Workers signal completion via a multiprocessing.Barrier or Event.
    - Main reads obs from shared memory after barrier.

  Worker process (one per env):
    - Receives action from its action Queue.
    - Runs env.step() / env.reset().
    - Writes obs arrays into its shared memory slots.
    - Posts to the result Pipe (only reward/done/info — small payloads).
"""

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import gymnasium as gym
from gymnasium.vector.utils import concatenate, create_empty_array
import time


def _worker(
    env_fn,
    action_queue: mp.Queue,
    result_pipe,        # sends back (reward, terminated, truncated, info)
    shm_names: dict,    # key -> shm name
    shm_shapes: dict,   # key -> shape
    shm_dtypes: dict,   # key -> dtype
    idx: int,
    seed: int,
):
    """Worker process: env lives here. Obs go to shared memory, scalars to pipe."""
    env = env_fn()
    obs, info = env.reset(seed=seed)

    # Attach to shared memory segments
    shms = {k: SharedMemory(name=shm_names[k]) for k in shm_names}
    bufs = {}
    for k in shm_names:
        arr = np.ndarray(shm_shapes[k], dtype=shm_dtypes[k], buffer=shms[k].buf)
        bufs[k] = arr

    def write_obs(obs):
        for k, arr in bufs.items():
            np.copyto(arr, obs[k])

    write_obs(obs)
    result_pipe.send(("reset_done", info))

    while True:
        msg = action_queue.get()
        if msg is None:  # sentinel: shutdown
            break
        cmd, payload = msg
        if cmd == "step":
            action = payload
            obs, reward, terminated, truncated, info = env.step(action)
            write_obs(obs)
            result_pipe.send(("step_done", reward, terminated, truncated, info))
        elif cmd == "reset":
            kwargs = payload
            obs, info = env.reset(**kwargs)
            write_obs(obs)
            result_pipe.send(("reset_done", info))

    env.close()
    for shm in shms.values():
        shm.close()


class SharedMemoryVectorEnv:
    """
    Vectorized env using SharedMemory for zero-copy obs transfer.
    API-compatible with gym.vector.AsyncVectorEnv for the subset used in agent1.py:
      - reset(seed=...)
      - step(actions)
      - single_observation_space / single_action_space
      - observation_space / action_space
      - close()
    """

    def __init__(self, env_fns, context="spawn"):
        self.num_envs = len(env_fns)
        self.ctx = mp.get_context(context)

        # Build observation/action spaces from a temp env
        dummy = env_fns[0]()
        self.single_observation_space = dummy.observation_space
        self.single_action_space = dummy.action_space
        dummy.close()

        # Build observation space for vector env (batch dim prepended)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        obs_space = self.single_observation_space

        # Allocate one SharedMemory block per obs key per env
        self._shm = {}       # (env_idx, key) -> SharedMemory
        self._bufs = {}      # (env_idx, key) -> np.ndarray view
        self._shm_meta = {}  # key -> (shape, dtype)

        for key, space in obs_space.spaces.items():
            shape = space.shape
            dtype = np.dtype(space.dtype)
            self._shm_meta[key] = (shape, dtype)
            for i in range(self.num_envs):
                nbytes = int(np.prod(shape)) * dtype.itemsize
                shm = SharedMemory(create=True, size=max(nbytes, 1))
                arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                self._shm[(i, key)] = shm
                self._bufs[(i, key)] = arr

        # Build shm_names dicts per worker (env_idx -> {key: shm_name})
        self._action_queues = []
        self._result_pipes = []
        self._processes = []

        for i, fn in enumerate(env_fns):
            shm_names = {k: self._shm[(i, k)].name for k in self._shm_meta}
            shm_shapes = {k: v[0] for k, v in self._shm_meta.items()}
            shm_dtypes = {k: v[1] for k, v in self._shm_meta.items()}

            aq = self.ctx.Queue(maxsize=2)
            parent_conn, child_conn = self.ctx.Pipe(duplex=False)

            p = self.ctx.Process(
                target=_worker,
                args=(fn, aq, child_conn, shm_names, shm_shapes, shm_dtypes, i, i),
                daemon=True,
            )
            p.start()
            child_conn.close()  # only parent reads from parent_conn

            self._action_queues.append(aq)
            self._result_pipes.append(parent_conn)
            self._processes.append(p)

        # Wait for all workers to finish their first reset
        for pipe in self._result_pipes:
            pipe.recv()  # ("reset_done", info)

    def _read_obs(self):
        """Gather obs from all shared memory slots into batched numpy arrays."""
        obs = {}
        for key, (shape, dtype) in self._shm_meta.items():
            batch = np.empty((self.num_envs,) + shape, dtype=dtype)
            for i in range(self.num_envs):
                np.copyto(batch[i], self._bufs[(i, key)])
            obs[key] = batch
        return obs

    def reset(self, seed=None, options=None):
        seeds = [None] * self.num_envs
        if seed is not None:
            if isinstance(seed, int):
                seeds = [seed + i for i in range(self.num_envs)]
            else:
                seeds = seed

        for i, aq in enumerate(self._action_queues):
            kwargs = {}
            if seeds[i] is not None:
                kwargs["seed"] = seeds[i]
            aq.put(("reset", kwargs))

        infos = {}
        for i, pipe in enumerate(self._result_pipes):
            msg = pipe.recv()
            infos[i] = msg[1]  # ("reset_done", info)

        obs = self._read_obs()
        return obs, infos

    def step(self, actions):
        for i, aq in enumerate(self._action_queues):
            aq.put(("step", actions[i]))

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        infos = {}

        for i, pipe in enumerate(self._result_pipes):
            msg = pipe.recv()  # ("step_done", reward, terminated, truncated, info)
            _, reward, terminated, truncated, info = msg
            rewards[i] = reward
            terminations[i] = terminated
            truncations[i] = truncated
            infos[i] = info

        obs = self._read_obs()
        return obs, rewards, terminations, truncations, infos

    def close(self):
        for aq in self._action_queues:
            aq.put(None)  # sentinel
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        # Free shared memory
        for shm in self._shm.values():
            shm.close()
            try:
                shm.unlink()
            except Exception:
                pass
