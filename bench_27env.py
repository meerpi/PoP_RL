import sys, time, gc
import numpy as np
import multiprocessing as mp
import gymnasium as gym

sys.path.insert(0, './experimental')
sys.path.insert(0, './PrincipiaDev')


class _EnvFn:
    def __call__(self):
        import sys
        sys.path.insert(0, './experimental')
        sys.path.insert(0, './PrincipiaDev')
        import env1
        e = env1.PoPEnv(headless=True, visual_mode=False)
        e = env1.FrameStackWrapper(e, n_frames=5, warmup_steps=3)
        return e


if __name__ == '__main__':
    from shm_vec_env import SharedMemoryVectorEnv

    NUM_ENVS     = 27
    NUM_STEPS    = 2048
    NUM_MINIBATCHES = 9
    BATCH_SIZE   = NUM_ENVS * NUM_STEPS
    MINIBATCH    = BATCH_SIZE // NUM_MINIBATCHES

    print(f"Config: num_envs={NUM_ENVS}  num_steps={NUM_STEPS}")
    print(f"  batch_size={BATCH_SIZE}  minibatch_size={MINIBATCH}")
    print()

    env_fns = [_EnvFn() for _ in range(NUM_ENVS)]
    action_batch = np.array([[4, 2]] * NUM_ENVS)

    # ── SharedMemoryVectorEnv rollout ────────────────────────────────────────
    print("Starting SharedMemoryVectorEnv with 27 envs ...")
    t_spawn = time.perf_counter()
    shm_envs = SharedMemoryVectorEnv(env_fns)
    shm_envs.reset(seed=42)
    t_spawn = time.perf_counter() - t_spawn
    print(f"  Spawn + reset: {t_spawn:.2f}s")

    # Warmup
    for _ in range(20):
        shm_envs.step(action_batch)

    # Full rollout timing (NUM_STEPS steps = 1 full PPO rollout)
    t0 = time.perf_counter()
    for step in range(NUM_STEPS):
        obs, rew, term, trunc, info = shm_envs.step(action_batch)
    rollout_time = time.perf_counter() - t0

    shm_sps = (NUM_STEPS * NUM_ENVS) / rollout_time
    print(f"\n  SharedMemoryVectorEnv SPS: {shm_sps:.0f}")
    print(f"  Full rollout ({NUM_STEPS} steps × {NUM_ENVS} envs): {rollout_time:.3f}s")
    shm_envs.close(); del shm_envs; gc.collect()
    print()

    # ── AsyncVectorEnv rollout ───────────────────────────────────────────────
    print("Starting AsyncVectorEnv with 27 envs ...")
    t_spawn = time.perf_counter()
    async_envs = gym.vector.AsyncVectorEnv(env_fns, context="spawn")
    async_envs.reset(seed=42)
    t_spawn = time.perf_counter() - t_spawn
    print(f"  Spawn + reset: {t_spawn:.2f}s")

    for _ in range(20):
        async_envs.step(action_batch)

    t0 = time.perf_counter()
    for step in range(NUM_STEPS):
        obs, rew, term, trunc, info = async_envs.step(action_batch)
    rollout_time_async = time.perf_counter() - t0

    async_sps = (NUM_STEPS * NUM_ENVS) / rollout_time_async
    print(f"\n  AsyncVectorEnv SPS:        {async_sps:.0f}")
    print(f"  Full rollout ({NUM_STEPS} steps × {NUM_ENVS} envs): {rollout_time_async:.3f}s")
    async_envs.close(); del async_envs; gc.collect()

    print()
    print("=" * 50)
    print(f"  Speedup: {shm_sps/async_sps:.2f}x  (SHM {shm_sps:.0f} vs Async {async_sps:.0f})")
    print(f"  Time per PPO iteration (rollout only):")
    print(f"    SharedMem: {rollout_time:.2f}s")
    print(f"    AsyncVec:  {rollout_time_async:.2f}s")
    print("=" * 50)
