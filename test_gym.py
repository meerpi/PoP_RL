import gymnasium as gym
import numpy as np

class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
        self.step_count = 0
    def reset(self, **kwargs):
        self.step_count = 0
        return np.array([0.0]), {"level": 1}
    def step(self, action):
        self.step_count += 1
        if self.step_count == 3:
            return np.array([0.0]), 1.0, True, False, {"level": 2}
        return np.array([0.0]), 0.0, False, False, {"level": 1}

env = gym.vector.AsyncVectorEnv([lambda: DummyEnv()], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
obs, info = env.reset()
print(f"reset info: {info}")
for i in range(4):
    obs, reward, terminated, truncated, info = env.step(np.array([0]))
    print(f"step {i+1} info: {info}")
