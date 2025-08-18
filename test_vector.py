import gymnasium as gym
import numpy as np

class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.step_cnt = 0
        self.ep = 0

    def reset(self, seed=None, options=None):
        self.step_cnt = 0
        self.ep += 1
        return 0, {"ep": self.ep, "step": self.step_cnt, "msg": "new"}

    def step(self, action):
        self.step_cnt += 1
        term = (self.step_cnt >= 3)
        return 0, 1.0, term, False, {"ep": self.ep, "step": self.step_cnt, "msg": "term" if term else "step"}

envs = gym.vector.AsyncVectorEnv([lambda: DummyEnv()], context="fork", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
obs, info = envs.reset()
print("Reset info:", info)
for i in range(4):
    obs, rew, term, trunc, info = envs.step([0])
    print(f"Step {i+1}: term={term[0]} info={info}")
