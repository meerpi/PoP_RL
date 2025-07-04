import gymnasium as gym
import numpy as np

class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 100, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
        self.step_count = 0
    def reset(self, **kwargs):
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {"my_val": 0, "status": "reset"}
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 3
        return np.array([self.step_count], dtype=np.float32), 1.0, done, False, {"my_val": self.step_count, "status": "terminal" if done else "step"}

if __name__ == '__main__':
    envs = gym.vector.AsyncVectorEnv([lambda: DummyEnv() for _ in range(2)])
    obs, infos = envs.reset()
    for i in range(3):
        obs, rew, term, trunc, infos = envs.step([0, 0])
        print(f"Step {i+1} obs: {obs.flatten()}")
        print(f"Step {i+1} infos keys: {list(infos.keys())}")
        if "final_observation" in infos:
            # Note: final_observation might be an array or tuple depending on dict space
            print(f"Step {i+1} final obs: {infos['final_observation']}")
        if "final_info" in infos:
            print(f"Step {i+1} final info: {infos['final_info']}")
        print(f"Step {i+1} my_val: {infos['my_val']}")
    envs.close()
