import gymnasium as gym
from final_env import PoPEnv

env = PoPEnv()
obs, info = env.reset()
for i in range(100):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        env.reset()
print("Run successful")
