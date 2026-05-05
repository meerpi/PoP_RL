import PoP_env.envs.PoP_env as pop
import numpy as np
env = pop.PoPEnv(headless=True)
obs, info = env.reset()
print("init frame:", env.obs_builder.kid.frame)
for act in [0, 0, 0, 4, 4, 4, 0, 0]:
    obs, r, term, trunc, info = env.step(act)
    print(f"action: {act} -> elapsed: {env.last_tau}, frame: {env.obs_builder.kid.frame}")
