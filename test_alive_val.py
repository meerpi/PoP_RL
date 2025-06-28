import env1
from ctypes import *
import time

env = env1.PoPEnv(headless=True)
obs, info = env.reset()
print(f"Alive after reset: {env.data.kid.alive}")
for i in range(200):
   env.lib.rl_sync_wait(1)
   env.lib.rl_get_data(byref(env.data))
   if i % 20 == 0:
       print(f"Frame {i}, alive: {env.data.kid.alive}")
