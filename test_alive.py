import env1

env = env1.PoPEnv(headless=True)
obs, info = env.reset()
print(f"After reset, alive status: {env.data.kid.alive}")
for _ in range(5):
    obs, reward, term, trunc, info = env.step(0)
    print(f"After step, alive status: {env.data.kid.alive}")
