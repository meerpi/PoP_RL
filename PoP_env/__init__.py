from gymnasium.envs.registration import register

register(
    id="PoP_env/GridWorld-v0",
    entry_point="PoP_env.envs:GridWorldEnv",
)
