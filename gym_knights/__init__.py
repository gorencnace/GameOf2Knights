from gym.envs.registration import register

register(
    id='knights-v0',
    entry_point='gym_knights.envs:KnightsEnv',
)