from gym.envs.registration import register
import numpy as np

register(
    id='lqr-v0',
    entry_point='gym_lqr.envs:LqrEnv',
    max_episode_steps=1000,
    kwargs={'dim' : 1, 'x_bound' : np.inf, 'u_bound' : 10},
)

register(
    id='lqr-2d-v0',
    entry_point='gym_lqr.envs:LqrEnv2d',
    max_episode_steps=1000,
    kwargs={'dim' : 2, 'x_bound' : np.inf, 'u_bound' : 10},
)

register(
    id='lqr-stochastic-v0',
    entry_point='gym_lqr.envs:LqrEnvStochastic',
    max_episode_steps=1000,
    kwargs={'dim_state' : 3, 'dim_action': 3, 'x_bound' : np.inf, 'u_bound' : 10},
)
