from gym.envs.registration import register
import numpy as np

register(
    id='lqr-v0',
    entry_point='gym_lqr.envs:LqrEnv',
    max_episode_steps=1000,
    kwargs={'dim' : 1, 'init_x' : 10., 'x_bound' : np.inf, 'u_bound' : 5},
)
