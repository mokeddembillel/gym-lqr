from gym.envs.registration import register

register(
    id='lqr-v0',
    entry_point='gym_lqr.envs:FooEnv',
    max_episode_steps=150,
    kwargs={'dim' : 1, 'init_x' : 10., 'x_bound' : np.inf},
)
