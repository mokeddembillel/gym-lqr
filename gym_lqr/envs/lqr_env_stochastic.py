import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class LqrEnvStochastic(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dim_state, dim_action, x_bound, u_bound):
        
        # Dimention n
        self.n = dim_state
        # Current State
        self.x = None
        # Dimention m
        self.m = dim_action
        # Action
        self.u = None
        # Cost
        self.c = None
        
        # Noise parameter
        self.w = None
        
        # Constant matricies
        self.A = None
        self.S = None
        self.B = None
        self.R = None
        
        # Set State space
        self.observation_space = spaces.Box(low=-x_bound, high=x_bound, shape=(self.n,))
        # Set Action space
        self.action_space = spaces.Box(low=-u_bound, high=u_bound, shape=(self.m,))

        # Max number of steps in an episode (initialized using the reset function)
        self.max_steps = None
        # number of steps in an episode
        self.steps = None
        
    # def _randomSDM(matrixSize):
    #     A = np.random.rand(matrixSize, matrixSize)
    #     B = np.dot(A, A.transpose())
    #     print B
    
    # def _randomPDM(matrixSize):
    #     A = np.random.rand(matrixSize, matrixSize)
    #     B = np.dot(A, A.transpose())
    #     B[i][i] += 1 for i in range(len(B))
    #     print B
    

        
     
    def reset(self, init_x, max_steps):
        self.x = np.array([init_x])
        
        self.w = np.random.multivariate_normal([0, 0, 0], np.identity(self.n))

        self.S = np.identity(self.n)*0.001  
        self.R = np.identity(self.m)

        self.A = np.array([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0.0, 0.01, 1.01]])
        self.B = np.identity(self.m)

        self.steps = 0
        self.max_steps = max_steps
        return self.x
        
     
    def step(self, action):
        self.u = np.array(action)        
        # Calculating the cost
        
        self.c = np.dot(np.dot(np.squeeze(self.x.transpose()), self.S), np.squeeze(self.x)) + \
            np.dot(np.dot(np.squeeze(self.u.transpose()), self.R), np.squeeze(self.u))
        # Calculating the new State
        self.x = np.dot(self.A, np.squeeze(self.x)) + np.dot(self.B, np.squeeze(self.u)) + np.squeeze(self.w)

        self.steps += 1
        return self.x, -self.c, 1 if self.steps > self.max_steps else 0, None
        

    def render(self, mode='human'):
        pass
    def close(self):
        pass

    







































