import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class LqrEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dim, init_x, x_bound, u_bound):
        
        # Dimention n
        self.n = dim
        # Initial State
        self.init_x = init_x
        # Current State
        self.x = self.init_x
        # Old State
        self.x_old = None
        # Dimention m
        self.m = dim
        # Action
        self.u = None
        # Cost
        self.c = None
        
        # Constant matricies
        self.A = None
        self.S = None
        self.B = None
        self.R = None
        
        # Set Action space
        self.action_space = spaces.Box(low=-x_bound, high=x_bound, shape=(self.n,))
        # Set State space
        self.observation_space = spaces.Box(low=-u_bound, high=u_bound, shape=(self.m,))
        
        
    # def _randomSDM(matrixSize):
    #     A = np.random.rand(matrixSize, matrixSize)
    #     B = np.dot(A, A.transpose())
    #     print B
    
    # def _randomPDM(matrixSize):
    #     A = np.random.rand(matrixSize, matrixSize)
    #     B = np.dot(A, A.transpose())
    #     B[i][i] += 1 for i in range(len(B))
    #     print B
        
     
    def reset(self):
        self.x = self.init_x
        self.u = None
        self.S = np.identity(self.n)
        self.R = np.identity(self.m)
        return self.x
        
     
    def step(self, action):
        self.u = action
        self.x_old = self.x
        
        # Calculating the cost
        self.c = np.dot(np.dot(self.x.transpose(), self.S), self.x) + \
            np.dot(np.dot(self.u.transpose(), self.R), self.u)
        
        # Calculating the new State
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        return self.x, -self.c, 0, None
        
    
        
    def render(self, mode='human'):
        pass
    def close(self):
        pass

    







































