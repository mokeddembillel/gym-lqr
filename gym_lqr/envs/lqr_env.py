import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class LqrEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dim, x_bound, u_bound):
        
        # Dimention n
        self.n = dim
        # Current State
        self.x = None
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

        self.gamma = 0.9
        self.P = np.ones((self.n, self.n))

        
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
    
    def get_Q(self, x, u):
        q = self.get_cost(x, u) + self.gamma * self.get_V(x)
        return -q
    
    def get_V(self, x):
        v = np.dot(np.dot(np.squeeze(x.transpose()), self.P), np.squeeze(x))
        return v

    def get_cost(self, x, u):
        # print(x)
        # print(u)
        c = np.dot(np.dot(np.squeeze(x.transpose()), self.S), np.squeeze(x)) + \
            np.dot(np.dot(np.squeeze(u.transpose()), self.R), np.squeeze(u))
        return c
    
    def get_P(self):
        return self.P
    
    def set_P(self, P):
        self.P = P

     
    def reset(self, init_x, max_steps):
        self.x = init_x

        self.S = np.identity(self.n)
        self.R = np.identity(self.m)

        self.A = np.ones((self.n,self.n))
        self.B = np.ones((self.n,self.m))

        self.steps = 0
        self.max_steps = max_steps
        return self.x
        
     
    def step(self, action):
        self.u = np.array(action)
        self.x_old = self.x
        
        # Calculating the cost
        self.c = self.get_cost(self.x, self.u)
        # print(self.c)
        # Calculating the new State
        self.x = np.dot(self.A, np.squeeze(self.x)) + np.dot(self.B, np.squeeze(self.u))
        # print(self.x)

        self.P = self.S \
            + self.gamma * np.dot(np.dot(np.squeeze(self.A.transpose()), self.P), self.A) \
            - np.power(self.gamma, 2) * np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.squeeze(self.A.transpose()), self.P), self.B), \
            np.linalg.inv(self.R + self.gamma * np.dot(np.dot(np.squeeze(self.B.transpose()), self.P), self.B))), \
            np.squeeze(self.B.transpose())), self.P), self.A)

        #print(self.P)

        self.steps += 1
        return self.x, -self.c, 1 if self.steps > self.max_steps else 0, None
        

    def render(self, mode='human'):
        pass
    def close(self):
        pass

    







































