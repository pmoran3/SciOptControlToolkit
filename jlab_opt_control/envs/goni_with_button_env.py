import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import tensorflow as tf

class PolarizedBeamButtonEnv(gym.Env):
    def __init__(self):

        self.Ebeam = 11600 #MeV

        self.action_space = spaces.Box(low=-10**-3, high=10**-3, shape=(1,), dtype=np.float32)

        low_bounds = np.array([8000, 8000])
        high_bounds = np.array([9000, 9000])

        #self.observation_space = spaces.Tuple(spaces.Discrete(3))
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)        

        
        self.df = pd.read_csv('../data/spring2023_nudge_final.csv')
        self.df = self.df[self.df['start_edge'] > 0]
        #self.df = self.df[self.df['plane'] == 2]
        #self.df = self.df[self.df['phi022'] == 0]        
        
        
        self.nsteps = 0
        self.iterations = 0        
        self._max_episode_steps = 20 # Should be able to calculate this (not a problem as we can get the proper solution in a single step)

        self.states, _ = self.reset()
        

    def _get_obs(self):
        return np.array([self.edge, self.req_edge])

        
    def reset(self, seed=None):
        super().reset(seed=seed)

        
        # Utilize historical data to set parameters on the reset (within the dataset)        
        sample = self.df.sample()
        
        self.pitch = np.deg2rad(sample.iloc[0]['start_pitch'])
        self.yaw = np.deg2rad(sample.iloc[0]['start_yaw'])
        self.roll = np.deg2rad(sample.iloc[0]['start_roll'])        

        self.plane = sample.iloc[0]['plane']
        if self.plane == 1:
            self.mode = sample.iloc[0]['para_mode']
        else:
            self.mode = sample.iloc[0]['perp_mode']            
        self.phi022 = np.deg2rad(sample.iloc[0]['phi022']) # Planar polarization (direction of one of the vectors) 

        self.edge = sample.iloc[0]['start_edge']
        self.req_edge = sample.iloc[0]['start_req_edge'] # If you know this value then you can calculate the c angle value
        
        self.beam_pos_x = sample.iloc[0]['start_beam_xpos']
        self.beam_pos_y = sample.iloc[0]['start_beam_ypos']
        
        observation = self._get_obs()

        self.nsteps = 0
        
        return observation, {}


    def new_edge(self, delta_c):

        k = 26.5601 #MeV
        g = 2
        E0 = self.Ebeam
        Ei = self.edge

        Ef = E0*(1-1/((delta_c*g*E0)/k + 1/(1-Ei/E0)))

        return Ef


    def getDeltaC(self, action):
        delta_c=0.0001
        if action<0:
            delta_c*=-1
        return delta_c
    
    def moveCbrem(self, c):
        
        cosphi=np.cos(self.phi022)
        sinphi=np.sin(self.phi022)

        if self.plane==1:
            if self.mode==2 or self.mode==3:
                v = + c*cosphi
                h = + c*sinphi
            else:
                v = - c*cosphi
                h = - c*sinphi
        else:
            if self.mode==1 or self.mode==4:
                v = + c*sinphi
                h = - c*cosphi
            else:
                v = - c*sinphi
                h = + c*cosphi
                
        pitch_change = h
        yaw_change = v        

        return np.array([pitch_change, yaw_change])

    
    def step(self, action):

        self.nsteps += 1
        self.iterations += 1

        goni_change=self.moveCbrem(action[0])
        
        self.pitch += goni_change[0]
        self.yaw += goni_change[1] 

        new_edge = self.new_edge(action[0])
                
        tf.summary.scalar(
            "Current Edge", data=self.edge, step=self.iterations)

        tf.summary.scalar(
            "Nominal Edge", data=self.req_edge, step=self.iterations)        

        tf.summary.scalar(
            "Distance to Nominal Edge", data=abs(self.edge-self.req_edge), step=self.iterations)

        tf.summary.scalar(
            "New Edge", data=new_edge, step=self.iterations)                
        
        reward = np.absolute(self.edge - self.req_edge) - np.absolute(new_edge - self.req_edge) # Change in delta E
        # Reward is a function of your current step and your previous step
        #reward = 1./(np.absolute(new_edge - self.req_edge)+0.00001)


        self.edge = new_edge

        # Episode is done if the edge is within some window of the required edge
        if np.absolute(self.edge-self.req_edge)/self.req_edge < 0.001:
            done=True
        else:
            done=False

        if self.nsteps >= self._max_episode_steps:
            done=True

        self.state = self._get_obs()
        

        return self.state, reward, done, False, {} # Look into truncation
        

        
    def render(self):
        pass

