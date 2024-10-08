import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import tensorflow as tf

class PolarizedBeamEnv(gym.Env):
    def __init__(self):

        self.Ebeam = 11600 #MeV

        #self.action_space = spaces.Box(low=-np.pi/180, high=np.pi/180, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.pi/(500000), high=np.pi/(500000), shape=(1,), dtype=np.float32)        

        low_bounds = np.array([8000, 8600])
        high_bounds = np.array([9000, 8700])

        #low_bounds = np.array([-np.pi/90, 8000, 8600])
        #high_bounds = np.array([np.pi/90, 9000, 8700])
        
        #low_bounds = np.array([-np.pi/36, -np.pi/36, -np.pi, 4000])
        #high_bounds = np.array([np.pi/36, np.pi/36, np.pi, self.Ebeam])        


        self.observation_space = spaces.Tuple(spaces.Discrete(3),)
        #self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)        

        
        self.df = pd.read_csv('../data/spring2023_nudge_final.csv')
        self.df = self.df[self.df['start_edge'] > 0]
        self.df = self.df[self.df['plane'] == 2]
        self.df = self.df[self.df['phi022'] == 0]        
        
        
        self.nsteps = 0
        self.iterations = 0        
        self._max_episode_steps = 20 # Should be able to calculate this (not a problem as we can get the proper solution in a single step)

        self.states, _ = self.reset()
        

    def _get_obs(self):
        #return np.array([self.pitch, self.edge, self.req_edge]) # beam position is two variables
        #return np.array([self.edge, self.req_edge]) # beam position is two variables    
        #return np.array([self.pitch, self.yaw, self.roll, self.edge]) # beam position is two variables    
        obs = (self.edge-self.req_edge)/np.absolute(self.edge-self.req_edge)
        return np.array([obs])
        
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
        if self.phi022!=0:
            print("Invalid phi022 value.")
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


    def get_delta_c_from_delta_pitch_yaw(self, pitch, yaw):

        if self.plane==1:
            if self.mode==2 or self.mode==3:
                if pitch>=0 and yaw>=0:
                    c = np.sqrt(pitch*pitch + yaw*yaw)
                elif pitch<=0 and yaw<=0:
                    c = -np.sqrt(pitch*pitch + yaw*yaw)

            else:
                if pitch>=0 and yaw>=0:
                    c = -np.sqrt(pitch*pitch + yaw*yaw)
                elif pitch<=0 and yaw<=0:
                    c = np.sqrt(pitch*pitch + yaw*yaw)

        else:
            if self.mode==1 or self.mode==4:
                c=-pitch
                #if pitch>=0 and yaw<=0:
                    #c = -np.sqrt(pitch*pitch + yaw*yaw)
                #elif pitch<=0 and yaw>=0:
                    #c = np.sqrt(pitch*pitch + yaw*yaw)

            else:
                c=pitch
                #if pitch>=0 and yaw<=0:
                    #c = np.sqrt(pitch*pitch + yaw*yaw)
                #elif pitch<=0 and yaw>=0:
                    #c = -np.sqrt(pitch*pitch + yaw*yaw)

        tf.summary.scalar(
            "delta c", data=c, step=self.iterations)
                    
        return c
    
    def get_delta_c_from_delta_E(self):
        k = 26.5601 #MeV
        g = 2
        E0 = self.Ebeam
        Ei = self.edge
        Ef = self.req_edge
        
        delta_c = (k/g)*(Ef-Ei)/((E0-Ei)*(E0-Ef)) #in radians

        return delta_c

    
    def get_optimal_action(self):
        #from moveCbrem.sh script
        c=self.get_delta_c_from_delta_E() #radians

        phi=self.phi022        
        cosphi=np.cos(phi)
        sinphi=np.sin(phi)

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

        action_optimal = self.get_optimal_action()

        tf.summary.scalar(
            "Action Env 0", data=action_optimal[0], step=self.iterations)

        tf.summary.scalar(
            "Action Env 1", data=action_optimal[1], step=self.iterations)

        #self.pitch += action_optimal[0]
        #self.yaw += action_optimal[1]

        delta_pitch = action[0]
        #delta_yaw = action[1]
        delta_yaw = 0        
        
        self.pitch += delta_pitch
        self.yaw += delta_yaw        

        new_edge = self.new_edge(self.get_delta_c_from_delta_pitch_yaw(delta_pitch, delta_yaw))
        #new_edge = self.new_edge(self.get_delta_c_from_delta_pitch_yaw(action_optimal[0], action_optimal[1]))        

        #print("Step %i, edge %f" % (self.nsteps, self.edge))
        
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

