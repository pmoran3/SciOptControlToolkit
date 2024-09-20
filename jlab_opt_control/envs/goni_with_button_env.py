import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import tensorflow as tf

class PolarizedBeamButtonEnv(gym.Env):
    def __init__(self):
        self.Ebeam = 11600  # MeV

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.edge_low = 8000
        self.edge_high = 9000
        self.observation_space = spaces.Box(
            low=np.array([self.edge_low, self.edge_low, -1, -1]), 
            high=np.array([self.edge_high, self.edge_high, 1, 1]), 
            dtype=np.float32
        )

        self.df = pd.read_csv('../data/spring2023_nudge_final.csv')
        self.df = self.df[self.df['start_edge'] > 0]

        self.nsteps = 0
        self.iterations = 0
        self._max_episode_steps = 20

        self.reset()

    def _get_obs(self):
        relative_diff = np.clip((self.edge - self.req_edge) / (self.req_edge + 1e-8), -1, 1)
        return np.array([
            self.edge,
            self.req_edge,
            np.sign(self.edge - self.req_edge),
            relative_diff
        ])

    def _normalize_obs(self, obs):
        normalized = (obs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        return np.clip(normalized, 0, 1)

    def reset(self, seed=None):
        super().reset(seed=seed)

        sample = self.df.sample()
        
        self.pitch = np.deg2rad(sample.iloc[0]['start_pitch'])
        self.yaw = np.deg2rad(sample.iloc[0]['start_yaw'])
        self.roll = np.deg2rad(sample.iloc[0]['start_roll'])        

        self.plane = sample.iloc[0]['plane']
        self.mode = sample.iloc[0]['para_mode'] if self.plane == 1 else sample.iloc[0]['perp_mode']
        self.phi022 = np.deg2rad(sample.iloc[0]['phi022'])

        self.edge = np.clip(sample.iloc[0]['start_edge'], self.edge_low, self.edge_high)
        self.req_edge = np.clip(sample.iloc[0]['start_req_edge'], self.edge_low, self.edge_high)
        
        self.beam_pos_x = sample.iloc[0]['start_beam_xpos']
        self.beam_pos_y = sample.iloc[0]['start_beam_ypos']
        
        self.nsteps = 0

        self._log_to_tensorboard()
        
        return self._normalize_obs(self._get_obs()), {}


    def new_edge(self, delta_c):

        k = 26.5601 #MeV (Constant which is a function of many other variables)
        g = 2 # Reciprical lattice vector producing the primary peak (022 vector) described in paper
        E0 = self.Ebeam
        Ei = self.edge

        Ef = E0*(1-1/((delta_c*g*E0)/k + 1/(1-Ei/E0)))

        return Ef

    
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

        scaled_action = np.clip(action[0], -1, 1) * 0.001
        delta_c = np.deg2rad(scaled_action)
        
        goni_change = self.moveCbrem(delta_c)
        
        self.pitch += goni_change[0]
        self.yaw += goni_change[1] 

        new_edge = np.clip(self.new_edge(delta_c), self.edge_low, self.edge_high)
        
        relative_error = abs(new_edge - self.req_edge) / (self.req_edge + 1e-8)
        reward = self.calculate_reward(relative_error)

        action_penalty = 0.1 * abs(action[0])
        reward -= action_penalty

        self.edge = new_edge

        done = relative_error < 0.001 or self.nsteps >= self._max_episode_steps

        self._log_to_tensorboard()

        return self._normalize_obs(self._get_obs()), reward, done, False, {}

    def calculate_reward(self, relative_error):
        a = 10  # Maximum reward
        b = 0.01  # Controls the steepness of the curve
        c = 0.001  # Controls the center point of the curve
        safe_exp = np.clip((relative_error - c) / b, -709, 709)  # Prevent exp overflow
        return a / (1 + np.exp(safe_exp))
        
    def render(self):
        pass

    def _log_to_tensorboard(self):
        tf.summary.scalar('Current Edge', self.edge, step=self.iterations)
        tf.summary.scalar('Required Edge', self.req_edge, step=self.iterations)
        tf.summary.scalar('Edge Difference', self.edge - self.req_edge, step=self.iterations)
        tf.summary.scalar('Relative Error', abs(self.edge - self.req_edge) / (self.req_edge + 1e-8), step=self.iterations)

# Uncomment for GP solution
# import matplotlib.pyplot as plt
# from scipy import stats

# def calculate_edge_shift(env, action):
#     delta_c = np.deg2rad(action[0])
#     new_edge = env.new_edge(delta_c)
#     shift = new_edge - env.edge
#     return shift

# def derive_relationship(env):
#     actions = np.linspace(-1e-3, 1e-3, 1000)
#     shifts = []
#     for action in actions:
#         env.reset()
#         shift = (env, [action])
#         shifts.append(shift)
    
#     # Fit a linear regression
#     slope, intercept, r_value, p_value, std_err = stats.linregress(actions, shifts)
    
#     print(f"Linear fit: shift = {slope:.6f} * action + {intercept:.6f}")
#     print(f"R-squared: {r_value**2:.6f}")
    
#     return slope, intercept

# def optimal_action(current_edge, required_edge, slope):
#     needed_shift = required_edge - current_edge
#     action = needed_shift / slope
#     return np.clip(action, -1e-3, 1e-3)  # Clip to action space bounds

# def main():
#     env = PolarizedBeamButtonEnv()
    
#     slope, intercept = derive_relationship(env)
    
#     # Plot the relationship
#     actions = np.linspace(-1e-3, 1e-3, 1000)
#     shifts = slope * actions + intercept
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(actions, shifts, label='Fitted Line')
#     plt.scatter(actions, [calculate_edge_shift(env, [a]) for a in actions], alpha=0.5, label='Actual Data')
#     plt.title('Action vs Edge Shift')
#     plt.xlabel('Action (radians)')
#     plt.ylabel('Edge Shift (MeV)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('action_shift_relationship.png')
#     print("\nPlot saved as 'action_shift_relationship.png'")
#     plt.close()
    
#     # Example usage of optimal_action function
#     current_edge = 8500  # Example current edge
#     required_edge = 8503  # Example required edge
#     action = optimal_action(current_edge, required_edge, slope)
#     print(f"\nFor current edge {current_edge} MeV and required edge {required_edge} MeV:")
#     print(f"Optimal action: {action:.6f}")
#     print(f"Predicted shift: {slope * action:.6f} MeV")

# if __name__ == "__main__":
#     main()