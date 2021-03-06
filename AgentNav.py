from os import name
import numpy as np 
import csv
from matplotlib import pyplot as plt

from TD3 import TD3
import LibFunctions as lib
from HistoryStructs import TrainHistory



class BaseNav:
    def __init__(self, agent_name, sim_conf) -> None:
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer

        self.distance_scale = 10 # max meters for scaling

    def transform_obs(self, obs):
        max_angle = 3.14

        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        target_angle = [obs[5]/max_angle]
        target_distance = [obs[6]/self.distance_scale]

        scan = obs[7:-1]

        nn_obs = np.concatenate([cur_v, cur_d, target_angle, target_distance, scan])
        nn_obs = np.concatenate([cur_d, target_angle, scan])

        return nn_obs

    

class AgentNav(BaseNav):
    def __init__(self, agent_name, sim_conf, load=False) -> None:
        BaseNav.__init__(self, agent_name, sim_conf)
        self.path = 'Vehicles/' + agent_name
        state_space = 2 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        h_size = 200
        self.agent.try_load(load, h_size, self.path)

        self.t_his = TrainHistory(agent_name, load)
        self.velocity = 4

        self.state = None
        self.action = None
        self.nn_state = None
        self.nn_action = None

    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        self.add_memory_entry(obs, nn_obs)

        nn_action = self.agent.act(nn_obs)
        
        self.state = obs
        self.nn_state = nn_obs
        self.nn_action = nn_action

        steering_angle = self.max_steer * nn_action[0]
        self.action = np.array([steering_angle, self.velocity])

        return self.action

    def calcualte_reward(self, s_prime):
        reward = (s_prime[6] - self.state[6]) / self.distance_scale
        reward += s_prime[-1]

        return reward

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = self.calcualte_reward(s_prime)

            self.t_his.add_step_data(reward)
            mem_entry = (self.nn_state, self.nn_action, nn_s_prime, reward, False)

            self.agent.replay_buffer.add(mem_entry)

    def done_entry(self, s_prime):
        reward = self.calcualte_reward(s_prime)
        nn_s_prime = self.transform_obs(s_prime)
        if len(self.t_his.ep_rewards) % 10 == 0:
            self.t_his.print_update()
            self.agent.save(self.path)
        self.state = None
        mem_entry = (self.nn_state, self.nn_action, nn_s_prime, reward, True)
        self.agent.replay_buffer.add(mem_entry)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)

    def reset_lap(self):
        self.state = None
        self.action = None
        self.nn_state = None
        self.nn_action = None


