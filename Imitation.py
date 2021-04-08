import numpy as np 
from matplotlib import pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import LibFunctions as lib

MEMORY_SIZE = 100000


# hyper parameters
BATCH_SIZE = 100
GAMMA = 0.99
tau = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
POLICY_NOISE = 0.2


class BufferIL(object):
    def __init__(self, max_size=1000000):     
        #TODO: change from list to array
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions = [], []

        for i in ind: 
            s, a = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))

        return np.array(states), np.array(actions)

    def size(self):
        return len(self.storage)

    def load_data(self, name):
        filename = "Vehicles/" + name + ".npy"
        self.storage = np.load(filename, allow_pickle=True)





class Actor(nn.Module):   
    def __init__(self, state_dim, action_dim, max_action, h_size):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) 
        return x


class ImitationNet:
    def __init__(self, name, state_dim=12):
        self.name = name
        self.state_dim = state_dim
        self.max_action = 1
        self.act_dim = 1

        self.actor = None
        self.actor_optimizer = None

        self.create_agent()

    def create_agent(self, h_size=200):
        state_dim = self.state_dim
        action_dim = self.act_dim
        max_action = self.max_action
        self.actor = Actor(state_dim, action_dim, max_action, h_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def nn_act(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))

        action = self.actor(state).data.numpy().flatten()
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=self.act_dim))
            
        return action.clip(-self.max_action, self.max_action)

    def save(self, directory="./saves"):
        filename = self.name

        torch.save(self.actor, '%s/%s_actor.pth' % (directory, filename))

    def load(self, directory="./saves"):
        filename = self.name
        self.actor = torch.load('%s/%s_actor.pth' % (directory, filename))

        print("Agent Loaded")

    def try_load(self, load=True, h_size=300, path=None):
        if load:
            try:
                self.load(path)
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Unable to load model")
                pass
        else:
            print(f"Not loading - restarting training")
            self.create_agent(h_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def train(self, replay_buffer, batches=10000):
        losses = np.zeros(batches)
        print(f"Training agent")
        for i in range(batches):
            x, u = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)

            #TODO check that the actions are correctly scaled from steering to -1, 1
            actor_loss = F.mse_loss(self.actor(state), action)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            losses[i] = actor_loss

            if i % 100 == 0:
                print(f"Batch: {i}: Loss: {actor_loss}")

        return losses 


class ImitationVehicle:
    def __init__(self, sim_conf, name) -> None:
        self.actor = torch.load('%s/%s_actor.pth' % ("Vehicles/", name))

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.distance_scale = 10

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

    def act(self, obs):
        v = 1

        nn_obs = self.transform_obs(obs)
        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_act = self.actor(nn_obs).data.numpy().flatten()
        steering = nn_act[0] * self.max_steer

        action = np.array([steering, v])

        return action

    def reset_lap(self):
        pass

