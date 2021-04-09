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
        storage = np.load(filename, allow_pickle=True)

        self.storage = list(storage)
        print(f"Data loaded: type ({type(self.storage)})")




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

    def save(self, directory="Vehicles"):
        filename = self.name

        torch.save(self.actor, '%s/%s_actor.pth' % (directory, filename))

    def train(self, replay_buffer, batches=200000):
        losses = np.zeros(batches)
        print(f"Training agent")
        for i in range(batches):
            x, u = replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)

            #TODO check that the actions are correctly scaled from steering to -1, 1
            action_guesses = self.actor(state)
            actor_loss = F.mse_loss(action_guesses, action)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            losses[i] = actor_loss

            if i % 500 == 0:
                print(f"Batch: {i}: Loss: {actor_loss}")

                plt.figure(1)
                plt.clf()
                plt.plot(losses)
                plt.pause(0.0001)

        return losses 

    def train2(self, replay_buffer, batches=200000):
        losses = np.zeros(batches)
        batch_size = 100

        loss = nn.MSELoss()
        optimiser = optim.SGD(self.actor.parameters(), lr=0.001)

        for i in range(batches):
            x, u = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)

            optimiser.zero_grad()

            outputs = self.actor(state)
            actor_loss = loss(outputs[:,0], action)
            actor_loss.backward()
            optimiser.step()

            losses[i] = actor_loss

            if i % 500 == 0:
                print(f"Batch: {i}: Loss: {actor_loss}")

                lib.plot(losses, 100)

                self.save()

                # plt.figure(1)
                # plt.clf()
                # plt.plot(losses)
                # plt.pause(0.0001)

        return losses 



class ImitationVehicle:
    def __init__(self, sim_conf, name) -> None:
        self.actor = torch.load('%s/%s_actor.pth' % ("Vehicles/", name))
        # self.actor = Actor(12, 1, 1, 200)

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.distance_scale = 10

    def transform_obs(self, obs):
        max_angle = 3.14

        # cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        target_angle = [obs[5]/max_angle]
        # target_distance = [obs[6]/self.distance_scale]

        scan = obs[7:-1]

        # nn_obs = np.concatenate([cur_v, cur_d, target_angle, target_distance, scan])
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


class DaggerVehicle:
    def __init__(self, name, sim_conf):
        self.name = name
        self.buffer = BufferIL()

        filename = '%s/%s_actor.pth' % ("Vehicles", self.name)
        self.actor = torch.load(filename)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.distance_scale = 10

    def save(self, directory="Vehicles"):
        filename = '%s/%s_actor.pth' % (directory, self.name)

        torch.save(self.actor, filename)

    def load_buffer(self, buffer_name):
        self.buffer.load_data(buffer_name)

    def aggregate_buffer(self, new_buffer):
        for sample in new_buffer.storage:
            self.buffer.add(sample)

        new_buffer.storage.clear()
        new_buffer.ptr = 0

    def train(self, batches=5000):
        losses = np.zeros(batches)
        batch_size = 100

        loss = nn.MSELoss()
        optimiser = optim.SGD(self.actor.parameters(), lr=0.001)

        for i in range(batches):
            x, u = self.buffer.sample(batch_size)
            state = torch.FloatTensor(x)
            action = torch.FloatTensor(u)

            optimiser.zero_grad()

            outputs = self.actor(state)
            actor_loss = loss(outputs[:,0], action)
            actor_loss.backward()
            optimiser.step()

            losses[i] = actor_loss

            if i % 500 == 0:
                print(f"Batch: {i}: Loss: {actor_loss}")

                lib.plot(losses, 100)

                self.save()

        return losses 

    def transform_obs(self, obs):
        max_angle = 3.14

        # cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        target_angle = [obs[5]/max_angle]
        # target_distance = [obs[6]/self.distance_scale]

        scan = obs[7:-1]

        # nn_obs = np.concatenate([cur_v, cur_d, target_angle, target_distance, scan])
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

