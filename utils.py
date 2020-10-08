import torch
import math
import numpy as np
import logging
from torch.autograd import Variable
from collections import namedtuple
from CartPole.continuous_cartpole import angle_normalize

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

logging.basicConfig(filename='logfile.log', filemode='w', level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def tt(array):
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return Variable(torch.from_numpy(array).float(), requires_grad=False)
    else:
        raise ValueError(f'Given array must be from type torch.Tensor or np.ndarray but is {type(array)}')


def reward(cart_pole):
    if cart_pole.state[0] < -cart_pole.x_threshold or cart_pole.state[0] > cart_pole.x_threshold:
        return -1
    if 0 <= math.fabs(angle_normalize(cart_pole.state[2])) <= 0.1:
        return 1
    elif 0.1 < math.fabs(angle_normalize(cart_pole.state[2])) <= 0.5:
        return 0.5
    elif 0.5 < math.fabs(angle_normalize(cart_pole.state[2])) <= 1:
        return 0.3
    elif 1 < math.fabs(angle_normalize(cart_pole.state[2])) <= 2:
        return 0.2
    elif 2 < math.fabs(angle_normalize(cart_pole.state[2])) <= 3:
        return 0.1
    else:
        return 0


class ReplayMemory:
    def __init__(self, capacity=100000, batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.trajectories = []

    def append(self, state, action, reward, next_state, done):
        # Remove first element when capacity reached
        if len(self.trajectories) > self.capacity:
            self.trajectories = self.trajectories[1:self.capacity - 1]

        # Convert all inputs to torch tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor([1-int(done)])

        # Append new trajectory
        self.trajectories.append(Transition(state, action, reward, next_state, done))

    def sample_batch(self):
        # Generate random integers in range (0, len(self.trajectories) - 1)
        indices = torch.randint(high=len(self.trajectories) - 1, size=(self.batch_size, 1))

        states = torch.empty((self.batch_size, *self.trajectories[0].state.shape))
        actions = torch.empty((self.batch_size, *self.trajectories[0].action.shape))
        rewards = torch.empty((self.batch_size, *self.trajectories[0].reward.shape))
        next_states = torch.empty((self.batch_size, *self.trajectories[0].next_state.shape))
        dones = torch.empty((self.batch_size, *self.trajectories[0].done.shape))

        for k, idx in enumerate(indices):
            states[k] = self.trajectories[idx].state
            actions[k] = self.trajectories[idx].action
            rewards[k] = self.trajectories[idx].reward
            next_states[k] = self.trajectories[idx].next_state
            dones[k] = self.trajectories[idx].done

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.trajectories)
