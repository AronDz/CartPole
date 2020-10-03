import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch.autograd import Variable
from collections import namedtuple
from CartPole.continuous_cartpole import angle_normalize

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

logging.basicConfig(filename='td3.log', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def tt(array):
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return Variable(torch.from_numpy(array).float().to(DEVICE), requires_grad=False)
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
        state = torch.FloatTensor(state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor([reward]).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        done = torch.FloatTensor([1-int(done)]).to(DEVICE)

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


class Stats:
    def __init__(self, name, win_size=10):
        self.rewards_per_episode = list()
        self.steps_per_episode = list()
        self.angle = list()
        self.name = name

        self.win_size = win_size

    def update(self, reward, step, angle):
        self.rewards_per_episode.append(reward)
        self.steps_per_episode.append(step)
        self.angle.append(angle)

    def plot(self):
        smoothed_rewards = list()
        for i in range(len(self.rewards_per_episode) - self.win_size):
            smoothed_rewards.append(np.mean(self.rewards_per_episode[i: i + self.win_size - 1]))
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(smoothed_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title(f"Episode Reward over Time (Smoothed over window size {self.win_size})")
        fig1.savefig(f'{self.name}_smoothed_reward.png')

        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Reward over Time")
        fig2.savefig(f'{self.name}_reward.png')

        fig3 = plt.figure(figsize=(10, 5))
        plt.plot(self.steps_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Number of Steps")
        plt.title("Steps in each Episode")
        fig3.savefig(f'{self.name}_steps.png')

        fig4 = plt.figure(figsize=(10, 5))
        plt.plot(self.angle)
        plt.xlabel("Episode")
        plt.ylabel("Number of Times Steps where Angle between 0 and 0.1")
        plt.title("Number of Times where Angle is between 0 and 0.1")
        fig4.savefig(f'{self.name}_angle.png')
