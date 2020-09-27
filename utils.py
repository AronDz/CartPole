import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from collections import namedtuple

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def tt(array):
    return Variable(torch.from_numpy(array).float().to(DEVICE), requires_grad=False)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.non_linearity = F.relu
        self.sp = F.softplus
        self.tanh = torch.tanh

    def forward(self, state):
        output = self.non_linearity(self.fc1(state))
        output = self.non_linearity(self.fc2(output))
        mu = self.tanh(self.fc3(output))  # map between -1 and 1
        sigma = self.sp(self.fc3(output))  # map to positive value
        # Return mean and std of predicted prob. dist.
        return mu, sigma


class StateValueFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super(StateValueFunction, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.non_linearity = F.relu

    def forward(self, state):
        output = self.non_linearity(self.fc1(state))
        output = self.non_linearity(self.fc2(output))
        return self.fc3(output)


class ReplayMemory:
    def __init__(self, capacity=10000, batch_size=32):
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
    def __init__(self, win_size=10):
        self.rewards_per_episode = list()
        self.steps_per_episode = list()

        self.win_size = win_size

    def update(self, reward, step):
        self.rewards_per_episode.append(reward)
        self.steps_per_episode.append(step)

    def plot(self):
        smoothed_rewards = list()
        for i in range(len(self.rewards_per_episode) - self.win_size):
            smoothed_rewards.append(np.mean(self.rewards_per_episode[i: i + self.win_size - 1]))
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(smoothed_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title(f"Episode Reward over Time (Smoothed over window size {self.win_size})")
        fig1.savefig('smoothed_reward.png')

        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Reward over Time")
        fig2.savefig('reward.png')

        fig3 = plt.figure(figsize=(10, 5))
        plt.plot(self.steps_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Number of Steps")
        plt.title("Steps in each Episode")
        fig3.savefig('steps.png')

