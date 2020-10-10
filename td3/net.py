import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.non_linearity = F.relu
        self.tanh = torch.tanh


    def forward(self, state):
        output = self.non_linearity(self.fc1(state))
        output = self.non_linearity(self.fc2(output))
        output = self.tanh(self.fc3(output))  # map between -1 and 1
        return output


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(Critic, self).__init__()

        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, action_dim)

        self.non_linearity = F.relu
        

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        q1 = self.non_linearity(self.fc1(input))
        q1 = self.non_linearity(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = self.non_linearity(self.fc4(input))
        q2 = self.non_linearity(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2
