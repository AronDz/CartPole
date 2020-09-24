from scripts.continuous_cartpole import ContinuousCartPoleEnv

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn

ENTROPY_BETA = 1e-4
LEARNING_RATE = 0.0001
SEED = 41
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear = nn.Linear(self.state_size, 128)
        self.linear_2 = nn.Linear(128, 1)
        self.softplus = nn.Softplus()

    def forward(self, state):
        output = self.linear(state)
        output = self.linear_2(output)
        mu = torch.tanh(output)  # map between -1 and 1
        var = self.softplus(output)  # map to positive value
        # Return mean and variance of predicted prob. dist.
        return mu, var


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.linear = nn.Linear(self.state_size, 128)
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, state):
        output = self.linear(state)
        output = self.linear_2(output)
        return output


class ActorCriticAgent:
    def __init__(self, env, batch_size=32, n_episodes=100, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.actor = Actor(state_size=self.state_size, action_size=self.action_size)
        self.critic = Critic(state_size=self.state_size)

        self.n_episodes = n_episodes

    def _calc_logprob(self, mu_v, var_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
        return p1 + p2

    def train(self):
        actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        avg_episod_rewards = list()

        for i in range(self.n_episodes):
            state = self.env.reset()
            values = list()
            rewards = list()
            steps = 0
            while True:
                self.env.render()
                state = torch.FloatTensor(state).to(device)

                mu, var = self.actor(state)
                value = self.critic(state)

                sigma = torch.sqrt(var).data.cpu().numpy()
                action = np.random.normal(mu.detach().numpy(), sigma)
                action = np.clip(action, -1, 1)
                next_state, reward, done, _ = self.env.step(action)
                values.append(value)
                rewards.append(reward)

                ns_value = 0 if done else self.critic(torch.FloatTensor(next_state))
                td_target = reward + self.gamma * ns_value - value

                delta = (td_target - value).detach()

                loss_actor = torch.FloatTensor(-(delta * self._calc_logprob(mu, var, torch.FloatTensor(action))))

                # Optimize critic
                loss_critic = torch.nn.functional.mse_loss(value, td_target)
                critic_optim.zero_grad()
                loss_critic.backward()
                critic_optim.step()

                # Optimize actor
                actor_optim.zero_grad()
                loss_actor.backward()
                actor_optim.step()

                state = next_state
                steps += 1
                # Break if current state is terminal state.
                if done:
                    print(f'Episode: {i+1}, Num. Steps: {steps}')
                    print(f'Avg. reward: {sum(rewards) / len(rewards)}')
                    avg_episod_rewards.append(sum(rewards) / len(rewards))
                    break
        plt.plot(avg_episod_rewards)
        plt.show()
        self.env.close()


if __name__ == '__main__':
    env = ContinuousCartPoleEnv()
    n_episodes = 500
    agent = ActorCriticAgent(env=env, n_episodes=n_episodes)
    agent.train()
