import torch
import math
import numpy as np
from .net import Actor, Critic
from ..utils import tt, Stats, ReplayMemory, logging
from CartPole.continuous_cartpole import angle_normalize


class DDPGAgent:
    def __init__(self, env, n_episodes=3000, time_steps=500, gamma=0.99, batch_size=32,
                 memory_capacity=100000, tau=1e-2, eps=0.1, lr=0.00001, render=False):
        self.env = env
        self.gamma = gamma
        self.time_steps = time_steps
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.tau = tau
        self.eps = eps
        self.lr = lr
        self.render = render

        # Same weights for target network as for original network
        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim)
        self.actor_target = Actor(state_dim=self.state_dim, action_dim=self.action_dim)

        self.critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic_target = Critic(state_dim=self.state_dim, action_dim=self.action_dim)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.critic_loss_fct = torch.nn.MSELoss()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr * 10)

        self.n_episodes = n_episodes

        self.replay_memory = ReplayMemory(capacity=self.memory_capacity, batch_size=batch_size)

        self.stats = Stats(name='DDPG')

    def train(self):
        for i in range(self.n_episodes):
            reward_per_time_step = list()
            angle = 0
            steps = 0
            state = self.env.reset()

            for step in range(self.time_steps):
                if self.render:
                    self.env.render()

                state = tt(state)
                action = self.actor(state).detach().numpy()

                # Exploration
                p = np.random.random()
                if p < self.eps:
                    action = np.random.uniform(low=-1, high=1, size=(1,))
                # Do one step in env
                next_state, reward, done, _ = self.env.step(action)

                if 0 <= math.fabs(angle_normalize(state[2])) <= 0.1:
                    angle += 1

                # Save step in memory
                self.replay_memory.append(state=state, action=action, reward=reward, next_state=next_state, done=done)

                # Start training, if batch size reached
                if len(self.replay_memory) < self.batch_size:
                    continue

                # Sample batch from memory
                states, actions, rewards, next_states, dones = self.replay_memory.sample_batch()

                # Critic loss
                q_values = self.critic(states, actions)
                next_actions = self.actor_target(next_states)
                q_values_ns = self.critic_target(next_states, next_actions.detach())
                td_target = rewards + self.gamma * q_values_ns
                loss_critic = self.critic_loss_fct(q_values, td_target)

                # Actor loss
                loss_actor = -(self.critic(states, self.actor(states)).mean())

                # Optimize actor
                self.actor_optim.zero_grad()
                loss_actor.backward()
                self.actor_optim.step()

                # Optimize critic
                self.critic_optim.zero_grad()
                loss_critic.backward()
                self.critic_optim.step()

                # update target networks
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

                state = next_state
                reward_per_time_step.append(reward)
                steps += 1

                if done:
                    break

            avg_episode_reward = sum(reward_per_time_step) / len(reward_per_time_step)
            logging.info(f'Episode {i + 1}:')
            logging.info(f'\t Steps: {steps}')
            logging.info(f'\t Reward: {sum(reward_per_time_step)}')
            # Update stats
            self.stats.update(reward=sum(reward_per_time_step), step=steps, angle=angle)

        self.stats.plot()
        self.env.close()
