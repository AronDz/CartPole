import torch
import numpy as np
import pandas as pd
from .net import Actor, Critic
from ..utils import tt, ReplayMemory, logging


class TD3Agent:
    def __init__(self, env, n_episodes=3000, time_steps=500, gamma=0.99, batch_size=64,
                 memory_capacity=100000, tau=1e-2, lr=0.00001, pi_update_steps=2, render=False):
        self.env = env
        self.gamma = gamma
        self.time_steps = time_steps
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.tau = tau
        self.lr = lr
        self.pi_update_steps = pi_update_steps
        self.render = render

        # Create actor and critic network
        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim)
        self.actor_target = Actor(state_dim=self.state_dim, action_dim=self.action_dim)

        self.critic = Critic(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic_target = Critic(state_dim=self.state_dim, action_dim=self.action_dim)

        # Same weights for target network as for original network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.critic_loss_fct = torch.nn.MSELoss()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr * 10)

        self.n_episodes = n_episodes
        self.replay_memory = ReplayMemory(capacity=self.memory_capacity, batch_size=batch_size)

        self.res = pd.DataFrame({
                'episodes': [],
                'states': [],
                'rewards': [],
                'steps': [],
                'actor_losses': [],
                'critic_losses': [],
                 })

    def train(self):
        for i in range(self.n_episodes):
            state = self.env.reset()

            for step in range(self.time_steps):
                if self.render:
                    self.env.render()

                state = tt(state)
                action = self.actor(state).detach().numpy()

                noise = np.random.normal(0, 0.1, size=self.env.action_space.shape[0])
                action = np.clip(action + noise, self.env.action_space.low[0], self.env.action_space.high[0])
                next_state, reward, done, _ = self.env.step(action)

                # Save step in memory
                self.replay_memory.append(state=state, action=action, reward=reward, next_state=next_state, done=done)

                res = {'episodes': i + 1,
                       'states': state.tolist(),
                       'rewards': reward,
                       'steps': step + 1}

                # Start training, if batch size reached
                if len(self.replay_memory) < self.batch_size:
                    self.res = self.res.append([res])
                    continue

                # Sample batch from memory
                states, actions, rewards, next_states, dones = self.replay_memory.sample_batch()

                # Critic loss
                q1, q2 = self.critic(states, actions)
                next_actions = self.actor_target(next_states)

                noise = torch.FloatTensor(actions).data.normal_(0, 0.2)
                noise = noise.clamp(-0.5, 0.5)
                next_actions = (next_actions + noise).clamp(self.env.action_space.low[0], self.env.action_space.high[0])
                # Get next state q values by Clipped Double Q-Learning
                q1_ns,  q2_ns = self.critic_target(next_states, next_actions.detach())
                q_ns = torch.min(q1_ns, q2_ns)
                td_target = rewards + self.gamma * q_ns

                loss_critic = self.critic_loss_fct(q1, td_target) + self.critic_loss_fct(q2, td_target)
                res['critic_losses'] = float(loss_critic)

                # Optimize critic
                self.critic_optim.zero_grad()
                loss_critic.backward()
                self.critic_optim.step()

                # Delayed Policy Updates
                if step % self.pi_update_steps == 0:
                    q1, _ = self.critic(states, self.actor(states))
                    # Actor loss
                    loss_actor = -q1.mean()
                    res['actor_losses'] = float(loss_actor)

                    # Optimize actor
                    self.actor_optim.zero_grad()
                    loss_actor.backward()
                    self.actor_optim.step()

                    # update target networks
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                self.res = self.res.append([res])
                state = next_state
                if done:
                    break

            logging.info(f'Episode {i + 1}:')
            logging.info(f'\t Steps: {self.res.loc[self.res["episodes"] == i + 1]["steps"].max()}')
            logging.info(f'\t Reward: {self.res.loc[self.res["episodes"] == i + 1]["rewards"].sum()}')

        self.env.close()
        return self.res
