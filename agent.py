from continuous_cartpole import ContinuousCartPoleEnv
import torch
import numpy as np
from utils import tt, Policy, StateValueFunction, Stats

LEARNING_RATE = 0.00001
SEED = 0


class ActorCriticAgent:
    def __init__(self, env, n_episodes=100, time_steps=500, gamma=0.99, render=False):
        self.env = env
        self.gamma = gamma
        self.time_steps = time_steps
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.render = render

        self.actor = Policy(state_dim=self.state_dim, action_dim=self.action_dim)
        self.critic = StateValueFunction(state_dim=self.state_dim, action_dim=self.action_dim)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        self.n_episodes = n_episodes

        self.stats = Stats()

    def train(self):
        for i in range(self.n_episodes):
            state = self.env.reset()
            reward_per_time_step = list()
            steps = 0
            for _ in range(self.time_steps):
                if self.render:
                    self.env.render()
                state = tt(state)

                # Predict mean and variance of normal dist
                mu, sigma = self.actor(state)
                # Predict value of current state
                value = self.critic(state)

                # Sample action from predicted normal dist
                action = np.random.normal(mu.detach().numpy(), sigma.data.cpu().numpy())
                action = np.clip(action, -1, 1)

                # Do one step in env by taking the sampled action
                next_state, reward, done, _ = self.env.step(action)

                reward_per_time_step.append(reward)

                # Compute td-target (r + gamma * v(s'))
                td_target = reward + self.gamma * self.critic(tt(next_state)) * (1 - int(done))
                td_error = (td_target - value).detach()

                pdf = (1 / (torch.sqrt(2 * np.pi * sigma))) * torch.exp(-torch.square((tt(action) - mu) / (2 * sigma)))
                log_pdf = torch.log(pdf)
                loss_actor = -(log_pdf * td_error).mean()

                # Optimize critic
                loss_critic = torch.nn.functional.mse_loss(value, td_target)
                self.critic_optim.zero_grad()
                loss_critic.backward()
                self.critic_optim.step()

                # Optimize actor
                self.actor_optim.zero_grad()
                loss_actor.backward()
                self.actor_optim.step()

                state = next_state
                steps += 1
                # Break if current state is terminal state.
                if done:
                    avg_episode_reward = sum(reward_per_time_step) / len(reward_per_time_step)
                    print(f'Episode: {i+1}, Num. Steps: {steps}')
                    print(f'Avg. reward: {avg_episode_reward}')
                    # Update stats
                    self.stats.update(reward=avg_episode_reward, step=steps)
                    break

        self.stats.plot()
        self.env.close()


class REINFORCEAgent:
    def __init__(self, env, n_episodes=3000, time_steps=500, gamma=0.99, render=False):
        self.env = env
        self.gamma = gamma
        self.time_steps = time_steps
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.render = render

        self.policy = Policy(state_dim=self.state_dim, action_dim=self.action_dim)
        self.value_fct = StateValueFunction(state_dim=self.state_dim, action_dim=self.action_dim)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.value_fct_optim = torch.optim.Adam(self.value_fct.parameters(), lr=LEARNING_RATE)

        self.critic_loss_fct = torch.nn.MSELoss()

        self.n_episodes = n_episodes

        self.stats = Stats()

    def train(self):
        for i in range(self.n_episodes):
            state = self.env.reset()
            episodes = list()
            reward_per_time_step = list()
            steps = 0

            # Generate episode
            for _ in range(self.time_steps):
                if self.render:
                    self.env.render()
                state = tt(state)

                # Predict mean and variance of normal dist
                mu, sigma = self.policy(state)

                # Sample action from predicted normal dist
                action = np.random.normal(mu.detach().numpy(), sigma.data.cpu().numpy())
                action = np.clip(action, -1, 1)

                # Do one step in env by taking the sampled action
                next_state, reward, done, _ = self.env.step(action)
                episodes.append((state, action, reward))
                reward_per_time_step.append(reward)

                steps += 1

                if done:
                    break
                state = next_state

            avg_episode_reward = sum(reward_per_time_step) / len(reward_per_time_step)
            print(f'Episode: {i + 1}, Num. Steps: {steps}')
            print(f'Avg. reward: {avg_episode_reward}')

            # Update stats
            self.stats.update(reward=avg_episode_reward, step=steps)

            # Learn from episode
            for t in range(len(episodes)):
                state, action, reward = episodes[t]
                G = tt(np.array([sum([e[2]*(self.gamma**i) for i, e in enumerate(episodes[t:])])]))

                # Optimize critic
                loss_critic = self.critic_loss_fct(self.value_fct(state), G)
                self.value_fct_optim.zero_grad()
                loss_critic.backward()
                self.value_fct_optim.step()

                mu, sigma = self.policy(state)

                # Compute log prob. density function
                pdf = (1 / (torch.sqrt(2 * np.pi * sigma))) * torch.exp(-torch.square((tt(action) - mu) / (2 * sigma)))
                log_pdf = torch.log(pdf)
                loss_policy = -(log_pdf * (G - self.value_fct(state))).mean()
                # Optimize actor
                self.policy_optim.zero_grad()
                loss_policy.backward()
                self.policy_optim.step()

        self.stats.plot()
        self.env.close()


if __name__ == '__main__':
    env = ContinuousCartPoleEnv()
    env.seed(seed=SEED)
    n_episodes = 3000
    # agent = ActorCriticAgent(env=env, n_episodes=n_episodes, gamma=0.99)
    agent = REINFORCEAgent(env=env, n_episodes=n_episodes, gamma=0.99, render=False)
    agent.train()
