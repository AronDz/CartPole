import torch
import numpy as np
from CartPole.utils import reward
from CartPole.continuous_cartpole import ContinuousCartPoleEnv
from CartPole.ac.agent import ActorCriticAgent
from CartPole.ddpg.agent import DDPGAgent
from CartPole.reinforce.agent import REINFORCEAgent
from CartPole.td3.agent import TD3Agent

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    env = ContinuousCartPoleEnv(reward_function=reward)
    env.seed(seed=SEED)
    n_episodes = 2000
    time_steps = 500
    gamma = 0.99
    batch_size = 64
    render = False
    alg = 'ddpg'
    if alg == 'ddpg':
        lr = 0.0001
        tau = 0.001
        memory_capacity = 1000000
        agent = DDPGAgent(env=env, n_episodes=n_episodes, time_steps=time_steps, gamma=gamma, batch_size=batch_size,
                          memory_capacity=memory_capacity, tau=tau, lr=lr, render=render)
    elif alg == 'ac':
        agent = ActorCriticAgent(env=env, n_episodes=n_episodes, gamma=gamma, render=render)
    elif alg == 'reinforce':
        agent = REINFORCEAgent(env=env, n_episodes=n_episodes, gamma=gamma, render=render)
    elif alg == 'td3':
        lr = 0.0001
        tau = 0.001
        memory_capacity = 1000000
        pi_update_steps = 2
        agent = TD3Agent(env=env, n_episodes=n_episodes, time_steps=time_steps, gamma=gamma, batch_size=batch_size,
                         memory_capacity=memory_capacity, tau=tau, lr=lr,
                         pi_update_steps=pi_update_steps, render=render)
    else:
        raise ValueError(f'agent must be ddpg, ac, reinforce or td3 but is {alg}')

    agent.train()
