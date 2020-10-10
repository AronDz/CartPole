import torch
import numpy as np
from utils import reward
from continuous_cartpole import ContinuousCartPoleEnv
from ddpg.agent import DDPGAgent
from td3.agent import TD3Agent

SEED = 5
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    env = ContinuousCartPoleEnv(reward_function=reward)
    env.seed(seed=SEED)
    n_episodes = 3000
    time_steps = 500

    lr = 0.0001
    tau = 0.001
    memory_capacity = 1000000
    gamma = 0.99
    batch_size = 64
    render = False
    method = 'td3'
    if method == 'ddpg':
        agent = DDPGAgent(env=env, n_episodes=n_episodes, time_steps=time_steps, gamma=gamma, batch_size=batch_size,
                          memory_capacity=memory_capacity, tau=tau, lr=lr, render=render)
    elif method == 'td3':
        pi_update_steps = 2
        agent = TD3Agent(env=env, n_episodes=n_episodes, time_steps=time_steps, gamma=gamma, batch_size=batch_size,
                         memory_capacity=memory_capacity, tau=tau, lr=lr,
                         pi_update_steps=pi_update_steps, render=render)
    else:
        raise ValueError(f'agent must be ddpg or td3 but is {method}')

    res = agent.train()
    res.to_csv(f'{method}_SEED{SEED}_res.csv')
