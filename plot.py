import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_reward_and_steps(df):
    # REWARD PLOT
    window_size = 50
    episode_df = pd.DataFrame({'Episode': [], 'Reward': [], 'Steps': []})
    for episode, grp_df in df.groupby(['episodes']):
        new_df = {'Episode': episode,
                  'Reward': grp_df['rewards'].sum(),
                  'Steps': grp_df['steps'].max()}
        episode_df = episode_df.append([new_df])

    fig, _ = plt.subplots(1, 1, figsize=(10, 8))
    smoothed_episode_df = episode_df.rolling(window_size).mean()
    sns.lineplot(data=smoothed_episode_df, x="Episode", y="Reward")
    plt.title(f'Episode Reward over Time\n(Smoothed over window size of {window_size})')
    plt.savefig(f'results/{fname.split(".")[0]}_episode_reward.pdf')

    # STEPS PLOT
    fig2, _ = plt.subplots(1, 1, figsize=(10, 8))
    sns.lineplot(data=smoothed_episode_df, x="Episode", y="Steps")
    plt.title(f'Number of Steps in each Episode\n(Smoothed over window size of {window_size})')
    plt.savefig(f'results/{fname.split(".")[0]}_episode_steps.pdf')


def plot_states(df, state, state_idx):
    dfs = []
    for episode, grp_df in df.groupby(['episodes']):
        if grp_df['rewards'].sum() > 420:
            state_df = pd.DataFrame({'Episode': [int(x) for x in grp_df['episodes'].tolist()],
                                     'Time Steps': grp_df['steps'].tolist(),
                                     state: np.array(grp_df['states'].tolist())[:, state_idx].tolist()})
            dfs.append(state_df)
            if len(dfs) >= 10:
                break
    state_df = pd.concat(dfs)

    fig3, _ = plt.subplots(1, 1, figsize=(10, 8))
    sns.lineplot(data=state_df, x="Time Steps", y=state, hue='Episode')
    plt.title(f'{state} in each Time Step over Different Episodes')
    plt.savefig(f'results/{fname.split(".")[0]}_{state}.pdf')


if __name__ == '__main__':
    fname = 'ddpg_res.csv'
    path = f'{fname}'
    df = pd.read_csv(path, converters={'states': eval})

    font = {'family': 'Times New Roman',
            'size': 16}

    plt.rc('font', **font)
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['grid.linewidth'] = 5
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # REWARD AND STEPS PLOT
    plot_reward_and_steps(df=df)
    # POSITION PLOT
    plot_states(df=df, state='Position', state_idx=0)
    # VELOCITY PLOT
    plot_states(df=df, state='Velocity', state_idx=1)
    # ANGLE PLOT
    plot_states(df=df, state='Angle', state_idx=2)

