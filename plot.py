import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fname = 'td3_res.csv'
    path = f'{fname}'
    sns.set()
    df = pd.read_csv(path)

    window_size = 10
    episode_df = pd.DataFrame({'Episode': [], 'Reward': [], 'Steps': []})
    for episode, grp_df in df.groupby(['episodes']):
        new_df = {'Episode': episode,
                  'Reward': grp_df['rewards'].sum(),
                  'Steps': grp_df['steps'].max()}
        episode_df = episode_df.append([new_df])

    fig, _ = plt.subplots(1, 1, figsize=(10, 8))
    smoothed_episode_df = episode_df.rolling(window_size).mean()
    sns.lineplot(data=smoothed_episode_df, x="Episode", y="Reward")
    plt.title(f'Episode Reward over Time (Smoothed over window size of {window_size})')
    plt.savefig('episode_reward.png')

    fig2, _ = plt.subplots(1, 1, figsize=(10, 8))
    sns.lineplot(data=smoothed_episode_df, x="Episode", y="Steps")
    plt.title(f'Number of Steps in each Episode (Smoothed over window size of {window_size})')
    plt.savefig('episode_steps.png')

    angle_df = pd.DataFrame({'Episode': [], 'Average Angle': []})
    for episode, grp_df in df.groupby(['episodes']):
        print(type(grp_df['states'][0]))
        # new_df = {'Episode': episode,
        #           'Average Angle': grp_df['rewards'].mean()}
        # episode_df = episode_df.append([new_df])
