import os
import pandas as pd
import seaborn as sns


games = ["Breakout", "Pong", "DemonAttack"]
for game in games:
    path_logical = f'results_sig_0_0001_decay_500000/{game}/logical_version/train'
    path_comparative = f'results_sig_0_0001_decay_500000/{game}/comparative_version/train'

    # Get data logical splits
    all_dfs_logical = []
    for root, dirs, files in os.walk(path_logical):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('train.csv'):
                df = pd.read_csv(filepath, low_memory=False)
                df = df[["Run", "Iteration", "Episode", "Reward"]].reset_index(drop=True)
                all_dfs_logical.append(df)
    df_logical = pd.concat(all_dfs_logical, ignore_index=True)
    df_logical['Model version'] = 'Logical'

    # Get data comparative splits
    all_dfs_comparative = []
    for root, dirs, files in os.walk(path_comparative):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('train.csv'):
                df = pd.read_csv(filepath, low_memory=False)
                df = df[["Run", "Iteration", "Episode", "Reward"]].reset_index(drop=True)
                all_dfs_comparative.append(df)
    df_comparative = pd.concat(all_dfs_comparative, ignore_index=True)
    df_comparative['Model version'] = 'Comparative'

    # All data together
    df = pd.concat([df_logical, df_comparative])
    df = df[["Run", "Model version", "Iteration", "Episode", "Reward"]]

    df.to_csv(f"datasets/train_data_{game}.csv")