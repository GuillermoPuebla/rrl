import os
import pandas as pd


games = ["Breakout", "Pong", "DemonAttack"]
for game in games:
    path_logical = f'results/{game}/logical_version/test'
    path_comparative = f'results/{game}/comparative_version/test'

    # Get data logical splits
    all_dfs_logical = []
    for root, dirs, files in os.walk(path_logical):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('test.csv'):
                df = pd.read_csv(filepath, low_memory=False)
                df = df[['Run', 'Mean return', 'Std Return']].reset_index(drop=True)
                df = df.rename(columns={
                    "Run": "Run",
                    "Mean return": "Mean return",
                    "Std Return": "Std return",
                    })
                all_dfs_logical.append(df)
    df_logical = pd.concat(all_dfs_logical, ignore_index=True)
    df_logical['Model version'] = 'Logical'

    # Get data comparative splits
    all_dfs_comparative = []
    for root, dirs, files in os.walk(path_comparative):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath.endswith('test.csv'):
                df = pd.read_csv(filepath, low_memory=False)
                df = df[['Run', 'Mean return', 'Std Return']].reset_index(drop=True)
                df = df.rename(columns={
                    "Run": "Run",
                    "Mean return": "Mean return",
                    "Std Return": "Std return",
                    })
                all_dfs_comparative.append(df)
    df_comparative = pd.concat(all_dfs_comparative, ignore_index=True)
    df_comparative['Model version'] = 'Comparative'

    # All data together
    df = pd.concat([df_logical, df_comparative]).reset_index(drop=True)
    # Save
    df.to_csv(f"datasets/test_data_{game}.csv")