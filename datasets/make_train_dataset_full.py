import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


df_breakout = pd.read_csv(f"datasets/train_data_Breakout.csv", index_col=0)
df_breakout = df_breakout[["Run", "Model version", "Iteration", "Episode", "Reward"]]

df_breakout = df_breakout.groupby(by=['Model version', 'Run', 'Episode'], as_index=False)["Reward"].sum()
df_breakout.rename(columns={'Reward': 'Return'}, inplace=True)
df_breakout.sort_values(by=['Model version', 'Run', 'Episode'], inplace=True)
df_breakout["Cumulative return"] = df_breakout.groupby(by=['Model version', 'Run'], as_index=False)["Return"].cumsum()
df_breakout["Game"] = "Breakout"
print(df_breakout)
 

df_pong = pd.read_csv(f"datasets/train_data_Pong.csv", index_col=0)
df_pong = df_pong[["Run", "Model version", "Iteration", "Episode", "Reward"]]
#df_pong["Reward"] -= 0.1 

df_pong = df_pong.groupby(by=['Model version', 'Run', 'Episode'], as_index=False)["Reward"].sum()
df_pong.rename(columns={'Reward': 'Return'}, inplace=True)
df_pong.sort_values(by=['Model version', 'Run', 'Episode'], inplace=True)
df_pong["Cumulative return"] = df_pong.groupby(by=['Model version', 'Run'], as_index=False)["Return"].cumsum()
df_pong["Game"] = "Pong"
print(df_pong)

df_demon = pd.read_csv(f"datasets/train_data_DemonAttack.csv", index_col=0)
df_demon = df_demon[["Run", "Model version", "Iteration", "Episode", "Reward"]]

df_demon = df_demon.groupby(by=['Model version', 'Run', 'Episode'], as_index=False)["Reward"].sum()
df_demon.rename(columns={'Reward': 'Return'}, inplace=True)
df_demon.sort_values(by=['Model version', 'Run', 'Episode'], inplace=True)
df_demon["Cumulative return"] = df_demon.groupby(by=['Model version', 'Run'], as_index=False)["Return"].cumsum()
df_demon["Game"] = "Demon Attack"
print(df_demon)

df = pd.concat([df_breakout, df_pong, df_demon]).reset_index()
df.to_csv(f"datasets/train_data_by_episode_cumulative_return_full.csv", index=False)