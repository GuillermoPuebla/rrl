import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


#mpl.rcParams['savefig.facecolor'] = "FAFAFA"
window_divider = 20
min_periods_divider = 50

df = pd.read_csv(f"datasets/train_data_by_episode_cumulative_return_full.csv", index_col=0)
#df = df[df["Run"] % 2 == 0]

df_breakout = df[df["Game"] == "Breakout"].copy() 
window_size_breakout = int(df_breakout["Episode"].max() / window_divider)
min_periods_breakout = int(df_breakout["Episode"].max() / min_periods_divider)
df_breakout['MAR'] = df_breakout.groupby(['Model version', 'Run'])['Return'].transform(lambda x: x.rolling(window_size_breakout, min_periods_breakout).mean())
df_breakout['Std return'] = df_breakout.groupby(["Model version", "Run"])["Return"].transform('std')

df_breakout_reduced = df_breakout[df_breakout["Episode"] == 0][["Model version", "Run", "Std return"]].copy()
df_breakout_reduced["Rank"] = df_breakout_reduced.sort_values(by=["Model version", "Std return"]).groupby(by=["Model version"])["Std return"].rank(method="first",ascending=True).astype(int)
df_breakout = df_breakout.merge(df_breakout_reduced, on=["Model version", "Run", "Std return"], how="left")
print(df_breakout)


df_pong = df[df["Game"] == "Pong"].copy() 
window_size_pong = int(df_pong["Episode"].max() / window_divider)
min_periods_pong = int(df_pong["Episode"].max() / min_periods_divider)
df_pong['MAR'] = df_pong.groupby(['Model version', 'Run'])['Return'].transform(lambda x: x.rolling(window_size_pong, min_periods_pong).mean())
df_pong['Std return'] = df_pong.groupby(["Model version", "Run"])["Return"].transform('std')

df_pong_reduced = df_pong[df_pong["Episode"] == 0][["Model version", "Run", "Std return"]].copy()
df_pong_reduced["Rank"] = df_pong_reduced.sort_values(by=["Model version", "Std return"]).groupby(by=["Model version"])["Std return"].rank(method="first",ascending=True).astype(int)
df_pong = df_pong.merge(df_pong_reduced, on=["Model version", "Run", "Std return"], how="left")
print(df_pong)

df_demon = df[df["Game"] == "Demon Attack"].copy()
window_size_demon = int(df_demon["Episode"].max() / window_divider)
min_periods_demon = int(df_demon["Episode"].max() / min_periods_divider)
df_demon['MAR'] = df_demon.groupby(['Model version', 'Run'])['Return'].transform(lambda x: x.rolling(window_size_demon, min_periods_demon).mean())
df_demon['Std return'] = df_demon.groupby(["Model version", "Run"])["Return"].transform('std')

df_demon_reduced = df_demon[df_demon["Episode"] == 0][["Model version", "Run", "Std return"]].copy()
df_demon_reduced["Rank"] = df_demon_reduced.sort_values(by=["Model version", "Std return"]).groupby(by=["Model version"])["Std return"].rank(method="first",ascending=True).astype(int)
df_demon = df_demon.merge(df_demon_reduced, on=["Model version", "Run", "Std return"], how="left")
print(df_demon)

df = pd.concat([df_breakout, df_pong, df_demon]).reset_index()
#df = df[df["Run"] % 2 == 0]

(
    so.Plot(df, x="Episode", y="MAR", group="Run", color="Rank")
    .facet(col="Model version", row="Game", order={"col": ["Logical", "Comparative"], "row": ["Breakout", "Pong", "Demon Attack"]})
    .add(so.Line(linewidth=1.0), legend=False)
    .label(y="Moving average return")
    .share(x="row")
    .share(y="row")
    .scale(color="crest")
    .layout(size=(13, 12))
    .save("plots/train_plot_all_games_MAE.pdf")
)
