import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import os


def normalize_score(score, game):
    "Calculates human-normalized scores"
    # Random game performance in Mnih et al. (2015)
    breakout_random = 1.7
    pong_random = -20.7
    demon_attack_random = 152.1
    # Human game performance in Mnih et al. (2015)
    breakout_human = 31.8
    pong_human = 9.3
    demon_attack_human = 3401

    if game == "Breakout":
        score_random = breakout_random
        score_human = breakout_human
    elif game == "Pong":
        score_random = pong_random
        score_human = pong_human
    elif game == "DemonAttack":
        score_random = demon_attack_random
        score_human = demon_attack_human
    else:
        raise ValueError("Unrecognized game!")

    return (score - score_random) / (score_human - score_random)

# Plot constants
jitter = 0.25

df_breakout = pd.read_csv("datasets/test_data_Breakout.csv")
df_pong = pd.read_csv("datasets/test_data_Pong.csv")
df_demon_attack = pd.read_csv("datasets/test_data_DemonAttack.csv")
df_breakout["Human-normalized score"] = df_breakout["Mean return"].apply(normalize_score, game="Breakout")
df_pong["Human-normalized score"] = df_pong["Mean return"].apply(normalize_score, game="Pong")
df_demon_attack["Human-normalized score"] = df_demon_attack["Mean return"].apply(normalize_score, game="DemonAttack")


fig = plt.figure(figsize=(8.5, 12.75))
(sf1, sf2), (sf3, sf4), (sf5, sf6) = fig.subfigures(3, 2)
# fig = plt.figure(figsize=(12.75, 8.5))
# (sf1, sf2, sf3), (sf4, sf5, sf6) = fig.subfigures(2, 3)
sf1.suptitle("Breakout medians", y=0.94, fontsize="x-large")
sf2.suptitle("Breakout frequencies", y=0.94, fontsize="x-large")
sf3.suptitle("Pong medians", y=0.94, fontsize="x-large")
sf4.suptitle("Pong frequencies", y=0.94, fontsize="x-large")
sf5.suptitle("Demon Attack medians", y=0.94, fontsize="x-large")
sf6.suptitle("Demon Attack frequencies", y=0.94, fontsize="x-large")

(
    so.Plot(df_breakout, x="Model version", y="Human-normalized score")
    .add(so.Dots(color="C4", alpha=1.0), so.Jitter(jitter), legend=False)
    .add(so.Line(marker="_", color=".0"), so.Agg('median'))
    .add(so.Range(capsize=0.1, color=".0"), so.Est('median', errorbar=("ci", 95)))
    .on(sf1)
    .plot()
    )
(
    so.Plot(df_breakout, x="Human-normalized score", color="Model version")
    .add(so.Bars(edgewidth=1.25, alpha=0.5), so.Hist(bins=16), legend=False)
    .scale(color=["C2", "C1"])
    .label(y="Frequency")
    .on(sf2)
    #.theme(theme_dict)
    .plot()
    )

(
    so.Plot(df_pong, x="Model version", y="Human-normalized score")
    .add(so.Dots(color="C4", alpha=1.0), so.Jitter(jitter), legend=False)
    .add(so.Line(marker="_", color=".0"), so.Agg('median'))
    .add(so.Range(capsize=0.1, color=".0"), so.Est('median', errorbar=("ci", 95)))
    .on(sf3)
    .plot()
    )
(
    so.Plot(df_pong, x="Human-normalized score", color="Model version")
    .add(so.Bars(edgewidth=1.25, alpha=0.5), so.Hist(bins=16), legend=True)
    .scale(color=["C2", "C1"])
    .label(y="Frequency")
    .on(sf4)
    #.theme(theme_dict)
    .plot()
    )
(
    so.Plot(df_demon_attack, x="Model version", y="Human-normalized score")
    .add(so.Dots(color="C4", alpha=1.0), so.Jitter(jitter), legend=False)
    .add(so.Line(marker="_", color=".0"), so.Agg('median'))
    .add(so.Range(capsize=0.1, color=".0"), so.Est('median', errorbar=("ci", 95)))
    .on(sf5)
    .plot()
    )
(
    so.Plot(df_demon_attack, x="Human-normalized score", color="Model version")
    .add(so.Bars(edgewidth=1.25, alpha=0.5), so.Hist(bins=16), legend=False)
    .scale(color=["C2", "C1"])
    .label(y="Frequency")
    .on(sf6)
    #.theme(theme_dict)
    .plot()
)

fig.legends[0].set_bbox_to_anchor((0.775, 0.6))
# fig.axes[0].axhline(breakout_random, ls='dotted', color="#5144D3", alpha=1.0)
fig.axes[2].axhline(normalize_score(0.0, "Pong"), ls='dashed', color="C3", alpha=1.0)
# fig.axes[4].axhline(demon_attack_random, ls='dotted', color="#5144D3", alpha=1.0)

plt.savefig("plots/test_plot_all_games.pdf", bbox_inches='tight')
