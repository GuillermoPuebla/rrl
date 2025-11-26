import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


df_breakout = pd.read_csv("datasets/test_data_Breakout.csv")
df_breakout = df_breakout[df_breakout["Model version"] == "Comparative"]

df_pong = pd.read_csv("datasets/test_data_Pong.csv")
df_pong = df_pong[df_pong["Model version"] == "Comparative"]

df_demon_attack = pd.read_csv("datasets/test_data_DemonAttack.csv")
df_demon_attack = df_demon_attack[df_demon_attack["Model version"] == "Comparative"]

# Get the run of the comparative version with the higest mean return
sorted_breakout = df_breakout.sort_values(by=['Mean return'], ascending=False)
print("Breakout", sorted_breakout["Run"].iloc[0])

sorted_pong = df_pong.sort_values(by=['Mean return'], ascending=False)
print("Pong", sorted_pong["Run"].iloc[0])

sorted_demon_attack = df_demon_attack.sort_values(by=['Mean return'], ascending=False)
print("Demon Attack",sorted_demon_attack["Run"].iloc[0])

print(sorted_demon_attack)