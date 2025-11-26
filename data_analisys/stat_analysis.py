import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg


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

games = ["Breakout", "Pong", "DemonAttack"]
for game in games:
    df = pd.read_csv(f"datasets/test_data_{game}.csv")
    return_l = df.loc[df["Model version"]=="Logical", "Mean return"].values
    return_l = normalize_score(return_l, game)
    return_c = df.loc[df["Model version"]=="Comparative", "Mean return"].values
    return_c = normalize_score(return_c, game)
    # Seeds equal or above 1.0
    eq_above_1_l = len([x for x in return_l if x >= 1.0])
    eq_above_1_c = len([x for x in return_c if x >= 1.0])
    # Means
    mean_l = np.mean(return_l)
    mean_c = np.mean(return_c)
    # Medians
    median_l = np.median(return_l)
    median_c = np.median(return_c)
    # Skewness
    skewness_l = stats.skew(return_l)
    skewness_c = stats.skew(return_c)
    # Mann–Whitney U test
    results = pg.mwu(return_c, return_l, alternative='two-sided')
    print(game)
    print("Median: ", round(median_c, 3), round(median_l, 3))
    print(">= 1.0: ", eq_above_1_c, eq_above_1_l)
    print("Skewness", round(skewness_c, 3), round(skewness_l, 3))
    print(results)
    print("")

# The Mann–Whitney U test is a non-parametric test of the null hypothesis that it is equally likely that a randomly selected value from one sample will be less than or greater than a randomly selected value from a second sample. 

# Common Language Effect Size (CL): This represents the probability that a randomly selected value from one group will be greater than a randomly selected value from the other group. For example, a CL of 0.75 means there is a 75% chance that a score from group 1 is higher than a score from group 2. 