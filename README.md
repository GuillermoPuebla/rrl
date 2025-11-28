# Learning Rules from Rewards

Codebase for the paper [Learning Rules from Rewards](https://arxiv.org/abs/2203.13599).

The `train_and_test.py` script allows to reproduce the primary results in the paper. The script requires three arguments:
1. Game name. Either 'Breakout', 'Pong' or 'DemmonAttack'
2. Model version. Either 'comparative' or 'logical'.
3. Number of runs.

For example, to replicate the results of the paper for the comparative version of RRTL on Breakout run:
```
train_and_test.py --game Breakout --model_version comparative --runs 100
```
This script uses the default values for the simulations reported in the paper.

The data used in the results of the paper can be found in: https://osf.io/m4yf9/

## Prerequisites

- Python 3
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PIL](https://pillow.readthedocs.io/en/3.1.x/installation.html)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV-python](https://anaconda.org/conda-forge/opencv)
- [Graphviz-python](https://anaconda.org/conda-forge/graphviz)
