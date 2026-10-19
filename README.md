# Decentralized Proximal Stochastic Gradient Langevin Dynamics  

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](experiments/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/rispace/de-psgld)](https://github.com/rispace/de-psgld/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/rispace/de-psgld)](https://github.com/rispace/de-psgld)
[![Website](https://img.shields.io/badge/Website-rispace.github.io-0A66C2)](https://rispace.github.io)

**Authors** Mohammad Rafiqul Islam, Lingjiong Zhu  

Code for the numerical experiments of the paper *Decentralized Proximal Stochastic Gradient Langevin Dynamics* (DE-PSGLD).
DE-PSGLD samples from a log-concave distribution restricted to a convex set $\mathcal K$ when the data are distributed over $N$ agents
that communicate over a network. Each agent runs

$$
x_i^{(k+1)}=\sum_{j} W_{ij}x_j^{(k)}-\eta\Big[\tilde\nabla f_i(x_i^{(k)})+\tfrac{1}{N\gamma}\big(x_i^{(k)}-\mathcal P_{\mathcal K}(x_i^{(k)})\big)\Big]+\sqrt{2\eta}\,w_i^{(k+1)},
$$

where $W$ is the mixing matrix of the network, $\gamma$ is the proximal (Moreau–Yosida) parameter, and $\mathcal P_{\mathcal K}$ is the projection onto $\mathcal K$.

## Repository structure

```
code/
├── src/                      # library code
│   ├── depsgld_core.py       # networks, DE-PSGLD and PSGLD samplers, projections, W2 distances
│   ├── problems.py           # the four sampling problems of the paper
│   ├── data.py               # loaders for the Adult and California housing data
│   └── plot_style.py
├── experiments/              # one notebook per experiment
│   ├── exp_univariate.ipynb
│   ├── exp_linear_regression.ipynb
│   ├── exp_adult_income.ipynb
│   └── exp_california_housing.ipynb
├── data/                     # adult-all.csv, housing.csv
└── figure/                   # figures written by the notebooks
```

| Notebook | Paper | Problem | Constraint $\mathcal K$ |
|---|---|---|---|
| `exp_univariate` | Sec. 5.1 | $\pi\propto e^{-x^2-x^4/8+x}$ on $[-1,1]$, heterogeneous split over agents | interval |
| `exp_linear_regression` | Sec. 5.2 | Bayesian linear regression, synthetic data, exact posterior samples | $\ell_2$-ball |
| `exp_adult_income` | Sec. 5.3 | Bayesian logistic regression, Adult income data | $\ell_2$-ball |
| `exp_california_housing` | Sec. 5.4 | Bayesian lasso regression, California housing, one agent per region | $\ell_1$-ball |

All experiments use $N=10$ agents on the fully connected, circular, and star networks, with
$W=I-\delta L$ and $\delta=1/(d_{\max}+1)$, and compare against the centralized proximal SGLD (PSGLD) with stepsize $\eta/N$.

## Setup

Activate your Python environment (Python 3.10 or newer) and install the requirements:

```bash
pip install -r requirements.txt
```

## Running the experiments

Start Jupyter from the `code/` folder and open a notebook in `experiments/`:

```bash
jupyter lab        # or open the notebooks in VS Code
```

Each notebook has a **Parameters** cell at the top (number of agents, $\gamma$, stepsize fraction, iterations, number of copies $M$, networks, ...).
Change the values there and run the notebook from top to bottom. The stepsize is set as `ETA_FRAC` $\times\,\eta_{\max}$, where $\eta_{\max}$ is the bound of the main theorem.
Figures are saved to `figure/` when `SAVE_FIGS = True`.

Approximate run times with the default (paper) parameters on a laptop:
univariate 5–10 min (including the $\gamma$ sweep), linear regression ~10 min, Adult ~5 min, California housing ~6 min.
To try settings quickly, reduce `N_ITER` and `M`.

Notes on the parameters:
- `M` is the number of independent copies of the whole network. The distribution of each agent at each iteration is estimated from these copies, so `M` controls the Monte Carlo error of the reported $\mathcal W_2$ distances.
- Smaller `GAMMA` makes $\pi^\gamma$ closer to $\pi$ but also makes $\eta_{\max}$ smaller, so more iterations are needed.
- `RECORD_EVERY` controls how often the metrics are computed; in the linear regression notebook each record solves optimal transport problems of size `M`, which is the slowest part.

## Data

- `data/adult-all.csv`: Adult (Census Income) data, B. Becker and R. Kohavi, UCI Machine Learning Repository, 1996, https://doi.org/10.24432/C5XW20.
- `data/housing.csv`: California housing data, R. K. Pace and R. Barry, *Sparse spatial autoregressions*, Statistics & Probability Letters 33 (1997), 291–297.

## License

MIT License, see `LICENSE`.
