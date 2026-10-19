"""Decentralized Proximal Stochastic Gradient Langevin Dynamics (DE-PSGLD)."""
from .depsgld_core import (NETWORK_NAMES, adjacency, mixing_matrix, eta_max, de_psgld, psgld,
                           proj_interval, proj_l2_ball, proj_l1_ball, w2_1d, w2_empirical)
from . import problems, data
from .plot_style import apply_style, COLORS
