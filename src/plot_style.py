import matplotlib.pyplot as plt
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
def apply_style():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.3, "legend.framealpha": 0.9})
