"""Plotting utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")


def save_figure(fig, filename: str, output_dir: str = "results/figures"):
    """Save figure with consistent formatting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
