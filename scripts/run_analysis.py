#!/usr/bin/env python3
"""Main analysis pipeline script."""

import sys
from pathlib import Path

# Add src to path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from idd_manuscripts.data.loaders import load_csv
from idd_manuscripts.analysis.statistics import basic_summary
from idd_manuscripts.visualization.plots import setup_plot_style


def main():
    """Run the main analysis pipeline."""
    print("Starting analysis pipeline...")
    
    # Set up plotting
    setup_plot_style()
    
    print("Analysis pipeline completed!")


if __name__ == "__main__":
    main()
