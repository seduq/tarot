"""
Simple script to generate plots from simulation metrics.

Usage:
    python run_plots.py simulation_metrics_seed_7264828.json
"""

import sys
from plot_basic import create_all_plots


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_plots.py <metrics_file.json>")
        print("Example: python run_plots.py simulation_metrics_seed_7264828.json")
        return

    metrics_file = sys.argv[1]
    output_dir = "basic_plots"

    print(f"Creating plots from {metrics_file}...")
    create_all_plots(metrics_file, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
