"""
Base plotting utilities and common functions for Tarot strategy visualization.
"""

import warnings
import matplotlib
import matplotlib.font_manager as fm
import os
from typing import Dict, Any
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib to handle font issues and avoid font warnings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Suppress font warnings
warnings.filterwarnings("ignore", message=".*font.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="matplotlib.font_manager")

# Set style for better-looking plots
try:
    plt.style.use('default')
except:
    pass

sns.set_palette("husl")


def setup_plotting_environment():
    """Setup the plotting environment for non-interactive use"""
    matplotlib.use('Agg')
    os.makedirs("plot", exist_ok=True)


def load_results_from_json(file_path: str = "plot/simulation_results.json") -> Dict[str, Any]:
    """Load results from JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Results file {file_path} not found. Please run the simulation first.")

    with open(file_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded results for {len(results)} strategies")
    return results


def filter_mcts_strategies(results: Dict[str, Any]) -> Dict[str, Any]:
    """Filter results to only include IS-MCTS strategies"""
    mcts_strategies = ['is_mcts_single',
                       'is_mcts_per_action', 'is_mcts_per_trick']

    filtered_results = {}
    for strategy, data in results.items():
        if strategy in mcts_strategies:
            filtered_results[strategy] = data

    print(
        f"Filtered to {len(filtered_results)} IS-MCTS strategies: {list(filtered_results.keys())}")
    return filtered_results


def save_results_to_json(results: Dict[str, Any], save_path: str = "plot/simulation_results.json"):
    """Save results to JSON for future analysis"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")


def get_strategy_colors(strategies: list) -> list:
    """Get consistent colors for strategies"""
    return sns.color_palette("Set2", len(strategies))


def format_strategy_name(strategy: str, multiline: bool = False) -> str:
    """Format strategy name for display"""
    if multiline:
        return strategy.replace('_', '\n')
    else:
        return strategy.replace('_', ' ').title()


def create_value_labels_on_bars(axes, bars, values, format_string: str = "{:.1f}"):
    """Add value labels on top of bars"""
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                  format_string.format(value), ha='center', va='bottom', fontweight='bold')


def apply_log_scale_to_data(data: list, min_value: float = 1.0) -> list:
    """Apply log10 transformation to data for better visualization"""
    return [np.log10(max(min_value, x)) for x in data]
