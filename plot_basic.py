"""
Basic Plotting Module for Tarot Game Metrics

This module provides simple plotting functions for visualizing game metrics.
Each metric gets its own figure for clear comparison across strategies.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from metrics import MetricsCollector


def plot_win_rates(collector: MetricsCollector, strategies: List[str], save_path: Optional[str] = None):
    """
    Create bar plots for taker and defender win rates.

    Args:
        collector: MetricsCollector with game data
        strategies: List of strategy names to compare
        save_path: Optional path to save the plot
    """
    # Create subplots for taker and defender win rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    taker_rates = []
    defender_rates = []

    for strategy in strategies:
        win_rates = collector.get_win_rates(strategy)
        taker_rates.append(win_rates["taker_win_rate"])
        defender_rates.append(win_rates["defender_win_rate"])

    # Taker win rates
    bars1 = ax1.bar(strategies, taker_rates, color=[
                    '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
    ax1.set_title('Taker Win Rate by Strategy')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for bar, rate in zip(bars1, taker_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{rate:.3f}', ha='center', va='bottom')

    # Defender win rates
    bars2 = ax2.bar(strategies, defender_rates, color=[
                    '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
    ax2.set_title('Defender Win Rate by Strategy')
    ax2.set_ylabel('Win Rate')
    ax2.set_ylim(0, 1)

    # Add value labels on bars
    for bar, rate in zip(bars2, defender_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{rate:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_legal_moves_growth(collector: MetricsCollector, strategies: List[str], save_path: Optional[str] = None):
    """
    Create line plot for legal moves growth over game turns with smoothing.

    Args:
        collector: MetricsCollector with game data
        strategies: List of strategy names to compare
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']

    for i, strategy in enumerate(strategies):
        # Use smoothing to reduce peaks and noise
        legal_moves_avg = collector.get_average_legal_moves(
            strategy, normalize=False, smoothing_window=5)
        if legal_moves_avg:
            turns = range(1, len(legal_moves_avg) + 1)
            plt.plot(turns, legal_moves_avg,
                     label=strategy.replace('_', ' ').title(),
                     color=colors[i % len(colors)],
                     linewidth=2, marker='o', markersize=3, alpha=0.8)

    plt.title('Legal Moves Growth During Game (Smoothed)')
    plt.xlabel('Game Turn')
    plt.ylabel('Average Number of Legal Moves (5-turn smoothing)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_decision_times(collector: MetricsCollector, save_path: Optional[str] = None):
    """
    Create line plot for RIS-MCTS average decision times through game progress.

    Args:
        collector: MetricsCollector with game data
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Only plot RIS-MCTS strategy with progressive averaging
    decision_times_avg = collector.get_average_decision_times(
        "ris_mcts", progressive=False)

    if decision_times_avg:
        decisions = range(1, len(decision_times_avg) + 1)
        plt.plot(decisions, decision_times_avg,
                 label='RIS-MCTS Progressive Average',
                 color='#1f77b4',
                 linewidth=2, marker='o', markersize=4)

        plt.title('RIS-MCTS Average Decision Time Through MCTS Decisions')
        plt.xlabel('MCTS Decision Number')
        plt.ylabel('Cumulative Average Decision Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No RIS-MCTS decision time data available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('RIS-MCTS Decision Time Growth - No Data')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_mcts_comparison(collector: MetricsCollector, mcts_configs: List[Dict], save_path: Optional[str] = None):
    """
    Compare different RIS-MCTS configurations.

    Args:
        collector: MetricsCollector with game data
        mcts_configs: List of MCTS configuration dicts to compare
        save_path: Optional path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(mcts_configs)))

    for i, config in enumerate(mcts_configs):
        # Filter games by MCTS config
        config_games = [
            game for game in collector.get_strategy_metrics("ris_mcts")
            if game.mcts_config == config
        ]

        if not config_games:
            continue

        config_name = f"iter={config['iterations']}, exp={config['exploration_constant']}"

        # Win rates
        total_games = len(config_games)
        taker_wins = sum(1 for game in config_games if game.taker_won)
        defender_wins = sum(1 for game in config_games if game.defender_won)

        ax1.bar(i, taker_wins / total_games,
                color=colors[i], label=config_name)
        ax2.bar(i, defender_wins / total_games, color=colors[i])

        # Decision times (if available)
        decision_times = []
        for game in config_games:
            if game.decision_times:
                decision_times.extend(game.decision_times)

        if decision_times:
            ax3.boxplot([decision_times], positions=[i], widths=0.6)

        # Legal moves (average per game)
        legal_moves_per_game = [
            sum(game.legal_moves_history) / len(game.legal_moves_history)
            for game in config_games
            if game.legal_moves_history
        ]

        if legal_moves_per_game:
            ax4.boxplot([legal_moves_per_game], positions=[i], widths=0.6)

    ax1.set_title('Taker Win Rate by MCTS Config')
    ax1.set_ylabel('Win Rate')
    ax1.legend()

    ax2.set_title('Defender Win Rate by MCTS Config')
    ax2.set_ylabel('Win Rate')

    ax3.set_title('Decision Times by MCTS Config')
    ax3.set_ylabel('Decision Time (seconds)')

    ax4.set_title('Average Legal Moves by MCTS Config')
    ax4.set_ylabel('Average Legal Moves per Turn')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_all_plots(metrics_file: str, output_dir: str = "plots"):
    """
    Create all basic plots from a metrics file.

    Args:
        metrics_file: Path to the metrics JSON file
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    collector = MetricsCollector.load_from_file(metrics_file)
    strategies = ["random", "max_card", "min_card", "ris_mcts"]

    # Win rates plot
    plot_win_rates(collector, strategies, f"{output_dir}/win_rates.png")

    # Legal moves growth plot
    plot_legal_moves_growth(collector, strategies,
                            f"{output_dir}/legal_moves_growth.png")

    # Decision times plot (RIS-MCTS only)
    plot_decision_times(collector, f"{output_dir}/decision_times.png")

    plot_mcts_comparison(
        collector,
        [
            {"iterations": 50, "exploration_constant": 1.4},
            {"iterations": 100, "exploration_constant": 1.4},
            {"iterations": 200, "exploration_constant": 1.4}
        ],
        f"{output_dir}/mcts_comparison.png"
    )

    print(f"All plots saved to {output_dir}/")
