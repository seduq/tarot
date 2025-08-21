"""
Basic Plotting Module for Tarot Game Metrics

This module provides simple def plot_legal_moves_growth(collector: MetricsCodef plot_decision_times(collector: MetricsCollector, save_path: Optional[str] = None, show_plots: bool = False, 
                        expected_games: Optional[int] = None):ector, strategies: List[str], save_path: Optional[str] = None, 
                            show_plots: bool = False, expected_games: Optional[int] = None):otting functions for visualizing game metrics.
Each metric gets its own figure for clear comparison across strategies.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
from metrics import MetricsCollector


def plot_win_rates(collector: MetricsCollector, strategies: List[str], save_path: Optional[str] = None,
                   show_plots: bool = False, expected_games: Optional[int] = None):
    """
    Create bar plots for taker and defender win rates.

    Args:
        collector: MetricsCollector with game data
        strategies: List of strategy names to compare
        save_path: Optional path to save the plot
        show_plots: Whether to display the plot
        expected_games: Expected number of games per strategy
    """
    # Create subplots for taker and defender win rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    taker_rates = []
    defender_rates = []
    game_counts = []

    for strategy in strategies:
        win_rates = collector.get_win_rates(strategy)
        taker_rates.append(win_rates["taker_win_rate"])
        defender_rates.append(win_rates["defender_win_rate"])

        # Get actual game count for this strategy
        strategy_games = collector.get_strategy_metrics(strategy)
        game_counts.append(len(strategy_games))

    # Taker win rates
    bars1 = ax1.bar(strategies, taker_rates, color=[
                    '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
    ax1.set_title('Taker Win Rate by Strategy')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1.2)

    # Add value labels on bars with game counts
    for bar, rate, count in zip(bars1, taker_rates, game_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{(rate * 100):.1f}%', ha='center', va='bottom', fontsize=9)

    # Defender win rates
    bars2 = ax2.bar(strategies, defender_rates, color=[
                    '#ff7f0e', '#2ca02c', '#d62728', '#1f77b4'])
    ax2.set_title('Defender Win Rate by Strategy')
    ax2.set_ylabel('Win Rate')
    ax2.set_ylim(0, 1.2)

    # Add value labels on bars with game counts
    for bar, rate, count in zip(bars2, defender_rates, game_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{(rate * 100):.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_legal_moves_growth(collector: MetricsCollector, strategies: List[str], save_path: Optional[str] = None,
                            show_plots: bool = True, expected_games: Optional[int] = None):
    """
    Create line plot for legal moves growth over game turns with smoothing.

    Args:
        collector: MetricsCollector with game data
        strategies: List of strategy names to compare
        save_path: Optional path to save the plot
        show_plots: Whether to display the plot
        expected_games: Expected number of games per strategy
    """
    plt.figure(figsize=(10, 6))

    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']

    # Count total games across all strategies for title
    total_games = sum(len(collector.get_strategy_metrics(strategy))
                      for strategy in strategies)

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

    plt.title(
        f'Legal Moves Growth During Game (Smoothed) - {total_games} total games')
    plt.xlabel('Game Turn')
    plt.ylabel('Average Number of Legal Moves (5-turn smoothing)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_decision_times(collector: MetricsCollector, save_path: Optional[str] = None, show_plots: bool = True,
                        expected_games: Optional[int] = None):
    """
    Create line plot for RIS-MCTS average decision times through game progress.

    Args:
        collector: MetricsCollector with game data
        save_path: Optional path to save the plot
        show_plots: Whether to display the plot
        expected_games: Expected number of games per strategy
    """
    plt.figure(figsize=(10, 6))

    # Get RIS-MCTS game count for title
    ris_mcts_games = collector.get_strategy_metrics("ris_mcts")
    game_count = len(ris_mcts_games)

    # Only plot RIS-MCTS strategy with progressive averaging
    decision_times_avg = collector.get_average_decision_times(
        "ris_mcts", progressive=False)

    if decision_times_avg:
        decisions = range(1, len(decision_times_avg) + 1)
        plt.plot(decisions, decision_times_avg,
                 label='RIS-MCTS Progressive Average',
                 color='#1f77b4',
                 linewidth=2, marker='o', markersize=4)

        plt.title(
            f'RIS-MCTS Average Decision Time Through MCTS Decisions ({game_count} games)')
        plt.xlabel('Progression of MCTS Decisions')
        plt.ylabel('Average Decision Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No RIS-MCTS decision time data available',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(
            f'RIS-MCTS Decision Time Growth - No Data ({game_count} games found)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close()


def create_all_plots(metrics_file: str, output_dir: str = "plots", strategies: Optional[List[str]] = None,
                     plot_types: Optional[List[str]] = None, show_plots: bool = False,
                     expected_games: Optional[int] = None):
    """
    Create all basic plots from a metrics file.

    Args:
        metrics_file: Path to the metrics JSON file
        output_dir: Directory to save plots
        strategies: List of strategies to include (default: all available)
        plot_types: List of plot types to generate (default: all)
        show_plots: Whether to display plots interactively
        expected_games: Expected number of games per strategy (for validation/display)
    """
    os.makedirs(output_dir, exist_ok=True)

    collector = MetricsCollector.load_from_file(metrics_file)

    # Use provided strategies or default ones
    if strategies is None:
        strategies = ["random", "max_card", "min_card", "ris_mcts"]

    # Filter strategies to only include those that have data
    available_strategies = collector.get_available_strategies()
    strategies = [s for s in strategies if s in available_strategies]

    if not strategies:
        print("Warning: No valid strategies found in the data")
        return

    # Default plot types
    all_plot_types = ["win_rates", "legal_moves", "decision_times"]
    if plot_types is None:
        plot_types = all_plot_types

    print(f"Creating plots for strategies: {strategies}")
    print(f"Plot types: {plot_types}")

    # Report game counts and validate if expected_games is provided
    if expected_games:
        print(f"Expected games per strategy: {expected_games}")

    for strategy in strategies:
        strategy_games = collector.get_strategy_metrics(strategy)
        game_count = len(strategy_games)
        print(f"  {strategy}: {game_count} games", end="")

        if expected_games and game_count != expected_games:
            print(f" (WARNING: Expected {expected_games})")
        else:
            print()

    # Temporarily disable interactive plotting if show_plots is False
    if not show_plots:
        plt.ioff()

    # Win rates plot
    if "win_rates" in plot_types:
        print("Creating win rates plot...")
        plot_win_rates(collector, strategies,
                       f"{output_dir}/win_rates.png", show_plots, expected_games)

    # Legal moves growth plot
    if "legal_moves" in plot_types:
        print("Creating legal moves growth plot...")
        plot_legal_moves_growth(
            collector, strategies, f"{output_dir}/legal_moves_growth.png", show_plots, expected_games)

    # Decision times plot (RIS-MCTS only)
    if "decision_times" in plot_types and "ris_mcts" in strategies:
        print("Creating decision times plot...")
        plot_decision_times(
            collector, f"{output_dir}/decision_times.png", show_plots, expected_games)

    # Re-enable interactive plotting
    if not show_plots:
        plt.ion()

    print(f"All plots saved to {output_dir}/")


def parse_arguments():
    """Parse command line arguments for basic plotting."""
    parser = argparse.ArgumentParser(
        description='Generate basic plots from Tarot game metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from default metrics file (save only)
  python plot_basic.py

  # Use custom metrics file and output directory with interactive display
  python plot_basic.py --input results/my_metrics.json --output my_plots --show

  # Generate only specific plot types with game count validation
  python plot_basic.py --plot-types win_rates legal_moves --games 200

  # Include only specific strategies with interactive display
  python plot_basic.py --strategies random --games 100 --show

  # Batch processing (default behavior - save only)
  python plot_basic.py --output batch_plots --games 300
        """)

    parser.add_argument('--input', '-i', type=str, default='results/simulation_metrics.json',
                        help='Path to input metrics JSON file (default: results/simulation_metrics.json)')

    parser.add_argument('--output', '-o', type=str, default='plots',
                        help='Output directory for plots (default: plots)')

    parser.add_argument('--games', type=int,
                        help='Expected number of games per strategy (for display purposes)')

    parser.add_argument('--strategies', nargs='+',
                        choices=['random', 'max_card', 'min_card', 'ris_mcts'],
                        help='Strategies to include in plots (default: all available)')

    parser.add_argument('--plot-types', nargs='+',
                        choices=['win_rates', 'legal_moves', 'decision_times'],
                        help='Types of plots to generate (default: all)')

    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: save only)')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def main():
    """Main function to generate plots with command line arguments."""
    args = parse_arguments()

    if args.verbose:
        print(f"Input file: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Show plots: {args.show}")
        if args.games:
            print(f"Expected games: {args.games}")
        if args.strategies:
            print(f"Strategies: {args.strategies}")
        if args.plot_types:
            print(f"Plot types: {args.plot_types}")

    try:
        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            return

        create_all_plots(
            metrics_file=args.input,
            output_dir=args.output,
            strategies=args.strategies,
            plot_types=args.plot_types,
            show_plots=args.show,
            expected_games=args.games
        )

        print("Plotting completed successfully!")

    except Exception as e:
        print(f"Error during plotting: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
