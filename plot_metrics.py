import os
from simulation import (
    StrategyType, compare_strategies, GameMetrics,
    aggregate_metrics, main as simulation_main
)
import json
from typing import Dict, List, Any
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Configure matplotlib to handle font issues and avoid font warnings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Suppress font warnings
warnings.filterwarnings("ignore", message=".*font.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="matplotlib.font_manager")


def run_simulation_and_collect_data(iterations: int = 200, games_per_strategy: int = 10) -> Dict[str, Any]:
    """Run the simulation and collect comprehensive data"""
    print("Running simulation to collect plotting data...")

    strategies = [
        StrategyType.RANDOM,
        StrategyType.MAX_CARD,
        StrategyType.MIN_CARD,
        StrategyType.IS_MCTS_SINGLE,
        StrategyType.IS_MCTS_PER_ACTION,
        StrategyType.IS_MCTS_PER_TRICK
    ]

    results = compare_strategies(
        strategies=strategies,
        iterations=iterations,
        games_per_strategy=games_per_strategy,
        verbose=False  # Reduce output for cleaner plotting
    )

    return results


def create_time_metrics_plot(results: Dict[str, Any], save_path: str = "plot/time_metrics.png"):
    """Create plots for time-related metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Time Performance Metrics Comparison',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())

    # Total game time
    total_times = [results[strategy]['avg_total_time']
                   for strategy in strategies]
    axes[0, 0].bar(strategies, total_times,
                   color=sns.color_palette("husl", len(strategies)))
    axes[0, 0].set_title('Average Total Game Time')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Average decision time
    decision_times = [results[strategy]['avg_decision_time']
                      for strategy in strategies]
    axes[0, 1].bar(strategies, decision_times,
                   color=sns.color_palette("husl", len(strategies)))
    axes[0, 1].set_title('Average Decision Time')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Decisions per second
    decisions_per_sec = [results[strategy]
                         ['avg_decisions_per_second'] for strategy in strategies]
    axes[1, 0].bar(strategies, decisions_per_sec,
                   color=sns.color_palette("husl", len(strategies)))
    axes[1, 0].set_title('Average Decisions Per Second')
    axes[1, 0].set_ylabel('Decisions/sec')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Cumulative time comparison
    # cumulative_times = np.cumsum(total_times)
    # axes[1, 1].plot(strategies, cumulative_times,
    #                 marker='o', linewidth=2, markersize=8)
    # axes[1, 1].set_title('Cumulative Total Game Time')
    # axes[1, 1].set_ylabel('Cumulative Time (seconds)')
    # axes[1, 1].tick_params(axis='x', rotation=45)
    # axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time metrics plot saved as {save_path}")
    return fig


def create_decision_metrics_plot(results: Dict[str, Any], save_path: str = "plot/decision_metrics.png"):
    """Create plots for decision-related metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Decision Metrics Comparison', fontsize=16, fontweight='bold')

    strategies = list(results.keys())

    # Total decisions
    total_decisions = [results[strategy]['avg_total_decisions']
                       for strategy in strategies]
    axes[0, 0].bar(strategies, total_decisions,
                   color=sns.color_palette("viridis", len(strategies)))
    axes[0, 0].set_title('Average Total Decisions')
    axes[0, 0].set_ylabel('Number of Decisions')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Chance decisions
    chance_decisions = [results[strategy]['avg_chance_decisions']
                        for strategy in strategies]
    axes[0, 1].bar(strategies, chance_decisions,
                   color=sns.color_palette("viridis", len(strategies)))
    axes[0, 1].set_title('Average Chance Decisions')
    axes[0, 1].set_ylabel('Number of Chance Decisions')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Average legal actions
    legal_actions = [results[strategy]['avg_legal_actions']
                     for strategy in strategies]
    axes[1, 0].bar(strategies, legal_actions,
                   color=sns.color_palette("viridis", len(strategies)))
    axes[1, 0].set_title('Average Legal Actions per Decision')
    axes[1, 0].set_ylabel('Number of Legal Actions')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Cumulative decisions comparison
    # cumulative_decisions = np.cumsum(total_decisions)
    # axes[1, 1].plot(strategies, cumulative_decisions, marker='s',
    #                 linewidth=2, markersize=8, color='orange')
    # axes[1, 1].set_title('Cumulative Total Decisions')
    # axes[1, 1].set_ylabel('Cumulative Decisions')
    # axes[1, 1].tick_params(axis='x', rotation=45)
    # axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Decision metrics plot saved as {save_path}")
    return fig


def create_mcts_metrics_plot(results: Dict[str, Any], save_path: str = "plot/mcts_metrics.png"):
    """Create plots for MCTS-specific metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MCTS Performance Metrics', fontsize=16, fontweight='bold')

    strategies = list(results.keys())

    # Node visits
    node_visits = [results[strategy]['avg_node_visits']
                   for strategy in strategies]
    log_node_visits = [np.log10(max(1, x)) for x in node_visits]
    axes[0, 0].bar(strategies, log_node_visits,
                   color=sns.color_palette("plasma", len(strategies)))
    axes[0, 0].set_title('Average Node Visits')
    axes[0, 0].set_ylabel('Number of Node Visits (log10 scale)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    # Log scale due to potentially large differences

    # Nodes created
    nodes_created = [results[strategy]['avg_nodes_created']
                     for strategy in strategies]
    log_nodes_created = [np.log10(max(1, x)) for x in nodes_created]
    axes[0, 1].bar(strategies, log_nodes_created,
                   color=sns.color_palette("plasma", len(strategies)))
    axes[0, 1].set_title('Average Nodes Created')
    axes[0, 1].set_ylabel('Number of Nodes Created (log10 scale)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    # Log scale due to potentially large differences

    # Trees created
    trees_created = [results[strategy]['avg_trees_created']
                     for strategy in strategies]
    axes[1, 0].bar(strategies, trees_created,
                   color=sns.color_palette("plasma", len(strategies)))
    axes[1, 0].set_title('Average Trees Created')
    axes[1, 0].set_ylabel('Number of Trees Created')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Cumulative node visits
    cumulative_visits = np.cumsum(node_visits)
    log_cumulative_visits = [np.log10(max(1, x)) for x in cumulative_visits]
    axes[1, 1].plot(strategies, log_cumulative_visits, marker='^',
                    linewidth=2, markersize=8, color='red')
    axes[1, 1].set_title('Cumulative Node Visits')
    axes[1, 1].set_ylabel('Cumulative Node Visits (log10 scale)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MCTS metrics plot saved as {save_path}")
    return fig


def create_memory_and_game_metrics_plot(results: Dict[str, Any], save_path: str = "plot/memory_game_metrics.png"):
    """Create plots for memory usage and game state metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Memory Usage and Game State Metrics',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())

    # Peak memory usage
    peak_memory = [results[strategy]['avg_peak_memory']
                   for strategy in strategies]
    axes[0, 0].bar(strategies, peak_memory,
                   color=sns.color_palette("coolwarm", len(strategies)))
    axes[0, 0].set_title('Average Peak Memory Usage')
    axes[0, 0].set_ylabel('Memory (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Cards played
    cards_played = [results[strategy]['avg_cards_played']
                    for strategy in strategies]
    axes[0, 1].bar(strategies, cards_played,
                   color=sns.color_palette("coolwarm", len(strategies)))
    axes[0, 1].set_title('Average Cards Played')
    axes[0, 1].set_ylabel('Number of Cards')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Tricks played
    tricks_played = [results[strategy]['avg_tricks_played']
                     for strategy in strategies]
    axes[1, 0].bar(strategies, tricks_played,
                   color=sns.color_palette("coolwarm", len(strategies)))
    axes[1, 0].set_title('Average Tricks Played')
    axes[1, 0].set_ylabel('Number of Tricks')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Cumulative memory usage
    # cumulative_memory = np.cumsum(peak_memory)
    # axes[1, 1].plot(strategies, cumulative_memory, marker='d',
    #                 linewidth=2, markersize=8, color='purple')
    # axes[1, 1].set_title('Cumulative Peak Memory Usage')
    # axes[1, 1].set_ylabel('Cumulative Memory (MB)')
    # axes[1, 1].tick_params(axis='x', rotation=45)
    # axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Memory and game metrics plot saved as {save_path}")
    return fig


def create_comprehensive_comparison_plot(results: Dict[str, Any], save_path: str = "plot/comprehensive_comparison.png"):
    """Create a comprehensive comparison plot showing all key metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Comprehensive Strategy Comparison Against Random Opponents',
                 fontsize=18, fontweight='bold')

    strategies = list(results.keys())
    colors = sns.color_palette("Set2", len(strategies))

    # 1. Performance efficiency (decisions per second vs memory)
    decisions_per_sec = [results[strategy]
                         ['avg_decisions_per_second'] for strategy in strategies]
    peak_memory = [results[strategy]['avg_peak_memory']
                   for strategy in strategies]

    scatter = axes[0, 0].scatter(
        peak_memory, decisions_per_sec, c=colors, s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Peak Memory Usage (MB)')
    axes[0, 0].set_ylabel('Decisions Per Second')
    axes[0, 0].set_title('Performance Efficiency')
    axes[0, 0].grid(True, alpha=0.3)

    # Add labels for each point
    for i, strategy in enumerate(strategies):
        axes[0, 0].annotate(strategy, (peak_memory[i], decisions_per_sec[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 2. MCTS complexity (nodes created vs trees created)
    nodes_created = [results[strategy]['avg_nodes_created']
                     for strategy in strategies]
    trees_created = [results[strategy]['avg_trees_created']
                     for strategy in strategies]

    # Filter out zero values for log scale
    mcts_strategies = []
    mcts_nodes = []
    mcts_trees = []
    mcts_colors = []

    for i, strategy in enumerate(strategies):
        if nodes_created[i] > 0 and trees_created[i] > 0:
            mcts_strategies.append(strategy)
            mcts_nodes.append(nodes_created[i])
            mcts_trees.append(trees_created[i])
            mcts_colors.append(colors[i])

    if mcts_nodes:
        log_mcts_nodes = [np.log10(max(1, x)) for x in mcts_nodes]
        axes[0, 1].scatter(mcts_trees, log_mcts_nodes,
                           c=mcts_colors, s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Trees Created')
        axes[0, 1].set_ylabel('Nodes Created (log10 scale)')
        axes[0, 1].set_title('MCTS Complexity')
        axes[0, 1].grid(True, alpha=0.3)

        for i, strategy in enumerate(mcts_strategies):
            axes[0, 1].annotate(strategy, (mcts_trees[i], log_mcts_nodes[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, 'No MCTS data available',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('MCTS Complexity')

    # 3. Time breakdown
    total_times = [results[strategy]['avg_total_time']
                   for strategy in strategies]
    decision_times = [results[strategy]['avg_decision_time']
                      for strategy in strategies]

    x_pos = np.arange(len(strategies))
    axes[1, 0].bar(x_pos, total_times, color=colors,
                   alpha=0.7, label='Total Game Time')
    axes[1, 0].set_xlabel('Strategy')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Average Game Duration')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(strategies, rotation=45)

    # 4. Decision complexity
    legal_actions = [results[strategy]['avg_legal_actions']
                     for strategy in strategies]
    total_decisions = [results[strategy]['avg_total_decisions']
                       for strategy in strategies]

    axes[1, 1].bar(x_pos, legal_actions, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('Average Legal Actions')
    axes[1, 1].set_title('Decision Space Complexity')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(strategies, rotation=45)

    # 5. Cumulative performance metrics
    axes[2, 0].plot(strategies, np.cumsum(total_times), 'o-',
                    linewidth=2, label='Cumulative Time', color='blue')
    axes[2, 0].plot(strategies, np.cumsum(peak_memory), 's-',
                    linewidth=2, label='Cumulative Memory', color='red')
    axes[2, 0].set_xlabel('Strategy')
    axes[2, 0].set_ylabel('Cumulative Value')
    axes[2, 0].set_title('Cumulative Resource Usage')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Performance radar chart (relative to random strategy)
    # Normalize metrics relative to random strategy
    random_idx = strategies.index('random') if 'random' in strategies else 0

    metrics_to_compare = ['avg_decisions_per_second', 'avg_total_time', 'avg_peak_memory',
                          'avg_node_visits', 'avg_nodes_created']
    metric_labels = ['Dec/Sec', 'Time', 'Memory', 'Visits', 'Nodes']

    # Create normalized data for radar chart
    normalized_data = {}
    for strategy in strategies:
        normalized_data[strategy] = []
        for metric in metrics_to_compare:
            value = results[strategy][metric]
            random_value = results[strategies[random_idx]][metric]
            if random_value > 0:
                if metric in ['avg_total_time', 'avg_peak_memory']:  # Lower is better
                    normalized_value = random_value / max(value, 0.001)
                else:  # Higher is better
                    normalized_value = value / max(random_value, 0.001)
            else:
                normalized_value = 1.0
            normalized_data[strategy].append(normalized_value)

    # Simple bar chart instead of radar for simplicity
    x_metrics = np.arange(len(metric_labels))
    width = 0.15
    for i, strategy in enumerate(strategies):
        offset = (i - len(strategies)/2) * width
        axes[2, 1].bar(x_metrics + offset, normalized_data[strategy],
                       width, label=strategy, alpha=0.7, color=colors[i])

    axes[2, 1].set_xlabel('Metrics')
    axes[2, 1].set_ylabel('Performance Relative to Random')
    axes[2, 1].set_title('Relative Performance Comparison')
    axes[2, 1].set_xticks(x_metrics)
    axes[2, 1].set_xticklabels(metric_labels)
    axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison plot saved as {save_path}")
    return fig


def save_results_to_json(results: Dict[str, Any], save_path: str = "plot/simulation_results.json"):
    """Save results to JSON for future analysis"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")


def main():
    """Main function to run simulation and create all plots"""
    print("Tarot Strategy Metrics Visualization")
    print("=" * 50)

    os.makedirs("plot", exist_ok=True)  # Ensure plot directory exists

    # Run simulation to collect data
    results = run_simulation_and_collect_data(
        iterations=200,  # MCTS iterations per decision
        games_per_strategy=5  # Number of games per strategy for faster plotting
    )

    # Save results for future use
    save_results_to_json(results)

    # Create all metric plots
    print("\nCreating visualization plots...")

    # Create individual metric plots
    create_time_metrics_plot(results)
    create_decision_metrics_plot(results)
    create_mcts_metrics_plot(results)
    create_memory_and_game_metrics_plot(results)

    # Create comprehensive comparison
    create_comprehensive_comparison_plot(results)

    print("\nAll plots created successfully!")
    print("Generated files:")
    print("- time_metrics.png")
    print("- decision_metrics.png")
    print("- mcts_metrics.png")
    print("- memory_game_metrics.png")
    print("- comprehensive_comparison.png")
    print("- simulation_results.json")

    # Show plots if running interactively
    plt.show()


if __name__ == "__main__":
    main()
