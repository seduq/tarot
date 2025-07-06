"""
Performance metrics plotting module for Tarot strategy analysis.

This module contains functions for creating detailed performance analysis plots,
including MCTS-specific metrics, decision metrics, memory usage, and game state metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from plotting_base import setup_plotting_environment, get_strategy_colors, format_strategy_name, create_value_labels_on_bars, apply_log_scale_to_data

# Initialize plotting environment
setup_plotting_environment()


def create_decision_metrics_plot(results: Dict[str, Any], save_path: str = "plot/decision_metrics.png"):
    """Create plots for decision-related metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IS-MCTS Decision Metrics Comparison',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)
    strategy_labels = [format_strategy_name(s) for s in strategies]

    # Total decisions
    total_decisions = [results[strategy].get(
        'avg_total_decisions', 0) for strategy in strategies]
    bars1 = axes[0, 0].bar(strategy_labels, total_decisions, color=colors)
    axes[0, 0].set_title('Average Total Decisions')
    axes[0, 0].set_ylabel('Number of Decisions')
    axes[0, 0].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[0, 0], bars1, total_decisions, "{:.0f}")

    # Chance decisions
    chance_decisions = [results[strategy].get(
        'avg_chance_decisions', 0) for strategy in strategies]
    bars2 = axes[0, 1].bar(strategy_labels, chance_decisions, color=colors)
    axes[0, 1].set_title('Average Chance Decisions')
    axes[0, 1].set_ylabel('Number of Chance Decisions')
    axes[0, 1].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[0, 1], bars2, chance_decisions, "{:.0f}")

    # Average legal actions
    legal_actions = [results[strategy].get(
        'avg_legal_actions', 0) for strategy in strategies]
    bars3 = axes[1, 0].bar(strategy_labels, legal_actions, color=colors)
    axes[1, 0].set_title('Average Legal Actions per Decision')
    axes[1, 0].set_ylabel('Number of Legal Actions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[1, 0], bars3, legal_actions, "{:.1f}")

    # Decision efficiency (ratio of total to chance decisions)
    decision_efficiency = [t/c if c > 0 else 0 for t,
                           c in zip(total_decisions, chance_decisions)]
    bars4 = axes[1, 1].bar(strategy_labels, decision_efficiency, color=colors)
    axes[1, 1].set_title('Decision Efficiency Ratio')
    axes[1, 1].set_ylabel('Total / Chance Decisions')
    axes[1, 1].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(
        axes[1, 1], bars4, decision_efficiency, "{:.2f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Decision metrics plot saved as {save_path}")
    return fig


def create_mcts_metrics_plot(results: Dict[str, Any], save_path: str = "plot/mcts_metrics.png"):
    """Create plots for MCTS-specific metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IS-MCTS Performance Metrics', fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)
    strategy_labels = [format_strategy_name(s) for s in strategies]

    # Node visits
    node_visits = [results[strategy].get(
        'avg_node_visits', 0) for strategy in strategies]
    log_node_visits = apply_log_scale_to_data(node_visits)
    bars1 = axes[0, 0].bar(strategy_labels, log_node_visits, color=colors)
    axes[0, 0].set_title('Average Node Visits')
    axes[0, 0].set_ylabel('Number of Node Visits (log10 scale)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[0, 0], bars1, log_node_visits, "{:.1f}")

    # Nodes created
    nodes_created = [results[strategy].get(
        'avg_nodes_created', 0) for strategy in strategies]
    log_nodes_created = apply_log_scale_to_data(nodes_created)
    bars2 = axes[0, 1].bar(strategy_labels, log_nodes_created, color=colors)
    axes[0, 1].set_title('Average Nodes Created')
    axes[0, 1].set_ylabel('Number of Nodes Created (log10 scale)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[0, 1], bars2, log_nodes_created, "{:.1f}")

    # Trees created
    trees_created = [results[strategy].get(
        'avg_trees_created', 0) for strategy in strategies]
    bars3 = axes[1, 0].bar(strategy_labels, trees_created, color=colors)
    axes[1, 0].set_title('Average Trees Created')
    axes[1, 0].set_ylabel('Number of Trees Created')
    axes[1, 0].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[1, 0], bars3, trees_created, "{:.0f}")

    # Node efficiency (visits per node created)
    node_efficiency = [v/n if n > 0 else 0 for v,
                       n in zip(node_visits, nodes_created)]
    bars4 = axes[1, 1].bar(strategy_labels, node_efficiency, color=colors)
    axes[1, 1].set_title('Node Efficiency (Visits per Node)')
    axes[1, 1].set_ylabel('Visits / Nodes Created')
    axes[1, 1].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[1, 1], bars4, node_efficiency, "{:.1f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MCTS metrics plot saved as {save_path}")
    return fig


def create_memory_and_game_metrics_plot(results: Dict[str, Any], save_path: str = "plot/memory_game_metrics.png"):
    """Create plots for memory usage and game state metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IS-MCTS Memory Usage and Game State Metrics',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)
    strategy_labels = [format_strategy_name(s) for s in strategies]

    # Peak memory usage
    peak_memory = [results[strategy].get(
        'avg_peak_memory', 0) for strategy in strategies]
    bars1 = axes[0, 0].bar(strategy_labels, peak_memory, color=colors)
    axes[0, 0].set_title('Average Peak Memory Usage')
    axes[0, 0].set_ylabel('Memory (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[0, 0], bars1, peak_memory, "{:.1f}")

    # Cards played
    cards_played = [results[strategy].get(
        'avg_cards_played', 0) for strategy in strategies]
    bars2 = axes[0, 1].bar(strategy_labels, cards_played, color=colors)
    axes[0, 1].set_title('Average Cards Played')
    axes[0, 1].set_ylabel('Number of Cards')
    axes[0, 1].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[0, 1], bars2, cards_played, "{:.0f}")

    # Tricks played
    tricks_played = [results[strategy].get(
        'avg_tricks_played', 0) for strategy in strategies]
    bars3 = axes[1, 0].bar(strategy_labels, tricks_played, color=colors)
    axes[1, 0].set_title('Average Tricks Played')
    axes[1, 0].set_ylabel('Number of Tricks')
    axes[1, 0].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[1, 0], bars3, tricks_played, "{:.0f}")

    # Memory efficiency (memory per card played)
    memory_efficiency = [m/c if c > 0 else 0 for m,
                         c in zip(peak_memory, cards_played)]
    bars4 = axes[1, 1].bar(strategy_labels, memory_efficiency, color=colors)
    axes[1, 1].set_title('Memory Efficiency (MB per Card)')
    axes[1, 1].set_ylabel('Memory / Cards Played')
    axes[1, 1].tick_params(axis='x', rotation=45)
    create_value_labels_on_bars(axes[1, 1], bars4, memory_efficiency, "{:.2f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Memory and game metrics plot saved as {save_path}")
    return fig


def create_comprehensive_comparison_plot(results: Dict[str, Any], save_path: str = "plot/comprehensive_comparison.png"):
    """Create a comprehensive comparison plot showing all key metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Comprehensive IS-MCTS Strategy Comparison',
                 fontsize=18, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)

    # 1. Performance efficiency (decisions per second vs memory)
    decisions_per_sec = [results[strategy].get(
        'avg_decisions_per_second', 0) for strategy in strategies]
    peak_memory = [results[strategy].get(
        'avg_peak_memory', 0) for strategy in strategies]

    scatter = axes[0, 0].scatter(
        peak_memory, decisions_per_sec, c=colors, s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Peak Memory Usage (MB)')
    axes[0, 0].set_ylabel('Decisions Per Second')
    axes[0, 0].set_title('Performance Efficiency')
    axes[0, 0].grid(True, alpha=0.3)

    # Add labels for each point
    for i, strategy in enumerate(strategies):
        axes[0, 0].annotate(format_strategy_name(strategy),
                            (peak_memory[i], decisions_per_sec[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 2. MCTS complexity (nodes created vs trees created)
    nodes_created = [results[strategy].get(
        'avg_nodes_created', 0) for strategy in strategies]
    trees_created = [results[strategy].get(
        'avg_trees_created', 0) for strategy in strategies]

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
        log_mcts_nodes = apply_log_scale_to_data(mcts_nodes)
        axes[0, 1].scatter(mcts_trees, log_mcts_nodes,
                           c=mcts_colors, s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Trees Created')
        axes[0, 1].set_ylabel('Nodes Created (log10 scale)')
        axes[0, 1].set_title('MCTS Complexity')
        axes[0, 1].grid(True, alpha=0.3)

        for i, strategy in enumerate(mcts_strategies):
            axes[0, 1].annotate(format_strategy_name(strategy),
                                (mcts_trees[i], log_mcts_nodes[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, 'No MCTS data available',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('MCTS Complexity')

    # 3. Time breakdown
    total_times = [results[strategy].get(
        'avg_total_time', 0) for strategy in strategies]
    decision_times = [results[strategy].get(
        'avg_decision_time', 0) for strategy in strategies]

    x_pos = np.arange(len(strategies))
    strategy_labels = [format_strategy_name(s) for s in strategies]
    axes[1, 0].bar(x_pos, total_times, color=colors,
                   alpha=0.7, label='Total Game Time')
    axes[1, 0].set_xlabel('Strategy')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Average Game Duration')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(strategy_labels, rotation=45)

    # 4. Decision complexity
    legal_actions = [results[strategy].get(
        'avg_legal_actions', 0) for strategy in strategies]
    total_decisions = [results[strategy].get(
        'avg_total_decisions', 0) for strategy in strategies]

    axes[1, 1].bar(x_pos, legal_actions, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('Average Legal Actions')
    axes[1, 1].set_title('Decision Space Complexity')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(strategy_labels, rotation=45)

    # 5. Cumulative performance metrics
    axes[2, 0].plot(strategy_labels, np.cumsum(total_times), 'o-',
                    linewidth=2, label='Cumulative Time', color='blue')
    axes[2, 0].plot(strategy_labels, np.cumsum(peak_memory), 's-',
                    linewidth=2, label='Cumulative Memory', color='red')
    axes[2, 0].set_xlabel('Strategy')
    axes[2, 0].set_ylabel('Cumulative Value')
    axes[2, 0].set_title('Cumulative Resource Usage')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Performance radar chart (relative to first strategy as baseline)
    baseline_idx = 0
    baseline_strategy = strategies[baseline_idx]

    metrics_to_compare = ['avg_decisions_per_second', 'avg_total_time', 'avg_peak_memory',
                          'avg_node_visits', 'avg_nodes_created']
    metric_labels = ['Dec/Sec', 'Time', 'Memory', 'Visits', 'Nodes']

    # Create normalized data for comparison
    normalized_data = {}
    for strategy in strategies:
        normalized_data[strategy] = []
        for metric in metrics_to_compare:
            value = results[strategy].get(metric, 0)
            baseline_value = results[baseline_strategy].get(metric, 0)
            if baseline_value > 0:
                if metric in ['avg_total_time', 'avg_peak_memory']:  # Lower is better
                    normalized_value = baseline_value / max(value, 0.001)
                else:  # Higher is better
                    normalized_value = value / max(baseline_value, 0.001)
            else:
                normalized_value = 1.0
            normalized_data[strategy].append(normalized_value)

    # Simple bar chart for relative performance
    x_metrics = np.arange(len(metric_labels))
    width = 0.15
    for i, strategy in enumerate(strategies):
        offset = (i - len(strategies)/2) * width
        axes[2, 1].bar(x_metrics + offset, normalized_data[strategy],
                       width, label=format_strategy_name(strategy), alpha=0.7, color=colors[i])

    axes[2, 1].set_xlabel('Metrics')
    axes[2, 1].set_ylabel(
        f'Performance Relative to {format_strategy_name(baseline_strategy)}')
    axes[2, 1].set_title('Relative Performance Comparison')
    axes[2, 1].set_xticks(x_metrics)
    axes[2, 1].set_xticklabels(metric_labels)
    axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison plot saved as {save_path}")
    return fig
