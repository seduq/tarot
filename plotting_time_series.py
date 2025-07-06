"""
Time-series and cumulative metrics plotting module for Tarot strategy analysis.

This module contains functions for creating plots that show metrics evolving over time,
including cumulative metrics, progression analysis, and growth rate comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from plotting_base import setup_plotting_environment, get_strategy_colors, format_strategy_name

# Initialize plotting environment
setup_plotting_environment()


def create_cumulative_metrics_plot(results: Dict[str, Any], save_path: str = "plot/cumulative_metrics_over_time.png"):
    """Create a plot showing cumulative metrics over time for all IS-MCTS strategies"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cumulative Metrics Over Time - IS-MCTS Strategies',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)

    # 1. Cumulative Decisions Over Time (Top Left)
    axes[0, 0].set_title('Cumulative Decisions Over Time',
                         fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Cumulative Number of Decisions')
    axes[0, 0].set_xlabel('Decision Step')

    for i, strategy in enumerate(strategies):
        cumulative_decisions = results[strategy].get(
            'cumulative_decisions', [])
        if cumulative_decisions:
            decision_steps = range(1, len(cumulative_decisions) + 1)
            axes[0, 0].plot(decision_steps, cumulative_decisions, 'o-',
                            linewidth=2, markersize=3, label=format_strategy_name(strategy),
                            color=colors[i], alpha=0.8)

    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Cumulative Legal Actions Over Time (Top Right)
    axes[0, 1].set_title('Cumulative Legal Actions Over Time',
                         fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('Cumulative Legal Actions Encountered')
    axes[0, 1].set_xlabel('Decision Step')

    for i, strategy in enumerate(strategies):
        cumulative_legal_actions = results[strategy].get(
            'cumulative_legal_actions', [])
        if cumulative_legal_actions:
            decision_steps = range(1, len(cumulative_legal_actions) + 1)
            axes[0, 1].plot(decision_steps, cumulative_legal_actions, 's-',
                            linewidth=2, markersize=3, label=format_strategy_name(strategy),
                            color=colors[i], alpha=0.8)

    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Cumulative Nodes Created Over Time (Bottom Left)
    axes[1, 0].set_title(
        'Cumulative MCTS Nodes Created Over Time', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Nodes Created (log scale)')
    axes[1, 0].set_xlabel('Decision Step')

    for i, strategy in enumerate(strategies):
        cumulative_nodes = results[strategy].get(
            'cumulative_nodes_created', [])
        # Only plot if there are actually nodes created
        if cumulative_nodes and max(cumulative_nodes) > 0:
            decision_steps = range(1, len(cumulative_nodes) + 1)
            # Use log scale for better visualization
            log_cumulative_nodes = [np.log10(max(1, x))
                                    for x in cumulative_nodes]
            axes[1, 0].plot(decision_steps, log_cumulative_nodes, '^-',
                            linewidth=2, markersize=3, label=format_strategy_name(strategy),
                            color=colors[i], alpha=0.8)

    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Growth Rate Analysis (Bottom Right)
    axes[1, 1].set_title('Growth Rate Comparison',
                         fontweight='bold', fontsize=12)
    axes[1, 1].set_ylabel('Rate of Change')
    axes[1, 1].set_xlabel('Decision Step')

    for i, strategy in enumerate(strategies):
        cumulative_decisions = results[strategy].get(
            'cumulative_decisions', [])
        cumulative_nodes = results[strategy].get(
            'cumulative_nodes_created', [])

        if len(cumulative_decisions) > 1:
            # Calculate decision rate (decisions per step - should be mostly 1)
            decision_rate = np.diff(cumulative_decisions)
            decision_steps = range(2, len(cumulative_decisions) + 1)

            # Calculate node creation rate
            if cumulative_nodes and max(cumulative_nodes) > 0:
                node_rate = np.diff(cumulative_nodes)
                # Normalize to make comparable with decision rate
                max_node_rate = max(node_rate) if max(node_rate) > 0 else 1
                normalized_node_rate = node_rate / max_node_rate

                axes[1, 1].plot(decision_steps, normalized_node_rate, 'o',
                                linewidth=2, markersize=2,
                                label=f'{format_strategy_name(strategy)} - Nodes (normalized)',
                                color=colors[i], alpha=0.6, linestyle='--')

    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cumulative metrics over time plot saved as {save_path}")
    return fig


def create_time_metrics_plot(results: Dict[str, Any], save_path: str = "plot/time_metrics.png"):
    """Create plots for time-related metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IS-MCTS Time Performance Metrics Comparison',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)
    strategy_labels = [format_strategy_name(s) for s in strategies]

    # Total game time
    total_times = [results[strategy].get(
        'avg_total_time', 0) for strategy in strategies]
    axes[0, 0].bar(strategy_labels, total_times, color=colors)
    axes[0, 0].set_title('Average Total Game Time')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Average decision time
    decision_times = [results[strategy].get(
        'avg_decision_time', 0) for strategy in strategies]
    axes[0, 1].bar(strategy_labels, decision_times, color=colors)
    axes[0, 1].set_title('Average Decision Time')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Decisions per second
    decisions_per_sec = [results[strategy].get(
        'avg_decisions_per_second', 0) for strategy in strategies]
    axes[1, 0].bar(strategy_labels, decisions_per_sec, color=colors)
    axes[1, 0].set_title('Average Decisions Per Second')
    axes[1, 0].set_ylabel('Decisions/sec')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Time efficiency comparison
    if total_times and decision_times:
        efficiency_ratios = [d/t if t > 0 else 0 for d,
                             t in zip(decision_times, total_times)]
        axes[1, 1].bar(strategy_labels, efficiency_ratios, color=colors)
        axes[1, 1].set_title('Decision Time Efficiency Ratio')
        axes[1, 1].set_ylabel('Decision Time / Total Time')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time metrics plot saved as {save_path}")
    return fig
