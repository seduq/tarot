"""
Win rate plotting module for Tarot strategy analysis.

This module contains functions for creating plots that visualize win rates,
specifically comparing performance when the agent is the taker vs. defender.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from plotting_base import setup_plotting_environment, get_strategy_colors, format_strategy_name

# Initialize plotting environment
setup_plotting_environment()


def create_winrate_comparison_plot(results: Dict[str, Any], save_path: str = "plot/winrate_comparison.png"):
    """Create a comprehensive win rate comparison plot"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Win Rate Analysis - Taker vs Defender Performance',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)
    strategy_labels = [format_strategy_name(s) for s in strategies]

    # 1. Overall Win Rate Comparison (Top Left)
    axes[0, 0].set_title('Overall Win Rate by Strategy',
                         fontweight='bold', fontsize=12)

    overall_win_rates = [results[strategy].get(
        'overall_win_rate', 0) * 100 for strategy in strategies]
    bars = axes[0, 0].bar(
        strategy_labels, overall_win_rates, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars, overall_win_rates):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    axes[0, 0].set_ylabel('Win Rate (%)')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Taker vs Defender Win Rates (Top Right)
    axes[0, 1].set_title('Win Rate: Taker vs Defender',
                         fontweight='bold', fontsize=12)

    taker_win_rates = [results[strategy].get(
        'taker_win_rate', 0) * 100 for strategy in strategies]
    defender_win_rates = [results[strategy].get(
        'defender_win_rate', 0) * 100 for strategy in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = axes[0, 1].bar(x - width/2, taker_win_rates, width, label='As Taker',
                           color=colors, alpha=0.8)
    bars2 = axes[0, 1].bar(x + width/2, defender_win_rates, width, label='As Defender',
                           color=colors, alpha=0.6, hatch='//')

    # Add value labels
    for bar, rate in zip(bars1, taker_win_rates):
        height = bar.get_height()
        if height > 0:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    for bar, rate in zip(bars2, defender_win_rates):
        height = bar.get_height()
        if height > 0:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(strategy_labels, rotation=45)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Game Distribution: Taker vs Defender (Bottom Left)
    axes[1, 0].set_title('Game Distribution: Taker vs Defender',
                         fontweight='bold', fontsize=12)

    games_as_taker = [results[strategy].get(
        'total_games_as_taker', 0) for strategy in strategies]
    games_as_defender = [results[strategy].get(
        'total_games_as_defender', 0) for strategy in strategies]

    bars1 = axes[1, 0].bar(x - width/2, games_as_taker, width, label='Games as Taker',
                           color=colors, alpha=0.8)
    bars2 = axes[1, 0].bar(x + width/2, games_as_defender, width, label='Games as Defender',
                           color=colors, alpha=0.6, hatch='//')

    # Add value labels
    for bar, count in zip(bars1, games_as_taker):
        height = bar.get_height()
        if height > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(count)}', ha='center', va='bottom', fontsize=9)

    for bar, count in zip(bars2, games_as_defender):
        height = bar.get_height()
        if height > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(count)}', ha='center', va='bottom', fontsize=9)

    axes[1, 0].set_ylabel('Number of Games')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategy_labels, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Win Rate Efficiency Score (Bottom Right)
    axes[1, 1].set_title('Win Rate Efficiency Analysis',
                         fontweight='bold', fontsize=12)

    # Calculate efficiency scores (weighted by game counts and role performance)
    efficiency_scores = []
    for strategy in strategies:
        taker_wr = results[strategy].get('taker_win_rate', 0)
        defender_wr = results[strategy].get('defender_win_rate', 0)
        games_taker = results[strategy].get('total_games_as_taker', 0)
        games_defender = results[strategy].get('total_games_as_defender', 0)
        total_games = games_taker + games_defender

        if total_games > 0:
            # Weighted efficiency considering both roles
            taker_weight = games_taker / total_games
            defender_weight = games_defender / total_games
            efficiency = (taker_wr * taker_weight +
                          defender_wr * defender_weight) * 100
        else:
            efficiency = 0

        efficiency_scores.append(efficiency)

    bars = axes[1, 1].bar(
        strategy_labels, efficiency_scores, color=colors, alpha=0.8)

    # Add value labels
    for bar, score in zip(bars, efficiency_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')

    axes[1, 1].set_ylabel('Efficiency Score (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Win rate comparison plot saved as {save_path}")
    return fig


def create_winrate_heatmap(results: Dict[str, Any], save_path: str = "plot/winrate_heatmap.png"):
    """Create a heatmap showing win rates for different scenarios"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Win Rate Heatmap: Strategy Performance Matrix',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    strategy_labels = [format_strategy_name(s) for s in strategies]

    # Create data matrix: rows = strategies, columns = [Taker, Defender, Overall]
    scenarios = ['As Taker', 'As Defender', 'Overall']
    win_rate_matrix = []

    for strategy in strategies:
        taker_wr = results[strategy].get('taker_win_rate', 0) * 100
        defender_wr = results[strategy].get('defender_win_rate', 0) * 100
        overall_wr = results[strategy].get('overall_win_rate', 0) * 100
        win_rate_matrix.append([taker_wr, defender_wr, overall_wr])

    win_rate_matrix = np.array(win_rate_matrix)

    # Create heatmap
    im = ax.imshow(win_rate_matrix, cmap='RdYlGn',
                   vmin=0, vmax=100, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(scenarios)
    ax.set_yticklabels(strategy_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Win Rate (%)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(scenarios)):
            value = win_rate_matrix[i, j]
            color = 'white' if value < 50 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                    color=color, fontweight='bold')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Strategy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Win rate heatmap saved as {save_path}")
    return fig


def create_winrate_radar_plot(results: Dict[str, Any], save_path: str = "plot/winrate_radar.png"):
    """Create a radar plot comparing win rate performance across dimensions"""

    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('Win Rate Performance Radar Chart',
                 fontsize=16, fontweight='bold', y=0.95)

    # Define categories for radar plot
    categories = ['Taker Win Rate', 'Defender Win Rate', 'Overall Win Rate',
                  'Game Balance', 'Consistency']

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)

    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, len(categories),
                         endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for i, strategy in enumerate(strategies):
        # Calculate metrics (normalized to 0-100 scale)
        taker_wr = results[strategy].get('taker_win_rate', 0) * 100
        defender_wr = results[strategy].get('defender_win_rate', 0) * 100
        overall_wr = results[strategy].get('overall_win_rate', 0) * 100

        # Game balance: how evenly distributed are games between taker/defender
        games_taker = results[strategy].get('total_games_as_taker', 0)
        games_defender = results[strategy].get('total_games_as_defender', 0)
        total_games = games_taker + games_defender

        if total_games > 0:
            balance_ratio = min(games_taker, games_defender) / max(games_taker,
                                                                   games_defender) if max(games_taker, games_defender) > 0 else 0
            game_balance = balance_ratio * 100
        else:
            game_balance = 0

        # Consistency: how close are taker and defender win rates
        if games_taker > 0 and games_defender > 0:
            consistency = 100 - abs(taker_wr - defender_wr)
        else:
            consistency = 0

        values = [taker_wr, defender_wr, overall_wr, game_balance, consistency]
        values += values[:1]  # Complete the circle

        # Plot the radar chart
        ax.plot(angles, values, 'o-', linewidth=2, label=format_strategy_name(strategy),
                color=colors[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Win rate radar plot saved as {save_path}")
    return fig
