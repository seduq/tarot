"""
MCTS Configuration Comparison Plotting

This module provides functions to compare different RIS-MCTS configurations
by running simulations with various hyperparameters and visualizing results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from simulation import ISMCTSConfig, StrategyType, play_with_strategy
from metrics import MetricsCollector, GameMetrics
import random


def run_mcts_comparison(configs: List[Dict], games_per_config: int = 100, seed: int = 42):
    """
    Run simulations with different MCTS configurations and collect metrics.

    Args:
        configs: List of MCTS configuration dictionaries
        games_per_config: Number of games to run per configuration
        seed: Random seed

    Returns:
        MetricsCollector with results
    """
    random.seed(seed)
    collector = MetricsCollector()
    game_id = 0

    for i, config_dict in enumerate(configs):
        config = ISMCTSConfig(**config_dict)

        print(f"Testing MCTS config {i+1}/{len(configs)}: {config_dict}")

        for game_num in range(games_per_config):
            print(f"  Game {game_num + 1}/{games_per_config}", end='\r')

            metrics = play_with_strategy(
                StrategyType.RIS_MCTS,
                config,
                max_mcts_agents=1,
                game_id=game_id
            )
            collector.add_game(metrics)
            game_id += 1

        print(f"  Completed {games_per_config} games")

    return collector


def plot_mcts_parameter_sweep():
    """
    Create a parameter sweep comparison for common MCTS hyperparameters.
    """
    # Define configurations to test
    configs = [
        # Varying iterations
        {"iterations": 100, "exploration_constant": 1.4,
            "pw_alpha": 0.5, "pw_constant": 2.0},
        {"iterations": 200, "exploration_constant": 1.4,
            "pw_alpha": 0.5, "pw_constant": 2.0},
        {"iterations": 300, "exploration_constant": 1.4,
            "pw_alpha": 0.5, "pw_constant": 2.0},

        # Varying exploration constant
        {"iterations": 100, "exploration_constant": 1.0,
            "pw_alpha": 0.5, "pw_constant": 2.0},
        {"iterations": 100, "exploration_constant": 1.4,
            "pw_alpha": 0.5, "pw_constant": 2.0},
        {"iterations": 100, "exploration_constant": 2.0,
            "pw_alpha": 0.5, "pw_constant": 2.0},

        # Varying pw_alpha
        {"iterations": 100, "exploration_constant": 1.4,
            "pw_alpha": 0.3, "pw_constant": 2.0},
        {"iterations": 100, "exploration_constant": 1.4,
            "pw_alpha": 0.5, "pw_constant": 2.0},
        {"iterations": 100, "exploration_constant": 1.4,
            "pw_alpha": 0.7, "pw_constant": 2.0},
    ]

    # Run comparison
    collector = run_mcts_comparison(configs, games_per_config=30)

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    config_names = []
    taker_rates = []
    avg_decision_times = []

    for config in configs:
        # Create readable config name
        name = f"i={config['iterations']}, e={config['exploration_constant']}, a={config['pw_alpha']}"
        config_names.append(name)

        # Filter games for this config
        config_games = [
            game for game in collector.games
            if game.mcts_config == config
        ]

        # Calculate win rate
        if config_games:
            taker_wins = sum(1 for game in config_games if game.taker_won)
            taker_rate = taker_wins / len(config_games)
            taker_rates.append(taker_rate)

            # Calculate average decision time
            all_times = []
            for game in config_games:
                all_times.extend(game.decision_times)
            avg_time = np.mean(all_times) if all_times else 0
            avg_decision_times.append(avg_time)
        else:
            taker_rates.append(0)
            avg_decision_times.append(0)

    # Plot taker win rates
    bars = ax1.bar(range(len(configs)), taker_rates, color='skyblue')
    ax1.set_title('Taker Win Rate by MCTS Configuration')
    ax1.set_ylabel('Win Rate')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')

    # Add value labels
    for bar, rate in zip(bars, taker_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{rate:.3f}', ha='center', va='bottom')

    # Plot average decision times
    bars2 = ax2.bar(range(len(configs)),
                    avg_decision_times, color='lightcoral')
    ax2.set_title('Average Decision Time by MCTS Configuration')
    ax2.set_ylabel('Decision Time (seconds)')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')

    # Group by parameter type for easier comparison
    iterations_idx = [0, 1, 2]
    exploration_idx = [3, 4, 5]
    pw_alpha_idx = [6, 7, 8]

    # Iterations comparison
    ax3.bar(range(3), [taker_rates[i] for i in iterations_idx],
            color=['lightblue', 'blue', 'darkblue'])
    ax3.set_title('Win Rate vs Iterations')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['50', '100', '200'])
    ax3.set_ylabel('Taker Win Rate')

    # Exploration constant comparison
    ax4.bar(range(3), [taker_rates[i] for i in exploration_idx],
            color=['lightgreen', 'green', 'darkgreen'])
    ax4.set_title('Win Rate vs Exploration Constant')
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['1.0', '1.4', '2.0'])
    ax4.set_ylabel('Taker Win Rate')

    plt.tight_layout()
    plt.savefig('results/mcts_parameter_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    collector.save_to_file('results/mcts_comparison_metrics.json')
    print("MCTS comparison results saved to results/mcts_comparison_metrics.json")


if __name__ == "__main__":
    print("Running MCTS parameter sweep comparison...")
    plot_mcts_parameter_sweep()
    print("Done!")
