"""
MCTS Configuration Comparison Plotting

This module provides functions to compare different RIS-MCTS configurations
by running simulations with various hyperparameters and visualizing results.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from simulation import ISMCTSConfig, StrategyType, play_with_strategy
from metrics import MetricsCollector, GameMetrics
import random


def run_mcts_comparison(configs: List[Dict], games_per_config: int = 200, seed: int = 42):
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
                num_mcts_agents=1,
                game_id=game_id
            )
            collector.add_game(metrics)
            game_id += 1

        print(f"  Completed {games_per_config} games")

    return collector


def plot_mcts_parameter_sweep(games_per_config: int = 200, seed: int = 42,
                              output_dir: str = 'results', custom_configs: Optional[List[Dict]] = None,
                              show_plots: bool = False):
    """
    Create a parameter sweep comparison for common MCTS hyperparameters.

    Args:
        games_per_config: Number of games to run per configuration
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        custom_configs: Optional list of custom configurations to test
        show_plots: Whether to display plots interactively
    """
    # Use custom configs if provided, otherwise use default parameter sweep
    if custom_configs:
        configs = custom_configs
    else:
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
    collector = run_mcts_comparison(
        configs, games_per_config=games_per_config, seed=seed)

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    config_names = []
    taker_rates = []
    avg_decision_times = []

    iterations_set = set()
    exploration_set = set()
    pw_alpha_set = set()

    for config in configs:
        # Create readable config name
        name = f"i={config['iterations']}, e={config['exploration_constant']}, a={config['pw_alpha']}"
        config_names.append(name)
        iterations_set.add(config['iterations'])
        exploration_set.add(config['exploration_constant'])
        pw_alpha_set.add(config['pw_alpha'])

        # Filter games for this config
        config_games = [
            game for game in collector.games
            if game.mcts_config == config
        ]

        # Calculate win rate
        if config_games:
            taker_wins = sum(1 for game in config_games if game.taker_won)
            taker_games = sum(1 for game in config_games if game.taker_game)
            taker_rate = (taker_wins / max(1, taker_games))
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
    ax1.set_ylim(0, 1.2)
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
    start = 0
    end = len(iterations_set)
    iterations_idx = range(start, end)
    start = end
    end += len(exploration_set)
    exploration_idx = range(start, end)
    start = end
    end += len(pw_alpha_set)
    pw_alpha_idx = range(start, end)

    # Iterations comparison
    ax3.bar(range(len(iterations_idx)), [taker_rates[i] for i in iterations_idx],
            color=['lightblue', 'blue', 'darkblue'])
    ax3.set_title('Win Rate vs Iterations')
    ax1.set_ylim(0, 1.2)
    ax3.set_xticks(range(len(iterations_set)))
    ax3.set_xticklabels(iterations_set)
    ax3.set_ylabel('Taker Win Rate')

    # Exploration constant comparison
    ax4.bar(range(len(exploration_idx)), [taker_rates[i] for i in exploration_idx],
            color=['lightgreen', 'green', 'darkgreen'])
    ax4.set_title('Win Rate vs Exploration Constant')
    ax1.set_ylim(0, 1.2)
    ax4.set_xticks(range(len(exploration_set)))
    ax4.set_xticklabels(exploration_set)
    ax4.set_ylabel('Taker Win Rate')

    plt.tight_layout()

    # Ensure results directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/mcts_parameter_comparison.png',
                dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close()

    # Generate and save readable text comparison
    text_report = generate_text_comparison(collector, configs)
    with open(f'{output_dir}/mcts_comparison_report.txt', 'w') as f:
        f.write(text_report)
    print(
        f"Text comparison report saved to {output_dir}/mcts_comparison_report.txt")

    # Save detailed results
    collector.save_to_file(output_dir, 'mcts_comparison_metrics.json')
    print(
        f"MCTS comparison results saved to {output_dir}/mcts_comparison_metrics.json")

    # Print summary to console
    print("\nQuick Summary:")
    print("-" * 50)
    for i, config in enumerate(configs):
        config_games = [
            game for game in collector.games if game.mcts_config == config]
        if config_games:
            taker_wins = sum(1 for game in config_games if game.taker_won)
            taker_rate = (taker_wins /
                          max(1, sum(1 for game in config_games if game.taker_game)))

            all_nodes = []
            for game in config_games:
                all_nodes.extend([n for n in game.nodes_created if n > 0])
            avg_nodes = np.mean(all_nodes) if all_nodes else 0

            print(f"Config {i+1} (iter={config['iterations']}, exp={config['exploration_constant']}): "
                  f"Win Rate={taker_rate:.3f}, Avg Nodes={avg_nodes:.1f}")
    print("-" * 50)


def generate_text_comparison(collector: MetricsCollector, configs: List[Dict]) -> str:
    """
    Generate a readable text comparison of MCTS configurations.

    Args:
        collector: MetricsCollector with results from all configurations
        configs: List of MCTS configuration dictionaries

    Returns:
        Formatted text comparison string
    """
    report = []
    report.append("="*80)
    report.append("MCTS PARAMETER COMPARISON REPORT")
    report.append("="*80)
    report.append("")

    for i, config in enumerate(configs):
        report.append(f"Configuration {i+1}:")
        report.append(f"  Iterations: {config['iterations']}")
        report.append(
            f"  Exploration Constant: {config['exploration_constant']}")
        report.append(f"  Progressive Widening Alpha: {config['pw_alpha']}")
        report.append(
            f"  Progressive Widening Constant: {config['pw_constant']}")
        report.append("")

        # Filter games for this config
        config_games = [
            game for game in collector.games
            if game.mcts_config == config
        ]

        if config_games:
            # Calculate win rate
            taker_wins = sum(1 for game in config_games if game.taker_won)
            taker_games = sum(1 for game in config_games if game.taker_game)
            taker_rate = (taker_wins / max(1, taker_games))

            # Calculate average decision time
            all_times = []
            for game in config_games:
                all_times.extend([t for t in game.decision_times if t > 0])
            avg_time = np.mean(all_times) if all_times else 0

            # Calculate average nodes created
            all_nodes = []
            for game in config_games:
                all_nodes.extend([n for n in game.nodes_created if n > 0])
            avg_nodes = np.mean(all_nodes) if all_nodes else 0

            # Calculate average illegal moves
            all_illegal = []
            for game in config_games:
                all_illegal.extend(game.missed_moves)
            avg_illegal = np.mean(all_illegal) if all_illegal else 0

            report.append(f"  Results ({len(config_games)} games):")
            report.append(
                f"    Taker Win Rate: {taker_rate:.3f} ({taker_wins}/{taker_games})")
            report.append(f"    Avg Decision Time: {avg_time:.4f} seconds")
            report.append(
                f"    Avg Nodes Created per Decision: {avg_nodes:.1f}")
            report.append(
                f"    Avg Illegal Move Attempts per Decision: {avg_illegal:.1f}")
        else:
            report.append("  No games found for this configuration")

        report.append("")
        report.append("-"*60)
        report.append("")

    # Add summary section
    report.append("SUMMARY ANALYSIS")
    report.append("="*40)
    report.append("")

    # Find best performing configs by different metrics
    config_results = []
    for i, config in enumerate(configs):
        config_games = [
            game for game in collector.games if game.mcts_config == config]
        if config_games:
            taker_wins = sum(1 for game in config_games if game.taker_won)
            taker_games = sum(1 for game in config_games if game.taker_game)
            taker_rate = (taker_wins / max(1, taker_games))

            all_times = []
            for game in config_games:
                all_times.extend([t for t in game.decision_times if t > 0])
            avg_time = np.mean(all_times) if all_times else 0

            all_nodes = []
            for game in config_games:
                all_nodes.extend([n for n in game.nodes_created if n > 0])
            avg_nodes = np.mean(all_nodes) if all_nodes else 0

            config_results.append(
                (i+1, taker_rate, avg_time, avg_nodes, config))

    if config_results:
        # Best win rate
        best_winrate = max(config_results, key=lambda x: x[1])
        report.append(
            f"Best Win Rate: Configuration {best_winrate[0]} with {best_winrate[1]:.3f}")
        report.append(
            f"  Settings: iterations={best_winrate[4]['iterations']}, exploration={best_winrate[4]['exploration_constant']}")
        report.append("")

        # Fastest decisions
        fastest = min(config_results, key=lambda x: x[2])
        report.append(
            f"Fastest Decisions: Configuration {fastest[0]} with {fastest[2]:.4f}s avg")
        report.append(
            f"  Settings: iterations={fastest[4]['iterations']}, exploration={fastest[4]['exploration_constant']}")
        report.append("")

        # Most efficient (fewest nodes)
        most_efficient = min(config_results, key=lambda x: x[3])
        report.append(
            f"Most Efficient (fewest nodes): Configuration {most_efficient[0]} with {most_efficient[3]:.1f} avg nodes")
        report.append(
            f"  Settings: iterations={most_efficient[4]['iterations']}, exploration={most_efficient[4]['exploration_constant']}")

    report.append("")
    report.append("="*80)

    return "\n".join(report)


def create_iteration_sweep(base_config: Dict, iterations_list: List[int]) -> List[Dict]:
    """Create a configuration sweep varying iterations."""
    configs = []
    for iterations in iterations_list:
        config = base_config.copy()
        config['iterations'] = iterations
        configs.append(config)
    return configs


def create_exploration_sweep(base_config: Dict, exploration_constants: List[float]) -> List[Dict]:
    """Create a configuration sweep varying exploration constant."""
    configs = []
    for exp_const in exploration_constants:
        config = base_config.copy()
        config['exploration_constant'] = exp_const
        configs.append(config)
    return configs


def create_pw_alpha_sweep(base_config: Dict, pw_alphas: List[float]) -> List[Dict]:
    """Create a configuration sweep varying progressive widening alpha."""
    configs = []
    for pw_alpha in pw_alphas:
        config = base_config.copy()
        config['pw_alpha'] = pw_alpha
        configs.append(config)
    return configs


def create_full_grid_search(iterations: List[int], exploration_constants: List[float],
                            pw_alphas: List[float], pw_constants: List[float]) -> List[Dict]:
    """Create a full grid search of all parameter combinations."""
    configs = []
    for iter_val in iterations:
        for exp_const in exploration_constants:
            for pw_alpha in pw_alphas:
                for pw_constant in pw_constants:
                    configs.append({
                        'iterations': iter_val,
                        'exploration_constant': exp_const,
                        'pw_alpha': pw_alpha,
                        'pw_constant': pw_constant
                    })
    return configs


def parse_arguments():
    """Parse command line arguments for MCTS comparison."""
    parser = argparse.ArgumentParser(
        description='Compare different MCTS configurations for Tarot gameplay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default parameter sweep (save only)
  python mcts_comparison.py

  # Run with custom number of games and interactive display
  python mcts_comparison.py --games 100 --seed 123 --show

  # Run iteration sweep only
  python mcts_comparison.py --sweep-type iterations --iterations 50 100 200 500

  # Run exploration constant sweep with interactive display
  python mcts_comparison.py --sweep-type exploration --exploration 0.5 1.0 1.4 2.0 --show

  # Run progressive widening alpha sweep
  python mcts_comparison.py --sweep-type pw_alpha --pw-alpha 0.2 0.5 0.8

  # Run full grid search with interactive display
  python mcts_comparison.py --sweep-type grid --iterations 100 200 --exploration 1.0 1.4 --pw-alpha 0.5 0.7 --show

  # Save to custom directory (batch mode - save only)
  python mcts_comparison.py --output-dir my_results
        """)

    parser.add_argument('--games', type=int, default=50,
                        help='Number of games to run per configuration (default: 50)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')

    parser.add_argument('--sweep-type', choices=['default', 'iterations', 'exploration', 'pw_alpha', 'grid'],
                        default='default',
                        help='Type of parameter sweep to perform (default: default)')

    # Base configuration parameters
    parser.add_argument('--base-iterations', type=int, default=100,
                        help='Base iterations for parameter sweeps (default: 100)')

    parser.add_argument('--base-exploration', type=float, default=1.4,
                        help='Base exploration constant for parameter sweeps (default: 1.4)')

    parser.add_argument('--base-pw-alpha', type=float, default=0.5,
                        help='Base progressive widening alpha for parameter sweeps (default: 0.5)')

    parser.add_argument('--base-pw-constant', type=float, default=2.0,
                        help='Base progressive widening constant for parameter sweeps (default: 2.0)')

    # Sweep parameters
    parser.add_argument('--iterations', type=int, nargs='+', default=[50, 100, 200],
                        help='Iterations values for sweep (default: [50, 100, 200])')

    parser.add_argument('--exploration', type=float, nargs='+', default=[1.0, 1.4, 2.0],
                        help='Exploration constant values for sweep (default: [1.0, 1.4, 2.0])')

    parser.add_argument('--pw-alpha', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                        help='Progressive widening alpha values for sweep (default: [0.3, 0.5, 0.7])')

    parser.add_argument('--pw-constant', type=float, nargs='+', default=[2.0],
                        help='Progressive widening constant values for sweep (default: [2.0])')

    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: save only)')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def create_configs_from_args(args) -> Optional[List[Dict]]:
    """Create MCTS configurations based on command line arguments."""
    base_config = {
        'iterations': args.base_iterations,
        'exploration_constant': args.base_exploration,
        'pw_alpha': args.base_pw_alpha,
        'pw_constant': args.base_pw_constant
    }

    if args.sweep_type == 'iterations':
        return create_iteration_sweep(base_config, args.iterations)
    elif args.sweep_type == 'exploration':
        return create_exploration_sweep(base_config, args.exploration)
    elif args.sweep_type == 'pw_alpha':
        return create_pw_alpha_sweep(base_config, args.pw_alpha)
    elif args.sweep_type == 'grid':
        return create_full_grid_search(args.iterations, args.exploration,
                                       args.pw_alpha, args.pw_constant)
    else:  # default
        return None  # Will use the default configs in plot_mcts_parameter_sweep


def print_configuration_summary(configs: List[Dict]):
    """Print a summary of the configurations to be tested."""
    print(f"\nConfiguration Summary ({len(configs)} configs):")
    print("-" * 60)
    for i, config in enumerate(configs):
        print(f"Config {i+1}: iter={config['iterations']}, "
              f"exp={config['exploration_constant']}, "
              f"alpha={config['pw_alpha']}, "
              f"constant={config['pw_constant']}")
    print("-" * 60)


def validate_config(config: Dict) -> bool:
    """Validate that a configuration has all required parameters."""
    required_params = ['iterations',
                       'exploration_constant', 'pw_alpha', 'pw_constant']
    for param in required_params:
        if param not in config:
            print(
                f"Warning: Missing required parameter '{param}' in config: {config}")
            return False
        if not isinstance(config[param], (int, float)) or config[param] <= 0:
            print(f"Warning: Invalid value for '{param}' in config: {config}")
            return False
    return True


def validate_configs(configs: List[Dict]) -> List[Dict]:
    """Validate and filter out invalid configurations."""
    valid_configs = [config for config in configs if validate_config(config)]
    if len(valid_configs) != len(configs):
        print(
            f"Warning: {len(configs) - len(valid_configs)} invalid configurations were removed")
    return valid_configs


def main():
    """Main function to run MCTS comparison with command line arguments."""
    args = parse_arguments()

    if args.verbose:
        print(f"Running MCTS comparison with {args.games} games per config")
        print(f"Random seed: {args.seed}")
        print(f"Output directory: {args.output_dir}")
        print(f"Sweep type: {args.sweep_type}")

    # Create configurations based on arguments
    custom_configs = create_configs_from_args(args)

    if custom_configs:
        custom_configs = validate_configs(custom_configs)
        if not custom_configs:
            print("Error: No valid configurations generated. Exiting.")
            return

    if args.sweep_type != 'default' and custom_configs:
        print(
            f"Generated {len(custom_configs)} configurations for {args.sweep_type} sweep")
        if args.verbose:
            print_configuration_summary(custom_configs)

    print("Running MCTS parameter sweep comparison...")

    # Always run comparison and save data, show plots only if requested
    if custom_configs:
        configs = custom_configs
    else:
        # Use default configs
        configs = [
            {"iterations": 50, "exploration_constant": 1.4,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 1.4,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 200, "exploration_constant": 1.4,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 1.0,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 1.4,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 2.0,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 1.4,
                "pw_alpha": 0.3, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 1.4,
                "pw_alpha": 0.5, "pw_constant": 2.0},
            {"iterations": 100, "exploration_constant": 1.4,
                "pw_alpha": 0.7, "pw_constant": 2.0},
        ]

    # Run with plotting (generates plots and saves data)
    plot_mcts_parameter_sweep(
        games_per_config=args.games,
        seed=args.seed,
        output_dir=args.output_dir,
        custom_configs=configs,
        show_plots=args.show
    )

    print("Done!")


if __name__ == "__main__":
    main()
