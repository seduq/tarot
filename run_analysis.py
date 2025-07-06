"""
Simple runner script for Tarot strategy comparison and visualization
"""

import argparse
import sys
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt


def run_simulation_only(iterations=200, games=5, determinizations=10, verbose=False):
    """Run only the simulation without plotting"""
    from simulation import compare_strategies, StrategyType

    strategies = [
        StrategyType.RANDOM,
        StrategyType.MAX_CARD,
        StrategyType.MIN_CARD,
        StrategyType.IS_MCTS_PER_ACTION,
        StrategyType.IS_MCTS_PER_TRICK
    ]

    print(
        f"Running simulation with {iterations} iterations and {games} games per strategy...")
    results = compare_strategies(
        strategies=strategies,
        iterations=iterations,
        games_per_strategy=games,
        determinizations=determinizations,
        verbose=verbose
    )

    return results


def run_plotting_only():
    """Run only the plotting from existing results"""
    try:
        from plot_from_results import main as plot_main
        plot_main()
    except ImportError as e:
        print(f"Error importing plotting module: {e}")
    except Exception as e:
        print(f"Error creating plots: {e}")


def run_full_pipeline(iterations=200, games=5, verbose=False):
    """Run simulation and create plots"""
    try:
        from plot_metrics import main as full_main

        # Modify the main function to use our parameters
        from plot_metrics import run_simulation_and_collect_data, save_results_to_json
        from plot_metrics import (create_time_metrics_plot, create_decision_metrics_plot,
                                  create_mcts_metrics_plot, create_memory_and_game_metrics_plot,
                                  create_comprehensive_comparison_plot)

        print("Running full pipeline...")

        # Run simulation
        results = run_simulation_and_collect_data(
            iterations=iterations, games_per_strategy=games)

        # Save results
        save_results_to_json(results)

        # Create plots
        print("\nCreating visualization plots...")
        create_time_metrics_plot(results)
        create_decision_metrics_plot(results)
        create_mcts_metrics_plot(results)
        create_memory_and_game_metrics_plot(results)
        create_comprehensive_comparison_plot(results)

        # Also create dashboard plots
        from plot_from_results import create_summary_dashboard, create_comparison_analysis
        create_summary_dashboard(results)
        create_comparison_analysis(results)

        print("\nAll plots created successfully!")
        print("Generated files:")
        print("- time_metrics.png")
        print("- decision_metrics.png")
        print("- mcts_metrics.png")
        print("- memory_game_metrics.png")
        print("- comprehensive_comparison.png")
        print("- strategy_dashboard.png")
        print("- strategy_analysis.png")
        print("- simulation_results.json")

    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure matplotlib, seaborn, and numpy are installed.")
    except Exception as e:
        print(f"Error in pipeline: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Tarot Strategy Analysis Tool')

    parser.add_argument('--mode', choices=['sim', 'plot', 'full'], default='full',
                        help='Mode: sim=simulation only, plot=plotting only, full=both (default: full)')
    parser.add_argument('--iterations', type=int, default=200,
                        help='MCTS iterations per decision (default: 200)')
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games per strategy (default: 5)')
    parser.add_argument('--determinizations', type=int, default=10,
                        help='Number of determinizations for multiple (default: 10)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output during simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with reduced parameters (100 iterations, 3 games)')

    args = parser.parse_args()

    # Apply quick mode settings
    if args.quick:
        args.iterations = 100
        args.games = 3
        print("Quick mode enabled: 100 iterations, 3 games per strategy")

    print("Tarot Strategy Analysis Tool")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"MCTS iterations: {args.iterations}")
    print(f"Games per strategy: {args.games}")
    print(f"Verbose: {args.verbose}")
    print("=" * 50)

    # Use a backend that doesn't require display (for server environments)
    matplotlib.use('Agg')
    os.makedirs("plot", exist_ok=True)  # Ensure plot directory exists

    try:
        if args.mode == 'sim':
            results = run_simulation_only(
                args.iterations, args.games, args.determinizations, args.verbose)
            print("\nSimulation completed. Results saved to simulation_results.json")

        elif args.mode == 'plot':
            run_plotting_only()

        elif args.mode == 'full':
            run_full_pipeline(args.iterations, args.games, args.verbose)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
