"""
Focused IS-MCTS Strategy Analysis for French Tarot
"""

from simulation import *
import matplotlib.pyplot as plt
import numpy as np


def detailed_mcts_comparison():
    """Detailed comparison of IS-MCTS strategies with multiple games"""
    print("="*80)
    print("DETAILED IS-MCTS STRATEGY ANALYSIS")
    print("="*80)

    mcts_strategies = [
        StrategyType.IS_MCTS_SINGLE,
        StrategyType.IS_MCTS_PER_ACTION,
        StrategyType.IS_MCTS_PER_TRICK
    ]

    # Test with different iteration counts
    iteration_counts = [50, 100, 200, 500]

    results = {}

    for iterations in iteration_counts:
        print(f"\n{'-'*60}")
        print(f"Testing with {iterations} MCTS iterations per decision")
        print(f"{'-'*60}")

        iteration_results = compare_strategies(
            strategies=mcts_strategies,
            iterations=iterations,
            games_per_strategy=2,  # Multiple games for more reliable results
            verbose=False
        )

        results[iterations] = iteration_results

    # Analysis of results
    print(f"\n{'='*80}")
    print("ITERATION SCALING ANALYSIS")
    print(f"{'='*80}")

    print(f"{'Iterations':<12} {'Strategy':<20} {'Time/Decision':<15} {'Decisions/s':<12} {'Nodes Created':<12} {'Memory(MB)':<12}")
    print("-" * 90)

    for iterations in iteration_counts:
        for strategy_name, metrics in results[iterations].items():
            print(f"{iterations:<12} "
                  f"{strategy_name:<20} "
                  f"{metrics['avg_decision_time']:<15.4f} "
                  f"{metrics['avg_decisions_per_second']:<12.1f} "
                  f"{metrics['avg_nodes_created']:<12.0f} "
                  f"{metrics['avg_peak_memory']:<12.1f}")

    return results


def analyze_strategy_scaling():
    """Analyze how strategies scale with problem complexity"""
    print(f"\n{'='*80}")
    print("STRATEGY SCALING ANALYSIS")
    print(f"{'='*80}")

    # Test different MCTS configurations
    configs = [
        {"name": "Fast", "iterations": 50, "determinizations": 3},
        {"name": "Balanced", "iterations": 200, "determinizations": 6},
        {"name": "Deep", "iterations": 500, "determinizations": 10}
    ]

    strategies = [StrategyType.IS_MCTS_SINGLE, StrategyType.IS_MCTS_PER_TRICK]

    for config in configs:
        print(f"\n{'-'*60}")
        print(
            f"Configuration: {config['name']} ({config['iterations']} iterations)")
        print(f"{'-'*60}")

        for strategy in strategies:
            print(f"\nTesting {strategy.value}...")

            # Create custom agent with specific config
            if strategy == StrategyType.IS_MCTS_SINGLE:
                agent = TarotISMCTSAgent(
                    player_id=0,
                    iterations=config['iterations'],
                    multiple_determinization=False
                )
            else:
                agent = TarotISMCTSAgent(
                    player_id=0,
                    iterations=config['iterations'],
                    multiple_determinization=True,
                    num_determinizations=config['determinizations'],
                    strategy=ISMCTSStrategy.PER_TRICK
                )

            # Quick test
            start_time = time.time()
            metrics = play_with_strategy(
                strategy, verbose=False, iterations=config['iterations'])
            test_time = time.time() - start_time

            print(f"  Total time: {test_time:.3f}s")
            print(f"  Decisions/sec: {metrics.decisions_per_second:.1f}")
            print(f"  Memory: {metrics.peak_memory_usage:.1f} MB")
            print(f"  Nodes created: {metrics.total_nodes_created:,}")
            print(
                f"  Simulations: {metrics.simulation_counts[-1] if metrics.simulation_counts else 0:,}")


def memory_analysis():
    """Analyze memory usage patterns"""
    print(f"\n{'='*80}")
    print("MEMORY USAGE ANALYSIS")
    print(f"{'='*80}")

    strategies = [
        StrategyType.RANDOM,
        StrategyType.IS_MCTS_SINGLE,
        StrategyType.IS_MCTS_PER_TRICK
    ]

    for strategy in strategies:
        print(f"\n{'-'*40}")
        print(f"Strategy: {strategy.value}")
        print(f"{'-'*40}")

        # Run with memory tracking
        metrics = play_with_strategy(strategy, verbose=False, iterations=100)

        print(f"Peak memory: {metrics.peak_memory_usage:.1f} MB")
        print(f"Memory samples: {len(metrics.memory_samples)}")
        if metrics.memory_samples:
            print(f"Min memory: {min(metrics.memory_samples):.1f} MB")
            print(f"Max memory: {max(metrics.memory_samples):.1f} MB")
            print(
                f"Avg memory: {sum(metrics.memory_samples)/len(metrics.memory_samples):.1f} MB")


def performance_profiling():
    """Profile performance characteristics"""
    print(f"\n{'='*80}")
    print("PERFORMANCE PROFILING")
    print(f"{'='*80}")

    # Test decision speed vs quality trade-off
    iteration_ranges = [10, 25, 50, 100, 200, 500]

    print(f"{'Iterations':<12} {'Time/Decision':<15} {'Nodes Created':<15} {'Simulations':<12} {'Quality Est.':<12}")
    print("-" * 75)

    for iterations in iteration_ranges:
        start_time = time.time()

        # Create agent
        agent = TarotISMCTSAgent(
            player_id=0,
            iterations=iterations,
            multiple_determinization=True,
            strategy=ISMCTSStrategy.PER_TRICK
        )

        # Quick test with limited game
        state = Tarot()
        random.seed(SEED)

        # Skip to trick phase for consistent testing
        moves = 0
        while state.phase != Phase.TRICK and moves < 50:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = random.choices(actions, weights=probs)[0]
            else:
                legal_actions = state.legal_actions()
                action = random.choice(
                    legal_actions) if legal_actions else None

            if action is not None:
                state.apply_action(action)
                state.next()
                moves += 1

        # Test a few MCTS decisions
        decision_times = []
        total_nodes_created = 0
        total_simulations = 0

        for _ in range(3):  # Test 3 decisions
            if state.phase == Phase.TRICK:
                decision_start = time.time()
                action = agent.get_action(state)
                decision_time = time.time() - decision_start
                decision_times.append(decision_time)

                # Get stats after decision
                tree_stats = agent.get_tree_stats()
                total_nodes_created += tree_stats['nodes_created_this_decision']
                total_simulations += tree_stats['simulations_this_decision']

                # Apply action
                try:
                    state.apply_action(action)
                    state.next()
                except:
                    break

        avg_decision_time = sum(decision_times) / \
            len(decision_times) if decision_times else 0
        quality_est = min(1.0, iterations / 200.0)  # Rough quality estimate

        print(f"{iterations:<12} "
              f"{avg_decision_time:<15.4f} "
              f"{total_nodes_created:<15} "
              f"{total_simulations:<12} "
              f"{quality_est:<12.2f}")


def main():
    """Run comprehensive analysis"""
    print("French Tarot IS-MCTS Comprehensive Analysis")

    # Run all analyses
    detailed_mcts_comparison()
    analyze_strategy_scaling()
    memory_analysis()
    performance_profiling()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
