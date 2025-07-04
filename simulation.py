from tarot import Tarot
from tarot.mcts import TarotISMCTSAgent, ISMCTSStrategy
from tarot.constants import Phase, TRICK_FINISHED
import random
import time
import psutil
import os
from collections import defaultdict
from typing import Dict, List, Any, Callable
from enum import Enum


SEED = 7264828


class StrategyType(Enum):
    """Available strategy types"""
    RANDOM = "random"
    MAX_CARD = "max_card"
    MIN_CARD = "min_card"
    IS_MCTS_SINGLE = "is_mcts_single"
    IS_MCTS_PER_ACTION = "is_mcts_per_action"
    IS_MCTS_PER_TRICK = "is_mcts_per_trick"


class GameMetrics:
    """Class to track comprehensive game metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        # Time metrics
        self.total_time = 0.0
        self.decision_times = []
        self.chance_times = []

        # Decision metrics
        self.total_decisions = 0
        self.total_chance_decisions = 0
        self.decisions_per_phase = defaultdict(int)
        self.legal_actions_count = []
        self.chance_actions_count = []

        # MCTS specific metrics
        self.total_node_visits = 0
        self.total_nodes_created = 0
        self.tree_sizes = []
        self.simulation_counts = []
        self.total_trees_created = 0
        self.trees_created_counts = []

        # Memory metrics
        self.peak_memory_usage = 0
        self.memory_samples = []

        # Game state metrics
        self.phases_visited = set()
        self.tricks_played = 0
        self.cards_played = 0

        # Performance metrics
        self.decisions_per_second = 0
        self.average_decision_time = 0
        self.average_legal_actions = 0
        self.average_chance_actions = 0

    def add_decision(self, decision_time: float, num_legal_actions: int, phase: Phase):
        """Record a decision made"""
        self.decision_times.append(decision_time)
        self.total_decisions += 1
        self.decisions_per_phase[phase] += 1
        self.legal_actions_count.append(num_legal_actions)
        self.phases_visited.add(phase)

    def add_chance_decision(self, decision_time: float, num_chance_actions: int):
        """Record a chance decision made"""
        self.chance_times.append(decision_time)
        self.total_chance_decisions += 1
        self.chance_actions_count.append(num_chance_actions)

    def add_mcts_metrics(self, node_visits: int, nodes_created: int, tree_size: int, simulations: int, trees_created: int = 0):
        """Record MCTS-specific metrics"""
        self.total_node_visits += node_visits
        self.total_nodes_created += nodes_created
        self.tree_sizes.append(tree_size)
        self.simulation_counts.append(simulations)
        self.total_trees_created += trees_created
        self.trees_created_counts.append(trees_created)

    def sample_memory(self):
        """Sample current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)

    def finalize(self, total_game_time: float):
        """Calculate final metrics"""
        self.total_time = total_game_time

        if self.decision_times:
            self.average_decision_time = sum(
                self.decision_times) / len(self.decision_times)
            self.decisions_per_second = len(
                self.decision_times) / total_game_time if total_game_time > 0 else 0

        if self.legal_actions_count:
            self.average_legal_actions = sum(
                self.legal_actions_count) / len(self.legal_actions_count)

        if self.chance_actions_count:
            self.average_chance_actions = sum(
                self.chance_actions_count) / len(self.chance_actions_count)

    def print_summary(self, strategy_name: str):
        """Print comprehensive metrics summary"""
        print(f"\n{'='*60}")
        print(f"METRICS SUMMARY: {strategy_name}")
        print(f"{'='*60}")

        # Time metrics
        print(f"Total game time: {self.total_time:.3f}s")
        print(f"Average decision time: {self.average_decision_time:.4f}s")
        print(f"Decisions per second: {self.decisions_per_second:.1f}")

        # Decision metrics
        print(f"\nDecision Counts:")
        print(f"  Total decisions: {self.total_decisions}")
        print(f"  Total chance decisions: {self.total_chance_decisions}")
        print(f"  Average legal actions: {self.average_legal_actions:.1f}")
        print(f"  Average chance actions: {self.average_chance_actions:.1f}")

        # Phase breakdown
        print(f"\nDecisions per phase:")
        for phase, count in self.decisions_per_phase.items():
            print(f"  {phase}: {count}")

        # MCTS metrics (if applicable)
        if self.total_node_visits > 0 or self.total_nodes_created > 0:
            print(f"\nMCTS Metrics:")
            print(f"  Total node visits: {self.total_node_visits:,}")
            print(f"  Total nodes created: {self.total_nodes_created:,}")
            if self.tree_sizes:
                avg_tree_size = sum(self.tree_sizes) / len(self.tree_sizes)
                max_tree_size = max(self.tree_sizes)
                print(f"  Average tree size: {avg_tree_size:.0f}")
                print(f"  Maximum tree size: {max_tree_size}")
            else:
                print(f"  Tree sizes: N/A")
            if self.total_trees_created > 0:
                print(f"  Total trees created: {self.total_trees_created:,}")
                if self.trees_created_counts:
                    avg_trees_per_decision = sum(
                        self.trees_created_counts) / len(self.trees_created_counts)
                    print(
                        f"  Average trees per decision: {avg_trees_per_decision:.1f}")
            if self.simulation_counts:
                total_sims = sum(self.simulation_counts)
                avg_sims = total_sims / len(self.simulation_counts)
                print(f"  Total simulations: {total_sims:,}")
                print(f"  Average simulations per decision: {avg_sims:.0f}")
            else:
                print(f"  Simulations: N/A")

        # Memory metrics
        print(f"\nMemory Usage:")
        print(f"  Peak memory: {self.peak_memory_usage:.1f} MB")
        print(
            f"  Average memory: {sum(self.memory_samples)/len(self.memory_samples):.1f} MB" if self.memory_samples else "N/A")

        # Game state metrics
        print(f"\nGame State:")
        print(f"  Phases visited: {[str(p) for p in self.phases_visited]}")
        print(f"  Cards played: {self.cards_played}")
        print(f"  Tricks completed: {self.tricks_played}")


class StrategyAgent:
    """Base class for strategy agents"""

    def __init__(self, player_id: int, strategy_type: StrategyType):
        self.player_id = player_id
        self.strategy_type = strategy_type
        self.metrics = GameMetrics()

    def get_action(self, state: Tarot) -> int:
        """Get action from the strategy"""
        start_time = time.time()
        legal_actions = state.legal_actions()

        # Sample memory
        self.metrics.sample_memory()

        if self.strategy_type == StrategyType.RANDOM:
            action = random.choice(
                legal_actions) if legal_actions else TRICK_FINISHED

        elif self.strategy_type == StrategyType.MAX_CARD:
            # Choose the highest valued card when possible
            if state.phase == Phase.TRICK:
                action = max(
                    legal_actions) if legal_actions else TRICK_FINISHED
            else:
                action = random.choice(
                    legal_actions) if legal_actions else TRICK_FINISHED

        elif self.strategy_type == StrategyType.MIN_CARD:
            # Choose the lowest valued card when possible
            if state.phase == Phase.TRICK:
                action = min(
                    legal_actions) if legal_actions else TRICK_FINISHED
            else:
                action = random.choice(
                    legal_actions) if legal_actions else TRICK_FINISHED

        else:
            # Should not reach here for base strategies
            action = random.choice(
                legal_actions) if legal_actions else TRICK_FINISHED

        decision_time = time.time() - start_time
        self.metrics.add_decision(
            decision_time, len(legal_actions), state.phase)

        return action


class MCTSStrategyAgent(StrategyAgent):
    """MCTS Strategy Agent using IS-MCTS"""

    def __init__(self, player_id: int, strategy_type: StrategyType, iterations: int = 1000):
        super().__init__(player_id, strategy_type)
        self.iterations = iterations

        # Create appropriate IS-MCTS agent
        if strategy_type == StrategyType.IS_MCTS_SINGLE:
            self.mcts_agent = TarotISMCTSAgent(
                player_id=player_id,
                iterations=iterations,
                multiple_determinization=False
            )
        elif strategy_type == StrategyType.IS_MCTS_PER_ACTION:
            self.mcts_agent = TarotISMCTSAgent(
                player_id=player_id,
                iterations=iterations,
                multiple_determinization=True,
                strategy=ISMCTSStrategy.PER_ACTION
            )
        elif strategy_type == StrategyType.IS_MCTS_PER_TRICK:
            self.mcts_agent = TarotISMCTSAgent(
                player_id=player_id,
                iterations=iterations,
                multiple_determinization=True,
                strategy=ISMCTSStrategy.PER_TRICK
            )

    def get_action(self, state: Tarot) -> int:
        """Get action using IS-MCTS"""
        start_time = time.time()
        legal_actions = state.legal_actions()

        # Sample memory
        self.metrics.sample_memory()

        # Get action from MCTS agent
        action = self.mcts_agent.get_action(state)

        decision_time = time.time() - start_time
        self.metrics.add_decision(
            decision_time, len(legal_actions), state.phase)

        # Record MCTS specific metrics using new tree stats
        tree_stats = self.mcts_agent.get_tree_stats()
        self.metrics.add_mcts_metrics(
            node_visits=tree_stats['total_visits'],
            nodes_created=tree_stats['nodes_created_this_decision'],
            tree_size=tree_stats['total_nodes'],
            simulations=tree_stats['simulations_this_decision'],
            trees_created=tree_stats.get('trees_created_this_decision', 0)
        )

        return action


def create_agent(player_id: int, strategy_type: StrategyType, iterations: int = 1000) -> StrategyAgent:
    """Factory function to create appropriate agent"""
    if strategy_type in [StrategyType.IS_MCTS_SINGLE, StrategyType.IS_MCTS_PER_ACTION, StrategyType.IS_MCTS_PER_TRICK]:
        return MCTSStrategyAgent(player_id, strategy_type, iterations)
    else:
        return StrategyAgent(player_id, strategy_type)


def play_with_strategy(strategy_type: StrategyType, verbose: bool = True, iterations: int = 1000) -> GameMetrics:
    """Play a single game with the specified strategy"""
    state = Tarot()
    random.seed(SEED)

    # Create agent for player 0 (we'll track this player's metrics)
    agent = create_agent(0, strategy_type, iterations)

    # Create simple random agents for other players
    other_agents = [create_agent(i, StrategyType.RANDOM, 100)
                    for i in range(1, 4)]
    all_agents = [agent] + other_agents

    game_start_time = time.time()

    if verbose:
        print(f"\nStarting game with strategy: {strategy_type.value}")
        print(state)
        print("=" * 50)

    iteration = 0
    while not state.is_terminal() and iteration < 500:  # Limit iterations for safety
        iteration += 1

        if verbose:
            print("--" * 25)
            print(f"Iteration: {iteration}")
            print(f"Current player: {state.current}")
            print(f"Current phase: {state.phase}")

        # Sample memory periodically
        if iteration % 10 == 0:
            agent.metrics.sample_memory()

        if state.is_chance_node():
            # Handle chance nodes
            start_time = time.time()

            if verbose:
                print(f"Chance outcomes: {state.chance_outcomes()}")

            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = random.choices(actions, weights=probs)[0]

            decision_time = time.time() - start_time
            agent.metrics.add_chance_decision(decision_time, len(actions))

        else:
            # Handle regular decisions
            legal_actions = state.legal_actions()

            if verbose:
                print(f"Legal actions: {legal_actions}")

            if not legal_actions:
                if verbose:
                    print("No legal actions available")
                break

            # Get action from appropriate agent
            current_agent = all_agents[state.current]
            action = current_agent.get_action(state)

            # Track additional metrics for our main agent
            if state.current == 0:
                if state.phase == Phase.TRICK:
                    agent.metrics.cards_played += 1

                # Count tricks (when trick is completed)
                if (state.phase == Phase.TRICK_FINISHED or
                        (state.phase == Phase.TRICK and len([c for c in state.trick if c != -1]) == 4)):
                    agent.metrics.tricks_played += 1

        if verbose:
            print(f'Taking action {action} {state.action_to_string(action)}')

        # Apply action and advance state
        try:
            state.apply_action(action)
            state.next()
        except Exception as e:
            if verbose:
                print(f"Error applying action: {e}")
            break

    # Finalize metrics
    total_game_time = time.time() - game_start_time
    agent.metrics.finalize(total_game_time)

    if verbose:
        print(state)
        agent.metrics.print_summary(strategy_type.value)

    return agent.metrics


def compare_strategies(strategies: List[StrategyType], iterations: int = 500, games_per_strategy: int = 1, verbose: bool = False) -> Dict[str, Any]:
    """Compare multiple strategies across multiple games"""
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON - {games_per_strategy} game(s) per strategy")
    print(f"MCTS iterations per decision: {iterations}")
    print(f"{'='*80}")

    results = {}

    for strategy in strategies:
        print(f"\n{'-'*60}")
        print(f"Testing strategy: {strategy.value.upper()}")
        print(f"{'-'*60}")

        strategy_metrics = []

        for game_num in range(games_per_strategy):
            if games_per_strategy > 1:
                print(f"\nGame {game_num + 1}/{games_per_strategy}")

            metrics = play_with_strategy(
                strategy, verbose=verbose, iterations=iterations)
            strategy_metrics.append(metrics)

        # Aggregate metrics across games
        if strategy_metrics:
            avg_metrics = aggregate_metrics(strategy_metrics, strategy.value)
            results[strategy.value] = avg_metrics

            if games_per_strategy > 1:
                print_aggregated_results(avg_metrics, strategy.value)

    # Print final comparison
    print_strategy_comparison(results)

    return results


def aggregate_metrics(metrics_list: List[GameMetrics], strategy_name: str) -> Dict[str, Any]:
    """Aggregate metrics across multiple games"""
    if not metrics_list:
        return {}

    n_games = len(metrics_list)

    return {
        'strategy': strategy_name,
        'games_played': n_games,
        'avg_total_time': sum(m.total_time for m in metrics_list) / n_games,
        'avg_decision_time': sum(m.average_decision_time for m in metrics_list) / n_games,
        'avg_decisions_per_second': sum(m.decisions_per_second for m in metrics_list) / n_games,
        'avg_total_decisions': sum(m.total_decisions for m in metrics_list) / n_games,
        'avg_chance_decisions': sum(m.total_chance_decisions for m in metrics_list) / n_games,
        'avg_legal_actions': sum(m.average_legal_actions for m in metrics_list) / n_games,
        'avg_chance_actions': sum(m.average_chance_actions for m in metrics_list) / n_games,
        'avg_node_visits': sum(m.total_node_visits for m in metrics_list) / n_games,
        'avg_nodes_created': sum(m.total_nodes_created for m in metrics_list) / n_games,
        'avg_trees_created': sum(m.total_trees_created for m in metrics_list) / n_games,
        'avg_peak_memory': sum(m.peak_memory_usage for m in metrics_list) / n_games,
        'avg_cards_played': sum(m.cards_played for m in metrics_list) / n_games,
        'avg_tricks_played': sum(m.tricks_played for m in metrics_list) / n_games,
    }


def print_aggregated_results(metrics: Dict[str, Any], strategy_name: str):
    """Print aggregated results for a strategy"""
    print(f"\nAGGREGATED RESULTS: {strategy_name}")
    print(f"Games played: {metrics['games_played']}")
    print(f"Avg game time: {metrics['avg_total_time']:.3f}s")
    print(f"Avg decision time: {metrics['avg_decision_time']:.4f}s")
    print(f"Avg decisions/sec: {metrics['avg_decisions_per_second']:.1f}")
    print(f"Avg peak memory: {metrics['avg_peak_memory']:.1f} MB")
    if metrics['avg_node_visits'] > 0:
        print(f"Avg node visits: {metrics['avg_node_visits']:,.0f}")
        print(f"Avg nodes created: {metrics['avg_nodes_created']:,.0f}")
        if metrics['avg_trees_created'] > 0:
            print(f"Avg trees created: {metrics['avg_trees_created']:,.0f}")


def print_strategy_comparison(results: Dict[str, Any]):
    """Print final comparison table"""
    if not results:
        return

    print(f"\n{'='*100}")
    print("FINAL STRATEGY COMPARISON")
    print(f"{'='*100}")

    # Table header
    print(f"{'Strategy':<20} {'Time(s)':<8} {'Dec/s':<8} {'Memory(MB)':<12} {'Nodes':<10} {'Visits':<10} {'Trees':<10}")
    print("-" * 100)

    # Sort by decision speed (descending)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]['avg_decisions_per_second'], reverse=True)

    for strategy_name, metrics in sorted_results:
        nodes_str = f"{metrics['avg_nodes_created']:,.0f}" if metrics['avg_nodes_created'] > 0 else "N/A"
        visits_str = f"{metrics['avg_node_visits']:,.0f}" if metrics['avg_node_visits'] > 0 else "N/A"
        tree_created_str = f"{metrics['avg_trees_created']:,.0f}" if metrics['avg_trees_created'] > 0 else "N/A"

        print(f"{strategy_name:<20} "
              f"{metrics['avg_total_time']:<8.3f} "
              f"{metrics['avg_decisions_per_second']:<8.1f} "
              f"{metrics['avg_peak_memory']:<12.1f} "
              f"{nodes_str:<10} "
              f"{visits_str:<10}"
              f"{tree_created_str:<10}")


def play(verbose: bool = True):
    """Original play function for backward compatibility"""
    return play_with_strategy(StrategyType.RANDOM, verbose)


def main():
    """Main function to run comprehensive strategy comparison"""
    print("French Tarot Strategy Comparison")
    print("This will test all available strategies with comprehensive metrics")

    # Define strategies to test
    strategies = [
        StrategyType.RANDOM,
        StrategyType.MAX_CARD,
        StrategyType.MIN_CARD,
        StrategyType.IS_MCTS_SINGLE,
        StrategyType.IS_MCTS_PER_ACTION,
        StrategyType.IS_MCTS_PER_TRICK
    ]

    # Run comparison
    results = compare_strategies(
        strategies=[StrategyType.IS_MCTS_PER_TRICK],
        iterations=200,  # MCTS iterations per decision
        games_per_strategy=1,  # Number of games per strategy
        verbose=False  # Set to True for detailed output
    )

    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
