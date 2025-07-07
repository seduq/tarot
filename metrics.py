from tarot import Tarot
from tarot.is_mcts import TarotISMCTSAgent
from tarot.constants import Phase
from tarot import Const
import random
import time
import psutil
import os
from collections import defaultdict
from typing import Dict, List, Any, Callable
from enum import Enum


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

        # Cumulative metrics over time
        self.cumulative_decisions = []  # Running total of decisions
        self.cumulative_legal_actions = []  # Running total of legal actions encountered
        self.cumulative_nodes_created = []  # Running total of nodes created

        # Win rate metrics
        self.wins_as_taker = 0
        self.losses_as_taker = 0
        self.wins_as_defender = 0
        self.losses_as_defender = 0
        self.total_games_as_taker = 0
        self.total_games_as_defender = 0

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

        # Update cumulative metrics
        self.cumulative_decisions.append(self.total_decisions)
        total_legal_actions = sum(self.legal_actions_count)
        self.cumulative_legal_actions.append(total_legal_actions)

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

        # Update cumulative nodes created
        self.cumulative_nodes_created.append(self.total_nodes_created)

    def add_game_result(self, player_id: int, taker_id: int, player_won: bool):
        """Record a game result for win rate tracking"""
        if player_id == taker_id:
            # Player was the taker
            self.total_games_as_taker += 1
            if player_won:
                self.wins_as_taker += 1
            else:
                self.losses_as_taker += 1
        else:
            # Player was a defender
            self.total_games_as_defender += 1
            if player_won:
                self.wins_as_defender += 1
            else:
                self.losses_as_defender += 1

    def get_win_rates(self):
        """Calculate win rates for taker and defender scenarios"""
        taker_win_rate = self.wins_as_taker / \
            self.total_games_as_taker if self.total_games_as_taker > 0 else 0.0
        defender_win_rate = self.wins_as_defender / \
            self.total_games_as_defender if self.total_games_as_defender > 0 else 0.0
        overall_win_rate = (self.wins_as_taker + self.wins_as_defender) / (self.total_games_as_taker +
                                                                           self.total_games_as_defender) if (self.total_games_as_taker + self.total_games_as_defender) > 0 else 0.0

        return {
            'taker_win_rate': taker_win_rate,
            'defender_win_rate': defender_win_rate,
            'overall_win_rate': overall_win_rate,
            'games_as_taker': self.total_games_as_taker,
            'games_as_defender': self.total_games_as_defender,
            'total_games': self.total_games_as_taker + self.total_games_as_defender
        }

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

        # Win rate metrics
        win_rates = self.get_win_rates()
        if win_rates['total_games'] > 0:
            print(f"\nWin Rate Metrics:")
            print(f"  Overall win rate: {win_rates['overall_win_rate']:.1%}")
            if win_rates['games_as_taker'] > 0:
                print(
                    f"  Win rate as taker: {win_rates['taker_win_rate']:.1%} ({self.wins_as_taker}/{win_rates['games_as_taker']})")
            if win_rates['games_as_defender'] > 0:
                print(
                    f"  Win rate as defender: {win_rates['defender_win_rate']:.1%} ({self.wins_as_defender}/{win_rates['games_as_defender']})")
            print(f"  Games as taker: {win_rates['games_as_taker']}")
            print(f"  Games as defender: {win_rates['games_as_defender']}")


def aggregate_metrics(metrics_list: List[GameMetrics], strategy_name: str) -> Dict[str, Any]:
    """Aggregate metrics across multiple games"""
    if not metrics_list:
        return {}

    n_games = len(metrics_list)

    # Aggregate cumulative data (take the longest sequence and average across games)
    max_decisions = max(len(m.cumulative_decisions)
                        for m in metrics_list) if metrics_list else 0
    avg_cumulative_decisions = []
    avg_cumulative_legal_actions = []
    avg_cumulative_nodes_created = []

    for i in range(max_decisions):
        decisions_at_step = [m.cumulative_decisions[i]
                             for m in metrics_list if i < len(m.cumulative_decisions)]
        legal_actions_at_step = [m.cumulative_legal_actions[i]
                                 for m in metrics_list if i < len(m.cumulative_legal_actions)]
        nodes_at_step = [m.cumulative_nodes_created[i]
                         for m in metrics_list if i < len(m.cumulative_nodes_created)]

        if decisions_at_step:
            avg_cumulative_decisions.append(
                sum(decisions_at_step) / len(decisions_at_step))
        if legal_actions_at_step:
            avg_cumulative_legal_actions.append(
                sum(legal_actions_at_step) / len(legal_actions_at_step))
        if nodes_at_step:
            avg_cumulative_nodes_created.append(
                sum(nodes_at_step) / len(nodes_at_step))

    # Aggregate win rate metrics
    total_wins_as_taker = sum(m.wins_as_taker for m in metrics_list)
    total_losses_as_taker = sum(m.losses_as_taker for m in metrics_list)
    total_wins_as_defender = sum(m.wins_as_defender for m in metrics_list)
    total_losses_as_defender = sum(m.losses_as_defender for m in metrics_list)
    total_games_as_taker = sum(m.total_games_as_taker for m in metrics_list)
    total_games_as_defender = sum(
        m.total_games_as_defender for m in metrics_list)

    # Calculate aggregated win rates
    taker_win_rate = total_wins_as_taker / \
        total_games_as_taker if total_games_as_taker > 0 else 0.0
    defender_win_rate = total_wins_as_defender / \
        total_games_as_defender if total_games_as_defender > 0 else 0.0
    overall_win_rate = (total_wins_as_taker + total_wins_as_defender) / (total_games_as_taker +
                                                                         total_games_as_defender) if (total_games_as_taker + total_games_as_defender) > 0 else 0.0

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
        # Cumulative time-series data
        'cumulative_decisions': avg_cumulative_decisions,
        'cumulative_legal_actions': avg_cumulative_legal_actions,
        'cumulative_nodes_created': avg_cumulative_nodes_created,
        # Win rate metrics
        'taker_win_rate': taker_win_rate,
        'defender_win_rate': defender_win_rate,
        'overall_win_rate': overall_win_rate,
        'total_wins_as_taker': total_wins_as_taker,
        'total_losses_as_taker': total_losses_as_taker,
        'total_wins_as_defender': total_wins_as_defender,
        'total_losses_as_defender': total_losses_as_defender,
        'total_games_as_taker': total_games_as_taker,
        'total_games_as_defender': total_games_as_defender,
    }


def print_aggregated_results(metrics: Dict[str, Any], strategy_name: str):
    """Print aggregated results for a strategy"""
    print(f"\nAGGREGATED RESULTS: {strategy_name}")
    print(f"Games played: {metrics['games_played']}")
    print(f"Avg game time: {metrics['avg_total_time']:.3f}s")
    print(f"Avg decision time: {metrics['avg_decision_time']:.4f}s")
    print(f"Avg decisions/sec: {metrics['avg_decisions_per_second']:.1f}")
    print(f"Avg peak memory: {metrics['avg_peak_memory']:.1f} MB")

    # Win rate metrics
    print(f"\nWin Rate Summary:")
    print(f"Overall win rate: {metrics['overall_win_rate']:.1%}")
    if metrics['total_games_as_taker'] > 0:
        print(
            f"Win rate as taker: {metrics['taker_win_rate']:.1%} ({metrics['total_wins_as_taker']}/{metrics['total_games_as_taker']})")
    if metrics['total_games_as_defender'] > 0:
        print(
            f"Win rate as defender: {metrics['defender_win_rate']:.1%} ({metrics['total_wins_as_defender']}/{metrics['total_games_as_defender']})")

    if metrics['avg_node_visits'] > 0:
        print(f"\nMCTS Metrics:")
        print(f"Avg node visits: {metrics['avg_node_visits']:,.0f}")
        print(f"Avg nodes created: {metrics['avg_nodes_created']:,.0f}")
        if metrics['avg_trees_created'] > 0:
            print(f"Avg trees created: {metrics['avg_trees_created']:,.0f}")


def print_strategy_comparison(results: Dict[str, Any]):
    """Print final comparison table"""
    if not results:
        return

    print(f"\n{'='*120}")
    print("FINAL STRATEGY COMPARISON")
    print(f"{'='*120}")

    # Table header
    print(f"{'Strategy':<20} {'Time(s)':<8} {'Dec/s':<8} {'Memory(MB)':<12} {'Win Rate':<10} {'Taker WR':<10} {'Defender WR':<12} {'Nodes':<10}")
    print("-" * 120)

    # Sort by overall win rate (descending)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].get('overall_win_rate', 0), reverse=True)

    for strategy_name, metrics in sorted_results:
        nodes_str = f"{metrics['avg_nodes_created']:,.0f}" if metrics['avg_nodes_created'] > 0 else "N/A"
        overall_wr = f"{metrics['overall_win_rate']:.1%}" if metrics.get(
            'overall_win_rate') is not None else "N/A"
        taker_wr = f"{metrics['taker_win_rate']:.1%}" if metrics.get(
            'taker_win_rate') is not None and metrics['total_games_as_taker'] > 0 else "N/A"
        defender_wr = f"{metrics['defender_win_rate']:.1%}" if metrics.get(
            'defender_win_rate') is not None and metrics['total_games_as_defender'] > 0 else "N/A"

        print(f"{strategy_name:<20} "
              f"{metrics['avg_total_time']:<8.3f} "
              f"{metrics['avg_decisions_per_second']:<8.1f} "
              f"{metrics['avg_peak_memory']:<12.1f} "
              f"{overall_wr:<10} "
              f"{taker_wr:<10} "
              f"{defender_wr:<12} "
              f"{nodes_str:<10}")

    print(f"\nWin Rate Legend:")
    print(f"Win Rate: Overall win percentage across all games")
    print(f"Taker WR: Win rate when the agent is the taker")
    print(f"Defender WR: Win rate when the agent is a defender")
