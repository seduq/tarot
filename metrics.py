"""
Game Metrics Collection Module

This module provides utilities for collecting and storing game metrics
during Tarot simulations for later analysis and plotting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import time
import json


@dataclass
class GameMetrics:
    """
    Stores metrics for a single game.
    """
    strategy: str
    game_id: int
    # Win rates
    taker_won: bool
    defender_won: bool
    # Game progression
    legal_moves_history: List[int] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)  # Only for MCTS
    # Game info
    num_mcts_agents: int = 0
    mcts_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsCollector:
    """
    Collects and aggregates metrics across multiple games.
    """
    games: List[GameMetrics] = field(default_factory=list)

    def add_game(self, metrics: GameMetrics):
        """Add game metrics to the collection."""
        self.games.append(metrics)

    def get_strategy_metrics(self, strategy: str) -> List[GameMetrics]:
        """Get all metrics for a specific strategy."""
        return [game for game in self.games if game.strategy == strategy]

    def get_win_rates(self, strategy: str) -> Dict[str, float]:
        """Calculate win rates for a strategy."""
        strategy_games = self.get_strategy_metrics(strategy)
        if not strategy_games:
            return {"taker_win_rate": 0.0, "defender_win_rate": 0.0}

        total_games = len(strategy_games)
        taker_wins = sum(1 for game in strategy_games if game.taker_won)
        defender_wins = sum(1 for game in strategy_games if game.defender_won)

        return {
            "taker_win_rate": taker_wins / total_games,
            "defender_win_rate": defender_wins / total_games
        }

    def get_average_legal_moves(self, strategy: str, normalize: bool = False, smoothing_window: int = 3) -> List[float]:
        """Get average legal moves progression for a strategy."""
        strategy_games = self.get_strategy_metrics(strategy)
        if not strategy_games:
            return []

        # Find the maximum length to avoid index errors
        max_length = max(len(game.legal_moves_history)
                         for game in strategy_games)
        if max_length == 0:
            return []

        # Calculate average at each turn
        averages = []
        for turn in range(max_length):
            turn_moves = [
                game.legal_moves_history[turn]
                for game in strategy_games
                if turn < len(game.legal_moves_history)
            ]
            if turn_moves:
                averages.append(sum(turn_moves) / len(turn_moves))

        if not averages:
            return []

        # Apply normalization if requested
        if normalize:
            max_moves = max(averages)
            min_moves = min(averages)
            if max_moves > min_moves:
                averages = [(x - min_moves) / (max_moves - min_moves)
                            for x in averages]

        # Apply smoothing to reduce peaks
        if smoothing_window > 1 and len(averages) >= smoothing_window:
            smoothed = []
            for i in range(len(averages)):
                # Calculate window bounds
                window_start = max(0, i - smoothing_window // 2)
                window_end = min(len(averages), i + smoothing_window // 2 + 1)

                # Calculate average within window
                window_values = averages[window_start:window_end]
                smoothed.append(sum(window_values) / len(window_values))

            averages = smoothed

        return averages

    def get_average_decision_times(self, strategy: str, progressive: bool = True) -> List[float]:
        """Get average decision times for MCTS strategy through game progress."""
        strategy_games = self.get_strategy_metrics(strategy)
        mcts_games = [game for game in strategy_games if game.decision_times]

        if not mcts_games:
            return []

        # Filter out zero times (non-MCTS decisions) and get MCTS-only times
        mcts_only_times = []
        for game in mcts_games:
            # Only include non-zero decision times (actual MCTS decisions)
            game_mcts_times = [t for t in game.decision_times if t > 0.0]
            if game_mcts_times:
                mcts_only_times.append(game_mcts_times)

        if not mcts_only_times:
            return []

        # Find the maximum length among MCTS-only decision sequences
        max_length = max(len(times) for times in mcts_only_times)
        if max_length == 0:
            return []

        if progressive:
            # Calculate progressive average (cumulative average up to each decision)
            averages = []
            for decision in range(max_length):
                # Collect all MCTS decision times up to and including this decision
                cumulative_times = []
                for game_times in mcts_only_times:
                    if decision < len(game_times):
                        # Add all times from start up to current decision
                        cumulative_times.extend(game_times[:decision + 1])

                if cumulative_times:
                    averages.append(sum(cumulative_times) /
                                    len(cumulative_times))
        else:
            # Calculate average at each MCTS decision point
            averages = []
            for decision in range(max_length):
                decision_times = [
                    game_times[decision]
                    for game_times in mcts_only_times
                    if decision < len(game_times)
                ]
                if decision_times:
                    averages.append(sum(decision_times) / len(decision_times))

        return averages

    def save_to_file(self, filename: str):
        """Save metrics to JSON file."""
        data = {
            "games": [
                {
                    "strategy": game.strategy,
                    "game_id": game.game_id,
                    "taker_won": game.taker_won,
                    "defender_won": game.defender_won,
                    "legal_moves_history": game.legal_moves_history,
                    "decision_times": game.decision_times,
                    "num_mcts_agents": game.num_mcts_agents,
                    "mcts_config": game.mcts_config
                }
                for game in self.games
            ]
        }

        with open(f"results/{filename}", 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'MetricsCollector':
        """Load metrics from JSON file."""
        with open(f"results/{filename}", 'r') as f:
            data = json.load(f)

        collector = cls()
        for game_data in data["games"]:
            metrics = GameMetrics(
                strategy=game_data["strategy"],
                game_id=game_data["game_id"],
                taker_won=game_data["taker_won"],
                defender_won=game_data["defender_won"],
                legal_moves_history=game_data["legal_moves_history"],
                decision_times=game_data["decision_times"],
                num_mcts_agents=game_data["num_mcts_agents"],
                mcts_config=game_data["mcts_config"]
            )
            collector.add_game(metrics)

        return collector
