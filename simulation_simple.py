"""
Tarot Game Strategy Simulation Module

This module provides a framework for testing and comparing different AI strategies
in the Tarot card game. It includes implementations of basic heuristic strategies
and Monte Carlo Tree Search (MCTS) agents for performance analysis.

The simulation allows for systematic comparison of strategy effectiveness by running
multiple games with different agent configurations and tracking their performance.
"""

import random
import argparse
from typing import List
from tarot_rl.tarot import Tarot, Agent, TarotState
from tarot_rl.agents import RandomAgent
from tarot_rl.constants import NUM_PLAYERS


def play() -> None:
    # Initialize a new game state
    state = Tarot.new()
    agents: List[Agent] = []
    for i in range(NUM_PLAYERS):
        agents.append(RandomAgent(
            name=f"RandomAgent_{i}", player_id=i, state=state))
    Tarot.play(state, agents)


def parse_arguments():
    """
    Parse command line arguments for simulation configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Tarot Game Strategy Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Game configuration
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=100,
        help="Number of games to simulate per strategy (default: 100)"
    )

    # RIS-MCTS configuration
    parser.add_argument(
        "--mcts-iterations", "-i",
        type=int,
        default=100,
        help="Number of MCTS iterations per decision (default: 100)"
    )

    parser.add_argument(
        "--exploration-constant", "-e",
        type=float,
        default=1.4,
        help="UCB1 exploration constant for MCTS (default: 1.4)"
    )

    parser.add_argument(
        "--pw-alpha", "-a",
        type=float,
        default=0.5,
        help="Progressive widening alpha parameter (default: 0.5)"
    )

    parser.add_argument(
        "--pw-constant", "-c",
        type=float,
        default=2.0,
        help="Progressive widening constant parameter (default: 2.0)"
    )

    # Other options
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)"
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed for reproducible results
    random.seed(args.seed)

    # Run simulations for each strategy
    for i in range(args.games):
        play()


if __name__ == "__main__":
    main()
