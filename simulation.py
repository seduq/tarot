"""
Tarot Game Strategy Simulation Module

This module provides a framework for testing and comparing different AI strategies
in the Tarot card game. It includes implementations of basic heuristic strategies
and Monte Carlo Tree Search (MCTS) agents for performance analysis.

The simulation allows for systematic comparison of strategy effectiveness by running
multiple games with different agent configurations and tracking their performance.
"""

from dataclasses import dataclass
from typing import List
from tarot import Tarot
from tarot.is_mcts import TarotISMCTSAgent
from tarot.constants import Phase
from tarot import Const
import random
from enum import Enum
import argparse
import time
from metrics import GameMetrics, MetricsCollector


class StrategyType(Enum):
    """
    Available strategy types for Tarot game agents.

    Attributes:
        RANDOM: Randomly selects from legal actions
        MAX_CARD: Prefers playing highest value cards during tricks
        MIN_CARD: Prefers playing lowest value cards during tricks  
        RIS_MCTS: Uses Information Set Monte Carlo Tree Search algorithm
    """
    RANDOM = "random"
    MAX_CARD = "max_card"
    MIN_CARD = "min_card"
    RIS_MCTS = "ris_mcts"


@dataclass
class ISMCTSConfig:
    """
    Configuration parameters for Information Set Monte Carlo Tree Search.

    This class encapsulates the hyperparameters that control the behavior
    of the IS-MCTS algorithm, allowing for easy experimentation with
    different configurations.

    Attributes:
        iterations: Number of MCTS simulations to run per decision
        exploration_constant: UCB1 exploration parameter (typically √2)
        pw_alpha: Progressive widening alpha parameter for action selection
        pw_constant: Progressive widening constant for controlling expansion
    """
    iterations: int = 100
    exploration_constant: float = 1.4
    pw_alpha: float = 0.5
    pw_constant: float = 2.0


class StrategyAgent:
    """
    Base class for implementing different Tarot game strategies.

    This class provides a common interface for all strategy agents,
    implementing basic heuristic strategies that can be used as baselines
    for comparison with more sophisticated AI approaches.

    Attributes:
        player_id: Unique identifier for the player (0-4 in 5-player Tarot)
        strategy_type: The type of strategy this agent implements
    """

    def __init__(self, player_id: int, strategy_type: StrategyType):
        """
        Initialize a strategy agent.

        Args:
            player_id: The player's position in the game (0-4)
            strategy_type: The strategy type this agent will use
        """
        self.player_id = player_id
        self.strategy_type = strategy_type

    def get_action(self, state: Tarot) -> int:
        """
        Select an action based on the agent's strategy.

        Args:
            state: Current game state

        Returns:
            Selected action as an integer identifier
        """
        legal_actions = state.legal_actions()

        if self.strategy_type == StrategyType.RANDOM:
            # Baseline strategy: random selection from legal actions
            action = random.choice(legal_actions)

        elif self.strategy_type == StrategyType.MAX_CARD:
            # Aggressive strategy: play highest cards during tricks
            if state.phase == Phase.TRICK:
                action = max(legal_actions)
            else:
                # For non-trick phases, fall back to random selection
                action = random.choice(legal_actions)

        elif self.strategy_type == StrategyType.MIN_CARD:
            # Conservative strategy: play lowest cards during tricks
            if state.phase == Phase.TRICK:
                action = min(legal_actions)
            else:
                # For non-trick phases, fall back to random selection
                action = random.choice(legal_actions)

        else:
            # Fallback for undefined strategies
            action = random.choice(legal_actions)

        return action


class MCTSStrategyAgent(StrategyAgent):
    """
    Advanced strategy agent using Information Set Monte Carlo Tree Search.

    This agent uses the IS-MCTS algorithm to make decisions by simulating
    possible future game states and selecting actions that maximize expected
    utility. It's designed to handle imperfect information games like Tarot.

    Attributes:
        iterations: Number of MCTS simulations per decision
        exploration_constant: UCB1 exploration parameter
        pw_alpha: Progressive widening alpha parameter
        pw_constant: Progressive widening constant
        mcts_agent: The underlying IS-MCTS implementation
    """

    def __init__(self,
                 player_id: int,
                 strategy_type: StrategyType,
                 config: ISMCTSConfig
                 ):
        """
        Initialize an MCTS strategy agent.

        Args:
            player_id: The player's position in the game
            strategy_type: Should be StrategyType.RIS_MCTS
            config: Configuration parameters for the MCTS algorithm
        """
        super().__init__(player_id, strategy_type)
        self.iterations = config.iterations
        self.exploration_constant = config.exploration_constant
        self.pw_alpha = config.pw_alpha
        self.pw_constant = config.pw_constant

        # Initialize the IS-MCTS agent with the specified parameters
        self.mcts_agent = TarotISMCTSAgent(
            player=player_id,
            iterations=self.iterations,
            exploration_constant=self.exploration_constant,
            pw_alpha=self.pw_alpha,
            pw_constant=self.pw_constant,
        )

    def get_action(self, state: Tarot) -> int:
        """
        Select an action using Information Set Monte Carlo Tree Search.

        Args:
            state: Current game state

        Returns:
            Action selected by the MCTS algorithm
        """
        # Run MCTS to determine the best action
        action = self.mcts_agent.run(state)
        return action


def play_with_strategy(strategy_type: StrategyType, config: ISMCTSConfig, max_mcts_agents: int = 1, game_id: int = 0) -> GameMetrics:
    """
    Execute a single Tarot game with mixed strategy agents and collect metrics.

    Args:
        strategy_type: The strategy to test (assigned to selected players)
        config: Configuration for MCTS if the strategy requires it
        max_mcts_agents: Maximum number of RIS-MCTS agents to use (1-5)
        game_id: Unique identifier for this game

    Returns:
        GameMetrics: Collected metrics from the game
    """
    # Initialize a new game state
    state = Tarot()
    agents: List[StrategyAgent] = []

    # Initialize metrics collection
    metrics = GameMetrics(
        strategy=strategy_type.value,
        game_id=game_id,
        taker_won=False,
        defender_won=False,
        num_mcts_agents=0,
        mcts_config={
            "iterations": config.iterations,
            "exploration_constant": config.exploration_constant,
            "pw_alpha": config.pw_alpha,
            "pw_constant": config.pw_constant
        }
    )

    # Determine how many MCTS agents to use for this game
    if strategy_type == StrategyType.RIS_MCTS:
        num_mcts_agents = random.randint(
            1, min(max_mcts_agents, Const.NUM_PLAYERS))
        # Randomly select which player positions will use MCTS
        mcts_positions = random.sample(
            range(Const.NUM_PLAYERS), num_mcts_agents)
        metrics.num_mcts_agents = num_mcts_agents
    else:
        # For non-MCTS strategies, randomly select one player position
        mcts_positions = [random.randint(0, Const.NUM_PLAYERS - 1)]

    # Create agents for all players
    for i in range(Const.NUM_PLAYERS):
        if i in mcts_positions and strategy_type == StrategyType.RIS_MCTS:
            # Assign the MCTS strategy to selected players
            agents.append(MCTSStrategyAgent(i, strategy_type, config))
        elif i in mcts_positions:
            # Assign the test strategy to the selected player (for non-MCTS strategies)
            agents.append(StrategyAgent(i, strategy_type))
        else:
            # Assign random basic strategies to other players
            strategies = [
                StrategyType.RANDOM,
                StrategyType.MAX_CARD,
                StrategyType.MIN_CARD,]
            strategy = random.choice(strategies)
            agents.append(StrategyAgent(i, strategy))

    # Main game loop with safety iteration limit
    iteration = 0
    while not state.is_terminal() and iteration < 150:
        iteration += 1

        if state.is_chance_node():
            # Handle chance events (like card dealing)
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = random.choices(actions, weights=probs)[0]
        else:
            # Handle player decisions
            legal_actions = state.legal_actions()
            if not legal_actions:
                # No legal actions available, game should end
                break

            # Collect metrics for legal moves
            metrics.legal_moves_history.append(len(legal_actions))

            # Get action from the current player's agent
            current_agent = agents[state.current]

            # Time decision for all agents (MCTS gets actual time, others get 0)
            if isinstance(current_agent, MCTSStrategyAgent):
                start_time = time.time()
                action = current_agent.get_action(state)
                decision_time = time.time() - start_time
                metrics.decision_times.append(decision_time)

                # Collect MCTS-specific metrics
                nodes_created, illegal_moves = current_agent.mcts_agent.get_last_decision_stats()
                metrics.nodes_created.append(nodes_created)
                metrics.illegal_moves.append(illegal_moves)
            else:
                action = current_agent.get_action(state)
                # Add minimal time for non-MCTS agents to keep arrays same length
                metrics.decision_times.append(0.0)
                # Add zero values for MCTS-specific metrics for non-MCTS agents
                metrics.nodes_created.append(0)
                metrics.illegal_moves.append(0)

        # Apply the selected action and advance to the next state
        state.apply_action(action)
        state.next()

    # Determine game outcome (simplified - you may need to adjust based on your Tarot implementation)
    # This is a placeholder - replace with actual win condition checking
    if state.is_terminal():
        # Randomly assign win for demonstration - replace with actual game result logic
        taker_won = random.choice([True, False])
        metrics.taker_won = taker_won
        metrics.defender_won = not taker_won

    return metrics


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
        help="Number of games to simulate per strategy"
    )

    parser.add_argument(
        "--max-mcts-agents", "-m",
        type=int,
        default=1,
        choices=range(1, Const.NUM_PLAYERS + 1),
        help="Maximum number of RIS-MCTS agents in a single game"
    )

    # RIS-MCTS configuration
    parser.add_argument(
        "--mcts-iterations", "-i",
        type=int,
        default=100,
        help="Number of MCTS iterations per decision"
    )

    parser.add_argument(
        "--exploration-constant", "-e",
        type=float,
        default=1.4,
        help="UCB1 exploration constant for MCTS"
    )

    parser.add_argument(
        "--pw-alpha", "-a",
        type=float,
        default=0.5,
        help="Progressive widening alpha parameter"
    )

    parser.add_argument(
        "--pw-constant", "-c",
        type=float,
        default=2.0,
        help="Progressive widening constant parameter"
    )

    # Other options
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=7264828,
        help="Random seed for reproducible results"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["random", "max_card", "min_card", "ris_mcts"],
        default=["random", "max_card", "min_card", "ris_mcts"],
        help="Strategies to test in the simulation"
    )

    return parser.parse_args()


def main():
    """
    Main function to run comprehensive strategy comparison experiments.

    This function executes a systematic comparison of different AI strategies
    by running multiple games for each strategy type. The results can be used
    to evaluate the relative performance of different approaches.

    Command line arguments allow customization of:
    - Number of games per strategy
    - RIS-MCTS hyperparameters
    - Maximum number of MCTS agents per game
    - Random seed for reproducibility
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set random seed for reproducible results
    random.seed(args.seed)

    # Convert strategy strings to StrategyType enums
    strategy_map = {
        "random": StrategyType.RANDOM,
        "max_card": StrategyType.MAX_CARD,
        "min_card": StrategyType.MIN_CARD,
        "ris_mcts": StrategyType.RIS_MCTS
    }
    strategies = [strategy_map[s] for s in args.strategies]

    # Configure MCTS hyperparameters from command line arguments
    mcts_config = ISMCTSConfig(
        iterations=args.mcts_iterations,
        exploration_constant=args.exploration_constant,
        pw_alpha=args.pw_alpha,
        pw_constant=args.pw_constant
    )

    # Initialize metrics collector
    collector = MetricsCollector()

    # Print configuration summary
    print("=" * 60)
    print("TAROT STRATEGY SIMULATION")
    print("=" * 60)
    print(f"Games per strategy: {args.games}")
    print(f"Max MCTS agents: {args.max_mcts_agents}")
    print(f"Random seed: {args.seed}")
    print(f"Strategies to test: {[s.value for s in strategies]}")
    print("\nMCTS Configuration:")
    print(f"  Iterations: {mcts_config.iterations}")
    print(f"  Exploration constant: {mcts_config.exploration_constant}")
    print(f"  Progressive widening alpha: {mcts_config.pw_alpha}")
    print(f"  Progressive widening constant: {mcts_config.pw_constant}")
    print("=" * 60)

    # Execute the strategy comparison experiment
    game_counter = 0
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy.value}")
        if strategy == StrategyType.RIS_MCTS:
            print(f"  Using up to {args.max_mcts_agents} MCTS agents per game")

        for game_num in range(args.games):
            print(f"  Playing game {game_num + 1}/{args.games}", end='\r')
            metrics = play_with_strategy(
                strategy, mcts_config, args.max_mcts_agents, game_counter)
            collector.add_game(metrics)
            game_counter += 1
        print(f"  Completed {args.games} games" + " " * 20)  # Clear the line

    # Save metrics to file
    metrics_file = f"simulation_metrics.json"
    collector.save_to_file("results", metrics_file)
    print(f"\nMetrics saved to: results/{metrics_file}")

    # Generate readable reports
    try:
        from report_generator import MetricsReporter
        reporter = MetricsReporter(collector)

        # Generate text and CSV reports in results directory
        text_report = f"simulation_report_seed_{args.seed}.txt"
        csv_report = f"simulation_summary_seed_{args.seed}.csv"

        reporter.generate_summary_report("results", text_report)
        reporter.generate_csv_summary("results", csv_report)

        print(f"Reports generated:")
        print(f"  Detailed report: results/{text_report}")
        print(f"  CSV summary: results/{csv_report}")
    except ImportError:
        print("Report generator not available. Install required dependencies.")

    print("\nUse plot_basic.py to generate plots from this data.")


if __name__ == "__main__":
    main()
