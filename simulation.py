from dataclasses import dataclass
from tarot import Tarot
from tarot.is_mcts import TarotISMCTSAgent
from tarot.constants import Phase
from tarot import Const
import random
from enum import Enum


class StrategyType(Enum):
    """Available strategy types"""
    RANDOM = "random"
    MAX_CARD = "max_card"
    MIN_CARD = "min_card"
    RIS_MCTS = "ris_mcts"


@dataclass
class ISMCTSConfig:
    """Configuration for StrategyAgent"""
    iterations: int = 100
    exploration_constant: float = 1.4
    pw_alpha: float = 0.5
    pw_constant: float = 2.0


class StrategyAgent:
    """Base class for strategy agents"""

    def __init__(self, player_id: int, strategy_type: StrategyType):
        self.player_id = player_id
        self.strategy_type = strategy_type

    def get_action(self, state: Tarot) -> int:
        """Get action from the strategy"""
        legal_actions = state.legal_actions()

        if self.strategy_type == StrategyType.RANDOM:
            action = random.choice(legal_actions)

        elif self.strategy_type == StrategyType.MAX_CARD:
            # Choose the highest valued card when possible
            if state.phase == Phase.TRICK:
                action = max(legal_actions)
            else:
                action = random.choice(legal_actions)

        elif self.strategy_type == StrategyType.MIN_CARD:
            # Choose the lowest valued card when possible
            if state.phase == Phase.TRICK:
                action = min(legal_actions)
            else:
                action = random.choice(legal_actions)

        else:
            # Should not reach here for base strategies
            action = random.choice(legal_actions)

        return action


class MCTSStrategyAgent(StrategyAgent):
    """MCTS Strategy Agent using IS-MCTS"""

    def __init__(self,
                 player_id: int,
                 strategy_type: StrategyType,
                 config: ISMCTSConfig
                 ):
        super().__init__(player_id, strategy_type)
        self.iterations = config.iterations
        self.exploration_constant = config.exploration_constant
        self.pw_alpha = config.pw_alpha
        self.pw_constant = config.pw_constant

        self.mcts_agent = TarotISMCTSAgent(
            player=player_id,
            iterations=self.iterations,
            exploration_constant=self.exploration_constant,
            pw_alpha=self.pw_alpha,
            pw_constant=self.pw_constant,
        )

    def get_action(self, state: Tarot) -> int:
        """Get action using IS-MCTS"""
        # Get action from MCTS agent
        action = self.mcts_agent.run(state)
        return action


def play_with_strategy(strategy_type: StrategyType, config: ISMCTSConfig):
    """Play a single game with the specified strategy"""
    state = Tarot()
    agents = []

    # Randomly select a player position for the test agent
    position = random.randint(0, Const.NUM_PLAYERS - 1)

    # Create agent for player with the specified strategy
    for i in range(Const.NUM_PLAYERS):
        if i == position and strategy_type == StrategyType.RIS_MCTS:
            # Use MCTS agent for the specified player
            agents.append(MCTSStrategyAgent(i, strategy_type, config))
        else:
            # Use basic strategy agent for other players
            strategies = [
                StrategyType.RANDOM,
                StrategyType.MAX_CARD,
                StrategyType.MIN_CARD,]
            # Randomly assign a basic strategy to other players
            strategy = random.choice(strategies)
            agents.append(StrategyAgent(i, strategy))

    iteration = 0
    while not state.is_terminal() and iteration < 150:  # Limit iterations for safety
        iteration += 1

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = random.choices(actions, weights=probs)[0]
        else:
            # Handle regular decisions
            legal_actions = state.legal_actions()
            if not legal_actions:
                break

            # Get action from appropriate agent
            current_agent = agents[state.current]
            action = current_agent.get_action(state)

        # Apply action and advance state
        state.apply_action(action)
        state.next()


def main():
    """Main function to run comprehensive strategy comparison"""
    # Define strategies to test
    games_per_strategy = 100  # Number of games to play for each strategy
    strategies = [
        StrategyType.RANDOM,
        StrategyType.MAX_CARD,
        StrategyType.MIN_CARD,
        StrategyType.RIS_MCTS
    ]
    mcts_config = ISMCTSConfig(
        iterations=100,  # Number of MCTS iterations
        exploration_constant=1.4,  # Exploration constant for MCTS
        pw_alpha=0.5,  # Progressive widening alpha
        pw_constant=2.0  # Progressive widening constant
    )

    # Run comparison
    for strategy in strategies:
        for game_num in range(games_per_strategy):
            print(f"Playing game {game_num + 1}",
                  f"with strategy: {strategy.value}")
            play_with_strategy(strategy, mcts_config)


SEED = 7264828
if __name__ == "__main__":
    random.seed(SEED)
    main()
