import random
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .tarot import Tarot
from .state import TarotState
from . import constants as Const


@dataclass
class MctsConfig:
    """
    Configuration parameters for the MCTS algorithm.
    """
    exploration_constant: float = 1.4
    max_simulations: int = 1000
    pw_alpha: float = 0.5
    pw_constant: float = 2.0


@dataclass
class TarotInformationSet:
    """
    Represents an information set for the French Tarot game in IS-MCTS.

    An information set contains all the observable information available to a player
    at a given game state, including played cards, current hand, trick state, and
    game metadata. This is used as a key for MCTS nodes to ensure that states
    with identical observable information are grouped together.
    """
    information_set: List[int]

    def __hash__(self):
        return hash(tuple(self.information_set))

    def __eq__(self, other):
        if not isinstance(other, TarotInformationSet):
            return False
        return self.information_set == other.information_set


class TarotISMCTSNode:
    """
    Node in the Information Set Monte Carlo Tree Search for French Tarot.

    Each node represents a unique information set and maintains statistics
    for all actions that can be taken from this information set. Uses UCB1
    for action selection and progressive widening for exploration.
    """

    def __init__(self,
                 info_key: TarotInformationSet,
                 parent: Optional['TarotISMCTSNode'] = None,
                 ucb_c: float = 1.4):
        """
        Initialize a new MCTS node.

        Args:
            info_key: The information set this node represents
            parent: Parent node in the search tree (unused in flat structure)
            exploration_constant: UCB1 exploration parameter (default: 1.4)
        """
        self.info_key = info_key
        self.parent = parent

        # Visit and value statistics
        self.n_visits = 0

        # Action-specific statistics
        # Number of times each action was selected
        self.n_action = defaultdict(int)
        # Cumulative reward for each action
        self.total_action_value = defaultdict(float)
        self.expanded_actions = set()  # Actions that have been tried from this node

        self.ucb_c = ucb_c

    def ucb_score(self, action: List[int]) -> float:
        """
        Calculate the Upper Confidence Bound (UCB1) score for an action.

        UCB1 balances exploitation (choosing actions with high average rewards)
        and exploration (choosing actions that haven't been tried much).

        Args:
            action: The action to calculate the score for
            c: Exploration constant (higher values favor exploration)

        Returns:
            UCB1 score for the action (infinity if never tried)
        """
        if self.n_action[action] == 0:
            return float('inf')  # Unvisited actions have highest priority

        # Q-value: average reward for this action
        q = self.total_action_value[action] / self.n_action[action]

        # Exploration term: encourages trying less-visited actions
        u = self.ucb_c * math.sqrt(math.log(self.n_visits +
                                            1) / self.n_action[action])

        return q + u

    def select_action(self, legal_actions: List[int]) -> int:
        """
        Select the best action from this node using UCB1.

        If no actions have been expanded yet, select a random unexplored action.
        Otherwise, use UCB1 to balance exploitation and exploration among
        previously expanded actions.

        Args:
            legal_actions: List of legal actions in the current state

        Returns:
            Selected action index
        """
        # Only consider actions that have been expanded and are still legal
        valid_actions = self.expanded_actions.intersection(legal_actions)

        if not valid_actions:
            # No expanded actions are legal - pick a random legal action
            action = random.choice(legal_actions)
            self.expanded_actions.add(action)
            self.n_action[action] = 0
            return action

        # Use UCB1 to select among expanded actions
        return max(valid_actions, key=lambda a: self.ucb_score(a))

    def should_expand(self, k, alpha):
        """
        Determine if we should expand a new action using progressive widening.

        Progressive widening controls the rate at which new actions are added
        to the search tree. It prevents over-exploration early in the search
        while allowing more actions to be considered as we gain more information.

        Args:
            k: Progressive widening constant (controls expansion rate)
            alpha: Progressive widening exponent (controls expansion curve)

        Returns:
            True if we should expand a new action, False otherwise
        """
        if self.n_visits == 0:
            return True  # Always expand the first action
        return len(self.expanded_actions) < k * (self.n_visits ** alpha)


class IS_MCTS:
    """
    Information Set Monte Carlo Tree Search Agent for French Tarot.

    This agent uses IS-MCTS to handle the imperfect information in Tarot.
    It builds a flat structure over information sets (what the player can observe)
    rather than complete game states, and uses determinization to sample
    possible hidden information during simulations.

    Key features:
    - Information set-based flat tree structure
    - Determinization for handling hidden information
    - Progressive widening for action space exploration
    - UCB1 for action selection within nodes
    """

    def __init__(self,
                 player: int,
                 mcts_config: MctsConfig):
        """
        Initialize the IS-MCTS agent.

        Args:
            player: Player ID (0-4) this agent represents
            iterations: Number of MCTS simulations per decision (default: 300)
            exploration_constant: UCB1 exploration parameter (default: 1.4)
            pw_alpha: Progressive widening exponent (default: 0.5)
            pw_constant: Progressive widening constant (default: 2.0)
        """
        self.player = player
        self.iterations = mcts_config.max_simulations
        self.ucb_c = mcts_config.exploration_constant

        # Progressive widening parameters control how aggressively we explore new actions
        # Exponent: lower values = more conservative expansion
        self.pw_alpha = mcts_config.pw_alpha
        # Multiplier: higher values = more actions expanded
        self.pw_constant = mcts_config.pw_constant

        # Performance metrics (currently unused but available for analysis)
        self.total_nodes_created = 0
        self.total_simulations_run = 0
        self.nodes_created_this_decision = 0
        self.simulations_this_decision = 0

        # The search tree: maps information sets to their corresponding nodes
        self.tree: Dict[TarotInformationSet, TarotISMCTSNode] = {}

    def run(self, state: TarotState) -> int:
        """
        Run the IS-MCTS algorithm to select the best action.

        This is the main entry point for the agent. It performs the specified
        number of MCTS iterations, each consisting of:
        1. Determinization (sampling hidden information)
        2. Simulation (tree traversal and random rollout)
        3. Backpropagation (updating node statistics)

        Args:
            game: Current game state

        Returns:
            Index of the selected action
        """
        # Reset per-decision statistics
        self.nodes_created_this_decision = 0
        self.simulations_this_decision = 0
        nodes_before = len(self.tree)

        # Get or create the root node for the current information set
        root_info = self.info_key(state)
        if root_info not in self.tree:
            self.tree[root_info] = TarotISMCTSNode(
                root_info, ucb_c=self.ucb_c)
            self.nodes_created_this_decision += 1
        root_node = self.tree[root_info]

        # Run MCTS simulations
        for i in range(self.iterations):
            # Create a determinized version of the game state
            determinized_game = self.determinization(state)
            # Run one simulation from this determinized state
            self.simulate(determinized_game)
            self.simulations_this_decision += 1

        # Update total statistics
        self.total_nodes_created += self.nodes_created_this_decision
        self.total_simulations_run += self.simulations_this_decision

        # Calculate nodes created this decision (including new nodes from simulations)
        nodes_after = len(self.tree)
        self.nodes_created_this_decision = nodes_after - nodes_before

        # Select the action with the most visits (most promising)
        if not root_node.expanded_actions:
            # Fallback: if no actions have been expanded, choose randomly
            legal_actions = Tarot.legal_actions(state)
            if legal_actions:
                return random.choice(legal_actions)
            else:
                return 0  # Default action if no legal actions available

        return max(root_node.expanded_actions, key=lambda a: root_node.n_action[a])

    def choose_action(self, game: Tarot, legal_actions: List[int]) -> int:
        return random.choice(legal_actions) if legal_actions else 0

    def choose_chance_action(self, state: TarotState, legal_actions: List[int]) -> int:
        return random.choice(legal_actions) if legal_actions else 0

    def choose_unexplored(self, unexplored: List[int]) -> int:
        return random.choice(unexplored) if unexplored else 0

    def simulate(self, state: TarotState):
        """
        Perform one MCTS simulation: tree traversal, expansion, and backpropagation.

        The simulation consists of several phases:
        1. Tree traversal: Navigate through existing nodes using UCB1
        2. Expansion: Add new nodes when progressive widening allows
        3. Rollout: Play randomly to a terminal state
        4. Backpropagation: Update statistics along the path

        Args:
            game: Determinized game state to simulate from
        """
        # Track the path through the tree for backpropagation
        path: List[Tuple[TarotISMCTSNode, int]] = []

        # Tree traversal and rollout phase
        while not Tarot.is_terminal(state):
            # Handle chance nodes (random events in the game)
            if Tarot.is_chance_node(state):
                action = self.choose_chance_action(
                    state, Tarot.legal_actions(state))
                Tarot.apply_action(state, action)
                Tarot.next(state)
                continue

            # Get the information set for the current player
            info = self.info_key(state)
            legal = Tarot.legal_actions(state)
            if not legal:
                break  # No legal actions available

            # Get or create the node for this information set
            if info not in self.tree:
                self.tree[info] = TarotISMCTSNode(
                    info, ucb_c=self.ucb_c)

            node = self.tree[info]

            # Determine whether to expand a new action or select from existing ones
            unexplored = [a for a in legal if a not in node.expanded_actions]

            if unexplored and node.should_expand(self.pw_constant, self.pw_alpha):
                # Progressive widening suggests we should try a new action
                # Default to random action selection
                action = self.choose_unexplored(unexplored)
                node.expanded_actions.add(action)
            else:
                # Always have at least one expanded action
                # Use UCB1 to select among previously tried actions
                action = node.select_action(legal)

            # Apply the selected action and move to the next state
            Tarot.apply_action(state, action)
            path.append((node, action))
            Tarot.next(state)

        # Backpropagation phase: update statistics for all nodes in the path
        results = Tarot.returns(state)
        reward = results[self.player] if len(results) > self.player else 0.0

        for node, action in reversed(path):
            node.n_visits += 1
            node.n_action[action] += 1
            node.total_action_value[action] += reward

    def determinization(self, state: TarotState) -> TarotState:
        """
        Create a determinized version of the game state for simulation.

        In imperfect information games like Tarot, the agent doesn't know
        the complete state (e.g., other players' hands). Determinization
        samples a possible complete state that's consistent with the agent's
        observations, allowing MCTS to simulate forward.

        This method:
        1. Identifies cards that are known vs. unknown to the agent
        2. Randomly distributes unknown cards to other players
        3. Handles special cases like the taker's known cards from the chien

        Args:
            game: Original game state with hidden information

        Returns:
            Complete game state with sampled hidden information
        """
        determinized = Tarot.clone(state)

        # Collect all cards that are known to this player
        my_hand = state.hands[self.player]

        # Cards that have been played and are visible to everyone
        played_cards_set = state.know_cards.copy()

        # Cards currently visible in the ongoing trick
        current_trick_cards = set(
            card for card in state.trick if card != -1)

        # Find all cards that are unknown to this player
        unknown_cards = []
        for card_idx in range(Const.DECK_SIZE):
            card = card_idx + 1  # Tarot cards are 1-indexed
            if card not in [*my_hand, *played_cards_set, *current_trick_cards]:
                unknown_cards.append(card)

        # Randomly distribute unknown cards to other players
        random.shuffle(unknown_cards)
        card_index = 0

        # Special handling for garde sans/contre bids where chien cards aren't revealed
        if (determinized.taker_bid == Const.BID_GARDE_SANS
                or determinized.taker_bid == Const.BID_GARDE_CONTRE):
            # In these bids, chien cards count for the taker but aren't known
            # Remove chien-sized portion from unknown cards (they go to taker)
            unknown_cards = unknown_cards[Const.CHIEN_SIZE:]

        # Distribute cards to each other player
        for player in range(Const.NUM_PLAYERS):
            if player != self.player:
                original_hand_size = len(state.hands[player])
                determinized.hands[player] = []

                # Fill the remaining hand with random unknown cards
                cards_to_add = min(original_hand_size,
                                   len(unknown_cards) - card_index)
                for _ in range(cards_to_add):
                    if card_index < len(unknown_cards):
                        determinized.hands[player].append(
                            unknown_cards[card_index])
                        card_index += 1

        return determinized

    def info_key(self, game: TarotState) -> TarotInformationSet:
        """
        Create a unique key representing the player's information set.

        An information set captures everything this player can observe
        about the current game state. Two states with the same information
        set should be treated identically by the algorithm.

        The information set includes:
        - All played cards (public information)
        - This player's current hand
        - Current trick state
        - Current player turn
        - Who the taker is

        Args:
            game: Current game state

        Returns:
            Information set object that can be used as a dictionary key
        """
        information_set = []
        information_set += game.know_cards.copy()
        information_set += sorted(game.hands[self.player].copy())
        information_set += game.trick.copy()
        information_set += [game.current]
        information_set += [game.taker]
        return TarotInformationSet(information_set)
