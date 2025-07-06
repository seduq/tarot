from enum import Enum
import random
import math
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


from .tarot import Tarot
from .constants import Phase
from . import constants as Const
from .cards import Card


@dataclass
class TarotInformationSet:
    """Information set for French Tarot game"""
    player: int
    hand: List[int]  # Cards in player's hand
    played_cards: List[int]  # Cards played by each player (-1 if unknown)
    current_trick: List[int]  # Current trick cards
    played_tricks: List[int]  # Number of tricks won by each player
    phase: Phase
    taker: int  # Who is the taker
    current_player: int  # Whose turn it is

    def __hash__(self):
        # Use more stable hash that doesn't depend on exact trick order
        # Only consider essential information for decision-making
        return hash((
            self.player,
            tuple(sorted(self.hand)),
            tuple(self.played_cards),
            tuple(self.current_trick),
            tuple(self.played_tricks),
            self.phase,
            self.taker,
            self.current_player
        ))

    def __eq__(self, other):
        if not isinstance(other, TarotInformationSet):
            return False
        return (
            self.player == other.player and
            sorted(self.hand) == sorted(other.hand) and
            self.played_cards == other.played_cards and
            self.current_trick == other.current_trick and
            self.played_tricks == other.played_tricks and
            self.phase == other.phase and
            self.taker == other.taker and
            self.current_player == other.current_player
        )


class TarotISMCTSNode:
    """IS-MCTS Node for French Tarot"""

    def __init__(self, information_set: TarotInformationSet, parent=None, action=None):
        self.information_set = information_set
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0

        # RAVE values
        self.rave_visits = defaultdict(int)
        self.rave_rewards = defaultdict(float)

        # Progressive widening
        self.tried_actions = set()


class ISMCTSStrategy(Enum):
    """Enum for IS-MCTS strategies"""
    PER_ACTION = "per_action"  # Determinize per action
    PER_TRICK = "per_trick"  # Determinize per trick


class TarotISMCTSAgent:
    """IS-MCTS Agent for French Tarot game"""

    Strategy = ISMCTSStrategy  # Expose strategy enum for external use

    def __init__(self,
                 player_id: int,
                 iterations=1000,
                 exploration_constant=math.sqrt(2),
                 pw_alpha=0.5,
                 pw_constant=2.0,
                 rave_constant=200,
                 num_determinizations=10,
                 strategy: ISMCTSStrategy = ISMCTSStrategy.PER_TRICK):

        self.player_id = player_id
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        # Maps information sets to nodes
        self.tree: Dict[TarotInformationSet, TarotISMCTSNode] = {}

        # Progressive widening parameters
        self.pw_alpha = pw_alpha
        self.pw_constant = pw_constant

        # RAVE parameters
        self.rave_constant = rave_constant

        # Multiple determinization parameters
        self.num_determinizations = num_determinizations
        self.determinization_strategy = strategy

        # Per-trick determinization state
        self.current_trick_determinizations = []
        self.last_trick_state = None

        # Node counting metrics
        self.total_nodes_created = 0
        self.total_simulations_run = 0
        self.nodes_created_this_decision = 0
        self.simulations_this_decision = 0

        # Tree counting metrics for multiple determinization
        self.total_trees_created = 0
        self.trees_created_this_decision = 0

    def _get_key(self, game_state: Tarot) -> str:
        """Get a unique key for the tricks game state"""
        return str(game_state.played_tricks) + str(game_state.trick)

    def get_action(self, game_state: Tarot) -> int:
        """Get the best action using IS-MCTS with multiple determinization"""
        # Reset per-decision counters
        self.nodes_created_this_decision = 0
        self.simulations_this_decision = 0
        self.trees_created_this_decision = 0

        # Only work in TRICK phase, skip Const.TRICK_FINISHED
        if game_state.phase != Phase.TRICK:
            # For non-trick phases, return a random legal action or handle Const.TRICK_FINISHED
            legal_actions = game_state.legal_actions()
            return random.choice(legal_actions)

        if self.determinization_strategy == ISMCTSStrategy.PER_TRICK:
            return self._get_action_per_trick(game_state)
        else:
            return self._get_action_simple(game_state)

    def _get_action_simple(self, game_state: Tarot) -> int:
        """Get best action using multiple determinizations"""
        action_votes = defaultdict(int)
        action_scores = defaultdict(float)

        # Run MCTS on multiple possible worlds
        iterations_per_det = max(1, self.iterations //
                                 self.num_determinizations)

        for det_idx in range(self.num_determinizations):
            # Create a possible complete game state
            determinized_state = self._determinize_game_state(game_state)

            # Run MCTS on this determinized state
            best_action, action_quality = self._search(
                determinized_state, iterations_per_det)

            if best_action is not None:
                action_votes[best_action] += 1
                action_scores[best_action] += action_quality

        # Choose action with most votes, tie-break by quality
        if action_votes:
            best_action = max(action_votes.keys(),
                              key=lambda a: (action_votes[a], action_scores[a]))
            return best_action

        # Fallback to random action
        legal_actions = game_state.legal_actions()
        return random.choice(legal_actions)

    def _get_action_per_trick(self, game_state: Tarot) -> int:
        """Get best action using per-trick determinization strategy"""
        # Check if we need to create new determinizations for this trick
        if self._should_redeterminize(game_state):
            self._create_trick_determinizations(game_state)

        # Use existing trick determinizations
        action_votes = defaultdict(int)
        action_scores = defaultdict(float)

        iterations_per_det = max(1, self.iterations //
                                 len(self.current_trick_determinizations))

        for determinized_state in self.current_trick_determinizations:
            # Update the determinized state to match current trick state
            updated_state = self._update_determinized_state(
                determinized_state, game_state)

            # Run MCTS on this determinized state
            best_action, action_quality = self._search(
                updated_state, iterations_per_det)

            if best_action is not None:
                action_votes[best_action] += 1
                action_scores[best_action] += action_quality

        # Choose action with most votes, tie-break by quality
        if action_votes:
            best_action = max(action_votes.keys(),
                              key=lambda a: (action_votes[a], action_scores[a]))
            return best_action

        # Fallback to random action
        legal_actions = game_state.legal_actions()
        return random.choice(legal_actions)

    def _should_redeterminize(self, game_state: Tarot) -> bool:
        """Check if we should create new determinizations for French Tarot"""
        # Redeterminize if:
        # 1. No determinizations exist yet
        # 2. We're starting a new trick (trick is empty or all -1)
        # 3. Game state has changed significantly from last determinization

        if not self.current_trick_determinizations:
            return True

        # Check if this is the same trick state as when we last determinized
        current_trick_key = self._get_key(game_state)
        if self.last_trick_state != current_trick_key:
            return True

        return False

    def _create_trick_determinizations(self, game_state: Tarot):
        """Create determinizations for the current trick"""
        self.current_trick_determinizations = []

        for _ in range(self.num_determinizations):
            det_state = self._determinize_game_state(game_state)
            self.current_trick_determinizations.append(det_state)

        # Remember this trick state
        self.last_trick_state = self._get_key(game_state)

    def _update_determinized_state(self, determinized_state: Tarot, current_state: Tarot) -> Tarot:
        """Update a determinized state to match the current trick progress"""
        # Create a simple copy of the determinized state
        updated_state = Tarot()

        # Copy all the basic state from determinized state
        updated_state.phase = current_state.phase
        updated_state.current = current_state.current
        updated_state.taker = determinized_state.taker
        updated_state.taker_bid = determinized_state.taker_bid
        updated_state.bids = determinized_state.bids.copy()
        updated_state.chien = determinized_state.chien.copy()
        updated_state.discard = determinized_state.discard.copy()

        # Copy declarations from determinized state
        updated_state.chelem_declared_taker = determinized_state.chelem_declared_taker
        updated_state.chelem_declared_defenders = determinized_state.chelem_declared_defenders
        updated_state.poignee_declared_taker = determinized_state.poignee_declared_taker
        updated_state.poignee_declared_defenders = determinized_state.poignee_declared_defenders

        # Update with current trick information
        if current_state.phase == Phase.TRICK:
            updated_state.trick = [-1] * Const.NUM_PLAYERS
            # Copy existing cards from current trick
            for i in range(min(len(current_state.trick), Const.NUM_PLAYERS)):
                updated_state.trick[i] = current_state.trick[i]
        else:
            updated_state.trick = current_state.trick.copy() if current_state.trick else []

        updated_state.tricks = current_state.tricks.copy()
        updated_state.played_cards = current_state.played_cards.copy()

        # Start with determinized hands
        updated_state.hands = [hand.copy()
                               for hand in determinized_state.hands]

        # Remove cards that have been played in this trick from hands
        # In the trick array, trick[player] = card played by that player
        for player, card in enumerate(current_state.trick):
            if card != -1:  # Card has been played by this player
                if card in updated_state.hands[player]:
                    updated_state.hands[player].remove(card)

        # Ensure my hand matches exactly
        updated_state.hands[self.player_id] = (current_state.hands[self.player_id]
                                               .copy())

        return updated_state

    def _get_information_set(self, game_state: Tarot) -> TarotInformationSet:
        """Create information set for current player in French Tarot"""
        return TarotInformationSet(
            player=self.player_id,
            hand=game_state.hands[self.player_id].copy(),
            played_cards=game_state.played_cards.copy(),
            current_trick=game_state.trick.copy(),
            played_tricks=game_state.played_tricks.copy(),
            phase=game_state.phase,
            taker=game_state.taker,
            current_player=game_state.current
        )

    def _simulate(self, game_state: Tarot):
        """Run one MCTS simulation"""
        path = []
        current_state = game_state.clone()

        # Selection and expansion phase
        while not current_state.is_terminal():
            if current_state.current == self.player_id:
                # This is our decision node
                info_set = self._get_information_set(current_state)

                if info_set not in self.tree:
                    # Create new node and stop selection
                    self.tree[info_set] = TarotISMCTSNode(info_set)
                    self.total_nodes_created += 1
                    self.nodes_created_this_decision += 1
                    break

                node = self.tree[info_set]
                action = self._select_action(current_state, node)
                path.append((node, action))

                if action not in node.children:
                    # Expansion - create child node
                    current_state.apply_action(action)
                    current_state.next()
                    new_info_set = self._get_information_set(current_state)
                    node.children[action] = TarotISMCTSNode(
                        new_info_set, node, action)
                    self.total_nodes_created += 1
                    self.nodes_created_this_decision += 1
                    break
                else:
                    # Continue selection
                    current_state.apply_action(action)
                    current_state.next()
            else:
                # Opponent's turn - simulate random action
                legal_actions = current_state.legal_actions()
                if not legal_actions:
                    break
                action = random.choice(legal_actions)
                current_state.apply_action(action)
                current_state.next()

        # Rollout phase
        final_state = self._rollout(current_state)
        reward = self._get_reward(final_state)

        # Backpropagation
        self._backpropagate(path, reward)

        # Count simulation
        self.total_simulations_run += 1
        self.simulations_this_decision += 1

    def _select_action(self, game_state: Tarot, node: TarotISMCTSNode) -> int:
        """Select action using UCB1 with RAVE and progressive widening"""
        legal_actions = game_state.legal_actions()

        # Progressive widening: limit number of tried actions
        max_actions = max(
            1, int(self.pw_constant * (node.visits ** self.pw_alpha)))

        # Add new actions if we haven't reached the limit
        if len(node.tried_actions) < max_actions:
            untried = [a for a in legal_actions if a not in node.tried_actions]
            if untried:
                action = random.choice(untried)
                node.tried_actions.add(action)
                return action

        # Select among tried actions using UCB1 + RAVE
        best_value = float('-inf')
        best_action = None

        for action in node.tried_actions:
            if action in legal_actions:
                value = self._calculate_ucb_rave_value(node, action)
                if value > best_value:
                    best_value = value
                    best_action = action

        return best_action or random.choice(legal_actions)

    def _calculate_ucb_rave_value(self, node: TarotISMCTSNode, action: int) -> float:
        """Calculate UCB1 + RAVE value for an action"""
        if action not in node.children:
            return float('inf')

        child = node.children[action]

        # UCB1 value
        if child.visits == 0:
            ucb_value = float('inf')
        else:
            exploitation = child.total_reward / child.visits
            exploration = self.exploration_constant * math.sqrt(
                math.log(node.visits) / child.visits)
            ucb_value = exploitation + exploration

        # RAVE value
        if node.rave_visits[action] == 0:
            rave_value = 0.5  # Default value
        else:
            rave_value = node.rave_rewards[action] / node.rave_visits[action]

        # Combine UCB1 and RAVE
        beta = math.sqrt(self.rave_constant /
                         (3 * node.visits + self.rave_constant))
        combined_value = (1 - beta) * ucb_value + beta * rave_value

        return combined_value

    def _rollout(self, game_state: Tarot) -> Tarot:
        """Random rollout to terminal state"""
        # Create a deep copy for rollout
        current_state = game_state.clone()

        # Play randomly until terminal state
        while not current_state.is_terminal():
            legal_actions = current_state.legal_actions()
            if not legal_actions:
                break  # Defensive: no legal actions, cannot proceed
            action = random.choice(legal_actions)
            current_state.apply_action(action)
            current_state.next()
        return current_state

    def _backpropagate(self, path: List[Tuple[TarotISMCTSNode, int]], reward: float):
        """Backpropagate reward through the path"""
        for node, action in path:
            node.visits += 1
            node.total_reward += reward

            if action in node.children:
                child = node.children[action]
                child.visits += 1
                child.total_reward += reward

            # Update RAVE values for all actions in the path
            for _, rave_action in path:
                node.rave_visits[rave_action] += 1
                node.rave_rewards[rave_action] += reward

    def _determinize_game_state(self, game_state: Tarot) -> Tarot:
        """Create one possible complete game state from current information"""
        # Create a simple copy without using the complex tensor system
        determinized_state = game_state.clone()

        # What we know for certain
        my_hand = game_state.hands[self.player_id]
        played_cards_set = set(
            Card.from_idx(card) for card, player
            in enumerate(game_state.played_cards) if player != -1)
        current_trick_cards = set(
            card for card in game_state.trick if card != -1)
        taker_know_cards = game_state.taker_chien_hand

        # What we need to guess: other players' hands
        # Import Card class for card conversion

        unknown_cards = []

        # Generate all possible cards and check which are unknown
        for card_idx in range(Const.DECK_SIZE):
            card = Card.from_idx(card_idx)
            if (card not in my_hand and
                card not in played_cards_set and
                    card not in current_trick_cards and
                    card not in taker_know_cards):
                unknown_cards.append(card)

        # Randomly distribute unknown cards to other players
        random.shuffle(unknown_cards)
        card_index = 0

        # Clear other players' hands and redistribute
        # If the taker has a garde sans or garde contre bid,
        # we need to adjust the cards to account for the chien
        if (determinized_state.taker_bid == Const.BID_GARDE_SANS
                or determinized_state.taker_bid == Const.BID_GARDE_CONTRE):
            # In garde sans/contre, chien cards are not revealed but count for taker
            # Remove chien-sized portion from unknown cards
            unknown_cards = unknown_cards[Const.CHIEN_SIZE:]

        for player in range(Const.NUM_PLAYERS):
            if player != self.player_id:
                original_hand_size = len(game_state.hands[player])
                # Check if this player is the taker
                # Add cards that are known to the taker
                # Except the played cards
                if determinized_state.taker == player:
                    # Taker knows chien cards (except in garde sans/contre)
                    known_cards = []
                    if determinized_state.taker_bid < Const.BID_GARDE_SANS:
                        known_cards = list(
                            set(taker_know_cards).difference(played_cards_set))
                    determinized_state.hands[player] = known_cards
                    original_hand_size = max(
                        0, original_hand_size - len(known_cards))
                else:
                    determinized_state.hands[player] = []

                # Give this player the right number of remaining cards
                cards_to_add = min(original_hand_size, len(
                    unknown_cards) - card_index)
                for _ in range(cards_to_add):
                    if card_index < len(unknown_cards):
                        determinized_state.hands[player].append(
                            unknown_cards[card_index])
                        card_index += 1

        return determinized_state

    def _search(self, determinized_state: Tarot, iterations: int) -> Tuple[Optional[int], float]:
        """Run MCTS on a single determinized state"""
        # Use a separate tree for this determinization to avoid contamination
        temp_tree: Dict[TarotInformationSet, TarotISMCTSNode] = {}
        original_tree = self.tree
        self.tree = temp_tree

        # Track nodes created in this determinization
        nodes_before = self.total_nodes_created
        simulations_before = self.total_simulations_run

        # Count this tree creation
        self.total_trees_created += 1
        self.trees_created_this_decision += 1

        try:
            info_set = self._get_information_set(determinized_state)

            # Run MCTS iterations
            for _ in range(iterations):
                self._simulate(determinized_state)

            # Update per-decision counters with nodes created in this determinization
            nodes_created_here = self.total_nodes_created - nodes_before
            simulations_run_here = self.total_simulations_run - simulations_before

            self.nodes_created_this_decision += nodes_created_here
            self.simulations_this_decision += simulations_run_here

            # Get best action and its quality
            if info_set in self.tree:
                node = self.tree[info_set]
                if node.children:
                    best_action = max(node.children.keys(),
                                      key=lambda a: node.children[a].visits)

                    # Calculate action quality (average reward)
                    best_child = node.children[best_action]
                    action_quality = (best_child.total_reward / best_child.visits
                                      if best_child.visits > 0 else 0.0)

                    return best_action, action_quality

            return None, 0.0

        finally:
            # Restore original tree
            self.tree = original_tree

    def _get_reward(self, final_state: Tarot) -> float:
        """Get reward for the player from the final game state"""
        if not final_state.is_terminal():
            return 0.0

        # Get the final scores
        returns = final_state.returns()

        # Return the score for this player
        if len(returns) > self.player_id:
            return returns[self.player_id]
        else:
            return 0.0

    def get_tree_stats(self) -> Dict[str, int]:
        """Get current tree statistics"""
        # For multiple determinization strategies, we don't maintain a persistent tree
        # Return accumulated statistics from all determinizations
        return {
            'total_nodes': 0,  # No persistent tree
            'nodes_created_this_decision': self.nodes_created_this_decision,
            'total_nodes_created': self.total_nodes_created,
            'total_visits': 0,  # No persistent tree visits
            'total_simulations': self.total_simulations_run,
            'simulations_this_decision': self.simulations_this_decision,
            'max_depth': 0,  # No persistent tree depth
            'total_trees_created': self.total_trees_created,
            'trees_created_this_decision': self.trees_created_this_decision
        }

    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the tree"""
        if not self.tree:
            return 0

        max_depth = 0
        for node in self.tree.values():
            depth = self._get_node_depth(node)
            max_depth = max(max_depth, depth)

        return max_depth

    def _get_node_depth(self, node: TarotISMCTSNode) -> int:
        """Get depth of a specific node"""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def reset_tree(self):
        """Reset the tree (useful for new games)"""
        self.tree = {}
        self.total_nodes_created = 0
        self.total_simulations_run = 0
        self.nodes_created_this_decision = 0
        self.simulations_this_decision = 0
        self.total_trees_created = 0
        self.trees_created_this_decision = 0
