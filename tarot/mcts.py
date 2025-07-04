from enum import Enum
import random
import math
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


from .tarot import Tarot
from .constants import Phase
from . import constants as Const, Phase
from .cards import Card


@dataclass
class TarotInformationSet:
    """Information set for French Tarot game"""
    player: int
    hand: List[int]  # Cards in player's hand
    played_cards: List[int]  # Cards played by each player (-1 if unknown)
    current_trick: List[int]  # Current trick cards
    tricks_won: List[int]  # Number of tricks won by each player
    phase: Phase
    taker: int  # Who is the taker
    current_player: int  # Whose turn it is

    def __hash__(self):
        return hash((
            self.player,
            tuple(sorted(self.hand)),
            tuple(self.played_cards),
            tuple(self.current_trick),
            tuple(self.tricks_won),
            self.phase,
            self.taker,
            self.current_player
        ))

    def __eq__(self, other):
        return (self.player == other.player and
                sorted(self.hand) == sorted(other.hand) and
                self.played_cards == other.played_cards and
                self.current_trick == other.current_trick and
                self.tricks_won == other.tricks_won and
                self.phase == other.phase and
                self.taker == other.taker and
                self.current_player == other.current_player)


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
                 multiple_determinization=True,
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
        self.use_multiple_determinization = multiple_determinization
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
            if game_state.phase == Phase.TRICK_FINISHED:
                return Const.TRICK_FINISHED
            elif legal_actions:
                return random.choice(legal_actions)
            else:
                return Const.TRICK_FINISHED

        if self.use_multiple_determinization:
            if self.determinization_strategy == ISMCTSStrategy.PER_TRICK:
                return self._get_action_per_trick_determinization(game_state)
            else:
                return self._get_action_multiple_determinization(game_state)
        else:
            return self._get_action_single_determinization(game_state)

    def _get_action_multiple_determinization(self, game_state: Tarot) -> int:
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
            best_action, action_quality = self._run_single_determinization_mcts(
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
        return random.choice(legal_actions) if legal_actions else Const.TRICK_FINISHED

    def _get_action_single_determinization(self, game_state: Tarot) -> int:
        """Original single determinization approach"""
        info_set = self._get_information_set(game_state)

        for _ in range(self.iterations):
            self._simulate(game_state, info_set)

        # Choose action with highest visit count
        if info_set in self.tree:
            node = self.tree[info_set]
            if node.children:
                best_action = max(node.children.keys(),
                                  key=lambda a: node.children[a].visits)
                return best_action

        # Fallback to random action
        legal_actions = game_state.legal_actions()
        return random.choice(legal_actions) if legal_actions else Const.TRICK_FINISHED

    def _get_action_per_trick_determinization(self, game_state: Tarot) -> int:
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
            best_action, action_quality = self._run_single_determinization_mcts(
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
        return random.choice(legal_actions) if legal_actions else Const.TRICK_FINISHED

    def _should_redeterminize(self, game_state: Tarot) -> bool:
        """Check if we should create new determinizations for French Tarot"""
        # Redeterminize if:
        # 1. No determinizations exist yet
        # 2. We're starting a new trick (trick is empty)
        # 3. Game state has changed significantly from last determinization

        if not self.current_trick_determinizations:
            return True

        if len(game_state.trick) == 0:
            return True

        # Check if this is the same trick state as when we last determinized
        current_trick_key = (len(game_state.tricks), len(game_state.trick))
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
        self.last_trick_state = (len(game_state.tricks), len(game_state.trick))

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
        for i, card in enumerate(current_state.trick):
            if card != -1:  # Card has been played
                player_who_played = i
                if card in updated_state.hands[player_who_played]:
                    updated_state.hands[player_who_played].remove(card)

        # Ensure my hand matches exactly
        updated_state.hands[self.player_id] = current_state.hands[self.player_id].copy(
        )

        return updated_state

    def _get_information_set(self, game_state: Tarot) -> TarotInformationSet:
        """Create information set for current player in French Tarot"""
        return TarotInformationSet(
            player=self.player_id,
            hand=game_state.hands[self.player_id].copy(),
            played_cards=game_state.played_cards.copy(),
            current_trick=game_state.trick.copy(),
            tricks_won=[len([t for p, t in game_state.tricks if p == i])
                        for i in range(Const.NUM_PLAYERS)],
            phase=game_state.phase,
            taker=game_state.taker,
            current_player=game_state.current
        )

    def _simulate(self, game_state: Tarot, root_info_set: TarotInformationSet):
        """Run one MCTS simulation"""
        path = []

        # Just do a quick simulation for the TRICK phase
        if game_state.phase == Phase.TRICK and game_state.current == self.player_id:
            info_set = self._get_information_set(game_state)

            if info_set not in self.tree:
                self.tree[info_set] = TarotISMCTSNode(info_set)
                self.total_nodes_created += 1
                self.nodes_created_this_decision += 1

            node = self.tree[info_set]
            action = self._select_action(game_state, node)
            path.append((node, action))

            if action not in node.children:
                # Expansion
                new_info_set = self._get_information_set(game_state)
                node.children[action] = TarotISMCTSNode(
                    new_info_set, node, action)
                self.total_nodes_created += 1
                self.nodes_created_this_decision += 1

        # Simple rollout and reward
        final_state = self._rollout(game_state)
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

        return best_action or random.choice(legal_actions) if legal_actions else Const.TRICK_FINISHED

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
        # Create a simple copy for rollout
        current_state = Tarot()

        # Copy basic state
        current_state.phase = game_state.phase
        current_state.current = game_state.current
        current_state.taker = game_state.taker
        current_state.taker_bid = game_state.taker_bid
        current_state.bids = game_state.bids.copy()
        current_state.trick = current_state.trick.copy(
        ) if game_state.trick else [-1] * Const.NUM_PLAYERS
        current_state.tricks = game_state.tricks.copy()
        current_state.played_cards = game_state.played_cards.copy()
        current_state.chien = game_state.chien.copy()
        current_state.discard = game_state.discard.copy()
        current_state.hands = [hand.copy() for hand in game_state.hands]

        # Copy declarations
        current_state.chelem_declared_taker = game_state.chelem_declared_taker
        current_state.chelem_declared_defenders = game_state.chelem_declared_defenders
        current_state.poignee_declared_taker = game_state.poignee_declared_taker
        current_state.poignee_declared_defenders = game_state.poignee_declared_defenders

        # Simple random rollout - just return current state for scoring
        # In a real implementation, you would continue playing randomly
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
        determinized_state = Tarot()

        # Copy basic game state
        determinized_state.phase = game_state.phase
        determinized_state.current = game_state.current
        determinized_state.taker = game_state.taker
        determinized_state.taker_bid = game_state.taker_bid
        determinized_state.bids = game_state.bids.copy()

        # Properly initialize trick - ensure it has the right size
        determinized_state.trick = game_state.trick.copy()

        determinized_state.tricks = game_state.tricks.copy()
        determinized_state.played_cards = game_state.played_cards.copy()
        determinized_state.chien = game_state.chien.copy()
        determinized_state.discard = game_state.discard.copy()
        determinized_state.hands = [[] for _ in range(
            Const.NUM_PLAYERS)]  # Will be filled later

        # Copy declarations
        determinized_state.chelem_declared_taker = game_state.chelem_declared_taker
        determinized_state.chelem_declared_defenders = game_state.chelem_declared_defenders
        determinized_state.poignee_declared_taker = game_state.poignee_declared_taker
        determinized_state.poignee_declared_defenders = game_state.poignee_declared_defenders

        # What we know for certain
        my_hand = game_state.hands[self.player_id].copy()
        played_cards_set = set(
            card for card in game_state.played_cards if card != -1)
        current_trick_cards = set(
            card for card in game_state.trick if card != -1)

        # What we need to guess: other players' hands
        # Import Card class for card conversion

        unknown_cards = []

        # Generate all possible cards and check which are unknown
        for card_idx in range(Const.DECK_SIZE):
            card = Card.from_idx(card_idx)
            if (card not in my_hand and
                card not in played_cards_set and
                    card not in current_trick_cards):
                unknown_cards.append(card)

        # Randomly distribute unknown cards to other players
        random.shuffle(unknown_cards)
        card_index = 0

        # Clear other players' hands and redistribute
        # If the taker has a garde sans or garde contre bid,
        # we need to adjust the cards to account for the chien
        if (determinized_state.taker_bid == Const.BID_GARDE_SANS
                or determinized_state.taker_bid == Const.BID_GARDE_CONTRE):
            unknown_cards = unknown_cards[Const.CHIEN_SIZE:]
        for player in range(Const.NUM_PLAYERS):
            if player != self.player_id:
                original_hand_size = len(game_state.hands[player])
                determinized_state.hands[player] = []

                # Give this player the right number of cards
                for _ in range(original_hand_size):
                    if card_index < len(unknown_cards):
                        determinized_state.hands[player].append(
                            unknown_cards[card_index])
                        card_index += 1
            else:
                # Keep my hand exactly as it is
                determinized_state.hands[player] = my_hand.copy()

        return determinized_state

    def _run_single_determinization_mcts(self, determinized_state: Tarot, iterations: int) -> Tuple[Optional[int], float]:
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
                self._simulate(determinized_state, info_set)

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
        if self.use_multiple_determinization:
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
        else:
            # For single determinization, use the actual tree statistics
            total_visits = sum(
                node.visits for node in self.tree.values()) if self.tree else 0
            max_depth = self._calculate_max_depth() if self.tree else 0

            return {
                'total_nodes': len(self.tree),
                'nodes_created_this_decision': self.nodes_created_this_decision,
                'total_nodes_created': self.total_nodes_created,
                'total_visits': total_visits,
                'total_simulations': self.total_simulations_run,
                'simulations_this_decision': self.simulations_this_decision,
                'max_depth': max_depth,
                'total_trees_created': 1 if self.tree else 0,  # Single persistent tree
                'trees_created_this_decision': 0  # No trees created per decision in single mode
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
