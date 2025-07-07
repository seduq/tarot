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
    information_set: List[int]

    def __hash__(self):
        return hash(tuple(self.information_set))

    def __eq__(self, other):
        if not isinstance(other, TarotInformationSet):
            return False
        return self.information_set == other.information_set


class TarotISMCTSNode:
    def __init__(self,
                 info_key: TarotInformationSet,
                 parent: Optional['TarotISMCTSNode'] = None):
        self.info_key = info_key
        self.parent = parent
        self.children = dict()
        self.n_visits = 0
        self.total_value = 0.0
        self.n_action = defaultdict(int)
        self.total_action_value = defaultdict(float)
        self.expanded_actions = set()
        self.miss = 0

    def ucb_score(self, action, c=1.4):
        if self.n_action[action] == 0:
            return float('inf')
        q = self.total_action_value[action] / self.n_action[action]
        u = c * math.sqrt(math.log(self.n_visits + 1) / self.n_action[action])
        return q + u

    def select_action(self, legal_actions: List[int]) -> int:
        valid_actions = self.expanded_actions.intersection(legal_actions)
        if not valid_actions:
            action = random.choice(legal_actions)
            self.expanded_actions.add(action)
            self.n_action[action] = 0
            self.miss += 1
            return action
        return max(valid_actions, key=lambda a: self.ucb_score(a))

    def should_expand(self, k, alpha):
        if self.n_visits == 0:
            return True
        return len(self.expanded_actions) < k * (self.n_visits ** alpha)


class TarotISMCTSAgent:
    """IS-MCTS Agent for French Tarot game"""

    def __init__(self,
                 player: int,
                 iterations=300,
                 exploration_constant=1.4,
                 pw_alpha=0.5,
                 pw_constant=2.0,):

        self.player = player
        self.iterations = iterations
        self.exploration_constant = exploration_constant

        # Progressive widening parameters
        self.pw_alpha = pw_alpha
        self.pw_constant = pw_constant

        # Node counting metrics
        self.total_nodes_created = 0
        self.total_simulations_run = 0
        self.nodes_created_this_decision = 0
        self.simulations_this_decision = 0
        self.tree: Dict[TarotInformationSet, TarotISMCTSNode] = {}

    def run(self, game: Tarot) -> int:
        root_info = self.info_key(game)
        if root_info not in self.tree:
            self.tree[root_info] = TarotISMCTSNode(root_info)
        root_node = self.tree[root_info]

        for i in range(self.iterations):
            determinized_game = self.determinization(game)
            self.simulate(determinized_game)

        return max(root_node.expanded_actions, key=lambda a: root_node.n_action[a])

    def simulate(self, game: Tarot):
        path: List[Tuple[TarotISMCTSNode, int]] = []

        while not game.is_terminal():
            info = self.info_key(game)
            legal = game.legal_actions()
            if not legal:
                break
            action = random.choice(legal)

            if info not in self.tree:
                self.tree[info] = TarotISMCTSNode(info)

            node = self.tree[info]
            unexplored = [a for a in legal if a not in node.expanded_actions]
            if unexplored and node.should_expand(self.pw_constant, self.pw_alpha):
                action = random.choice(unexplored)
                node.expanded_actions.add(action)
            elif node.expanded_actions:
                action = node.select_action(legal)

            game.apply_action(action)
            path.append((node, action))
            game.next()

        # Backpropagation
        results = game.returns()
        reward = results[self.player] if len(results) > self.player else 0.0
        for node, action in reversed(path):
            node.n_visits += 1
            node.n_action[action] += 1
            node.total_action_value[action] += reward

    def determinization(self, game: Tarot) -> Tarot:
        """Re-determinize the hidden state for the perspective of player_id."""
        determinized = game.clone(self.player)

        # What we know for certain
        my_hand = game.hands[self.player]
        played_cards_set = set(
            Card.from_idx(card) for card, player
            in enumerate(game.played_cards) if player != -1)
        current_trick_cards = set(
            card for card in game.trick if card != -1)
        taker_know_cards = game.taker_chien_hand

        # Generate all possible cards and check which are unknown
        unknown_cards = []
        for card_idx in range(Const.DECK_SIZE):
            card = Card.from_idx(card_idx)
            if card not in [*my_hand, *played_cards_set, *current_trick_cards, *taker_know_cards]:
                unknown_cards.append(card)

        # Randomly distribute unknown cards to other players
        random.shuffle(unknown_cards)
        card_index = 0

        # Clear other players' hands and redistribute
        # If the taker has a garde sans or garde contre bid,
        # we need to adjust the cards to account for the chien
        if (determinized.taker_bid == Const.BID_GARDE_SANS
                or determinized.taker_bid == Const.BID_GARDE_CONTRE):
            # In garde sans/contre, chien cards are not revealed but count for taker
            # Remove chien-sized portion from unknown cards
            unknown_cards = unknown_cards[Const.CHIEN_SIZE:]

        for player in range(Const.NUM_PLAYERS):
            if player != self.player:
                # Check if this player is the taker
                # Add cards that are known to the taker
                # Except the played cards
                determinized.hands[player] = []
                if determinized.taker == player and determinized.taker_chien_hand:
                    determinized.hands[player] = (
                        determinized.taker_chien_hand.copy())
                    original_hand_size = len(game.hands[player])

                # Give this player the right number of remaining cards
                original_hand_size = len(game.hands[player])
                cards_to_add = min(original_hand_size,
                                   len(unknown_cards) - card_index)
                for _ in range(cards_to_add):
                    if card_index < len(unknown_cards):
                        determinized.hands[player].append(
                            unknown_cards[card_index])
                        card_index += 1

        return determinized

    def info_key(self, game: Tarot) -> TarotInformationSet:
        """Returns a hashable key representing the player's information set."""
        information_set = []
        information_set += game.played_cards.copy()
        information_set += sorted(game.hands[self.player].copy())
        information_set += game.trick.copy()
        information_set += [game.current]
        information_set += [game.taker]
        return TarotInformationSet(information_set)

    def retrieve_node_statistics(self) -> Tuple[int, int]:
        """Returns the total number of nodes created."""
        def count_nodes(node: TarotISMCTSNode) -> Tuple[int, int]:
            total_count = 1
            misses = node.miss
            for child in node.children.values():
                _count, misses = count_nodes(child)
                total_count += _count
                misses += misses
            return total_count, misses
        total = 0
        misses = 0
        for node in self.tree.values():
            _count, _misses = count_nodes(node)
            total += _count
            misses += _misses
        return total, misses
