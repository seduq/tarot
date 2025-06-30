import random
import math
from typing import Optional, Dict, Any, List
from .game import TarotGameState
from . import constants as Const, Phase
from .utils import Utils
from .cards import Card


class RIS_MCTS_Node:
    """
    Node for the RIS-MCTS search tree.
    Stores statistics and tree structure.
    """

    def __init__(self, player: int, parent: Optional['RIS_MCTS_Node'] = None, action: Optional[int] = None):
        self.visits: int = 0
        self.wins: float = 0.0
        self.children: Dict[int, 'RIS_MCTS_Node'] = {}
        self.parent: Optional['RIS_MCTS_Node'] = parent
        self.action: Optional[int] = action
        self.player: int = player


class RIS_MCTS:
    """
    RIS-MCTS for the French Tarot game.
    Implements Monte Carlo Tree Search with re-determinization for imperfect information games.
    """

    def __init__(self):
        self.tree: Dict[str, RIS_MCTS_Node] = {}
        self.exploration_constant: float = math.sqrt(2)
        self.num_players: int = Const.NUM_PLAYERS

    def search(self, initial_state: TarotGameState, player: int, iterations: int = 1000) -> Optional[int]:
        """
        Run RIS-MCTS from an initial state for a given player.
        Returns the best action found after the given number of iterations.
        If the state is not in the trick phase, returns the default legal actions.
        """
        if initial_state.phase != Phase.TRICK:
            legal_actions = self._get_legal_actions(initial_state)
            return random.choice(legal_actions) if legal_actions else None
        root_key = self._state_to_key(initial_state, player)
        if root_key not in self.tree:
            self.tree[root_key] = RIS_MCTS_Node(player)
        root_node = self.tree[root_key]
        best_action_votes = {}
        for _ in range(iterations):
            current_state = self._determinize_state(initial_state, player)
            current_player = player
            current_key = root_key
            node = root_node
            path = [node]
            # Selection
            while node.children:
                legal_actions = self._get_legal_actions(current_state)
                unexplored = [
                    a for a in legal_actions if a not in node.children]
                if unexplored:
                    break
                best_action = self._select_best_child(node)
                node = node.children[best_action]
                current_key = self._state_to_key(current_state, current_player)
                path.append(node)
                current_state = self._apply_action(
                    current_state.clone(), best_action)
                current_player = current_state.current_player()
            # Expansion
            if not self._is_terminal(current_state):
                legal_actions = self._get_legal_actions(current_state)
                unexplored = [
                    a for a in legal_actions if a not in node.children]
                if unexplored:
                    action = random.choice(unexplored)
                    new_state = self._apply_action(
                        current_state.clone(), action)
                    next_player = new_state.current_player()
                    new_node = RIS_MCTS_Node(
                        next_player, parent=node, action=action)
                    node.children[action] = new_node
                    key = self._state_to_key(new_state, current_player)
                    self.tree[key] = new_node
                    path.append(new_node)
                    current_state = new_state
                    current_player = next_player
                    node = new_node
            # Simulation
            simulation_state = current_state.clone()
            while not self._is_terminal(simulation_state):
                legal_actions = self._get_legal_actions(simulation_state)
                if not legal_actions:
                    break
                action = random.choice(legal_actions)
                simulation_state = self._apply_action(
                    simulation_state.clone(), action)
            # Backpropagation
            result = self._evaluate_terminal_state(simulation_state, player)
            for n in reversed(path):
                n.visits += 1
                if n.player == player:
                    n.wins += result
                else:
                    n.wins += (1.0 - result)
            # Voting
            if root_node.children:
                best_child_action = max(
                    root_node.children.keys(),
                    key=lambda a: root_node.children[a].visits
                )
                best_action_votes[best_child_action] = best_action_votes.get(
                    best_child_action, 0) + 1
        if not best_action_votes:
            legal_actions = self._get_legal_actions(initial_state)
            return random.choice(legal_actions) if legal_actions else None
        return max(best_action_votes.keys(), key=lambda a: best_action_votes[a])

    def _determinize_state(self, state: TarotGameState, player: int) -> TarotGameState:
        """
        Generate a determinization of the state for the player (sample opponents' hands).
        """
        if state.current_player() < 0:
            return state
        determinized_state = state.tensor_player(player)
        know_cards = Utils.get_mask(determinized_state, 'known_cards')
        unknown_cards = [card for card,
                         _player in enumerate(know_cards) if _player < 0]
        new_state = state.clone()
        for player_id in range(Const.NUM_PLAYERS):
            if player_id != player:
                current_hand_size = Const.HAND_SIZE - \
                    len([card for card in know_cards if card == player_id])
                if current_hand_size > 0 and len(unknown_cards) >= current_hand_size:
                    sampled_hand = random.sample(
                        unknown_cards, current_hand_size)
                    hand = [Card.from_idx(card) for card in sampled_hand]
                    new_state.hands[player_id] = hand
                    unknown_cards = [
                        c for c in unknown_cards if c not in sampled_hand]
        return new_state

    def _state_to_key(self, state: TarotGameState, player: int) -> str:
        """
        Generate a hashable key for the state, considering the player's view.
        """
        if player == state.current_player():
            tensor = state.tensor_player(player)
        else:
            tensor = state.tensor()
        return str(tensor)

    def _get_legal_actions(self, state: TarotGameState) -> List[int]:
        """
        Return the legal actions for the state.
        """
        return state.legal_actions()

    def _apply_action(self, state: TarotGameState, action: int) -> TarotGameState:
        """
        Apply an action to the state (modifies in-place and returns the state).
        """
        state.apply_action(action)
        return state

    def _is_terminal(self, state: TarotGameState) -> bool:
        """
        Check if the state is terminal.
        """
        return state.is_terminal()

    def _evaluate_terminal_state(self, state: TarotGameState, original_player: int) -> float:
        """
        Evaluate the terminal state from the original player's perspective.
        Returns 1 for win, 0 for loss, 0.5 for neutral/draw.
        """
        returns = state.returns()
        if len(returns) > original_player:
            return 0 if returns[original_player] < 0 else 1
        return 0.5

    def _select_best_child(self, node: RIS_MCTS_Node) -> int:
        """
        Select the best child of a node using the UCB1 criterion.
        """
        best_score = float('-inf')
        best_action = None
        for action, child in node.children.items():
            if child.visits == 0:
                return action  # Prioritize unvisited children
            exploitation = child.wins / child.visits
            exploration = self.exploration_constant * \
                math.sqrt(math.log(node.visits) / child.visits)
            ucb1_score = exploitation + exploration
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_action = action
        if best_action is None:
            raise ValueError("No best action found, check the tree structure.")
        return int(best_action)
