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
        self.invalid_action_count: int = 0

    def search(self, initial_state: TarotGameState, player: int, iterations: int = 100, verbose: bool = False) -> Optional[int]:
        """
        Run RIS-MCTS from an initial state for a given player.
        Returns the best action found after the given number of iterations.
        If the state is not in the trick phase, returns a random legal action.
        If verbose is True, prints debug information at each crucial step.
        """
        if initial_state.phase != Phase.TRICK:
            legal_actions = initial_state.legal_actions()
            if verbose:
                print("[RIS-MCTS] Not in trick phase. Returning random legal action.")
            return random.choice(legal_actions) if legal_actions else None
        root_key = str(initial_state.tensor_player(player))
        if root_key not in self.tree:
            self.tree[root_key] = RIS_MCTS_Node(player)
            if verbose:
                print(f"[RIS-MCTS] Created root node for key: {root_key}")
        root_node = self.tree[root_key]
        best_action_votes = {}
        for it in range(iterations):
            if verbose:
                print(f"\n[Iteration {it+1}/{iterations}] Starting new MCTS iteration.")
            # Determinization: sample hidden information for opponents
            current_state = self._determinize_state(initial_state, player)
            if verbose:
                print("[Determinization] Sampled determinization for player", player)
                for p in range(Const.NUM_PLAYERS):
                    print(f"Player {p} hand: {sorted(current_state.hands[p])} Initial hand: {sorted(initial_state.hands[p])}")
            current_player = player
            node = root_node
            path = [node]
            # --- SELECTION ---
            if verbose:
                print("[Selection] Starting selection phase.")
            while node.children:
                legal_actions = current_state.legal_actions()
                unexplored = [a for a in legal_actions if a not in node.children]
                if unexplored:
                    if verbose:
                        print(f"[Selection] Found unexplored actions: {unexplored}")
                    break
                # Select best child (UCB1)
                best_score = float('-inf')
                best_action = None
                for action, child in node.children.items():
                    if child.visits == 0:
                        best_action = action
                        if verbose:
                            print(f"[Selection] Child for action {action} has 0 visits. Selecting immediately.")
                        break
                    exploitation = child.wins / child.visits
                    exploration = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
                    ucb1_score = exploitation + exploration
                    if verbose:
                        print(f"[Selection] Action {action}: exploitation={exploitation:.3f}, exploration={exploration:.3f}, UCB1={ucb1_score:.3f}")
                    if ucb1_score > best_score:
                        best_score = ucb1_score
                        best_action = action
                if best_action is None:
                    raise ValueError("No best action found, check the tree structure.")
                
                if best_action in current_state.legal_actions():
                    node = node.children[best_action]
                    path.append(node)
                    current_state.apply_action(best_action)
                    current_player = current_state.current_player()
                    if verbose:
                        print(f"[Selection] Selected action {best_action}, moved to player {current_player}.")
                else:
                    if verbose:
                        print(f"[Selection] Best action {best_action} is not legal in the current state. Stopping selection.")
                        self.invalid_action_count += 1
                        node.children.pop(best_action, None)  # Remove invalid child
                    break
            # --- EXPANSION ---
            if not current_state.is_terminal():
                legal_actions = current_state.legal_actions()
                unexplored = [a for a in legal_actions if a not in node.children]
                if unexplored:
                    action = random.choice(unexplored)
                    new_state = current_state.clone()
                    new_state.apply_action(action)
                    next_player = new_state.current_player()
                    new_node = RIS_MCTS_Node(next_player, parent=node, action=action)
                    node.children[action] = new_node
                    key = str(new_state.tensor_player(next_player))
                    self.tree[key] = new_node
                    path.append(new_node)
                    current_state = new_state
                    current_player = next_player
                    node = new_node
                    if verbose:
                        print(f"[Expansion] Expanded with action {action}, created new node for player {next_player}.")
            # --- SIMULATION ---
            simulation_state = current_state.clone()
            if verbose:
                print("[Simulation] Starting simulation from expanded/leaf node.")
            sim_steps = 0
            while not simulation_state.is_terminal():
                legal_actions = simulation_state.legal_actions()
                if not legal_actions:
                    if verbose:
                        print("[Simulation] No legal actions left, breaking simulation loop.")
                    break
                action = random.choice(legal_actions)
                simulation_state.apply_action(action)
                sim_steps += 1
            if verbose:
                print(f"[Simulation] Simulation finished after {sim_steps} steps. Terminal state reached.")
            # --- BACKPROPAGATION ---
            returns = simulation_state.returns()
            if len(returns) > player:
                result = 0 if returns[player] < 0 else 1
            else:
                result = 0.5
            if verbose:
                print(f"[Backpropagation] Simulation result for player {player}: {returns[player] if len(returns) > player else 'N/A'} -> result={result}")
            for n in reversed(path):
                n.visits += 1
                if n.player == player:
                    n.wins += result
                else:
                    n.wins += (1.0 - result)
                if verbose:
                    print(f"[Backpropagation] Node for player {n.player}, visits={n.visits}, wins={n.wins:.2f}")
            # --- VOTING ---
            if root_node.children:
                best_child_action = max(
                    root_node.children.keys(),
                    key=lambda a: root_node.children[a].visits
                )
                best_action_votes[best_child_action] = best_action_votes.get(best_child_action, 0) + 1
            if verbose:
                print(f"[Voting] Current best action votes: {best_action_votes}")
        if not best_action_votes:
            legal_actions = initial_state.legal_actions()
            if verbose:
                print("[RIS-MCTS] No best action found by voting. Returning random legal action.")
            return random.choice(legal_actions) if legal_actions else None
        best = max(best_action_votes.keys(), key=lambda a: best_action_votes[a])
        if verbose:
            print(f"[RIS-MCTS] Best action after {iterations} iterations: {best} (votes: {best_action_votes[best]})")
        return best

    def _determinize_state(self, state: TarotGameState, player: int) -> TarotGameState:
        """
        Generate a determinization of the state for the player (sample opponents' hands).
        """
        if state.current_player() < 0:
            return state
        new_state = state.clone()
        # Get the player's view of the game
        player_view = state.tensor_player(player)
        played_cards_mask = Utils.get_mask(player_view, 'played_cards')

        # Identify all cards that are not in the current player's hand and have not been played yet.
        unknown_cards_indices = [
            i for i, p in enumerate(played_cards_mask) if p == -1
        ]
        random.shuffle(unknown_cards_indices)

        for p_id in range(Const.NUM_PLAYERS):
            if p_id != player:
                # Get cards already played by this opponent (and known to the current player)
                opponent_played_cards = [
                    Card.from_idx(i) for i, p in enumerate(played_cards_mask) if p == p_id
                    or (p == Const.CHIEN_ID + 1 and state.taker == p_id)
                ]
                
                # Determine how many cards we need to sample for this opponent's hand
                target_hand_size = Const.HAND_SIZE - len(opponent_played_cards)
                
                # We only need to sample if the hand is not complete
                if target_hand_size > 0:
                    if len(unknown_cards_indices) >= target_hand_size:
                        sampled_indices = unknown_cards_indices[:target_hand_size]
                        unknown_cards_indices = unknown_cards_indices[target_hand_size:]
                        
                        sampled_cards = [Card.from_idx(i) for i in sampled_indices]
                        new_state.hands[p_id] = sampled_cards

        return new_state