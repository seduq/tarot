import random
import math
import time
from typing import Optional, Dict, Any, List
from .tarot import Tarot
from . import constants as Const, Phase
from .utils import Utils
from .cards import Card


class RIS_MCTS_Node:
    """
    Node for the RIS-MCTS search tree.
    Stores statistics and tree structure.
    """

    def __init__(self, player: int, tensor: Optional[str] = None, parent: Optional['RIS_MCTS_Node'] = None, action: Optional[int] = None):
        self.visits: int = 0
        self.wins: float = 0.0
        self.children: Dict[int, 'RIS_MCTS_Node'] = {}
        self.parent: Optional['RIS_MCTS_Node'] = parent
        self.action: Optional[int] = action
        self.player: int = player
        self.tensor: Optional[str] = tensor


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

    def search(self, initial_state: Tarot, player: int, iterations: int = 100, max_time: Optional[float] = None, verbose: bool = False) -> Optional[int]:
        """
        Run RIS-MCTS from an initial state for a given player.
        Returns the best action found after the given number of iterations or time limit.
        If the state is not in the trick phase, returns a random legal action.
        If verbose is True, prints debug information at each crucial step.

        Args:
            initial_state: The initial game state
            player: The player to search for
            iterations: Maximum number of iterations to run
            max_time: Maximum time in seconds to run (None for no time limit)
            verbose: Whether to print debug information
        """
        if initial_state.phase != Phase.TRICK:
            legal_actions = initial_state.legal_actions()
            if verbose:
                print("[RIS-MCTS] Not in trick phase. Returning random legal action.")
            return random.choice(legal_actions) if legal_actions else None
        root_key = str(initial_state.tensor_player(player))
        if root_key not in self.tree:
            self.tree[root_key] = RIS_MCTS_Node(player, root_key)
            if verbose:
                print(f"[RIS-MCTS] Created root node for key: {root_key}")
        root_node = self.tree[root_key]
        best_action_votes = {}

        # Controle de tempo
        start_time = time.time()
        time_exceeded = False
        completed_iterations = 0

        for it in range(iterations):
            completed_iterations = it + 1
            # Verificar limite de tempo
            if max_time is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_time:
                    time_exceeded = True
                    completed_iterations = it  # Não conta a iteração atual se foi interrompida
                    if verbose:
                        print(
                            f"[RIS-MCTS] Time limit ({max_time:.2f}s) exceeded after {it} iterations (elapsed: {elapsed_time:.2f}s)")
                    break

            if verbose:
                print(
                    f"\n[Iteration {it+1}/{iterations}] Starting new MCTS iteration.")
            # Determinization: sample hidden information for opponents
            current_state = self._determinize_state(initial_state, player)
            if verbose:
                print("[Determinization] Sampled determinization for player", player)
                for p in range(Const.NUM_PLAYERS):
                    print(
                        f"Player {p} hand: {sorted(current_state.hands[p])} Initial hand: {sorted(initial_state.hands[p])}")
            current_player = player
            node = root_node
            path = [node]
            # --- SELECTION ---
            if verbose:
                print("[Selection] Starting selection phase.")
            selection_depth = 0
            max_selection_depth = 50  # Proteção contra loops infinitos
            while node.children and selection_depth < max_selection_depth:
                selection_depth += 1
                legal_actions = current_state.legal_actions()
                unexplored = [
                    a for a in legal_actions if a not in node.children]
                if unexplored:
                    if verbose:
                        print(
                            f"[Selection] Found unexplored actions: {unexplored}")
                    break
                # Select best child (UCB1)
                best_score = float('-inf')
                best_action = None
                for action, child in node.children.items():
                    if child.visits == 0:
                        # Prioritizar filhos não visitados
                        ucb1_score = float('inf')
                        exploitation = 0.0
                        exploration = 0.0
                    else:
                        exploitation = child.wins / child.visits
                        if node.visits > 0:
                            exploration = self.exploration_constant * \
                                math.sqrt(math.log(node.visits) / child.visits)
                        else:
                            exploration = self.exploration_constant
                        ucb1_score = exploitation + exploration

                    if verbose:
                        if child.visits == 0:
                            print(
                                f"[Selection] Action {action}: unvisited child, UCB1=inf")
                        else:
                            print(
                                f"[Selection] Action {action}: exploitation={exploitation:.3f}, exploration={exploration:.3f}, UCB1={ucb1_score:.3f}")

                    if ucb1_score > best_score:
                        best_score = ucb1_score
                        best_action = action
                if best_action is None:
                    raise ValueError(
                        "No best action found, check the tree structure.")

                if best_action in current_state.legal_actions():
                    node = node.children[best_action]
                    path.append(node)
                    current_state.apply_action(best_action)
                    current_player = current_state.current

                    # Re-determinização após transição para novo information set
                    if current_player != player:
                        # Se mudamos para um jogador diferente, re-determinizamos
                        current_state = self._determinize_state(
                            current_state, player)
                        if verbose:
                            print(
                                f"[Selection] Re-determinized state for new player {current_player}")

                    if verbose:
                        print(
                            f"[Selection] Selected action {best_action}, moved to player {current_player}.")
                else:
                    if verbose:
                        print(
                            f"[Selection] Best action {best_action} is not legal in the current state. Stopping selection.")
                        # Remove invalid child
                        self.invalid_action_count += 1
                        node.children.pop(best_action, None)
                    break
            # --- EXPANSION ---
            if not current_state.is_terminal():
                legal_actions = current_state.legal_actions()
                unexplored = [legal for legal in legal_actions
                              if legal not in node.children]
                if unexplored:
                    action = random.choice(unexplored)
                    new_state = current_state.clone()
                    new_state.apply_action(action)
                    next_player = new_state.current

                    # Re-determinização após expansão para novo information set
                    if next_player != player:
                        new_state = self._determinize_state(new_state, player)
                        if verbose:
                            print(
                                f"[Expansion] Re-determinized state for expanded player {next_player}")

                    # Usar perspectiva do player original
                    key = str(new_state.tensor_player(player))
                    new_node = RIS_MCTS_Node(
                        next_player, tensor=key, parent=node, action=action)
                    node.children[action] = new_node
                    self.tree[key] = new_node
                    path.append(new_node)
                    current_state = new_state
                    current_player = next_player
                    node = new_node
                    if verbose:
                        print(
                            f"[Expansion] Expanded with action {action}, created new node for player {next_player}.")
            # --- SIMULATION ---
            simulation_state = current_state.clone()
            if verbose:
                print("[Simulation] Starting simulation from expanded/leaf node.")
            sim_steps = 0
            while not simulation_state.is_terminal():
                # Verificação de tempo também durante a simulação (a cada 10 passos)
                if max_time is not None and sim_steps % 10 == 0:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= max_time:
                        if verbose:
                            print(
                                f"[Simulation] Time limit exceeded during simulation at step {sim_steps}")
                        break

                legal_actions = simulation_state.legal_actions()
                if not legal_actions:
                    if verbose:
                        print(
                            "[Simulation] No legal actions left, breaking simulation loop.")
                    break
                action = random.choice(legal_actions)
                simulation_state.apply_action(action)
                sim_steps += 1
            if verbose:
                print(
                    f"[Simulation] Simulation finished after {sim_steps} steps. Terminal state reached.")
            # --- BACKPROPAGATION ---
            returns = simulation_state.returns()
            if len(returns) > player:
                result = 0 if returns[player] < 0 else 1
            else:
                result = 0.5
            if verbose:
                print(
                    f"[Backpropagation] Simulation result for player {player}: {returns[player] if len(returns) > player else 'N/A'} -> result={result}")
            for n in reversed(path):
                n.visits += 1
                # Cada nó deve ser atualizado com o resultado do ponto de vista do player original
                n.wins += result
                if verbose:
                    print(
                        f"[Backpropagation] Node for player {n.player}, visits={n.visits}, wins={n.wins:.2f}")
            # --- VOTING ---
            if root_node.children:
                best_child_action = max(
                    root_node.children.keys(),
                    key=lambda a: root_node.children[a].visits
                )
                best_action_votes[best_child_action] = best_action_votes.get(
                    best_child_action, 0) + 1
            if verbose:
                print(
                    f"[Voting] Current best action votes: {best_action_votes}")

        # Log final statistics
        final_time = time.time() - start_time
        if verbose:
            print(
                f"[RIS-MCTS] Completed {completed_iterations} iterations in {final_time:.2f}s")
            if time_exceeded:
                print(f"[RIS-MCTS] Stopped due to time limit")

        if not best_action_votes:
            legal_actions = initial_state.legal_actions()
            if verbose:
                print(
                    "[RIS-MCTS] No best action found by voting. Returning random legal action.")
            return random.choice(legal_actions) if legal_actions else None
        best = max(best_action_votes.keys(),
                   key=lambda a: best_action_votes[a])
        if verbose:
            print(
                f"[RIS-MCTS] Best action after {iterations} iterations: {best} (votes: {best_action_votes[best]})")
        return best

    def _determinize_state(self, state: Tarot, player: int) -> Tarot:
        """
        Generate a determinization of the state for the player (sample opponents' hands).
        """
        if state.current < 0:
            return state
        new_state = state.clone()

        # Preservar a mão do player original
        new_state.hands[player] = state.hands[player][:]

        # Get the player's view of the game
        player_view = state.tensor_player(player)
        played_cards_mask = Utils.get_mask(player_view, 'played_cards')

        # Identify all cards that are not in the current player's hand and have not been played yet.
        player_cards = set(state.hands[player])
        all_cards = set(Card.from_idx(i)
                        for i in range(78))  # Assumindo 78 cartas no Tarot
        played_cards = set(Card.from_idx(i)
                           for i, p in enumerate(played_cards_mask) if p >= 0)

        unknown_cards = list(all_cards - player_cards - played_cards)
        random.shuffle(unknown_cards)

        unknown_idx = 0
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
                if target_hand_size > 0 and unknown_idx < len(unknown_cards):
                    cards_needed = min(target_hand_size, len(
                        unknown_cards) - unknown_idx)
                    sampled_cards = unknown_cards[unknown_idx:unknown_idx + cards_needed]
                    unknown_idx += cards_needed

                    new_state.hands[p_id] = sampled_cards
                else:
                    new_state.hands[p_id] = []

        return new_state
