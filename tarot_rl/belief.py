from typing import List
from .tarot import TarotState
from . import constants as Const


def belief(state: TarotState) -> List[int]:
    current_belief = [1 for _ in range(Const.DECK_SIZE * Const.NUM_PLAYERS)]
    for card in state.hands[state.current]:
        current_belief[card + state.current * Const.DECK_SIZE] = 1
    return current_belief


def update_belief(state: TarotState, belief: List[int], action: int, player: int) -> List[int]:
    """
    Updates the belief state based on the action taken by a player.
    The action is assumed to be a card played by the player.
    """
    if action < 0 or action >= Const.DECK_SIZE * Const.NUM_PLAYERS:
        raise ValueError(f"Invalid action {action} for player {player}")
    # TODO: Update belief based on the action, player, and game state
    # Matrix with all possible cards from all players.
    # If a player can't play a trick, we can assume they don't have a compatible card.
    # In this case, we can set the belief for that card to 0.
    return belief
