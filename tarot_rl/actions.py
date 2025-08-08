from typing import List, Optional, Tuple
from .cards import rank, suit, is_trump, value, count_bouts, count_trumps
from . import constants as Const
from .utils import trick_winner


def legal_trick_actions(legal_actions: List[int], trick: List[int]) -> List[int]:
    """
    Returns the list of legal cards that can be played from the hand given the current trick.
    Enforces following suit and trump rules according to Tarot rules.
    """
    trick = trick.copy()
    trick = [t for t in trick if t != -1]
    legal_actions = legal_actions.copy()
    if len(trick) == 0:
        return legal_actions
    lead_suit = suit(trick[0])
    has_lead = [card for card in legal_actions if suit(
        card) == lead_suit]
    if len(has_lead) > 0:
        return has_lead
    has_trumps = [card for card in legal_actions if is_trump(card)]
    if len(has_trumps) > 0:
        return has_trumps
    return legal_actions


def apply_trick_action(player: int, hand: List[int], trick: List[int], action: int) -> Optional[Tuple[int, List[int]]]:
    """
    Applies the given action (card play) to the hand and trick. Returns the winner and trick if the trick is complete, otherwise None.
    """
    if action not in hand:
        raise ValueError(f"Action {action} is not in hand: {hand}")
    _trick_winner = None
    hand.remove(action)
    trick[player] = action
    if len([card for card in trick if card > -1]) == Const.NUM_PLAYERS:
        winner = trick_winner(trick)
        _trick_winner = (winner, trick)
    return _trick_winner


def apply_fool_action(fool_trick: List[int], fool_player: int, taker: int, tricks: List[Tuple[int, List[int]]]) -> Optional[int]:
    """
    Applies the Fool card action by removing it from the hand.
    Returns the updated hand after removing the Fool card.
    """
    if Const.FOOL not in fool_trick:
        raise ValueError(
            f"Fool card {Const.FOOL} not in trick: {fool_trick}")

    substitute_card = None
    use_tricks = [trick for (player, trick)
                  in tricks if player == fool_player]
    if not use_tricks and fool_player != taker:
        use_tricks = [trick for player, trick in tricks for c in trick if value(
            c) == 0.5 and player != fool_player and player != taker and Const.FOOL not in trick]
    for idx, current_trick in enumerate(use_tricks):
        if any(value(card) == 0.5 for card in current_trick):
            substitute_card = next(card for card
                                   in current_trick if value(card) == 0.5)
            current_trick.remove(substitute_card)
            break
    if substitute_card:
        fool_trick.remove(Const.FOOL)
        fool_trick.append(substitute_card)
        tricks.append((fool_player, [Const.FOOL]))
    return substitute_card


def legal_chien_actions(cards: List[int]) -> List[int]:
    """
    Returns the list of cards that can be legally discarded to the chien (dog) according to Tarot rules.
    Trumps and kings cannot be discarded.
    """
    discard = [card for card in cards if not
               is_trump(card) and rank(card) != Const.KING]
    return discard


def apply_chien_action(hand: List[int], discard: List[int], action: int) -> None:
    """
    Applies the chien discard action by removing the specified card from the hand and adding it to the chien.
    Returns the updated hand and chien.
    """
    if action not in hand:
        raise ValueError(f"Action {action} is not in hand: {hand}")
    else:
        hand.remove(action)
    discard.append(action)


def legal_chelem_actions(hand: List[int]) -> List[Tuple[int, float]]:
    """
    Returns possible chelem (slam) declaration actions and their probabilities based on the hand.
    """
    probability = 0.01  # TODO: Add Reinforcement Learning to chelem probability
    return [(Const.DECLARE_CHELEM, probability), (Const.DECLARE_NONE, 1.0 - probability)]


def legal_poignee_actions(hand: List[int]) -> List[Tuple[int, float]]:
    """
    Returns possible poignee (handful) declaration actions and their probabilities based on the number of trumps in hand.
    """
    # TODO: Add Reinforcement Learning to poignee probability
    trumps = count_trumps(hand)
    probability = 0.0
    if trumps < 10:
        probability = 0.1
    if trumps < 13:
        probability = 0.1
    elif trumps < 15:
        probability = 0.1
    else:
        probability = 0.1

    return [(Const.DECLARE_POIGNEE, probability), (Const.DECLARE_NONE, 1.0 - probability)]
