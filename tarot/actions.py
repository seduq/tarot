import numpy as np
from typing import List, Optional, Tuple
from .cards import Card
from .utils import Utils
from . import constants as Const


class Action:
    """
    Provides static methods for determining legal actions and applying actions in the Tarot card game.
    Includes logic for trick actions, chien discards, excuse actions, chelem and poignee declarations.
    """
    @staticmethod
    def legal_trick_actions(legal_actions: List[int], trick: List[int], is_first: bool) -> List[int]:
        """
        Returns the list of legal cards that can be played from the hand given the current trick.
        Enforces following suit and trump rules according to Tarot rules.
        For simplicity, if it's the first card played in the trick, the Fool (Const.FOU) is removed from the hand.
        """
        legal_actions = legal_actions.copy()
        if is_first:
            if Const.FOU in legal_actions:
                legal_actions.remove(Const.FOU)
        if len(trick) == 0:
            return legal_actions
        lead_suit = Card.suit(trick[0])
        has_lead = [card for card in legal_actions if Card.suit(
            card) == lead_suit]
        if len(has_lead) > 0:
            return has_lead
        has_trumps = [card for card in legal_actions if Card.is_trump(card)]
        if len(has_trumps) > 0:
            return has_trumps
        return legal_actions

    @staticmethod
    def apply_trick_action(hand: List[int], trick: List[int], action: int) -> Optional[Tuple[int, List[int]]]:
        """
        Applies the given action (card play) to the hand and trick. Returns the winner and trick if the trick is complete, otherwise None.
        """
        trick_winner = None
        hand.remove(action)
        trick.append(action)
        if len(trick) == Const.NUM_PLAYERS:
            winner = Utils.trick_winner(trick)
            trick_winner = (winner, trick)
        return trick_winner

    @staticmethod
    def legal_chien_discards(cards: List[int]) -> List[int]:
        """
        Returns the list of cards that can be legally discarded to the chien (dog) according to Tarot rules.
        Trumps and kings cannot be discarded.
        """
        return Utils.select_discard_cards(cards)

    @staticmethod
    def apply_excuse_action(hand: List[int], player: int, taker: int, trick: List[int],
                            tricks: List[Tuple[int, List[int]]], fool: int) -> Optional[Tuple[int, List[int]]]:
        """
        Applies the Excuse (Fool) card action. Handles special rules for the Excuse, including substitution if the taker wins the trick.
        Returns the substitute card if applicable, otherwise None.
        """
        hand.remove(fool)
        substitute_card = fool
        player_is_taker = player == taker
        use_tricks = [(player, _trick) for (trick_player, _trick) in tricks
                      if (player_is_taker and trick_player == taker) or
                      (not player_is_taker and trick_player != taker)]
        for idx, (_, current_trick) in enumerate(use_tricks):
            if any(Card.value(card) == 0.5 for card in current_trick):
                substitute_card = next(card for card
                                       in current_trick if Card.value(card) == 0.5)
                current_trick.remove(substitute_card)
                break

        if substitute_card:
            tricks.append((player, [fool]))

        trick.append(substitute_card)
        if len(trick) == Const.NUM_PLAYERS:
            winner = Utils.trick_winner(trick)
            trick_winner = (winner, trick)
            return trick_winner
        return None

    @staticmethod
    def legal_chelem_actions(hand: List[int]) -> List[Tuple[int, float]]:
        """
        Returns possible chelem (slam) declaration actions and their probabilities based on the hand.
        """
        bouts = Card.count_bouts(hand)
        constant = Const.CHELEM_PROB[bouts]
        probability = constant
        return [(Const.DECLARE_CHELEM, probability), (Const.DECLARE_NONE, 1.0 - probability)]

    @staticmethod
    def legal_poignee_actions(hand: List[int]) -> List[Tuple[int, float]]:
        """
        Returns possible poignee (handful) declaration actions and their probabilities based on the number of trumps in hand.
        """
        trumps = Card.count_trumps(hand)
        constant = 0.0
        if trumps < 10:
            constant = Const.POIGNEE_PROB[0]
        if trumps < 13:
            constant = Const.POIGNEE_PROB[10]
        elif trumps < 15:
            constant = Const.POIGNEE_PROB[13]
        else:
            constant = Const.POIGNEE_PROB[15]

        probability = constant
        return [(Const.DECLARE_POIGNEE, probability), (Const.DECLARE_NONE, 1.0 - probability)]
