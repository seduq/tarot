import numpy as np
from typing import List, Optional, Tuple
from .cards import Card
from .utils import Utils
from . import constants as Const
from .constants import Phase


class Action:
    """
    Provides static methods for determining legal actions and applying actions in the Tarot card game.
    Includes logic for trick actions, chien discards, excuse actions, chelem and poignee declarations.
    """
    @staticmethod
    def legal_trick_actions(legal_actions: List[int], trick: List[int]) -> List[int]:
        """
        Returns the list of legal cards that can be played from the hand given the current trick.
        Enforces following suit and trump rules according to Tarot rules.
        """
        legal_actions = legal_actions.copy()
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
        if action not in hand:
            raise ValueError(f"Action {action} is not in hand: {hand}")
        trick_winner = None
        hand.remove(action)
        trick.append(action)
        if len(trick) == Const.NUM_PLAYERS:
            winner = Utils.trick_winner(trick)
            trick_winner = (winner, trick)
        return trick_winner

    @staticmethod
    def apply_fool_action(fool_trick: List[int], fool_player, tricks: List[Tuple[int, List[int]]]) -> int:
        """
        Applies the Fool card action by removing it from the hand.
        Returns the updated hand after removing the Fool card.
        """
        if Const.FOOL not in fool_trick:
            raise ValueError(
                f"Fool card {Const.FOOL} not in trick: {fool_trick}")
        fool_trick.remove(Const.FOOL)
        substitute_card = None
        use_tricks = [trick for (player, trick)
                      in tricks if player == fool_player]
        for idx, current_trick in enumerate(use_tricks):
            if any(Card.value(card) == 0.5 for card in current_trick):
                substitute_card = next(card for card
                                       in current_trick if Card.value(card) == 0.5)
                current_trick.remove(substitute_card)
                break
        if substitute_card:
            fool_trick.append(substitute_card)
            tricks.append((fool_player, [Const.FOOL]))
        else:
            raise ValueError(
                f"No substitute card found in tricks for player {fool_player}")
        return substitute_card

    @staticmethod
    def legal_chien_actions(cards: List[int]) -> List[int]:
        """
        Returns the list of cards that can be legally discarded to the chien (dog) according to Tarot rules.
        Trumps and kings cannot be discarded.
        """
        return Utils.select_discard_cards(cards)

    @staticmethod
    def apply_chien_action(hand: List[int], chien: List[int], discard: List[int], action: int) -> None:
        """
        Applies the chien discard action by removing the specified card from the hand and adding it to the chien.
        Returns the updated hand and chien.
        """
        if action not in hand:
            raise ValueError(f"Action {action} is not in hand: {hand}")
        else:
            hand.remove(action)
        discard.append(action)

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

    @staticmethod
    def is_action_of_type(action: int, phase: Phase) -> bool:
        if action > 100 and action < 600:
            return phase == Phase.TRICK or phase == Phase.CHIEN
        if action >= 600 and action < 700:
            return phase == Phase.BIDDING
        if action >= 700 and action < 800:
            return phase == Phase.DECLARE
        return False
