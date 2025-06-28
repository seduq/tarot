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
    def legal_trick_actions(hand: List[int], trick: List[int]) -> List[int]:
        """
        Returns the list of legal cards that can be played from the hand given the current trick.
        Enforces following suit and trump rules according to Tarot rules.
        """
        if len(trick) == 0:
            return hand
        lead_suit = Card.suit(trick[0])
        has_lead = [card for card in hand if Card.suit(card) == lead_suit]
        if len(has_lead) > 0:
            return has_lead
        has_trumps = [card for card in hand if Card.is_trump(card)]
        if len(has_trumps) > 0:
            return has_trumps
        return hand

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
        cards = [card for card in cards if not Card.is_trump(
            card) and not Card.is_king(card)]
        return cards

    @staticmethod
    def apply_excuse_action(hand: List[int], player: int, is_taker: bool, tricks: List[Tuple[int, List[int]]], action: int) -> Optional[int]:
        """
        Applies the Excuse (Fool) card action. Handles special rules for the Excuse, including substitution if the taker wins the trick.
        Returns the substitute card if applicable, otherwise None.
        """
        hand = hand.copy()
        tricks.append((player, [action]))
        substitute_card = None
        if action in hand:
            hand.remove(action)
        if is_taker:
            taker_tricks = [trick for trick in tricks if trick[0] == player]
            if taker_tricks:
                for idx, trick in enumerate(taker_tricks):
                    if any(Card.value(card) == 0.5 for card in trick[1]):
                        substitute_card = next(
                            card for card in trick[1] if Card.value(card) == 0.5)
                        new_trick = (
                            trick[0], [card for card in trick[1] if card != action])
                        tricks[tricks.index(trick)] = new_trick
                        break
        return substitute_card

    @staticmethod
    def legal_chelem_actions(hand: List[int]) -> List[Tuple[int, float]]:
        """
        Returns possible chelem (slam) declaration actions and their probabilities based on the hand.
        """
        points = Card.points(hand) / Const.MAX_POINTS
        bouts = Card.count_bouts(hand)
        constant = Const.BETA[bouts]
        probability = constant * points
        return [(Const.DECLARE_CHELEM, probability), (Const.DECLARE_NONE, 1.0 - probability)]

    @staticmethod
    def legal_poignee_actions(hand: List[int]) -> List[Tuple[int, float]]:
        """
        Returns possible poignee (handful) declaration actions and their probabilities based on the number of trumps in hand.
        """
        trumps = Card.count_trumps(hand)
        constant = 0.0
        if trumps < 10:
            constant = Const.GAMMA[0]
        if trumps < 13:
            constant = Const.GAMMA[10]
        elif trumps < 15:
            constant = Const.GAMMA[13]
        else:
            constant = Const.GAMMA[15]
        probability = constant * trumps / Const.NUM_TRUMPS
        return [(Const.DECLARE_POIGNEE, probability), (Const.DECLARE_NONE, 1.0 - probability)]
