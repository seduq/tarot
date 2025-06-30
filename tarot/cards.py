import random
import numpy as np
from typing import List, Tuple
from . import constants as Const


class Card:
    """
    Provides static methods for card operations in the Tarot card game, including card naming, suit/rank extraction, deck generation, dealing, and point calculation.
    """

    @staticmethod
    def name(card: int) -> str:
        """
        Returns the string name of a card given its integer representation.
        """
        suit = Card.suit(card)
        rank = Card.rank(card)
        if Card.is_trump(card):
            if rank == Const.PETIT:
                return "Le Petit"
            elif rank == Const.MONDE:
                return "Le Monde"
            elif rank == Const.FOU:
                return "Le Fou"
            return f"Trunfo {rank}"
        suit_names = ["Espadas", "Copas", "Paus", "Ouros"]
        rank_names = ["Ás", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                      "Valete", "Cavaleiro", "Rainha", "Rei"]
        return f"{rank_names[rank - 1]} de {suit_names[suit - 1]}"

    @staticmethod
    def suit(card: int) -> int:
        """
        Returns the suit of the card as an integer.
        """
        return card // 100

    @staticmethod
    def rank(card: int) -> int:
        """
        Returns the rank of the card as an integer.
        """
        return card % 100

    @staticmethod
    def from_idx(card_idx: int) -> int:
        """
        Converts a card index to its integer representation.
        """
        rank = (card_idx % 14 + 1)
        suit = (card_idx // 14 + 1)
        if card_idx >= (Const.DECK_SIZE - Const.NUM_TRUMPS):
            return (card_idx - (Const.DECK_SIZE - Const.NUM_TRUMPS)) + 100 * Const.TRUMP
        return suit * 100 + rank

    @staticmethod
    def to_idx(card_idx: int) -> int:
        """
        Converts a card integer representation to its index in the deck.
        """
        if Card.is_trump(card_idx):
            return (card_idx - 500) + (Const.DECK_SIZE - Const.NUM_TRUMPS)
        suit = Card.suit(card_idx) - 1
        rank = Card.rank(card_idx) - 1
        return suit * 14 + rank

    @staticmethod
    def is_trump(card: int) -> bool:
        """
        Returns True if the card is a trump, False otherwise.
        """
        return Card.suit(card) == Const.TRUMP

    @staticmethod
    def is_king(card: int) -> bool:
        """
        Returns True if the card is a king, False otherwise.
        """
        return Card.rank(card) == Const.KING

    @staticmethod
    def value(card: int) -> float:
        """
        Returns the point value of the card according to Tarot rules.
        """
        rank = Card.rank(card)
        if Card.is_trump(card):
            if (card == Const.PETIT or card == Const.MONDE or card == Const.FOU):
                return 4.5
            else:
                return 0.5
        if rank > 13:
            return 4.5
        elif rank > 12:
            return 3.5
        elif rank > 11:
            return 2.5
        elif rank > 10:
            return 1.5
        return 0.5

    @staticmethod
    def deck() -> List[int]:
        """
        Returns a list of all cards in the Tarot deck.
        """
        trumps = [Const.TRUMP * 100 + i for i in range(Const.NUM_TRUMPS)]
        hearts = [Const.HEART * 100 + (i + 1) for i in range(Const.KING)]
        diamonds = [Const.DIAMOND * 100 + (i + 1) for i in range(Const.KING)]
        spades = [Const.SPADE * 100 + (i + 1) for i in range(Const.KING)]
        clubs = [Const.CLUB * 100 + (i + 1) for i in range(Const.KING)]
        return trumps + hearts + diamonds + spades + clubs

    @staticmethod
    def deal(deck: List[int]) -> Tuple[List[int], List[List[int]]]:
        """
        Shuffles and deals the deck into the chien and player hands.
        Returns a tuple (chien, player_hands).
        """
        deck = deck.copy()
        random.shuffle(deck)
        chien = deck[:Const.CHIEN_SIZE]
        player_hands = [
            deck[(Const.CHIEN_SIZE + i * Const.HAND_SIZE):(Const.CHIEN_SIZE + (i+1)*Const.HAND_SIZE)]
            for i in range(Const.NUM_PLAYERS)]
        return chien, player_hands

    @staticmethod
    def count_trumps(hand: List[int]) -> int:
        """
        Returns the number of trumps in the given hand.
        """
        return sum([Card.is_trump(card) for card in hand])

    @staticmethod
    def count_bouts(hand: List[int]) -> int:
        """
        Returns the number of bouts (special trumps) in the given hand.
        """
        return sum([card in Const.BOUTS for card in hand])

    @staticmethod
    def points(hand: List[int]) -> float:
        """
        Returns the total point value of the given hand.
        """
        return sum([Card.value(card) for card in hand])
