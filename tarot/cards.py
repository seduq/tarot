import random
import numpy as np
from typing import List, Tuple
from . import constants as Const


class Card:
    @staticmethod
    def name(card: int) -> str:
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
        return card // 100

    @staticmethod
    def rank(card: int) -> int:
        return card % 100

    @staticmethod
    def from_idx(card_idx: int) -> int:
        if (card_idx + 1) >= Const.DECK_SIZE - Const.NUM_TRUMPS:
            return (card_idx - (Const.DECK_SIZE - Const.NUM_TRUMPS)) + 100 * Const.TRUMP
        card_idx -= Const.NUM_TRUMPS
        rank = card_idx % 14
        suit = card_idx - rank
        return (suit + 1) * 100 + (rank + 1)

    @staticmethod
    def to_idx(card_idx: int) -> int:
        if Card.is_trump(card_idx):
            return (card_idx - 500) + (Const.DECK_SIZE - Const.NUM_TRUMPS)
        suit = Card.suit(card_idx) - 1
        rank = Card.rank(card_idx) - 1
        return suit * 14 + rank

    @staticmethod
    def is_trump(card: int) -> bool:
        return Card.suit(card) == Const.TRUMP

    @staticmethod
    def is_king(card: int) -> bool:
        return Card.rank(card) == Const.KING

    @staticmethod
    def value(card: int) -> float:
        rank = Card.rank(card)
        if Card.is_trump(card):
            if (rank == Const.PETIT or rank == Const.MONDE or rank == Const.FOU):
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
        trumps = [Const.TRUMP * 100 + i for i in range(Const.NUM_TRUMPS)]
        hearts = [Const.HEART * 100 + (i + 1) for i in range(Const.KING)]
        diamonds = [Const.DIAMOND * 100 + (i + 1) for i in range(Const.KING)]
        spades = [Const.SPADE * 100 + (i + 1) for i in range(Const.KING)]
        clubs = [Const.CLUB * 100 + (i + 1) for i in range(Const.KING)]
        return trumps + hearts + diamonds + spades + clubs

    @staticmethod
    def deal(deck: List[int]) -> Tuple[List[int], List[List[int]]]:
        deck = deck.copy()
        np.random.shuffle(deck)
        chien = deck[:Const.CHIEN_SIZE]
        player_hands = [
            deck[(Const.CHIEN_SIZE + i * Const.HAND_SIZE):(Const.CHIEN_SIZE + (i+1)*Const.HAND_SIZE)]
            for i in range(Const.NUM_PLAYERS)]
        return chien, player_hands

    @staticmethod
    def count_trumps(hand: List[int]) -> int:
        return sum([Card.is_trump(card) for card in hand])

    @staticmethod
    def count_bouts(hand: List[int]) -> int:
        return sum([card in Const.BOUTS for card in hand])

    @staticmethod
    def points(hand: List[int]) -> float:
        return sum([Card.value(card) for card in hand])
