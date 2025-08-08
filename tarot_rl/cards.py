import random
from typing import List, Tuple
from . import constants as Const


def card_name(card: int) -> str:
    """
    Returns the string name of a card given its integer representation.
    """
    _suit = suit(card)
    _rank = rank(card)
    if is_trump(card):
        if _rank == Const.PETIT:
            return "Le Petit"
        elif _rank == Const.MONDE:
            return "Le Monde"
        elif _rank == Const.FOOL:
            return "Le Fou"
        return f"Trunfo {_rank}"
    suit_names = ["Espadas", "Copas", "Paus", "Ouros"]
    rank_names = ["Ás", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                  "Valete", "Cavaleiro", "Rainha", "Rei"]
    return f"{rank_names[_rank - 1]} de {suit_names[_suit - 1]}"


def suit(card: int) -> int:
    """
    Returns the suit of the card as an integer.
    """
    if card > 4 * Const.SUIT_SIZE:
        return Const.TRUMP
    return (card - 1) // Const.SUIT_SIZE + 1


@staticmethod
def rank(card: int) -> int:
    """
    Returns the rank of the card as an integer.
    """
    return (card - 1) % Const.SUIT_SIZE + 1


@staticmethod
def is_trump(card: int) -> bool:
    """
    Returns True if the card is a trump, False otherwise.
    """
    return suit(card) == Const.TRUMP


@staticmethod
def is_king(card: int) -> bool:
    """
    Returns True if the card is a king, False otherwise.
    """
    return rank(card) == Const.KING


@staticmethod
def value(card: int) -> float:
    """
    Returns the point value of the card according to Tarot rules.
    """
    _rank = rank(card)
    if is_trump(card):
        if (card == Const.PETIT or card == Const.MONDE or card == Const.FOOL):
            return 4.5
        else:
            return 0.5
    if _rank == Const.KING:
        return 4.5
    elif _rank == Const.QUEEN:
        return 3.5
    elif _rank == Const.KNIGHT:
        return 2.5
    elif _rank == Const.JACK:
        return 1.5
    return 0.5


@staticmethod
def deck() -> List[int]:
    """
    Returns a list of all cards in the Tarot deck.
    """
    return [i+1 for i in range(Const.DECK_SIZE)]


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
        deck[(Const.CHIEN_SIZE + i * Const.HAND_SIZE)
              :(Const.CHIEN_SIZE + (i+1)*Const.HAND_SIZE)]
        for i in range(Const.NUM_PLAYERS)]
    return chien, player_hands


@staticmethod
def count_trumps(hand: List[int]) -> int:
    """
    Returns the number of trumps in the given hand.
    """
    return sum([is_trump(card) for card in hand])


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
    return sum([value(card) for card in hand])
