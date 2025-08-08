from typing import List, Tuple
from . import constants as Const
from .cards import value, suit, is_trump, rank
from .bids import multiplier


def score(tricks: List[List[int]]) -> float:
    """
    Calculates the total score for a list of tricks.
    """
    return sum([value(card) for trick in tricks for card in trick])


def board_score(tricks: List[List[int]], chien: List[int], bid: int,
                chelem: bool, poignee: bool, taker: int, petit: bool) -> Tuple[float, List[float]]:
    """
    Distributes points among players based on bid, tricks, chien, chelem, and poignee.
    Returns the score and a list of points for each player.
    """
    score = total_score(
        tricks, chien, bid, chelem, poignee, petit)
    board = [0.0] * Const.NUM_PLAYERS
    taker_won = 1.0 if score > 0 else 0.0
    board[taker] = taker_won
    for player in range(Const.NUM_PLAYERS):
        if player != taker:
            board[player] = 1.0 - taker_won
    return score, board


def partial_score(tricks: List[List[int]], chien: List[int], bid: int) -> Tuple[float, bool]:
    """
    Calculates the partial score for a game and whether the points are sufficient for the bid.
    Returns the score and a boolean indicating if the bid was met.
    """
    points = 0
    bouts = (card for trick in tricks for card in trick if card in [
        Const.FOOL, Const.BID_PETIT, Const.MONDE])
    bouts = set(bouts)  # Remove duplicates
    bouts = len(bouts)
    required_points = Const.POINTS_PER_BOUT[bouts]
    points += (sum([value(card)
                    for trick in tricks for card in trick]))
    points += (sum([value(card) for card in chien])
               if bid in [Const.BID_PETIT, Const.BID_GARDE, Const.BID_GARDE_SANS] else 0)
    required = True if (points - required_points) >= 0 else False
    score = 25 + abs(points - required_points)
    return score, required


def total_score(tricks: List[List[int]], chien: List[int], bid: int,
                chelem_declared: bool, poignee_declared: bool, petit: bool) -> float:
    """
    Calculates the total score for a game, including bonuses for chelem, poignee, and petit.
    Returns the final score (positive if bid was met, negative otherwise).
    """
    cards = [card for trick in tricks for card in trick]
    points, required = partial_score(tricks, chien, bid)
    grand_chelem = len(cards) == Const.DECK_SIZE
    chelem_bonus = 0
    if grand_chelem and chelem_declared:
        chelem_bonus = 400
    elif (grand_chelem):
        chelem_bonus = 200
    trumps = sum(1 for card in cards if rank(card) == Const.TRUMP)
    poignee_bonus = 0
    if poignee_declared:
        for poignee_size, poignee_bonus in Const.POIGNEE_BONUS.items():
            if trumps >= poignee_size:
                poignee_bonus = poignee_bonus
    if petit:
        points += 10
    score = points * multiplier(bid) + chelem_bonus + poignee_bonus
    return score if required else -score


def trick_winner(trick: List[int]) -> int:
    """
    Determines the winner of a trick based on the cards played.
    Returns the index of the winning player.
    """
    lead_suit = suit(trick[0])
    trumps = [(i, c) for i, c in enumerate(trick) if is_trump(c)]
    if trumps:
        winner = max(trumps, key=lambda x: rank(x[1]))[0]
        return winner
    lead_cards = [(i, c) for i, c in enumerate(
        trick) if suit(c) == lead_suit]
    winner = max(lead_cards, key=lambda x: rank(x[1]))[0]
    return winner


def discard_by_distribution(hand: List[int]) -> List[int]:
    """
    Heuristic for selecting cards to discard into the chien.
    Selects cards that are not trumps and not high-ranking cards.
    """
    available_discard = [card for card in hand if not is_trump(card) and rank(
        card) not in [Const.KING, Const.QUEEN, Const.KNIGHT, Const.JACK]]
    available_discard.sort(key=lambda x: rank(x))

    spades = len(
        [card for card in available_discard if suit(card) == Const.SPADE])
    hearts = len(
        [card for card in available_discard if suit(card) == Const.HEART])
    diamonds = len(
        [card for card in available_discard if suit(card) == Const.DIAMOND])
    clubs = len(
        [card for card in available_discard if suit(card) == Const.CLUB])

    discard = []
    while len(discard) < Const.CHIEN_SIZE:
        spade_difference = (Const.SPADE, sum(
            [abs(spades - hearts), abs(spades - diamonds), abs(spades - clubs)]))
        heart_difference = (Const.HEART, sum(
            [abs(hearts - spades), abs(hearts - diamonds), abs(hearts - clubs)]))
        diamond_difference = (Const.DIAMOND, sum(
            [abs(diamonds - spades), abs(diamonds - hearts), abs(diamonds - clubs)]))
        club_difference = (Const.CLUB, sum(
            [abs(clubs - spades), abs(clubs - hearts), abs(clubs - diamonds)]))

        difference = [spade_difference, heart_difference,
                      diamond_difference, club_difference]
        suit_to_discard = max(difference, key=lambda x: x[1])[0]
        suit_cards = min([card for card in available_discard
                          if suit(card) == suit_to_discard], default=None)
        if suit_cards:
            discard.append(suit_cards)
            available_discard.remove(suit_cards)
        else:
            difference = [d for d in difference if d[0] != suit_to_discard]

    return discard


def discard_by_value(hand: List[int]) -> List[int]:
    """
    Heuristic for selecting cards to discard into the chien.
    Selects the lowest value cards that are not trumps.
    """
    available_discard = [card for card in hand if not is_trump(card) and rank(
        card) != Const.KING]
    available_discard.sort(key=lambda x: rank(x))

    return available_discard[:Const.CHIEN_SIZE]
