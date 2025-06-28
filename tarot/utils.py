from typing import List, Tuple
import numpy as np
from .cards import Card
from .bids import Bid
from . import constants as Const


class Utils:
    """
    Provides static utility methods for scoring, trick evaluation, and tensor/mask operations in the Tarot card game.
    """
    @staticmethod
    def score(tricks: List[List[int]]) -> float:
        """
        Calculates the total score for a list of tricks.
        """
        return float(np.sum([Card.value(card) for trick in tricks for card in trick]))

    @staticmethod
    def board_score(tricks: List[List[int]], chien: List[int], bid: int,
                    chelem: bool, poignee: bool, taker: int, petit: bool) -> Tuple[float, List[float]]:
        """
        Distributes points among players based on bid, tricks, chien, chelem, and poignee.
        Returns the score and a list of points for each player.
        """
        score = Utils.total_score(
            tricks, chien, bid, chelem, poignee, petit)
        points = 1 / (Const.NUM_PLAYERS - 1)
        board = [0.0] * Const.NUM_PLAYERS
        taker_won = 1.0 if score > 0 else -1.0
        board[taker] = taker_won
        for player in range(Const.NUM_PLAYERS):
            if player != taker:
                board[player] = - points * taker_won
        return score, board

    @staticmethod
    def partial_score(tricks: List[List[int]], chien: List[int], bid: int) -> Tuple[float, bool]:
        """
        Calculates the partial score for a game and whether the points are sufficient for the bid.
        Returns the score and a boolean indicating if the bid was met.
        """
        points = 0
        bouts = sum(1 for trick in tricks for card in trick if card in [
                    Const.FOU, Const.PETIT, Const.MONDE])
        required_points = Const.POINTS_PER_BOUT[bouts]
        points += (np.sum([Card.value(card)
                           for trick in tricks for card in trick]))
        points += (np.sum([Card.value(card) for card in chien])
                   if bid in [Const.PETIT, Const.GARDE, Const.GARDE_SANS] else 0)
        required = True if (points - required_points) > 0 else False
        score = 25 + abs(points - required_points)
        return score, required

    @staticmethod
    def total_score(tricks: List[List[int]], chien: List[int], bid: int,
                    chelem_declared: bool, poignee_declared: bool, petit: bool) -> float:
        """
        Calculates the total score for a game, including bonuses for chelem, poignee, and petit.
        Returns the final score (positive if bid was met, negative otherwise).
        """
        cards = [card for trick in tricks for card in trick]
        points, required = Utils.partial_score(tricks, chien, bid)
        grand_chelem = len(cards) == Const.DECK_SIZE
        chelem_bonus = 0
        if grand_chelem and chelem_declared:
            chelem_bonus = 400
        elif (grand_chelem):
            chelem_bonus = 200
        trumps = sum(1 for card in cards if Card.rank(card) == Const.TRUMP)
        poignee_bonus = 0
        if poignee_declared:
            for poignee_size, poignee_bonus in Const.POIGNEE_BONUS.items():
                if trumps >= poignee_size:
                    poignee_bonus = poignee_bonus
        if petit:
            points += 10
        score = points * Bid.multiplier(bid) + chelem_bonus + poignee_bonus
        return score if required else -score

    @staticmethod
    def trick_winner(trick: List[int]) -> int:
        """
        Determines the winner of a trick based on the cards played.
        Returns the index of the winning player.
        """
        lead_suit = Card.suit(trick[0])
        trumps = [(i, c) for i, c in enumerate(trick) if Card.is_trump(c)]
        if trumps:
            return max(trumps, key=lambda x: Card.rank(x[1]))[0]
        lead_cards = [(i, c) for i, c in enumerate(
            trick) if Card.suit(c) == lead_suit]
        return max(lead_cards, key=lambda x: Card.rank(x[1]))[0]

    @staticmethod
    def get_mask(tensor: List[int], name: str) -> List[int]:
        """
        Returns the mask slice from the tensor for the given mask name.
        """
        if name not in Const.MASK:
            raise ValueError(f"Mask '{name}' não existe.")
        mask_id = Const.MASK[name]
        start = 0
        size = Const.MASK_SIZE[mask_id]
        for i in range(mask_id):
            start += Const.MASK_SIZE[i]
        return tensor[start:start + size]

    @staticmethod
    def get_tricks_from_tensor(known_tricks: List[int]) -> List[Tuple[int, List[int]]]:
        """
        Extracts the list of tricks from the known_tricks tensor.
        Returns a list of (player, trick) tuples.
        """
        tricks: List[Tuple[int, List[int]]] = []
        for i in range(Const.NUM_TRICKS):
            idx = i * (Const.NUM_PLAYERS + 1)
            current_player = known_tricks[idx]
            if (current_player == - 1):
                break
            start = idx + 1
            current_trick = known_tricks[start:start + Const.NUM_PLAYERS]
            tricks.append((current_player, current_trick))
        return tricks
