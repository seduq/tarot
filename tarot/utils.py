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
        bouts = (card for trick in tricks for card in trick if card in [
            Const.FOOL, Const.BID_PETIT, Const.MONDE])
        bouts = set(bouts)  # Remove duplicates
        bouts = len(bouts)
        required_points = Const.POINTS_PER_BOUT[bouts]
        points += (sum([Card.value(card)
                        for trick in tricks for card in trick]))
        points += (sum([Card.value(card) for card in chien])
                   if bid in [Const.BID_PETIT, Const.BID_GARDE, Const.BID_GARDE_SANS] else 0)
        required = True if (points - required_points) >= 0 else False
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
    def select_discard_cards(hand: List[int], size=Const.CHIEN_SIZE) -> List[int]:
        """
        Selects cards to discard into the chien.
        """
        discard = [card for card in hand if not Card.is_trump(card) and Card.rank(
            card) not in [Const.KING, Const.QUEEN, Const.KNIGHT]]
        discard.sort(key=lambda x: Card.rank(x))
        discard = [card for card in discard]
        if len(discard) > size:
            discard = discard[:Const.CHIEN_SIZE]
        if len(discard) < size:
            trumps = [card for card in hand if Card.is_trump(card) and
                      Card.rank(card) not in [Const.FOOL, Const.PETIT, Const.MONDE]]
            trumps.sort(key=lambda x: Card.rank(x))
            while len(discard) < size and trumps:
                discard.append(trumps.pop(0))

        while len(discard) < size and hand:
            card = hand.pop(0)
            if card not in discard:
                discard.append(card)
        return discard

    @staticmethod
    def trick_winner(trick: List[int]) -> int:
        """
        Determines the winner of a trick based on the cards played.
        Returns the index of the winning player.
        """
        idx = -1
        if Const.FOOL in trick:
            idx = trick.index(Const.FOOL)
            trick[idx] = -1
        lead_suit = Card.suit(trick[0])
        trumps = [(i, c) for i, c in enumerate(trick) if Card.is_trump(c)]
        if trumps:
            winner = max(trumps, key=lambda x: Card.rank(x[1]))[0]
            if idx > -1:
                trick[idx] = Const.FOOL
            return winner
        lead_cards = [(i, c) for i, c in enumerate(
            trick) if Card.suit(c) == lead_suit]
        winner = max(lead_cards, key=lambda x: Card.rank(x[1]))[0]
        if idx > -1:
            trick[idx] = Const.FOOL
        return winner

    @staticmethod
    def get_mask(tensor: List[int], name: str) -> List[int]:
        """
        Get the mask slice in the tensor for the given mask name.
        Names:
        - 'played_cards': Played cards, each index is either player id or -1 (deck size)
        - 'taker_chien_hand': Know cards of the taker (hand size)
        - 'played_tricks': Known tricks of the player (1 + number of players) * (number of tricks + 1)
        - 'current_trick': Current trick being played (1 + number of players)
        - 'current_player': Current player index
        - 'taker_player': Taker player index
        - 'bids': Bids of all players
        - 'declarations': Declarations made by each team (2 for each team)
        - 'phase': Current phase of the game
        """
        mask_id = Const.MASK[name]
        start = 0
        size = Const.MASK_SIZE[mask_id]
        for i in range(mask_id):
            start += Const.MASK_SIZE[i]
        return tensor[start:start + size]

    @staticmethod
    def set_mask(tensor: List[int], name: str, mask: List[int]) -> List[int]:
        """
        Sets the mask slice in the tensor for the given mask name.
        Names:
        - 'played_cards': Played cards, each index is either player id or -1
        - 'taker_chien_hand': Know cards of the taker
        - 'played_tricks': Known tricks of the player
        - 'current_trick': Current trick being played
        - 'current_player': Current player index
        - 'taker_player': Taker player index
        - 'bid': Current bid of the player
        - 'declarations': Declarations made by each team
        - 'phase': Current phase of the game
        """
        mask_id = Const.MASK[name]
        start = 0
        size = Const.MASK_SIZE[mask_id]
        for i in range(mask_id):
            start += Const.MASK_SIZE[i]
        tensor[start:start + size] = mask
        return tensor

    @staticmethod
    def set_trick(tensor: List[int], know_tricks: int, trick: Tuple[int, List[int]]) -> List[int]:
        """
        Sets a trick in the trick_history tensor.
        The trick is a tuple of (player, trick).
        """
        player, _trick = trick
        start = know_tricks * (Const.NUM_PLAYERS + 1)
        tensor[start] = player
        tensor[start + 1:start + 1 + Const.NUM_PLAYERS] = _trick
        return tensor

    @staticmethod
    def get_tricks(trick_history: List[int]) -> List[Tuple[int, List[int]]]:
        """
        Extracts the list of tricks from the trick_history tensor.
        Returns a list of (player, trick) tuples.
        """
        tricks: List[Tuple[int, List[int]]] = []
        for i in range(Const.NUM_TRICKS):
            idx = i * (Const.NUM_PLAYERS + 1)
            current_player = trick_history[idx]
            start = idx + 1
            end = start + Const.NUM_PLAYERS
            if (current_player == - 1 or
                    -1 not in trick_history[start:end]):
                break
            current_trick = trick_history[start:end]
            tricks.append((current_player, current_trick))
        return tricks

    @staticmethod
    def get_trick_position(trick_history: List[Tuple[int, List[int]]]) -> int:
        """
        Get the first legal position of the trick in the trick history.
        """
        return (len(trick_history) - 1) * (Const.NUM_PLAYERS + 1)
