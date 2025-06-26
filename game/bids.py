from typing import List, Tuple
import numpy as np
from math import comb
from scipy.stats import norm
from game.cards import Card


class Bid:
    PASS = 600
    PETIT = 601
    GARDE = 602
    GARDE_SANS = 603
    GARDE_CONTRE = 604
    NUM_BIDS = len([PASS, PETIT, GARDE, GARDE_SANS, GARDE_CONTRE])

    DECLARE_NONE = 700
    DECLARE_CHELEM = 701
    DECLARE_POIGNEE = 702
    NUM_DECLARES = len([DECLARE_NONE, DECLARE_CHELEM, DECLARE_POIGNEE])

    @staticmethod
    def name(bid) -> Tuple[str, int]:
        """Returns the name and multiplier of the bid."""
        bid_name = ""
        if bid == Bid.PASS:
            bid_name = "Pass"
        elif bid == Bid.PETIT:
            bid_name = "Petit"
        elif bid == Bid.GARDE:
            bid_name = "Gade"
        elif bid == Bid.GARDE_SANS:
            bid_name = "Gade sans le chien"
        elif bid == Bid.GARDE_CONTRE:
            bid_name = "Gade contre le chien"
        else:
            bid_name = "Invalid bid"
        bid_multiplier = Bid.multiplier(bid)
        return bid_name, bid_multiplier

    @staticmethod
    def multiplier(bid) -> int:
        """Returns the multiplier for the bid."""
        if bid == Bid.PASS:
            return 0
        if bid == Bid.GARDE:
            return 2
        elif bid == Bid.GARDE_SANS:
            return 4
        elif bid == Bid.GARDE_CONTRE:
            return 6
        return 1

    @staticmethod
    def legal_bids(current_bids: List[int]) -> List[int]:
        """Returns the legal bids based on the current bids."""
        legal_bids = [Bid.PASS]
        bid = max(current_bids)
        if bid == Bid.PASS:
            legal_bids += [Bid.PETIT, Bid.GARDE,
                           Bid.GARDE_SANS, Bid.GARDE_CONTRE]
        elif bid == Bid.PETIT:
            legal_bids += [Bid.GARDE, Bid.GARDE_SANS, Bid.GARDE_CONTRE]
        elif bid == Bid.GARDE:
            legal_bids = [Bid.GARDE_SANS, Bid.GARDE_CONTRE]
        elif bid == Bid.GARDE_SANS:
            legal_bids += [Bid.GARDE_CONTRE]
        return legal_bids

    @staticmethod
    def finish_bidding(bids: List[int]) -> Tuple[int, int]:
        """Determina o tomador e o lance vencedor."""
        max_bid = max(bids)
        taker = next(i for i, b in enumerate(bids) if b == max_bid)
        return taker, max_bid
