import numpy as np
from typing import List, Tuple
from . import constants as Const


class Bid:

    @staticmethod
    def name(bid: int) -> str:
        bid_name = ""
        if bid == Const.PASS:
            bid_name = "Pass"
        elif bid == Const.PETIT:
            bid_name = "Petit"
        elif bid == Const.GARDE:
            bid_name = "Garde"
        elif bid == Const.GARDE_SANS:
            bid_name = "Garde sans le chien"
        elif bid == Const.GARDE_CONTRE:
            bid_name = "Garde contre le chien"
        else:
            bid_name = "Invalid bid"
        return bid_name

    @staticmethod
    def multiplier(bid: int) -> int:
        if bid == Const.PASS:
            return 0
        if bid == Const.GARDE:
            return 2
        elif bid == Const.GARDE_SANS:
            return 4
        elif bid == Const.GARDE_CONTRE:
            return 6
        return 1

    @staticmethod
    def legal_bids(current_bids: List[int]) -> List[Tuple[int, float]]:
        legal_bids = [Const.PASS]
        bid = max(current_bids) if current_bids else Const.PASS
        if bid == Const.PASS:
            legal_bids += [Const.PETIT, Const.GARDE,
                           Const.GARDE_SANS, Const.GARDE_CONTRE]
        elif bid == Const.PETIT:
            legal_bids += [Const.GARDE, Const.GARDE_SANS, Const.GARDE_CONTRE]
        elif bid == Const.GARDE:
            legal_bids += [Const.GARDE_SANS, Const.GARDE_CONTRE]
        elif bid == Const.GARDE_SANS:
            legal_bids += [Const.GARDE_CONTRE]
        legal_bid_outcomes = []
        sum_bids = sum(Const.GAMMA[b] for b in legal_bids)
        sum_all_bids = sum(Const.GAMMA[b] for b in Const.GAMMA)
        for b in legal_bids:
            legal_bid_outcomes.append(
                (b, Const.GAMMA[b] * sum_all_bids / sum_bids))
        return legal_bid_outcomes

    @staticmethod
    def finish_bidding(bids: List[int]) -> Tuple[int, int]:
        max_bid = max(bids)
        taker = bids.index(max_bid)
        return taker, max_bid
