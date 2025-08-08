from typing import List, Tuple
from . import constants as Const


def bid_name(bid: int) -> str:
    """
    Returns the string name of a bid value.
    """
    bid_name = ""
    if bid == Const.BID_PASS:
        bid_name = "Pass"
    elif bid == Const.BID_PETIT:
        bid_name = "Petit"
    elif bid == Const.BID_GARDE:
        bid_name = "Garde"
    elif bid == Const.BID_GARDE_SANS:
        bid_name = "Garde sans le chien"
    elif bid == Const.BID_GARDE_CONTRE:
        bid_name = "Garde contre le chien"
    else:
        bid_name = "Invalid bid"
    return bid_name


def multiplier(bid: int) -> int:
    """
    Returns the multiplier associated with a bid value.
    """
    if bid == Const.BID_PASS:
        return 0
    if bid == Const.BID_GARDE:
        return 2
    elif bid == Const.BID_GARDE_SANS:
        return 4
    elif bid == Const.BID_GARDE_CONTRE:
        return 6
    return 1


def legal_bids(current_bids: List[int]) -> List[Tuple[int, float]]:
    """
    Returns a list of legal bids and their probabilities given the current bidding history.
    """
    legal_bids = [Const.BID_PASS]
    bid = max(current_bids) if current_bids else Const.BID_PASS
    if bid == Const.BID_PASS:
        legal_bids += [Const.BID_PETIT, Const.BID_GARDE,
                       Const.BID_GARDE_SANS, Const.BID_GARDE_CONTRE]
    elif bid == Const.BID_PETIT:
        legal_bids += [Const.BID_GARDE,
                       Const.BID_GARDE_SANS, Const.BID_GARDE_CONTRE]
    elif bid == Const.BID_GARDE:
        legal_bids += [Const.BID_GARDE_SANS, Const.BID_GARDE_CONTRE]
    elif bid == Const.BID_GARDE_SANS:
        legal_bids += [Const.BID_GARDE_CONTRE]
    legal_bid_outcomes = []

    # TODO: Add Reinforcement Learning to bid probabilities
    probabilities = [0.1 for _ in legal_bids]
    sum_bids = sum(probabilities)

    for b in legal_bids:
        legal_bid_outcomes.append(
            (b, probabilities[legal_bids.index(b)] / sum_bids))
    return legal_bid_outcomes


def finish_bidding(bids: List[int]) -> Tuple[int, int]:
    """
    Determines the taker and the highest bid from the list of bids.
    Returns a tuple (taker, max_bid).
    """
    max_bid = max(bids)
    taker = bids.index(max_bid)
    return taker, max_bid
