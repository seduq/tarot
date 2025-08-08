
from dataclasses import dataclass
from typing import List, Tuple
from .constants import Phase


@dataclass
class TarotState:
    chien: List[int]
    hands: List[List[int]]
    phase: Phase
    bids: List[int]
    taker: int
    taker_bid: int
    current: int
    discard: List[int]
    tricks: List[List[int]]
    trick: List[int]
    trick_player: int
    trick_winners: List[int]
    know_cards: List[int]
    declared: List[Tuple[bool, bool]]
    chelem_def: bool
    poignee_def: List[bool]
    chelem_taker: bool
    poignee_taker: bool
    fool_paid: bool
    turn_counter: int
    fool_player: int
    fool_trick: int
