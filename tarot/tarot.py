from typing import List, Optional, Tuple
from .actions import Action
from .bids import Bid
from .cards import Card
from .utils import Utils
from . import constants as Const
from .constants import Phase


class Tarot:
    def __init__(self,):
        self.deck = Card.deck()
        self.chien, self.hands = Card.deal(self.deck)
        self.phase = Phase.BIDDING
        self.bids: List[int] = []
        self.taker: int = -1
        self.taker_bid = Const.BID_PASS
        self.current: int = 0
        self.discard: List[int] = []
        self.tricks: List[Tuple[int, List[int]]] = []
        self.trick: List[int] = []
        self.played_cards: List[int] = [-1] * Const.DECK_SIZE
        self.played_tricks: List[int] = [-1] * (Const.MASK_NUM_TRICKS_SIZE)
        self.declared: List[Tuple[bool, bool]] = [(False, False)]
        self.chelem_declared_defenders = False
        self.poignee_declared_defenders = False
        self.chelem_declared_taker = False
        self.poignee_declared_taker = False
        self.fool_state = Const.FOOL_NOT_PLAYED
        self.fool_player = -1
        self.fool_trick = []

    def legal_actions(self) -> List[int]:
        if self.phase == Phase.BIDDING:
            return [bid for bid, _ in Bid.legal_bids(self.bids)]
        elif self.phase == Phase.CHIEN:
            return Action.legal_chien_actions(self.hands[self.current])
        elif self.phase == Phase.TRICK:
            return Action.legal_trick_actions(self.hands[self.current], self.trick)
        elif self.phase == Phase.DECLARE:
            chelem_declared, poignee_declared = self.declared[-1]
            if not chelem_declared:
                return [Const.DECLARE_CHELEM, Const.DECLARE_NONE]
            if not poignee_declared:
                return [Const.DECLARE_POIGNEE, Const.DECLARE_NONE]
            return [Const.DECLARE_NONE]
        elif self.phase == Phase.END:
            return []
        else:
            raise ValueError(
                f"Phase {self.phase} does not support legal actions retrieval")

    def is_chance_node(self) -> bool:
        """
        Returns True if the current phase is a chance node (BIDDING, CHIEN, or DECLARE).
        """
        return self.phase in {Phase.BIDDING, Phase.CHIEN, Phase.DECLARE}

    def is_terminal(self) -> bool:
        """
        Returns True if the game is in the END phase, indicating a terminal state.
        """
        return self.phase == Phase.END

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        if self.phase == Phase.BIDDING:
            return Bid.legal_bids(self.bids)
        elif self.phase == Phase.CHIEN:
            legal_discards = Action.legal_chien_actions(
                self.hands[self.current])
            return [(action, 1.0 / len(legal_discards)) for action in legal_discards]
        elif self.phase == Phase.DECLARE:
            chelem_declared, poignee_declared = self.declared[-1]
            if not chelem_declared:
                return Action.legal_chelem_actions(self.hands[self.current])
            if not poignee_declared:
                return Action.legal_poignee_actions(self.hands[self.current])
            return [(Const.DECLARE_NONE, 1.0)]
        else:
            raise ValueError(
                f"Phase {self.phase} does not support chance outcomes retrieval")

    def apply_action(self, action: int) -> None:
        if self.phase == Phase.BIDDING:
            self.bids.append(action)
            if action > Const.BID_PASS:
                self.taker_bid = action
        elif self.phase == Phase.CHIEN:
            Action.apply_chien_action(
                self.hands[self.current], self.chien, action)
        elif self.phase == Phase.TRICK:
            trick_winner = Action.apply_trick_action(
                self.hands[self.current], self.trick, action)
            self.update_played_cards(action, trick_winner)
            if action == Const.FOOL:
                self.fool_player = self.current
                self.fool_state = Const.FOOL_NOT_PAID
                self.fool_trick = self.trick
        elif self.phase == Phase.DECLARE:
            Action.apply_declare_action(self.declared, action)
        else:
            raise ValueError(
                f"Phase {self.phase} does not support applying actions")

    def update_played_cards(self, action: int, trick_winner: Optional[Tuple[int, List[int]]]) -> None:
        """
        Updates the played cards and the trick history after a trick action is applied.
        """
        self.played_cards[action] = self.current
        self.trick.append(action)
        if trick_winner:
            winner, trick = trick_winner
            idx = Utils.get_trick_position(self.tricks, self.trick)
            self.played_tricks[idx] = winner
            idx += 1
            if len(self.played_tricks) <= idx + Const.NUM_PLAYERS:
                raise ValueError(
                    f"Trick history index {idx} out of bounds for trick history size {len(self.played_tricks)}")
            self.played_tricks[idx:idx + Const.NUM_PLAYERS] = trick
            self.trick = []

    def next(self) -> None:
        """
        Advances the game to the next phase based on the current phase and game state.
        """
        if self.phase == Phase.BIDDING:
            if len(self.bids) == Const.NUM_PLAYERS:
                self.phase = Phase.CHIEN
        elif self.phase == Phase.CHIEN:
            if len(self.chien) == Const.CHIEN_SIZE:
                self.phase = Phase.TRICK
                self.taker_bid = max(self.bids)
                self.taker = self.bids.index(self.taker_bid)
                self.current = self.taker
        elif self.phase == Phase.TRICK:
            if len(self.tricks) == 0:
                chelem, poignee = self.declared[-1]
                if not chelem or not poignee:
                    if len(self.declared) < Const.NUM_PLAYERS:
                        self.declared.append((False, False))
                        self.phase = Phase.DECLARE
                        self.next_player()
                else:
                    self.phase = Phase.TRICK
                    self.next_player()
            else:
                self.next_player()
        elif self.phase == Phase.DECLARE:
            chelem, poignee = self.declared[-1]
            if chelem:
                if not self.chelem_declared_taker and self.current == self.taker:
                    self.chelem_declared_taker = True
                elif not self.chelem_declared_defenders and self.current != self.taker:
                    self.chelem_declared_defenders = True
            if poignee:
                if not self.poignee_declared_taker and self.current == self.taker:
                    self.poignee_declared_taker = True
                elif not self.poignee_declared_defenders and self.current != self.taker:
                    self.poignee_declared_defenders = True
            if chelem and poignee:
                self.phase = Phase.TRICK
                self.next_player()
        elif self.phase == Phase.TRICK_FINISHED:
            if len(self.tricks) < Const.NUM_TRICKS:
                self.phase = Phase.TRICK
                self.next_player()
                self.trick = []
            else:
                self.fool_state = Const.FOOL_NOT_PAID
                Action.apply_fool_action(
                    self.fool_trick, self.fool_player, self.tricks)
                self.phase = Phase.END
        elif self.phase == Phase.END:
            pass

    def next_player(self) -> None:
        """
        Advances the current player to the next player in the game.
        """
        self.current = (self.current + 1) % Const.NUM_PLAYERS
