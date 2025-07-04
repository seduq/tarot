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
        self.trick: List[int] = [-1] * Const.NUM_PLAYERS
        self.played_cards: List[int] = [-1] * Const.DECK_SIZE
        self.played_tricks: List[int] = [-1] * (Const.MASK_NUM_TRICKS_SIZE)
        self.taker_chien_hand: List[int] = []
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
        elif self.phase == Phase.TRICK_FINISHED:
            return [Const.TRICK_FINISHED]
        elif self.phase == Phase.END:
            return []
        else:
            raise ValueError(
                f"Phase {self.phase} does not support legal actions retrieval")

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
            if action > self.taker_bid:
                self.taker_bid = action
        elif self.phase == Phase.CHIEN:
            Action.apply_chien_action(
                self.hands[self.current], self.discard, action)
            if len(self.discard) == Const.CHIEN_SIZE:
                self.phase = Phase.DECLARE
                self.update_taker_chien_hand()
                self.chien = self.discard.copy()
                self.discard.clear()
        elif self.phase == Phase.TRICK:
            # Don't apply trick action if it's TRICK_FINISHED
            if action != Const.TRICK_FINISHED:
                trick_winner = Action.apply_trick_action(
                    self.current, self.hands[self.current], self.trick, action)
                self.update_played_cards(action)
                if trick_winner:
                    self.tricks.append(trick_winner)
                if action == Const.FOOL:
                    self.fool_player = self.current
                    self.fool_state = Const.FOOL_NOT_PAID
                    self.fool_trick = self.trick
        elif self.phase == Phase.DECLARE:
            if action == Const.DECLARE_CHELEM:
                if not self.chelem_declared_taker and self.current == self.taker:
                    self.chelem_declared_taker = True
                elif not self.chelem_declared_defenders and self.current != self.taker:
                    self.chelem_declared_defenders = True
            if action == Const.DECLARE_POIGNEE:
                if not self.poignee_declared_taker and self.current == self.taker:
                    self.poignee_declared_taker = True
                elif not self.poignee_declared_defenders and self.current != self.taker:
                    self.poignee_declared_defenders = True
            chelem, poignee = self.declared[-1]
            if not chelem:
                chelem = True
            elif not poignee:
                poignee = True
            self.declared[-1] = (chelem, poignee)
        elif self.phase == Phase.TRICK_FINISHED:
            return
        else:
            raise ValueError(
                f"Phase {self.phase} does not support applying actions")

    def next(self) -> None:
        """
        Advances the game to the next phase based on the current phase and game state.
        """
        if self.phase == Phase.BIDDING:
            if len(self.bids) >= Const.NUM_PLAYERS:
                self.taker = self.bids.index(self.taker_bid)
                self.current = self.taker
                if self.taker_bid < Const.BID_GARDE_SANS:
                    self.hands[self.taker] += self.chien
                    self.phase = Phase.CHIEN
                else:
                    self.phase = Phase.DECLARE
                    self.current = self.taker
            else:
                self.next_player()
        elif self.phase == Phase.CHIEN:
            if len(self.discard) == Const.CHIEN_SIZE or self.taker_bid > Const.BID_GARDE:
                self.phase = Phase.DECLARE
        elif self.phase == Phase.TRICK:
            # Count cards played in current trick
            cards_played = len([card for card in self.trick if card != -1])
            if cards_played < Const.NUM_PLAYERS:
                self.next_player()
            else:
                self.phase = Phase.TRICK_FINISHED
        elif self.phase == Phase.DECLARE:
            chelem, poignee = self.declared[-1]
            if chelem and poignee:
                if len(self.declared) == Const.NUM_PLAYERS:
                    self.phase = Phase.TRICK
                    # Initialize trick properly
                    self.trick = [-1] * Const.NUM_PLAYERS
                self.next_player()
                self.declared.append((False, False))
        elif self.phase == Phase.TRICK_FINISHED:
            trick_winner = self.tricks[-1] if self.tricks else None
            if trick_winner is None:
                raise ValueError("No trick winner found")
            self.update_played_tricks(trick_winner)
            if len(self.tricks) < Const.NUM_TRICKS:
                self.phase = Phase.TRICK
                self.trick = [-1] * Const.NUM_PLAYERS
            else:
                # Game is over - we have all 18 tricks
                if len(self.tricks) == Const.NUM_TRICKS and self.fool_player is not None:
                    substitute_card = Action.apply_fool_action(
                        self.fool_trick, self.fool_player, self.tricks)
                    if substitute_card is None:
                        if self.fool_player == self.taker:
                            raise ValueError(
                                "No substitute card found for Fool in taker's tricks")
                        substitute_card = next(c for (player, trick) in self.tricks for c in trick if Card.value(
                            c) == 0.5 and player != self.fool_player and player != self.taker)

                    idx = self.played_tricks.index(Const.FOOL)
                    sub_idx = self.played_tricks.index(substitute_card)
                    self.played_tricks[sub_idx] = -1
                    self.played_tricks[idx] = substitute_card
                    self.update_played_tricks(
                        (self.fool_player, [Const.FOOL, -1, -1, -1]))
                self.phase = Phase.END
        elif self.phase == Phase.END:
            pass

    def returns(self) -> List[float]:
        if self.tricks == []:
            return [0.25] * Const.NUM_PLAYERS
        last_trick_winner, last_trick = self.tricks[-1]
        petit = False
        if Const.PETIT in last_trick and last_trick_winner == self.taker:
            petit = True
        tricks = [trick for player,
                  trick in self.tricks if player == self.taker]
        score, board = Utils.board_score(
            bid=self.bids[self.taker], taker=self.taker,
            tricks=tricks, chien=self.chien,
            chelem=self.chelem_declared_taker,
            poignee=self.poignee_declared_taker, petit=petit)
        self.score = score
        return board

    def update_played_cards(self, action: int) -> None:
        """
        Updates the played cards and the trick history after a trick action is applied.
        """
        card = Card.to_idx(action)
        self.played_cards[card] = self.current

    def update_played_tricks(self, trick_winner: Tuple[int, List[int]]) -> None:
        winner, trick = trick_winner
        idx = Utils.get_trick_position(self.tricks)
        self.played_tricks[idx] = winner
        idx += 1
        if len(self.played_tricks) <= idx + Const.NUM_PLAYERS - 1:
            raise ValueError(
                f"Trick history index {idx} out of bounds for trick history size {len(self.played_tricks)}")
        self.played_tricks[idx:idx + Const.NUM_PLAYERS] = trick

    def update_taker_chien_hand(self) -> None:
        self.taker_chien_hand = self.chien + self.discard
        chien_know_cards = set()
        for card in self.taker_chien_hand:
            if card not in chien_know_cards and card not in self.discard:
                chien_know_cards.add(card)
        self.taker_chien_hand = list(chien_know_cards)
        for card in self.discard:
            card_idx = Card.to_idx(card)
            self.played_cards[card_idx] = Const.CHIEN_ID

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

    def next_player(self) -> None:
        self.current = (self.current + 1) % Const.NUM_PLAYERS

    def tensor(self) -> List[int]:
        for hand in self.hands:
            hand.sort()

        current_trick = [-1] * (Const.NUM_PLAYERS)
        if self.trick:
            for i, card in enumerate(self.trick):
                current_trick[i] = card
        taker_know_cards = [-1] * Const.HAND_SIZE
        if self.taker >= 0:
            for i, card in enumerate(self.taker_chien_hand):
                taker_know_cards[i] = card

        tensor = [*self.played_cards, *taker_know_cards,
                  *self.played_tricks, *current_trick,
                  self.current, self.taker, *self.bids,
                  self.chelem_declared_taker, self.chelem_declared_defenders,
                  self.poignee_declared_taker, self.poignee_declared_defenders,
                  self.phase.value]
        return tensor

    def tensor_player(self, player: int) -> List[int]:
        tensor = self.tensor()
        played_cards = self.played_cards.copy()
        for hand in self.hands[player]:
            card_idx = Card.to_idx(hand)
            played_cards[card_idx] = player
        tensor[:Const.DECK_SIZE] = played_cards
        return tensor

    def from_tensor(self, tensor: List[int]) -> None:

        self.played_cards = Utils.get_mask(tensor, 'played_cards',)
        self.taker_chien_hand = Utils.get_mask(tensor, 'taker_chien_hand')
        self.played_tricks = Utils.get_mask(tensor, 'played_tricks')
        self.trick = Utils.get_mask(tensor, 'current_trick')
        self.current = Utils.get_mask(tensor, 'current_player')[0]
        self.taker = Utils.get_mask(tensor, 'taker_player')[0]
        self.bids = Utils.get_mask(tensor, 'bids')
        declarations = Utils.get_mask(tensor, 'declarations')
        phase = Utils.get_mask(tensor, 'phase')[0]

        self.tricks = Utils.get_tricks(self.played_tricks)
        self.chelem_declared_taker = bool(declarations[0])
        self.chelem_declared_defenders = bool(declarations[1])
        self.poignee_declared_taker = bool(declarations[2])
        self.poignee_declared_defenders = bool(declarations[3])
        self.phase = Const.Phase(phase)

    def clone(self):
        clone = Tarot()
        clone.from_tensor(self.tensor())
        clone.hands = [hand.copy() for hand in self.hands]
        clone.chien = self.chien.copy()
        clone.discard = self.discard.copy()
        clone.declared = self.declared.copy()
        return clone

    def action_to_string(self, action: int) -> str:
        if self.phase == Phase.BIDDING:
            return Bid.name(action)
        elif self.phase == Phase.DECLARE:
            if action == Const.DECLARE_NONE:
                return "No Declaration"
            if action == Const.DECLARE_CHELEM:
                return "Declare Chelem"
            if action == Const.DECLARE_POIGNEE:
                return "Declare Poignee"
        elif self.phase == Phase.CHIEN:
            return Card.name(action)
        elif self.phase == Phase.TRICK:
            return Card.name(action)
        return "N/A"

    def __str__(self) -> str:
        string = "=" * 40 + "\n"
        string += f"Current Player: {self.current}\n"
        string += f"Taker: {self.taker}\n"
        string += f"Phase: {self.phase}\n"
        string += f"Bids: {self.bids}\n"
        string += f"Chien [{len(self.chien)}]: {','.join([Card.name(card) for card in self.chien])}\n"
        for i in range(Const.NUM_PLAYERS):
            string += "=" * 10 + "\n"
            string += f"Player {i} {"[Taker]" if i == self.taker else "[Defender]"}\n"
            string += f"Bid: {Bid.name(self.bids[i])}\n" if len(
                self.bids) > i else "Bid: N/A\n"
            string += f"Hand [{len(self.hands[i])}]: {", ".join([Card.name(card) for card in self.hands[i]])}\n"
            string += f"Tricks [{len([trick for (p, trick) in self.tricks if p == i])}]: {', '.join([Card.name(card) for player, tricks in self.tricks if player == i for card in tricks])}\n"
        string += "=" * 10 + "\n"
        string += f"Taker Declared Chelem: {self.chelem_declared_taker}\n"
        string += f"Taker Declared Poignee: {self.poignee_declared_taker}\n"
        string += f"Defenders Declared Chelem: {self.chelem_declared_defenders}\n"
        string += f"Defenders Declared Poignee: {self.poignee_declared_defenders}\n"
        string += "=" * 40 + "\n"
        string += f"Player Cards: {', '.join([Card.name(card) if card != -1 else 'N/A' for card in self.played_cards])}\n"
        string += "=" * 40 + "\n"
        string += f"Trick History: {', '.join([Card.name(card) if card != -1 else 'N/A' for card in self.played_tricks])}\n"
        string += "=" * 40 + "\n"
        return string
