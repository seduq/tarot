from itertools import combinations
from typing import List, Tuple
import numpy as np
from .actions import Action
from .bids import Bid
from .cards import Card
from .utils import Utils
from . import constants as Const
from .constants import Phase


import pyspiel


class TarotGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(
            Const.GAME_TYPE,
            Const.GAME_INFO,
            params or dict()
        )

    def new_initial_state(self):
        return TarotGameState(self)


class TarotGameState(pyspiel.State):
    def __init__(self, game: TarotGame):
        super().__init__(game)
        self.deck = Card.deck()
        self.chien, self.hands = Card.deal(self.deck)
        self.phase = Phase.BIDDING
        self.bids: List[int] = []
        self.taker: int = -1
        self.taker_bid = Const.BID_PASS
        self.current: int = 0
        self.chien_discard: List[int] = []
        self.tricks: List[Tuple[int, List[int]]] = []
        self.trick: List[int] = []
        self.played_cards: List[int] = [-1] * Const.DECK_SIZE
        self.trick_history: List[int] = [-1] * (Const.MASK_NUM_TRICKS_SIZE)
        self.declared: List[Tuple[bool, bool]] = [(False, False)]
        self.chelem_declared_defenders = False
        self.poignee_declared_defenders = False
        self.chelem_declared_taker = False
        self.poignee_declared_taker = False
        self.game = game

    def next_player(self, current: int) -> int:
        if self.phase == Phase.DECLARE:
            chelem_declared, poignee_declared = self.declared[-1]
            if not chelem_declared or not poignee_declared:
                return self.current
        elif self.phase == Phase.CHIEN:
            return self.taker
        return (current + 1) % Const.NUM_PLAYERS

    def is_taker(self) -> bool:
        return self.current == self.taker

    def _update_played_card(self, card: int, player: int):
        card_idx = Card.to_idx(card)
        self.played_cards[card_idx] = player

    def _update_played_cards(self, cards: List[int], player: int):
        for card in list(cards):
            self._update_played_card(card, player)

    def _update_trick_history(self, cards: List[int], player: int):
        n_tricks = len(self.tricks) - 1
        idx = n_tricks * (Const.NUM_PLAYERS + 1)
        self.trick_history[idx] = player
        for i, card in enumerate(list(cards)):
            self.trick_history[idx + i + 1] = card

    def current_player(self):
        if self.phase == Phase.BIDDING:
            return pyspiel.PlayerId.CHANCE
        if self.phase == Phase.DECLARE:
            return pyspiel.PlayerId.CHANCE
        if self.phase == Phase.CHIEN:
            return self.taker
        if self.phase == Phase.END:
            return pyspiel.PlayerId.TERMINAL
        return self.current

    def legal_actions(self) -> List[int]:
        if self.phase == Phase.BIDDING:
            return [action for action, _ in self._legal_bidding_actions()]
        if self.phase == Phase.DECLARE:
            chelem_declared, _ = self.declared[-1]
            if not chelem_declared:
                return [action for action, _ in self._legal_chelem_action()]
            return [action for action, _ in self._legal_poignee_action()]
        if self.phase == Phase.CHIEN:
            return self._legal_chien_discards()
        if self.phase == Phase.TRICK:
            return Action.legal_trick_actions(self.hands[self.current], self.trick)
        return []

    def apply_action(self, action):
        if self.phase == Phase.BIDDING:
            self._apply_bidding_action(action)
        elif self.phase == Phase.DECLARE:
            chelem_declared, poignee_declared = self.declared[-1]
            if not chelem_declared:
                self._apply_chelem_action(action)
            elif not poignee_declared:
                self._apply_poignee_action(action)
            else:
                self.current = self.next_player(self.current)
                self.declared.append((False, False))
            if len(self.declared) > Const.NUM_PLAYERS:
                self.phase = Phase.TRICK
            return
        elif self.phase == Phase.CHIEN:
            self._apply_chien_discard(action)
        elif self.phase == Phase.TRICK:
            self._apply_trick_action(action)
        self.current = self.next_player(self.current)

    def _legal_bidding_actions(self):
        return Bid.legal_bids(self.bids)

    def _legal_chien_discards(self):
        return Action.legal_chien_actions(self.hands[self.taker])

    def _legal_chelem_action(self):
        if not self.is_taker() and self.poignee_declared_defenders:
            return [(Const.DECLARE_NONE, 1.0)]
        if self.poignee_declared_defenders:
            return [(Const.DECLARE_NONE, 1.0)]
        actions = Action.legal_chelem_actions(self.hands[self.current])
        return actions

    def _legal_poignee_action(self):
        if not self.is_taker() and self.poignee_declared_defenders:
            return [(Const.DECLARE_NONE, 1.0)]
        if self.poignee_declared_defenders:
            return [(Const.DECLARE_NONE, 1.0)]
        actions = Action.legal_poignee_actions(self.hands[self.current])
        return actions

    def _legal_trick_actions(self):
        return Action.legal_trick_actions(self.hands[self.current], self.trick)

    def _apply_chelem_action(self, action):
        if action == Const.DECLARE_CHELEM and self.current == self.taker:
            self.chelem_declared_taker = True
        elif action == Const.DECLARE_CHELEM and self.current != self.taker:
            self.chelem_declared_defenders = True
        self.declared[-1] = (True, False)

    def _apply_poignee_action(self, action):
        if action == Const.DECLARE_POIGNEE and self.current == self.taker:
            self.poignee_declared_taker = True
        elif action == Const.DECLARE_POIGNEE and self.current != self.taker:
            self.poignee_declared_defenders = True
        if action == Const.DECLARE_POIGNEE:
            trumps = [card for card in self.hands[self.current]
                      if Card.is_trump(card)]
            self._update_played_cards(trumps, self.current)
        self.declared[-1] = (True, True)

    def _apply_bidding_action(self, action):
        self.bids.append(action)
        if len(self.bids) == Const.NUM_PLAYERS:
            self.taker, self.taker_bid = Bid.finish_bidding(self.bids)
            if self.taker_bid == Const.BID_PASS:
                self.phase = Phase.END
                return
            if self.taker_bid in [Const.PETIT, Const.BID_GARDE]:
                self._update_played_cards(self.chien, (Const.CHIEN_ID + 1))
                self.hands[self.taker] += self.chien
                self.current = self.taker
                self.phase = Phase.CHIEN
            else:
                self.current = self.taker
                self.phase = Phase.DECLARE

    def _apply_chien_discard(self, action):
        self.chien_discard.append(action)
        self.hands[self.taker].remove(action)
        if len(self.chien_discard) == Const.CHIEN_SIZE:
            self._update_played_cards(
                self.chien_discard, Const.CHIEN_ID)
            self.chien = self.chien_discard.copy()
            self.chien_discard = []
            self.current = self.taker
            self.phase = Phase.DECLARE

    def _apply_trick_action(self, action):
        card_played = action
        hand = self.hands[self.current]
        trick_winner = None
        if card_played == Const.FOOL:
            trick_winner = Action.apply_excuse_action(
                hand, self.current, self.taker,
                self.trick, self.tricks, card_played)
        else:
            trick_winner = Action.apply_trick_action(
                hand, self.trick, card_played)
        if trick_winner:
            self.tricks.append(trick_winner)
            self.current = trick_winner[0]
            self._update_trick_history(self.trick, self.current)
            self.trick = []
        self._update_played_card(card_played, self.current)
        if all(len(h) == 0 for h in self.hands):
            self.phase = Phase.END

    def is_terminal(self):
        return self.phase == Phase.END

    def is_chance_node(self):
        return self.phase in [Phase.BIDDING, Phase.DECLARE]

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        if self.phase == Phase.BIDDING:
            return self._legal_bidding_actions()
        if self.phase == Phase.DECLARE:
            chelem_declared, poignee_declared = self.declared[-1]
            if not chelem_declared and not poignee_declared:
                return self._legal_chelem_action()
            if chelem_declared:
                return self._legal_poignee_action()
            return [(Const.DECLARE_NONE, 1.0)]
        return []

    def returns(self):
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

    def tensor(self) -> List[int]:
        for hand in self.hands:
            np.sort(hand)

        current_trick = [-1] * (Const.NUM_PLAYERS)
        if self.trick:
            for i, card in enumerate(self.trick):
                current_trick[i] = card

        tensor = [*self.played_cards, *current_trick, *self.trick_history,
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

        played_cards = Utils.get_mask(tensor, 'played_cards',)
        current_trick = Utils.get_mask(tensor, 'current_trick')
        trick_history = Utils.get_mask(tensor, 'trick_history')
        current_player = Utils.get_mask(tensor, 'current_player')
        taker_player = Utils.get_mask(tensor, 'taker_player')
        bids = Utils.get_mask(tensor, 'bid')
        declarations = Utils.get_mask(tensor, 'declarations')
        phase = Utils.get_mask(tensor, 'phase')

        self.played_cards = played_cards
        self.trick = [t for t in current_trick if t != -1]
        self.trick_history = trick_history
        self.tricks = Utils.get_tricks(trick_history)
        self.current = current_player[0]
        self.taker = taker_player[0]
        self.bids = bids
        self.chelem_declared_taker = bool(declarations[0])
        self.chelem_declared_defenders = bool(declarations[1])
        self.poignee_declared_taker = bool(declarations[2])
        self.poignee_declared_defenders = bool(declarations[3])
        self.phase = Const.Phase(phase[0])

    def clone(self):
        clone = TarotGameState(self.game)
        clone.from_tensor(self.tensor())
        clone.hands = [hand.copy() for hand in self.hands]
        clone.chien = self.chien.copy()
        clone.chien_discard = self.chien_discard.copy()
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

    @staticmethod
    def pretty_print(state: 'TarotGameState') -> str:
        string = f"Current Player: {state.current}\n"
        string += f"Taker: {state.taker}\n"
        string += f"Phase: {state.phase}\n"
        string += f"Bids: {state.bids}\n"
        string += f"Chien [{len(state.chien)}]: {','.join([Card.name(card) for card in state.chien])}\n"
        for i in range(Const.NUM_PLAYERS):
            string += "=" * 10 + "\n"
            string += f"Player {i} {"[Taker]" if i == state.taker else "[Defender]"}\n"
            string += f"Bid: {Bid.name(state.bids[i])}\n" if len(
                state.bids) > i else "Bid: N/A\n"
            string += f"Hand [{len(state.hands[i])}]: {", ".join([Card.name(card) for card in state.hands[i]])}\n"
            string += f"Tricks [{len([trick for (p, trick) in state.tricks if p == i])}]: {', '.join([Card.name(card) for player, tricks in state.tricks if player == i for card in tricks])}\n"
        string += "=" * 10 + "\n"
        string += f"Taker Declared Chelem: {state.chelem_declared_taker}\n"
        string += f"Taker Declared Poignee: {state.poignee_declared_taker}\n"
        string += f"Defenders Declared Chelem: {state.chelem_declared_defenders}\n"
        string += f"Defenders Declared Poignee: {state.poignee_declared_defenders}\n"
        string += "=" * 10 + "\n"
        string += f"Player Cards: {', '.join([Card.name(card) if card != -1 else 'N/A' for card in state.played_cards])}\n"
        string += f"Trick History: {', '.join([Card.name(card) if card != -1 else 'N/A' for card in state.trick_history])}\n"
        return string

    @staticmethod
    def possible_bid_combinations() -> List[List[int]]:
        bid_passes: List[List[int]] = [[Const.BID_PASS]
                                       * r for r in range(Const.NUM_PLAYERS)]
        unique_bids = []
        for _pass in bid_passes:
            size = max(Const.NUM_PLAYERS - len(_pass), Const.NUM_PLAYERS)
            bid_combinations = combinations(Const.BIDS, size)
            for bid in bid_combinations:
                player_bids = _pass + list(bid)
                bid_list_sorted = list(sorted(player_bids))
                unique_bids.append(bid_list_sorted)

        return unique_bids

    @staticmethod
    def possible_declare_combinations() -> List[Tuple[int, int]]:
        all_declarations = combinations(Const.DECLARES, 2)
        return list(all_declarations)
