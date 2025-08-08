
from typing import List, Optional
from .actions import legal_chien_actions, legal_trick_actions, apply_chien_action, apply_trick_action
from typing import List, Optional
from .bids import bid_name, legal_bids
from .cards import card_name, deck, deal
from . import constants as Const
from .constants import Phase
from .utils import board_score
from .agents import Agent
from .state import TarotState


class Tarot:
    @staticmethod
    def new(state: Optional[TarotState] = None) -> TarotState:
        if state is None:
            chien, hands = deal(deck())
            state = TarotState(
                chien=chien,
                hands=hands,
                phase=Phase.BIDDING,
                bids=[],
                taker=-1,
                taker_bid=Const.BID_PASS,
                current=0,
                discard=[],
                tricks=[],
                trick_player=-1,
                trick=[-1] * Const.NUM_PLAYERS,
                trick_winners=[-1] * Const.NUM_TRICKS,
                know_cards=[-1] * Const.DECK_SIZE,
                declared=[(False, False)],
                chelem_def=False,
                poignee_def=[False] * Const.NUM_PLAYERS,
                chelem_taker=False,
                poignee_taker=False,
                fool_paid=False,
                fool_player=-1,
                fool_trick=-1,
                turn_counter=0
            )
        return state

    @staticmethod
    def play(state: TarotState, agents: List[Agent]) -> None:
        """
        Simulates the game until it reaches a terminal state.
        This method will call the agents to choose actions based on the current state.
        """
        while not Tarot.is_terminal(state):
            legal_actions = Tarot.legal_actions(state)
            action = agents[state.current].choose(state, legal_actions)
            Tarot.apply_action(state, action)
            for agent in agents:
                agent.update(state, action, state.current)
            Tarot.next(state)

    @staticmethod
    def legal_actions(state: TarotState) -> List[int]:
        if state.phase == Phase.BIDDING:
            return [bid for bid, _ in legal_bids(state.bids)]
        elif state.phase == Phase.CHIEN:
            return legal_chien_actions(state.hands[state.current])
        elif state.phase == Phase.TRICK:
            return legal_trick_actions(
                state.hands[state.current], state.trick)
        elif state.phase == Phase.DECLARE:
            chelem_declared, poignee_declared = state.declared[-1]
            if not chelem_declared:
                return [Const.DECLARE_CHELEM, Const.DECLARE_NONE]
            if not poignee_declared:
                return [Const.DECLARE_POIGNEE, Const.DECLARE_NONE]
            return [Const.DECLARE_NONE]
        elif state.phase == Phase.END:
            return []
        else:
            raise ValueError(
                f"Phase {state.phase} does not support legal actions retrieval")

    @staticmethod
    def apply_action(state: TarotState, action: int) -> None:
        if state.phase == Phase.BIDDING:
            state.bids.append(action)
            if action > state.taker_bid:
                state.taker_bid = action
        elif state.phase == Phase.CHIEN:
            apply_chien_action(
                state.hands[state.current], state.discard, action)
            if len(state.discard) == Const.CHIEN_SIZE:
                state.phase = Phase.DECLARE
                state.chien = state.discard.copy()
                state.discard.clear()
        elif state.phase == Phase.TRICK:
            trick_winner = apply_trick_action(
                state.current, state.hands[state.current], state.trick, action)

            if state.current == state.taker and action in state.know_cards:
                state.know_cards.remove(action)

            if trick_winner is None:
                return

            winner, trick = trick_winner
            state.trick_winners.append(winner)
            state.tricks.append(trick)
            state.trick_player = winner
            if len(state.tricks) < Const.NUM_TRICKS:
                state.trick = [-1] * Const.NUM_PLAYERS
                state.turn_counter += 1
            if len(state.tricks) == Const.NUM_TRICKS:
                state.phase = Phase.END
        elif state.phase == Phase.DECLARE:
            if action == Const.DECLARE_CHELEM:
                if state.current == state.taker:
                    state.chelem_taker = True
                elif state.current != state.taker:
                    state.chelem_def = True
            if action == Const.DECLARE_POIGNEE:
                if state.current == state.taker:
                    state.poignee_taker = True
                elif state.current != state.taker:
                    state.poignee_def[state.current] = True
            chelem, poignee = state.declared[-1]
            if not chelem:
                chelem = True
            elif not poignee:
                poignee = True
            state.declared[-1] = (chelem, poignee)
        else:
            raise ValueError(
                f"Phase {state.phase} does not support applying actions")

    @staticmethod
    def next(state: TarotState) -> None:
        """
        Advances the game to the next phase based on the current phase and game state.
        """
        if state.phase == Phase.BIDDING:
            if len(state.bids) >= Const.NUM_PLAYERS:
                state.taker = state.bids.index(state.taker_bid)
                state.current = state.taker
                state.trick_player = state.taker
                if state.taker_bid < Const.BID_GARDE_SANS:
                    state.hands[state.taker] += state.chien
                    state.phase = Phase.CHIEN
                else:
                    state.phase = Phase.DECLARE
                    state.current = state.taker
            else:
                Tarot.next_player(state)
        elif state.phase == Phase.CHIEN:
            if len(state.discard) == Const.CHIEN_SIZE or state.taker_bid > Const.BID_GARDE:
                state.phase = Phase.DECLARE
        elif state.phase == Phase.TRICK:
            # Count cards played in current trick
            cards_played = len([card for card
                                in state.trick if card != -1])
            if cards_played < Const.NUM_PLAYERS:
                Tarot.next_player(state)
                return

            trick_winner = state.trick_winners[-1]
            state.current = trick_winner
            if trick_winner is None:
                raise ValueError("No trick winner found")
        elif state.phase == Phase.DECLARE:
            chelem, poignee = state.declared[-1]
            if chelem and poignee:
                if len(state.declared) == Const.NUM_PLAYERS:
                    state.phase = Phase.TRICK
                    state.trick = [-1] * Const.NUM_PLAYERS
                Tarot.next_player(state)
                state.declared.append((False, False))
        elif state.phase == Phase.END:
            pass

    @staticmethod
    def returns(state: TarotState) -> List[float]:
        # If something went wrong, return 0 for all players
        if state.tricks == []:
            return [0] * Const.NUM_PLAYERS
        if not state.fool_paid:
            return [0] * Const.NUM_PLAYERS
        last_trick = state.tricks[-1]
        last_trick_winner = state.trick_winners[-1]
        petit = False
        if Const.PETIT in last_trick and last_trick_winner == state.taker:
            petit = True
        tricks = [trick for i, trick in enumerate(state.tricks)
                  if state.taker == state.trick_winners[i]]
        score, board = board_score(
            bid=state.bids[state.taker], taker=state.taker,
            tricks=tricks, chien=state.chien,
            chelem=state.chelem_taker,
            poignee=state.poignee_taker, petit=petit)
        return board

    @staticmethod
    def is_chance_node(state: TarotState) -> bool:
        """
        Returns True if the current phase is a chance node (BIDDING, CHIEN, or DECLARE).
        """
        return state.phase in {Phase.BIDDING, Phase.CHIEN, Phase.DECLARE}

    @staticmethod
    def is_terminal(state: TarotState) -> bool:
        """
        Returns True if the game is in the END phase, indicating a terminal state.
        """
        return state.phase == Phase.END

    @staticmethod
    def next_player(state: TarotState) -> None:
        state.current = (state.current + 1) % Const.NUM_PLAYERS

    @staticmethod
    def clone(state: TarotState) -> TarotState:
        clone = TarotState(
            chien=state.chien.copy(),
            hands=[hand.copy() for hand in state.hands],
            phase=state.phase,
            bids=state.bids.copy(),
            taker=state.taker,
            taker_bid=state.taker_bid,
            current=state.current,
            discard=state.discard.copy(),
            tricks=state.tricks.copy(),
            trick=state.trick.copy(),
            trick_player=state.trick_player,
            trick_winners=state.trick_winners.copy(),
            know_cards=state.know_cards.copy(),
            declared=state.declared.copy(),
            chelem_def=state.chelem_def,
            poignee_def=state.poignee_def,
            chelem_taker=state.chelem_taker,
            poignee_taker=state.poignee_taker,
            fool_paid=state.fool_paid,
            turn_counter=state.turn_counter,
            fool_player=state.fool_player,
            fool_trick=state.fool_trick
        )
        return clone

    @staticmethod
    def action_to_string(state: TarotState, action: int) -> str:
        if state.phase == Phase.BIDDING:
            return bid_name(action)
        elif state.phase == Phase.DECLARE:
            if action == Const.DECLARE_NONE:
                return "No Declaration"
            if action == Const.DECLARE_CHELEM:
                return "Declare Chelem"
            if action == Const.DECLARE_POIGNEE:
                return "Declare Poignee"
        elif state.phase == Phase.CHIEN:
            return card_name(action)
        elif state.phase == Phase.TRICK:
            return card_name(action)
        return "N/A"

    @staticmethod
    def to_string(state: TarotState) -> str:
        string = "=" * 40 + "\n"
        string += f"Current Player: {state.current}\n"
        string += f"Taker: {state.taker}\n"
        string += f"Phase: {state.phase}\n"
        string += f"Bids: {state.bids}\n"
        string += f"Chien [{len(state.chien)}]: {','.join([card_name(card) for card in state.chien])}\n"
        for i in range(Const.NUM_PLAYERS):
            string += "=" * 10 + "\n"
            string += f"Player {i} {"[Taker]" if i == state.taker else "[Defender]"}\n"
            string += f"Bid: {bid_name(state.bids[i])}\n" if len(
                state.bids) > i else "Bid: N/A\n"
            string += f"Hand [{len(state.hands[i])}]: {", ".join([card_name(card) for card in state.hands[i]])}\n"
            string += f"Tricks [{len([trick for (p, trick) in state.tricks if p == i])}]: {', '.join(
                [card_name(card) for j, tricks in enumerate(state.tricks) if j == i for card in tricks])}\n"
        string += "=" * 10 + "\n"
        string += f"Taker Declared Chelem: {state.chelem_taker}\n"
        string += f"Taker Declared Poignee: {state.poignee_taker}\n"
        string += f"Defenders Declared Chelem: {state.chelem_def}\n"
        string += f"Defenders Declared Poignee: {state.poignee_def}\n"
        string += "=" * 40 + "\n"
        string += f"Know Cards: {', '.join([card_name(card) if card != -1 else 'N/A' for card in state.know_cards])}\n"
        string += "=" * 40 + "\n"
        string += f"Tricks [{len(state.tricks)}]:\n"
        for i, trick in enumerate(state.tricks):
            string += f"Trick {i + 1}: {', '.join([card_name(card) for card in trick])}\n"
        string += "=" * 40 + "\n"
        return string
