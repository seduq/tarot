from enum import Enum
import pyspiel
from typing import List, Tuple, Optional

from game.actions import Action
from game.bids import Bid
from game.cards import Card
from game.utils import Utils


class Phase(Enum):
    BIDDING = 1
    CHIEN = 2
    CHELEM = 3
    POIGNEE = 4
    TRICK = 5
    END = 6


class TarotGame(pyspiel.Game):
    NUM_PLAYERS = 4
    HAND_SIZE = 18
    CHIEN_SIZE = 6
    CHIEN_ID = NUM_PLAYERS + 1
    NUM_TRICKS = 18

    def __init__(self, params=None):
        super().__init__(
            pyspiel.GameType(
                short_name="french_tarot",
                long_name="French Tarot",
                dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
                chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
                information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
                utility=pyspiel.GameType.Utility.ZERO_SUM,
                reward_model=pyspiel.GameType.RewardModel.TERMINAL,
                max_num_players=TarotGame.NUM_PLAYERS,
                min_num_players=TarotGame.NUM_PLAYERS,
                provides_information_state_string=True,
                provides_information_state_tensor=False,
                provides_observation_string=False,
                provides_observation_tensor=False,
                parameter_specification={}
            ),
            pyspiel.GameInfo(
                num_distinct_actions=Card.DECK_SIZE,
                max_chance_outcomes=Bid.NUM_BIDS + Bid.NUM_DECLARES,
                num_players=TarotGame.NUM_PLAYERS,
                min_utility=-1.0,
                max_utility=1.0,
                utility_sum=0.0,
                max_game_length=TarotGame.NUM_PLAYERS * TarotGame.NUM_TRICKS +
                TarotGame.NUM_PLAYERS + TarotGame.CHIEN_SIZE + 2 * TarotGame.NUM_PLAYERS
                # 4 jogadores * 18 tricks + 4 apostas +  6 chien + 4 chelem + 4 poignee
            ),
            params or dict()
        )

    def new_initial_state(self):
        return TarotGameState(self)


class TarotGameState(pyspiel.State):
    def __init__(self, game: TarotGame):
        """
        Inicializa o estado do jogo de Tarot Francês.
        """
        super().__init__(game)
        self.deck = Card.deck()
        self.chien, self.hands = Card.deal(self.deck)
        self.phase = Phase.BIDDING
        self.bids = []
        self.taker: int = 0
        self.taker_bid = Bid.PASS
        self.current: int = 0
        self.chien_discard: List[int] = []
        self.tricks: List[Tuple[int, List[int]]] = []
        self.trick: List[int] = []
        self.known_cards = [-1] * Card.DECK_SIZE
        self.known_tricks = [-1] * \
            (TarotGame.NUM_PLAYERS + 1) * TarotGame.NUM_TRICKS
        self.chelem: List[Tuple[int, int]] = []
        self.poignee: List[Tuple[int, int]] = []

    def next_player(self, current: int) -> int:
        """Retorna o próximo jogador."""
        if self.phase == Phase.CHIEN:
            self.phase = Phase.TRICK
            return self.taker
        if self.phase == Phase.CHELEM:
            self.phase = Phase.POIGNEE
            return self.current
        if self.phase == Phase.POIGNEE:
            if len(self.poignee) < TarotGame.NUM_PLAYERS:
                self.phase = Phase.CHELEM
            else:
                self.phase = Phase.TRICK
        return (current + 1) % TarotGame.NUM_PLAYERS

    def _update_known_cards(self, card: int, player: int):
        """Atualiza o vetor known_cards quando uma carta é jogada."""
        self.known_cards[card] = player

    def _update_known_tricks(self, cards: List[int], player: int):
        """Atualiza o vetor known_tricks quando uma carta é jogada."""
        n_tricks = len(self.tricks)
        idx = n_tricks * (TarotGame.NUM_PLAYERS + 1)
        self.known_tricks[idx] = player
        for card_idx, card in enumerate(cards):
            self.known_tricks[idx + card_idx] = card

    def current_player(self):
        """
        Retorna o jogador atual, de acordo com a fase do jogo.
        """
        if self.phase == Phase.CHELEM:
            return pyspiel.PlayerId.CHANCE
        if self.phase == Phase.POIGNEE:
            return pyspiel.PlayerId.CHANCE
        if self.phase == Phase.CHIEN:
            return self.taker
        if self.phase == Phase.END:
            return pyspiel.PlayerId.TERMINAL
        return self.current

    def legal_actions(self):
        """
        Retorna as ações legais para o jogador atual, de acordo com a fase.
        """
        if self.phase == Phase.CHELEM:
            return [Bid.DECLARE_NONE, Bid.DECLARE_CHELEM]
        elif self.phase == Phase.POIGNEE:
            return [Bid.DECLARE_NONE, Bid.DECLARE_POIGNEE]
        elif self.phase == Phase.BIDDING:
            return self._legal_bidding_actions()
        elif self.phase == Phase.CHIEN:
            return self._legal_chien_discards()
        elif self.phase == Phase.TRICK:
            return self._legal_trick_actions()
        return []

    def apply_action(self, action):
        """
        Aplica a ação escolhida pelo jogador atual, atualizando o estado do jogo.
        """
        if self.phase == Phase.BIDDING:
            self._apply_bidding_action(action)
        elif self.phase == Phase.CHELEM:
            self._apply_chelem_action(action)
        elif self.phase == Phase.POIGNEE:
            self._apply_poignee_action(action)
        elif self.phase == Phase.CHIEN:
            self._apply_chien_discard(action)
        elif self.phase == Phase.TRICK:
            self._apply_trick_action(action)

        self.current = self.next_player(self.current)

    def _legal_bidding_actions(self):
        """Ações legais de aposta."""
        return Bid.legal_bids(self.bids)

    def _legal_chien_discards(self):
        """Retorna os índices das cartas que podem ser descartadas pelo tomador."""
        return Action.legal_chien_discards(self.hands[self.taker])

    def _legal_chelem_action(self):
        """Retorna as ações legais de chelem."""
        return [(Bid.DECLARE_NONE, 0.5), (Bid.DECLARE_CHELEM, 0.5)]

    def _legal_poignee_action(self):
        """Retorna as ações legais de poignee."""
        trumps = Card.count_trumps(self.hands[self.current])
        enough_trumps = trumps >= Utils.MIN_POIGNEE
        return [(Bid.DECLARE_NONE, 0.5 if enough_trumps else 1.0), (Bid.DECLARE_POIGNEE, 0.5 if enough_trumps else 0.0)]

    def _legal_trick_actions(self):
        """Retorna as cartas legais para jogar no trick atual."""
        return Action.legal_trick_actions(self.hands[self.current], self.trick)

    def _apply_chelem_action(self, action):
        """Processa uma ação de chelem."""
        self.chelem.append(action)
        self.phase = Phase.POIGNEE

    def _apply_poignee_action(self, action):
        """Processa uma ação de poignee."""
        self.poignee.append(action)
        if len(self.poignee) == TarotGame.NUM_PLAYERS:
            self.phase = Phase.TRICK
            self.current = self.taker

    def _apply_bidding_action(self, action):
        """Processa uma ação de aposta."""
        self.bids.append(action)
        if len(self.bids) == TarotGame.NUM_PLAYERS:
            self.taker, self.taker_bid = Bid.finish_bidding(self.bids)
            if self.taker_bid in [Bid.PETIT, Bid.GARDE, Bid.GARDE_SANS]:
                self.hands[self.taker] += self.chien
                self.phase = Phase.CHIEN
            else:
                self.tricks.append((-1, self.chien))
                self.phase = Phase.TRICK
            self.chien = []

    def _apply_chien_discard(self, action):
        """Processa o descarte e a troca de cartas com chien."""
        card_idx = self.hands[self.taker].index(action)
        card_to_discard = self.hands[self.taker][card_idx]
        self.chien_discard.append(card_to_discard)
        self.hands[self.taker].pop(card_idx)
        if len(self.chien_discard) == TarotGame.CHIEN_SIZE:
            self.phase = Phase.TRICK
            self.current = self.taker
            self.chien = self.chien_discard.copy()
            self.chien_discard = []

    def _apply_trick_action(self, action):
        """Processa a jogada de uma carta em um trick e atualiza known_cards."""
        card_played = action
        self.trick.append(card_played)

        self.hands[self.current], trick_winner = Action.apply_trick_action(
            self.hands[self.current], self.trick, card_played)

        if trick_winner:
            self.tricks.append(trick_winner)
            self.current = trick_winner[0]
            self.trick = []

        if all(len(h) == 0 for h in self.hands):
            self.phase = Phase.END

        self._update_known_cards(card_played, self.current)
        self._update_known_tricks(self.trick, self.current)

    def is_terminal(self):
        """
        Retorna True se o jogo terminou.
        """
        return self.phase == Phase.END

    def chance_outcome(self):
        """
        Se a fase atual for de apostas, retorna probabilidade de lances.
        Se a fase for inicial, observa se algum jogador pode declarar chelem ou poignee.
        """
        if self.phase == Phase.BIDDING:
            return self._legal_bidding_actions()

        if self.phase == Phase.CHELEM:
            return self._legal_chelem_action()

        if self.phase == Phase.POIGNEE:
            return self._legal_poignee_action()
        return []

    def returns(self):
        """
        Calcula o retorno dos jogadores ao final do jogo.
        """
        points = Utils.distribute_points(
            tricks=[trick for player,
                    trick in self.tricks if player == self.taker],
            bid=self.taker_bid or Bid.PETIT, taker=self.taker or 0,
            chien=self.chien_discard, chelem=False, poignee=False)
        return points

    def information_state_string(self):
        Utils.information_string(self)
