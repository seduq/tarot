from typing import List, Tuple
from game.cards import Card
from game.bids import Bid
from game.game import Phase, TarotGame, TarotGameState


class Utils:
    MIN_POIGNEE = 10
    MIN_POINTS_PER_BOUT = {
        3: 36,
        2: 41,
        1: 46,
        0: 51
    }
    POIGNEE_BONUS = {
        10: 20,
        13: 30,
        15: 40,
    }
    MASK_PLAYER = 1
    MASK_TAKER = 1
    MASK_PHASE = 1

    @staticmethod
    def score(tricks: List[List[int]]) -> float:
        """Calcula a pontuação total de uma lista de truques."""
        return sum([Card.value(card) for trick in tricks for card in trick])

    @staticmethod
    def distribute_points(tricks: List[List[int]], chien: List[int], bid: int,
                          chelem: bool, poignee: bool, taker: int) -> List[float]:
        """    
        Distribui os pontos entre os jogadores com base no lance, truques, chien, chelem e poignee.
        Retorna uma lista de pontos para cada jogador.
        """

        score = Utils.total_score(tricks, chien, bid, chelem, poignee)
        points = score / (TarotGame.NUM_PLAYERS - 1)
        board = [0.0] * TarotGame.NUM_PLAYERS
        board[taker] = score * 3
        for player in range(TarotGame.NUM_PLAYERS):
            if player != taker:
                board[player] = - points
        return board

    @staticmethod
    def partial_score(tricks: List[List[int]], chien: List[int], bid: int) -> Tuple[float, bool]:
        """Calcula a pontuação parcial de um jogo de game.
        Retorna a pontuação e se os pontos são suficientes para o lance.
        """

        points = 0
        bouts = sum(1 for trick in tricks for card in trick if card in [
                    Card.FOOL, Card.PETIT, Card.MONDE])

        required_points = Utils.MIN_POINTS_PER_BOUT[bouts]

        points += sum([Card.value(card) for trick in tricks for card in trick])

        # Apenas adiciona pontos do chien se o lance for Petit ou Garde
        # Garde Sans e Garde Contre não sabem do chien
        points += sum([Card.value(card) for card in chien]
                      ) if bid in [Bid.PETIT, Bid.GARDE] else 0

        required = True if (points - required_points) > 0 else False
        score = 25 + abs(points - required_points)

        return score, required

    @staticmethod
    def total_score(tricks: List[List[int]], chien: List[int], bid: int, chelem: bool, poignee: bool) -> float:
        """Calcula a pontuação total de um jogo de game."""
        bonus = 0
        cards = [card for trick in tricks for card in trick]

        # Adiciona o chien no final do jogo se for Garde sans
        points, required = Utils.partial_score(tricks, chien, bid)
        points += sum([Card.value(card) for card in chien]
                      ) if bid in [Bid.GARDE_SANS] else 0

        # Se foi declarado chelem, dobra o bônus
        grand_chelem = len(cards) == Card.DECK_SIZE
        if (grand_chelem and chelem):
            bonus += 400
        elif (grand_chelem):
            bonus += 200

        trumps = sum(1 for card in cards if Card.rank(card) == Card.TRUMP)

        # Se foi declarado poignee, adiciona o bônus
        for poignee_size, poignee_bonus in Utils.POIGNEE_BONUS.items():
            if (poignee and trumps >= poignee_size):
                bonus += poignee_bonus
                break

        # Petit au bout
        if (tricks[-1][-1] == Card.PETIT):
            points += 10

        # S = ((25 + points + petit_au_bout) * multiplier) + bonus
        score = points * Bid.multiplier(bid) + bonus

        return score if required else -score

    @staticmethod
    def trick_winner(trick: List[int]) -> int:
        """Determina o vencedor do trick."""

        lead_suit = Card.suit(trick[0])
        trumps = [(i, c) for i, c in enumerate(trick) if Card.is_trump(c)]
        if trumps:
            # Vence o maior trunfo
            return max(trumps, key=lambda x: Card.rank(x[1]))[0]
        # Vence a maior carta do naipe liderado
        lead_cards = [(i, c)
                      for i, c in enumerate(trick) if Card.suit(c) == lead_suit]
        return max(lead_cards, key=lambda x: Card.rank(x[1]))[0]

    # Estado do jogo, jogador atual, jogador tomador, lances, chelem, poignee, fase
    information_string_mask = [Card.DECK_SIZE,
                               (Card.DECK_SIZE + TarotGame.NUM_TRICKS),
                               MASK_PLAYER, MASK_TAKER, TarotGame.NUM_PLAYERS,
                               TarotGame.NUM_PLAYERS, TarotGame.NUM_PLAYERS, MASK_PHASE]

    @staticmethod
    def information_string(state: TarotGameState) -> str:
        """Gera uma string de observação para o jogador."""
        state_str = [*state.known_cards, *state.known_tricks,
                     state.current, state.taker, *state.bids,
                     *[value for sublist in state.chelem for value in sublist],
                     *[value for sublist in state.poignee for value in sublist],
                     state.phase.value]
        return str(state_str)

    @staticmethod
    def information_from_string(game: TarotGame, state_str: str) -> TarotGameState:
        """Cria um estado de jogo a partir de uma string de observação."""
        vector = eval(state_str)

        def get_v(vector, start, size) -> Tuple[int, List[int]]:
            return start + size, vector[start:start + size]

        start, known_cards = get_v(vector, 0,
                                   Card.DECK_SIZE)
        start, known_tricks = get_v(vector, start,
                                    (Card.DECK_SIZE + TarotGame.NUM_TRICKS))
        start, current = get_v(vector, start, 1)
        start, taker = get_v(vector, start, 1)
        start, bids = get_v(vector, start, TarotGame.NUM_PLAYERS)
        start, chelem = get_v(vector, start, 2 * TarotGame.NUM_PLAYERS)
        start, poignee = get_v(vector, start, 2 * TarotGame.NUM_PLAYERS)
        start, phase = get_v(vector, start, 1)

        state = game.new_initial_state()
        state.phase = Phase(phase)

        state.known_cards = known_cards
        state.known_tricks = known_tricks

        state.current = current[0]
        state.taker = taker[0]
        state.bids = bids

        state.chelem = []
        state.poignee = []

        for i in range(TarotGame.NUM_TRICKS):
            state.chelem.append((chelem[i * 2], chelem[i * 2 + 1]))
            state.poignee.append((poignee[i * 2], poignee[i * 2 + 1]))

            idx = i * (TarotGame.NUM_PLAYERS + 1)
            current_player = known_tricks[idx]
            if (current_player == - 1):
                break
            start = idx + 1
            current_trick = known_tricks[start:start +
                                         TarotGame.NUM_PLAYERS].copy()
            if -1 in current_trick:
                state.trick = current_trick[:current_trick.index(-1)]
                break
            else:
                state.tricks.append((current_player, current_trick))

        return state
