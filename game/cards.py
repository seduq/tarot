import random
from typing import List, Tuple
from game.game import TarotGame


class Card:
    """ 
    Classe para manipulação de cartas de game.
    O deck é composto por 78 cartas
    Cada naipe tem 14 cartas (de 1 a 14)
    e o trunfo tem 22 cartas (de 0 a 21)
    A array é formatada como naipe * 100 + valor
    Exemplo: 1 de coração = 1 * 100 + 1 = 101
    Exemplo: 22 de trunfo = 5 * 100 + 22 = 522
    Os naipes são da seguinte ordem: Espadas, Copas, Paus, Ouros
    """
    # Cartas especiais para o tarot
    FOOL = 500
    PETIT = 501
    MONDE = 521
    BOUTS = [FOOL, PETIT, MONDE]
    TRUMPS = 22

    TRUMP = 5
    HEART = 4
    DIAMOND = 3
    CLUB = 2
    SPADE = 1

    JACK = 11
    KNIGHT = 12
    QUEEN = 13
    KING = 14

    DECK_SIZE = 78

    # Pontuação máxima em mão: 60 para jogos de tarot com 4 jogadores.
    # - 3 * 4.5 (Petit, Monde, Fool)
    # - 4 * 4.5 (Kings)
    # - 4 * 3.5 (Queens)
    # - 4 * 2.5 (Knights)
    # - 3 * 1.5 (Jacks)
    MAX_POINTS = (3 * 4.5) + (4 * 4.5) + (4 * 3.5) + (4 * 2.5) + (3 * 1.5)
    # Pontuação mínima em mão: : 9 para jogos de tarot com 4 jogadores.
    MIN_POINTS = (DECK_SIZE // TarotGame.NUM_PLAYERS) * 0.5

    NUM_BOUTS = len(BOUTS)
    NUM_TRUMPS = TRUMPS

    @staticmethod
    def suit(card: int) -> int:
        return card // 100

    @staticmethod
    def rank(card: int) -> int:
        return card % 100

    @staticmethod
    def from_idx(card_idx: int) -> int:
        """Converte um índice de carta para o formato de carta."""
        if (card_idx + 1) >= Card.DECK_SIZE - Card.NUM_TRUMPS:
            return (card_idx - (Card.DECK_SIZE - Card.NUM_TRUMPS)) + 100 * Card.TRUMP
        card_idx -= Card.NUM_TRUMPS
        rank = card_idx % 14
        suit = card_idx - rank
        return (suit + 1) * 100 + (rank + 1)

    @staticmethod
    def to_idx(card_idx: int) -> int:
        """Converte uma carta para o índice do baralho."""
        if Card.is_trump(card_idx):
            return (card_idx - 500) + (Card.DECK_SIZE - Card.NUM_TRUMPS)
        suit = Card.suit(card_idx) - 1
        rank = Card.rank(card_idx) - 1
        return suit * 14 + rank

    @staticmethod
    def is_trump(card: int) -> bool:
        return Card.suit(card) == Card.TRUMP

    @staticmethod
    def is_king(card: int) -> bool:
        return Card.rank(card) == Card.KING

    @staticmethod
    def value(card: int) -> float:
        """ Retorna o valor da carta para o jogo."""
        rank = Card.rank(card)
        if Card.is_trump(card):
            if (rank == Card.PETIT or rank == Card.MONDE or rank == Card.FOOL):
                return 4.5
            else:
                return 0.5
        if rank > 13:
            return 4.5  # Rei
        elif rank > 12:
            return 3.5  # Rainha
        elif rank > 11:
            return 2.5  # Cavaleiro
        elif rank > 10:
            return 1.5  # Valete
        return 0.5

    @staticmethod
    def deck() -> List[int]:
        """ Retorna o baralho de tarot completo.
        O baralho é composto por 78 cartas:
        """
        # 22 cartas de trunfo
        trumps = [Card.TRUMP * 100 + i for i in range(22)].copy()
        # 14 cartas de copas
        hearts = [Card.HEART * 100 + (i + 1) for i in range(14)].copy()
        # 14 cartas de ouros
        diamonds = [Card.DIAMOND * 100 + (i + 1) for i in range(14)].copy()
        # 14 cartas de espadas
        spades = [Card.SPADE * 100 + (i + 1) for i in range(14)].copy()
        # 14 cartas de paus
        clubs = [Card.CLUB * 100 + (i + 1) for i in range(14)].copy()

        return trumps + hearts + diamonds + spades + clubs

    @staticmethod
    def deal(deck: List[int]) -> Tuple[List[int], List[List[int]]]:
        """Distribui as cartas para os jogadores e para o chien."""
        random.shuffle(deck)
        hand_size = (len(deck) - TarotGame.CHIEN_SIZE) // TarotGame.NUM_PLAYERS
        if hand_size * TarotGame.NUM_PLAYERS + TarotGame.CHIEN_SIZE != len(deck):
            raise ValueError(
                "Número de cartas não é divisível pelo número de jogadores.")
        chien = deck[:TarotGame.CHIEN_SIZE]
        player_hands = [deck[TarotGame.CHIEN_SIZE + i*hand_size:TarotGame.CHIEN_SIZE +
                             (i+1)*hand_size] for i in range(TarotGame.NUM_PLAYERS)]
        return chien, player_hands

    @staticmethod
    def count_trumps(hand: List[int]) -> int:
        """Conta o número de trunfos na mão."""
        return sum(1 for card in hand if Card.is_trump(card))
