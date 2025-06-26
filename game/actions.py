from typing import List, Optional, Tuple

from game.cards import Card
from game.game import Phase, TarotGame
from game.utils import Utils


class Action:
    @staticmethod
    def legal_trick_actions(hand: List[int], trick: List[int]) -> List[int]:
        """Retorna as cartas legais para jogar."""
        if not trick:
            return hand.copy()
        lead_suit = Card.suit(trick[0])
        has_lead = [c for c in hand if Card.suit(c) == lead_suit]
        if has_lead:
            return has_lead
        trumps = [c for c in hand if Card.is_trump(c)]
        if trumps:
            return trumps
        return hand.copy()

    @staticmethod
    def apply_trick_action(hand: List[int], trick: List[int], action: int) -> Tuple[List[int], Optional[Tuple[int, List[int]]]]:
        """Remove a carta jogada da mão do jogador."""
        trick_winner = None
        if action in hand:
            hand.remove(action)

        if len(trick) == TarotGame.NUM_PLAYERS:
            winner = Utils.trick_winner(trick)
            trick_winner = (winner, trick)

        return hand, trick_winner

    @staticmethod
    def legal_chien_discards(cards: List[int]) -> List[int]:
        """
        Retorna as cartas válidas para descartar no chien.
        As cartas válidas são aquelas que não são trunfos e não são o Rei.
        """
        return [card for card in cards if not Card.is_trump(card) and not Card.is_king(card)]
