import random
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from .state import TarotState
from .constants import Phase


class Agent(ABC):
    def __init__(self, name: str, player_id: int, state: TarotState):
        self.name = name
        self.player_id = player_id
        self.state = state
        self.metrics = Metrics()

    @abstractmethod
    def reset(self, player_id: int, state: TarotState):
        self.metrics = Metrics()
        self.player_id = player_id
        self.state = state

    @abstractmethod
    def choose(self, state: TarotState, legal_actions: List[int]) -> int:
        pass

    @abstractmethod
    def update(self, state: TarotState, action: int, player: int) -> None:
        self.metrics.record(action)
        self.state = state


class Metrics:
    def __init__(self):
        self.temporal: Dict[str, List[int]] = {}
        self.aggregate: Dict[str, List[int]] = {}
        self.data: Dict[str, float] = {}

    def record(self, action: int):
        key = f"{action}"
        if key not in self.data:
            self.data[key] = 0.0
        self.data[key] += 1.0


class RandomAgent(Agent):
    def reset(self, player_id: int, state: TarotState):
        return super().reset(player_id, state)

    def choose(self, state: TarotState, legal_actions: List[int]) -> int:
        return random.choice(legal_actions)

    def update(self, state: TarotState, action: int, player: int):
        super().update(state, action, player)


class MinCardAgent(Agent):
    def reset(self, player_id: int, state: TarotState):
        return super().reset(player_id, state)

    def choose(self, state: TarotState, legal_actions: List[int]) -> int:
        if (state.phase == Phase.BIDDING or state.phase == Phase.DECLARE):
            return random.choice(legal_actions)
        if (state.phase == Phase.TRICK or state.phase == Phase.CHIEN):
            return min(legal_actions)
        return 0

    def update(self, state: TarotState, action: int, player: int):
        super().update(state, action, player)


class MaxCardAgent(Agent):
    def reset(self, player_id: int, state: TarotState):
        return super().reset(player_id, state)

    def choose(self, state: TarotState, legal_actions: List[int]) -> int:
        if (state.phase == Phase.BIDDING or state.phase == Phase.DECLARE):
            return random.choice(legal_actions)
        if (state.phase == Phase.TRICK or state.phase == Phase.CHIEN):
            return max(legal_actions)
        return 0

    def update(self, state: TarotState, action: int, player: int):
        super().update(state, action, player)
