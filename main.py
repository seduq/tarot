from open_spiel.python import policy
from absl import flags
from absl import app
from tarot import Const, TarotGameState, TarotGame
import itertools as it
from matplotlib import pyplot as plt
import numpy as np
import pyspiel

from open_spiel.python.algorithms import ismcts
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import evaluate_bots


SEED = 129846127


def random_game():
    game = pyspiel.load_game('french_tarot')
    state = game.new_initial_state()
    np.random.seed(SEED)

    print(TarotGameState.pretty_print(state))
    print("=" * 20)
    for i in range(100):
        if (state.is_terminal()):
            print("--" * 10)
            print(f"Results: {state.returns()}")
            print("Game over")
            break
        print("--" * 10)
        print(f"Iteration: {i + 1}")
        print(f"Current player: {state.current}")
        print(f"Current phase: {state.phase}")
        if state.is_chance_node():
            print(f"Chances outcome: {state.chance_outcomes()}")
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
        else:
            print(f"Legal actions: {state.legal_actions()}")
            if (state.legal_actions() == []):
                print("No legal actions available")
                break
            action = np.random.choice(state.legal_actions())

        print(f'Taking action {action} {state.action_to_string(action)}')
        state.apply_action(action)
    print("=" * 20)
    print(TarotGameState.pretty_print(state))


def main(_):
    random_game()


if __name__ == "__main__":
    app.run(main)
