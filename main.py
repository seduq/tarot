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


def random_game(verbose=True):
    game = pyspiel.load_game('french_tarot')
    state = game.new_initial_state()
    np.random.seed(SEED)

    legal_actions_count = 0
    chance_outcomes_count = 0
    game_length = 0
    if verbose:
        print(TarotGameState.pretty_print(state))
        print("=" * 20)
    for i in range(100):
        game_length += 1
        if (state.is_terminal()):
            print("--" * 10)
            print(f"Results: {state.returns()}")
            print("Game over")
            break
        if verbose:
            print("--" * 10)
            print(f"Iteration: {i + 1}")
            print(f"Current player: {state.current}")
            print(f"Current phase: {state.phase}")
        if state.is_chance_node():
            if verbose:
                print(f"Chances outcome: {state.chance_outcomes()}")
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
            chance_outcomes_count += len(outcomes)
        else:
            if verbose:
                print(f"Legal actions: {state.legal_actions()}")
            if (state.legal_actions() == []):
                print("No legal actions available")
                break
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)
            legal_actions_count += len(state.legal_actions())

            if verbose:
                print(
                    f'Taking action {action} {state.action_to_string(action)}')
        state.apply_action(action)

    avg_legal_actions = legal_actions_count / game_length
    avg_chance_outcomes = chance_outcomes_count / game_length
    if verbose:
        print("=" * 20)
        print(TarotGameState.pretty_print(state))
        print(f"Game length: {game_length}")
        print(f"Legal actions count: {legal_actions_count}")
        print(f"Chance outcomes count: {chance_outcomes_count}")
        print(f"Average legal actions per turn: {avg_legal_actions}")
        print(f"Average chance outcomes per turn: {avg_chance_outcomes}")
    return legal_actions_count, chance_outcomes_count, game_length


def main(_):
    random_game()


if __name__ == "__main__":
    app.run(main)
