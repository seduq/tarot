from absl import app
from tarot import TarotGameState, Const, TarotSearch
import pyspiel
import random


SEED = 7264828


def ris_mcts_example(agents=1, verbose=True):
    game = pyspiel.load_game('french_tarot')
    state: TarotGameState = game.new_initial_state()
    random.seed(SEED)

    bots = ['random'] * (Const.NUM_PLAYERS - agents)
    bots += ['mcts'] * agents

    mcts = TarotSearch()

    legal_actions_count = 0
    chance_outcomes_count = 0
    game_length = 0
    win = 0
    if verbose:
        print(TarotGameState.pretty_print(state))
        print("=" * 20)
    for i in range(100):
        game_length += 1
        if (state.is_terminal()):
            if verbose:
                print("--" * 10)
                print(f"Results: {state.returns()}")
                print("Game over")
            result = state.returns()
            if result[state.taker] > 0:
                win += 1
            break
        if verbose:
            print("--" * 10)
            print(f"Iteration: {i + 1}")
            print(f"Current player: {state.current}")
            print(f"Current phase: {state.phase}")
        if state.is_chance_node():
            if verbose:
                print(f"Chances outcome: {state.chance_outcomes()}")
            # Handle chance nodes (bidding, declarations)
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = random.choices(actions, weights=probs)[0]
        else:
            if verbose:
                print(f"Legal actions: {state.legal_actions()}")
            if (state.legal_actions() == []):
                print("No legal actions available")
                break
            legal_actions = state.legal_actions()
            if bots[state.current] == 'random':
                action = random.choice(legal_actions)
            elif bots[state.current] == 'min':
                action = min(legal_actions)
            elif bots[state.current] == 'max':
                action = max(legal_actions)
            elif bots[state.current] == 'mcts':
                action = mcts.search(
                    state.clone(), state.current, iterations=1000)
            action = random.choice(legal_actions)
            action = int(action)
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
    return win, legal_actions_count, chance_outcomes_count, game_length


def main(_):
    won = 0
    for i in range(100000):
        win, _, _, _ = ris_mcts_example(verbose=True)
        won += win
        if i % 1000 == 0:
            print(f"Game {i}: Won {win} games so far")
    print(f"Won {won} out of 100000 games ({won / 1000:.2f}%)")


if __name__ == "__main__":
    app.run(main)
