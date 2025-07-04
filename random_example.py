from tarot import Tarot
import random


SEED = 7264828


def random_game(strategy=['random', 'random', 'random', 'random'], verbose=True):
    state = Tarot()
    random.seed(SEED)

    legal_actions_count = 0
    chance_outcomes_count = 0
    game_length = 0
    win = 0
    if verbose:
        print(state.pretty_print())
        print("=" * 20)
    while not state.is_terminal():
        game_length += 1
        if verbose:
            print("--" * 10)
            print(f"Iteration: {game_length}")
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
            if strategy[state.current] == 'random':
                action = random.choice(legal_actions)
            elif strategy[state.current] == 'min':
                action = min(legal_actions)
            elif strategy[state.current] == 'max':
                action = max(legal_actions)
            action = random.choice(legal_actions)
            action = int(action)
            legal_actions_count += len(state.legal_actions())

        if verbose:
            print(
                f'Taking action {action} {state.action_to_string(action)}')
        state.apply_action(action)
        state.next()

    avg_legal_actions = legal_actions_count / game_length
    avg_chance_outcomes = chance_outcomes_count / game_length
    if verbose:
        print("=" * 20)
        print(state.pretty_print())
        print(f"Results: {state.returns()}")
        print(f"Game length: {game_length}")
        print(f"Legal actions count: {legal_actions_count}")
        print(f"Chance outcomes count: {chance_outcomes_count}")
        print(f"Average legal actions per turn: {avg_legal_actions}")
        print(f"Average chance outcomes per turn: {avg_chance_outcomes}")
    return win, legal_actions_count, chance_outcomes_count, game_length


def main():
    won = 0
    n_games = 100000
    for i in range(n_games):
        win, _, _, _ = random_game(verbose=True)
        won += win
        if (i + 1) % 100 == 0:
            print(f"Game {i}: Won {win} games so far")
    print(f"Won {won} out of {n_games} games ({won / n_games:.2f}%)")


if __name__ == "__main__":
    main()
