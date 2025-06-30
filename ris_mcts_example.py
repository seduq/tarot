import os
from time import time
from absl import app
import numpy as np
from tarot import TarotGameState, Const, TarotSearch
import pyspiel
import random
import tracemalloc
import matplotlib.pyplot as plt
import ast

from tarot.bids import Bid
from tarot.constants import Phase


def ris_mcts_example(verbose=True, mcts_iterations=100):
    game = pyspiel.load_game('french_tarot')
    state: TarotGameState = game.new_initial_state()

    bots = ['random'] * Const.NUM_PLAYERS

    ris_mcts = TarotSearch()
    ris_mcts_position = random.randint(0, Const.NUM_PLAYERS - 1)
    bots[ris_mcts_position] = 'ris-mcts'
    results = {
        'win': False,
        'legal_actions_count': 0,
        'chance_outcomes_count': 0,
        'game_length': 0,
        'bids': [{}] * Const.NUM_PLAYERS,
        'invalid_action_count': 0,
        'ris_mcts_position': ris_mcts_position,
        'elapsed_time_mcts': 0.0
    }

    legal_actions_count = 0
    chance_outcomes_count = 0
    game_length = 0
    win = 0
    i = 0
    bids = [{} for _ in range(Const.NUM_PLAYERS)]
    invalid_action_count = 0
    elapsed_time_mcts = 0
    if verbose:
        print(TarotGameState.pretty_print(state))
        print("=" * 20)
    while state.is_terminal() is False or i < 100:
        game_length += 1
        if verbose:
            print("--" * 10)
            print(f"Iteration: {i + 1}")
            print(f"Current player: {state.current}")
            print(f"Current phase: {state.phase}")
        if state.is_chance_node():
            if verbose:
                print(f"Chances outcome: {state.chance_outcomes()}")
            # Handle chance nodes (bidding, declarations)
            if bots[state.current] != 'ris-mcts':
                action = Const.BID_PASS
            else:
                action = Const.BID_PETIT
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = int(np.random.choice(actions, p=probs))
            if state.phase == Phase.BIDDING:
                bids[state.current].setdefault(Bid.name(action), 0)
                bids[state.current][Bid.name(action)] += 1
        else:
            if verbose:
                print(f"Legal actions: {state.legal_actions()}")
            if (state.legal_actions() == [] and state.phase == Phase.TRICK or state.phase == Phase.END):
                break
            legal_actions = state.legal_actions()
            action = random.choice(legal_actions)
            if bots[state.current] == 'random':
                action = random.choice(legal_actions)
            elif bots[state.current] == 'ris-mcts':
                start = time()
                tracemalloc.start()
                mcts_action = ris_mcts.search(
                    state, state.current, iterations=mcts_iterations, verbose=verbose)
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                end = time()
                elapsed_time_mcts += (end - start)
                results['memory_peak_bytes'] = peak_mem
                results['memory_peak_mb'] = peak_mem / 1024 / 1024
                if mcts_action is not None:
                    action = mcts_action
                    if action not in legal_actions:
                        invalid_action_count += 1
                        action = random.choice(legal_actions)
            legal_actions_count += len(state.legal_actions())
        if verbose:
            print(
                f'Taking action {action} {state.action_to_string(action)}')
        state.apply_action(action)
        i += 1
    if 'memory_peak_mb' not in results:
        results['memory_peak_bytes'] = 0
        results['memory_peak_mb'] = 0.0
    if verbose:
        print("--" * 10)
        print(f"Results: {state.returns()}")
        print("Game over")
    result = state.returns()
    win = result[state.taker] > 0
    avg_legal_actions = legal_actions_count / game_length
    avg_chance_outcomes = chance_outcomes_count / game_length
    results['win'] = win
    results['legal_actions_count'] = legal_actions_count
    results['chance_outcomes_count'] = chance_outcomes_count
    results['game_length'] = game_length
    results['bids'] = bids
    results['invalid_action_count'] = invalid_action_count
    results['elapsed_time_mcts'] = elapsed_time_mcts
    if verbose:
        print("=" * 20)
        print(TarotGameState.pretty_print(state))
        print(f"Bids: {bids}")
        print(f"Game length: {game_length}")
        print(f"Legal actions count: {legal_actions_count}")
        print(f"Chance outcomes count: {chance_outcomes_count}")
        print(f"Average legal actions per turn: {avg_legal_actions}")
        print(f"Average chance outcomes per turn: {avg_chance_outcomes}")
        print(f"Invalid actions: {invalid_action_count}")
        print(f"RIS-MCTS position: {ris_mcts_position}")
        print(f"Win: {win}")
        print(
            f"Peak memory usage (RIS-MCTS): {results['memory_peak_mb']:.2f} MB")
    return results


def print_results(idx, results, time_elapsed=None):
    print(f"Game {idx+1} - {'Win' if results['win'] else 'Loss'}")
    print(f"  RIS-MCTS position: {results['ris_mcts_position']}")
    print(f"  Legal actions: {results['legal_actions_count']}")
    print(f"  Invalid actions: {results['invalid_action_count']}")
    print(f"  Game length: {results['game_length']}")
    print(f"  Chance outcomes: {results['chance_outcomes_count']}")
    print(f"  Bids: {results['bids']}")
    print(f"  Peak memory usage: {results['memory_peak_mb']:.2f} MB")
    if time_elapsed is not None:
        print(f"  Time: {time_elapsed:.2f} s")
    print("-" * 40)


def plot_stats_from_csv(csv_path, file_path):
    import csv
    import matplotlib.pyplot as plt
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    games = []
    wins = []
    win_rate = []
    legal_actions = []
    game_length = []
    invalid_actions = []
    chance_outcomes = []
    peak_memory = []
    elapsed_time_mcts = []
    # Bids per player and type
    bids = {p: {'Pass': [], 'Petit': [], 'Garde': []} for p in range(4)}
    for_win_rate = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append(int(row['Game']))
            wins.append(int(row['Win']))
            legal_actions.append(float(row['Legal Actions']))
            game_length.append(float(row['Game Length']))
            invalid_actions.append(float(row['Invalid Actions']))
            chance_outcomes.append(float(row['Chance Outcomes']))
            peak_memory.append(float(row['PeakMemoryMB']))
            elapsed_time_mcts.append(float(row.get('ElapsedTimeMCTS', 0)))
            for_win_rate.append(int(row['Win']))
            win_rate.append(sum(for_win_rate) / len(for_win_rate) * 100)
            for p in range(4):
                for bid in ['Pass', 'Petit', 'Garde']:
                    bids[p][bid].append(int(row[f'Bids_{p}_{bid}']))
    # Plot each stat in a separate file
    plt.figure()
    plt.plot(games, legal_actions)
    plt.xlabel('Games')
    plt.ylabel('Legal Actions')
    plt.title('Legal Actions per Log')
    plt.grid(True)
    plt.savefig(f'{file_path}/plot_legal_actions.png')
    plt.close()

    plt.figure()
    plt.plot(games, game_length)
    plt.xlabel('Games')
    plt.ylabel('Game Length (turns)')
    plt.title('Game Length per Log')
    plt.grid(True)
    plt.savefig(f'{file_path}/plot_game_length.png')
    plt.close()

    plt.figure()
    plt.plot(games, invalid_actions)
    plt.xlabel('Games')
    plt.ylabel('Invalid Actions')
    plt.title('Invalid Actions per Log')
    plt.grid(True)
    plt.savefig(f'{file_path}/plot_invalid_actions.png')
    plt.close()

    plt.figure()
    plt.plot(games, peak_memory)
    plt.xlabel('Games')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Peak Memory per Log')
    plt.grid(True)
    plt.savefig(f'{file_path}/plot_peak_memory.png')
    plt.close()

    plt.figure()
    plt.plot(games, win_rate)
    plt.xlabel('Games')
    plt.ylabel('Win Rate (%)')
    plt.title('Cumulative Win Rate')
    plt.grid(True)
    plt.savefig(f'{file_path}/plot_win_rate.png')
    plt.close()

    plt.figure()
    plt.plot(games, elapsed_time_mcts)
    plt.xlabel('Games')
    plt.ylabel('Elapsed Time RIS-MCTS (s)')
    plt.title('Cumulative Elapsed Time RIS-MCTS per Log')
    plt.grid(True)
    plt.savefig(f'{file_path}/plot_elapsed_time_mcts.png')
    plt.close()

    # Plot bids per player and type
    for p in range(4):
        plt.figure()
        for bid in ['Pass', 'Petit', 'Garde']:
            plt.plot(games, bids[p][bid], label=bid)
        plt.xlabel('Games')
        plt.ylabel('Bid Count')
        plt.title(f'Bids per Log - Player {p}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plot_bids_player_{p}.png')
        plt.close()


def run_tarot_simulation(n_games=100, n_log_games=10, verbose=False, mcts_iterations=100):
    total = {
        'won': 0,
        'legal_actions_count': 0,
        'chance_outcomes_count': 0,
        'game_length': 0,
        'invalid_actions': 0,
        'ris_positions': [0] * Const.NUM_PLAYERS,
        'bids': [{} for _ in range(Const.NUM_PLAYERS)],
        'time_sum': 0.0,
        'memory_sum': 0.0,
        'elapsed_time_mcts_sum': 0.0
    }
    # Define all possible bid types
    bid_types = ['Pass', 'Petit', 'Garde']
    # Write header once at the beginning, with expanded columns
    header = [
        "Game", "Win", "Legal Actions", "Chance Outcomes", "Game Length",
        "Invalid Actions", "PeakMemoryMB", "ElapsedTimeMCTS"
    ]
    # Add RIS-MCTS position columns
    header += [f"RIS_Position_{i}" for i in range(Const.NUM_PLAYERS)]
    # Add Bids columns for each player and bid type
    for p in range(Const.NUM_PLAYERS):
        for bid in bid_types:
            header.append(f"Bids_{p}_{bid}")
    with open(f'ris_mcts_results_{n_games}_{mcts_iterations}.csv', 'w') as f:
        f.write(",".join(header) + "\n")
    for i in range(n_games):
        start_time = time()
        results = ris_mcts_example(
            verbose=verbose, mcts_iterations=mcts_iterations)
        end_time = time()
        elapsed = end_time - start_time
        total['time_sum'] += elapsed
        total['legal_actions_count'] += results['legal_actions_count']
        total['chance_outcomes_count'] += results['chance_outcomes_count']
        total['game_length'] += results['game_length']
        total['invalid_actions'] += results['invalid_action_count']
        total['ris_positions'][results['ris_mcts_position']] += 1
        total['memory_sum'] += results['memory_peak_mb']
        total['elapsed_time_mcts_sum'] += results['elapsed_time_mcts']
        for key, value in results['bids'][results['ris_mcts_position']].items():
            total['bids'][results['ris_mcts_position']].setdefault(key, 0)
            total['bids'][results['ris_mcts_position']][key] += value
        if results['win']:
            total['won'] += 1
        j = i + 1
        if j % n_log_games == 0:
            # Prepare expanded columns
            row = [
                str(j),
                str(total['won']),
                str(total['legal_actions_count']),
                str(total['chance_outcomes_count']),
                str(total['game_length']),
                str(total['invalid_actions']),
                f"{total['memory_sum']:.2f}",
                f"{total['elapsed_time_mcts_sum']:.4f}"
            ]
            # RIS-MCTS positions
            row += [str(pos) for pos in total['ris_positions']]
            # Bids per player and bid type
            for p in range(Const.NUM_PLAYERS):
                for bid in bid_types:
                    row.append(str(total['bids'][p].get(bid, 0)))
            with open(f'ris_mcts_results_{n_games}_{mcts_iterations}.csv', 'a') as f:
                f.write(",".join(row) + "\n")
            with open(f'ris_mcts_results_{n_games}_{mcts_iterations}_{j}.txt', 'w') as ftxt:
                ftxt.write(
                    f"Wins: {total['won']} out of {j} games ({total['won'] / j:.2%})\n")
                ftxt.write(
                    f"Average legal actions per game: {total['legal_actions_count'] / j:.2f}\n")
                ftxt.write(
                    f"Average game length: {total['game_length'] / j:.2f} turns\n")
                ftxt.write(
                    f"Average time per game: {total['time_sum'] / j:.2f} seconds\n")
                ftxt.write(
                    f"Average invalid actions per game: {total['invalid_actions'] / j:.2f}\n")
                ftxt.write(
                    f"Average chance outcomes per game: {total['chance_outcomes_count'] / j:.2f}\n")
                ftxt.write(
                    f"Average peak memory per game: {total['memory_sum'] / j:.2f} MB\n")
                ftxt.write(
                    f"Average elapsed time RIS-MCTS per game: {total['elapsed_time_mcts_sum'] / j:.4f} s\n")
                ftxt.write(f"RIS-MCTS positions: {total['ris_positions']}\n")
                ftxt.write(f"Invalid actions: {total['invalid_actions']}\n")
                ftxt.write(f"Bids: {total['bids']}\n")


if __name__ == "__main__":
    n_games = [200, 500]
    mcts_iterations = [100, 500, 1000]
    for games in n_games:
        for iterations in mcts_iterations:
            print(
                f"Running simulation with {games} games and {iterations} MCTS iterations...")
            run_tarot_simulation(n_games=games, n_log_games=10,
                                 verbose=False, mcts_iterations=iterations)
            plot_stats_from_csv(
                f'ris_mcts_results_{games}_{iterations}.csv',
                f'plots_{games}_{iterations}')
