import os
import gc
import psutil
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
        'elapsed_time_mcts': 0.0,
        'ris_taker': False,
        'memory_stats': {
            'tracemalloc_peak_mb': 0.0,
            'tracemalloc_current_mb': 0.0,
            'psutil_peak_mb': 0.0,
            'psutil_current_mb': 0.0,
            'memory_growth_mb': 0.0,
            'gc_collections': {'gen0': 0, 'gen1': 0, 'gen2': 0},
            'memory_measurements': []
        }
    }

    legal_actions_count = 0
    chance_outcomes_count = 0
    game_length = 0
    win = 0
    i = 0
    bids = [{} for _ in range(Const.NUM_PLAYERS)]
    invalid_action_count = 0
    elapsed_time_mcts = 0

    # Inicialização do monitoramento de memória
    process = psutil.Process(os.getpid())
    memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
    gc_stats_baseline = gc.get_stats()

    if verbose:
        print(f"Baseline memory usage: {memory_baseline:.2f} MB")
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
                # Medições de memória ANTES do RIS-MCTS
                gc.collect()  # Força coleta de lixo para medição mais precisa
                memory_before = process.memory_info().rss / 1024 / 1024
                gc_before = gc.get_stats()

                # Inicia monitoramento detalhado
                tracemalloc.start()
                start = time()

                mcts_action = ris_mcts.search(
                    state, state.current, max_time=10, iterations=mcts_iterations, verbose=verbose)

                # Medições DURANTE e APÓS o RIS-MCTS
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                end = time()

                # Medições de memória APÓS o RIS-MCTS
                memory_after = process.memory_info().rss / 1024 / 1024
                gc_after = gc.get_stats()

                elapsed_time_mcts += (end - start)

                # Armazenar estatísticas detalhadas de memória
                memory_growth = memory_after - memory_before
                results['memory_stats']['tracemalloc_peak_mb'] = max(
                    results['memory_stats']['tracemalloc_peak_mb'], peak_mem / 1024 / 1024)
                results['memory_stats']['tracemalloc_current_mb'] = current_mem / 1024 / 1024
                results['memory_stats']['psutil_peak_mb'] = max(
                    results['memory_stats']['psutil_peak_mb'], memory_after)
                results['memory_stats']['psutil_current_mb'] = memory_after
                results['memory_stats']['memory_growth_mb'] += memory_growth

                # Estatísticas de garbage collection
                for i, (before, after) in enumerate(zip(gc_before, gc_after)):
                    gen_name = f'gen{i}'
                    collections_diff = after['collections'] - \
                        before['collections']
                    results['memory_stats']['gc_collections'][gen_name] += collections_diff

                # Registro detalhado da medição
                measurement = {
                    'iteration': game_length,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_growth_mb': memory_growth,
                    'tracemalloc_peak_mb': peak_mem / 1024 / 1024,
                    'tracemalloc_current_mb': current_mem / 1024 / 1024,
                    'time_elapsed': end - start,
                    'mcts_iterations_attempted': mcts_iterations
                }
                results['memory_stats']['memory_measurements'].append(
                    measurement)

                if verbose:
                    print(
                        f"[Memory] Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB, Growth: {memory_growth:.2f} MB")
                    print(
                        f"[Memory] Tracemalloc Peak: {peak_mem / 1024 / 1024:.2f} MB, Current: {current_mem / 1024 / 1024:.2f} MB")

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

    # Medições finais de memória
    memory_final = process.memory_info().rss / 1024 / 1024
    memory_total_growth = memory_final - memory_baseline
    results['memory_stats']['total_memory_growth_mb'] = memory_total_growth

    if 'tracemalloc_peak_mb' not in results['memory_stats'] or results['memory_stats']['tracemalloc_peak_mb'] == 0:
        results['memory_stats']['tracemalloc_peak_mb'] = 0.0
        results['memory_stats']['tracemalloc_current_mb'] = 0.0
        results['memory_stats']['psutil_peak_mb'] = memory_baseline
        results['memory_stats']['psutil_current_mb'] = memory_final
    if verbose:
        print("--" * 10)
        print(f"Results: {state.returns()}")
        print("Game over")
    result = state.returns()
    win = result[ris_mcts_position] > 0
    ris_taker_win = False
    if state.taker == ris_mcts_position:
        ris_taker_win = result[ris_mcts_position] > 0
    avg_legal_actions = legal_actions_count / game_length
    avg_chance_outcomes = chance_outcomes_count / game_length
    results['win'] = win
    results['ris_taker_win'] = ris_taker_win
    results['legal_actions_count'] = legal_actions_count
    results['chance_outcomes_count'] = chance_outcomes_count
    results['game_length'] = game_length
    results['bids'] = bids
    results['invalid_action_count'] = invalid_action_count
    results['elapsed_time_mcts'] = elapsed_time_mcts
    results['ris_taker'] = state.taker == ris_mcts_position
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
        print(f"RIS-MCTS taker win: {ris_taker_win}")
        print(f"Win: {win}")
        print("=" * 30)
        print("DETAILED MEMORY ANALYSIS:")
        print(f"Baseline memory: {memory_baseline:.2f} MB")
        print(f"Final memory: {memory_final:.2f} MB")
        print(f"Total memory growth: {memory_total_growth:.2f} MB")
        print(
            f"Tracemalloc peak: {results['memory_stats']['tracemalloc_peak_mb']:.2f} MB")
        print(
            f"PSUtil peak: {results['memory_stats']['psutil_peak_mb']:.2f} MB")
        print(
            f"Memory growth from RIS-MCTS: {results['memory_stats']['memory_growth_mb']:.2f} MB")
        print(f"GC collections: {results['memory_stats']['gc_collections']}")
        print(
            f"Number of memory measurements: {len(results['memory_stats']['memory_measurements'])}")
        if results['memory_stats']['memory_measurements']:
            avg_growth = sum(m['memory_growth_mb'] for m in results['memory_stats']
                             ['memory_measurements']) / len(results['memory_stats']['memory_measurements'])
            max_growth = max(m['memory_growth_mb']
                             for m in results['memory_stats']['memory_measurements'])
            print(
                f"Average memory growth per RIS-MCTS call: {avg_growth:.2f} MB")
            print(
                f"Maximum memory growth per RIS-MCTS call: {max_growth:.2f} MB")
    return results


def print_results(idx, results, time_elapsed=None):
    print(f"Game {idx+1} - {'Win' if results['win'] else 'Loss'}")
    print(f"  RIS-MCTS position: {results['ris_mcts_position']}")
    print(f"  Legal actions: {results['legal_actions_count']}")
    print(f"  Invalid actions: {results['invalid_action_count']}")
    print(f"  Game length: {results['game_length']}")
    print(f"  Chance outcomes: {results['chance_outcomes_count']}")
    print(f"  Bids: {results['bids']}")
    print(f"  Memory Analysis:")
    print(
        f"    Tracemalloc peak: {results['memory_stats']['tracemalloc_peak_mb']:.2f} MB")
    print(
        f"    PSUtil peak: {results['memory_stats']['psutil_peak_mb']:.2f} MB")
    print(
        f"    Total memory growth: {results['memory_stats'].get('total_memory_growth_mb', 0):.2f} MB")
    print(
        f"    RIS-MCTS memory growth: {results['memory_stats']['memory_growth_mb']:.2f} MB")
    print(f"    GC collections: {results['memory_stats']['gc_collections']}")
    if time_elapsed is not None:
        print(f"  Time: {time_elapsed:.2f} s")
    print("-" * 40)


def run_tarot_simulation(n_games=100, n_log_games=10, verbose=False, mcts_iterations=100):
    total = {
        'won': 0,
        'ris_taker_wins': 0,
        'ris_was_taker': 0,
        'legal_actions_count': 0,
        'chance_outcomes_count': 0,
        'game_length': 0,
        'invalid_actions': 0,
        'ris_positions': [0] * Const.NUM_PLAYERS,
        'bids': [{} for _ in range(Const.NUM_PLAYERS)],
        'time_sum': 0.0,
        'elapsed_time_mcts_sum': 0.0,
        'memory_stats': {
            'tracemalloc_peak_sum': 0.0,
            'psutil_peak_sum': 0.0,
            'memory_growth_sum': 0.0,
            'total_memory_growth_sum': 0.0,
            'gc_collections_sum': {'gen0': 0, 'gen1': 0, 'gen2': 0},
            'memory_measurements_count': 0
        }
    }
    # Define all possible bid types
    bid_types = ['Pass', 'Petit', 'Garde']
    # Write header once at the beginning, with expanded columns
    header = [
        "Game", "Win", "Legal Actions", "Chance Outcomes", "Game Length",
        "Invalid Actions", "TraceMallocPeakMB", "PSUtilPeakMB", "MemoryGrowthMB",
        "TotalMemoryGrowthMB", "ElapsedTimeMCTS", "GC_Gen0", "GC_Gen1", "GC_Gen2"
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
        total['elapsed_time_mcts_sum'] += results['elapsed_time_mcts']
        total['ris_was_taker'] += results['ris_taker']

        # Agregar estatísticas de memória
        total['memory_stats']['tracemalloc_peak_sum'] += results['memory_stats']['tracemalloc_peak_mb']
        total['memory_stats']['psutil_peak_sum'] += results['memory_stats']['psutil_peak_mb']
        total['memory_stats']['memory_growth_sum'] += results['memory_stats']['memory_growth_mb']
        total['memory_stats']['total_memory_growth_sum'] += results['memory_stats'].get(
            'total_memory_growth_mb', 0)
        total['memory_stats']['memory_measurements_count'] += len(
            results['memory_stats']['memory_measurements'])

        # Agregar estatísticas de garbage collection
        for gen in ['gen0', 'gen1', 'gen2']:
            total['memory_stats']['gc_collections_sum'][gen] += results['memory_stats']['gc_collections'][gen]
        for key, value in results['bids'][results['ris_mcts_position']].items():
            total['bids'][results['ris_mcts_position']].setdefault(key, 0)
            total['bids'][results['ris_mcts_position']][key] += value
        if results['win']:
            total['won'] += 1
        if results['ris_taker_win']:
            total['ris_taker_wins'] += 1
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
                f"{total['memory_stats']['tracemalloc_peak_sum']:.2f}",
                f"{total['memory_stats']['psutil_peak_sum']:.2f}",
                f"{total['memory_stats']['memory_growth_sum']:.2f}",
                f"{total['memory_stats']['total_memory_growth_sum']:.2f}",
                f"{total['elapsed_time_mcts_sum']:.4f}",
                str(total['memory_stats']['gc_collections_sum']['gen0']),
                str(total['memory_stats']['gc_collections_sum']['gen1']),
                str(total['memory_stats']['gc_collections_sum']['gen2'])
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
                    f"Taker Wins: {total['ris_taker_wins']} out of {j} games ({total['ris_taker_wins'] / j:.2%})\n")
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
                    f"Average elapsed time RIS-MCTS per game: {total['elapsed_time_mcts_sum'] / j:.4f} s\n")
                ftxt.write("=" * 50 + "\n")
                ftxt.write("DETAILED MEMORY ANALYSIS:\n")
                ftxt.write(
                    f"Average Tracemalloc peak per game: {total['memory_stats']['tracemalloc_peak_sum'] / j:.2f} MB\n")
                ftxt.write(
                    f"Average PSUtil peak per game: {total['memory_stats']['psutil_peak_sum'] / j:.2f} MB\n")
                ftxt.write(
                    f"Average memory growth from RIS-MCTS per game: {total['memory_stats']['memory_growth_sum'] / j:.2f} MB\n")
                ftxt.write(
                    f"Average total memory growth per game: {total['memory_stats']['total_memory_growth_sum'] / j:.2f} MB\n")
                ftxt.write(
                    f"Total memory measurements taken: {total['memory_stats']['memory_measurements_count']}\n")
                ftxt.write(
                    f"Average GC Gen0 collections per game: {total['memory_stats']['gc_collections_sum']['gen0'] / j:.1f}\n")
                ftxt.write(
                    f"Average GC Gen1 collections per game: {total['memory_stats']['gc_collections_sum']['gen1'] / j:.1f}\n")
                ftxt.write(
                    f"Average GC Gen2 collections per game: {total['memory_stats']['gc_collections_sum']['gen2'] / j:.1f}\n")
                ftxt.write("=" * 50 + "\n")
                ftxt.write(f"RIS-MCTS positions: {total['ris_positions']}\n")
                ftxt.write(f"Invalid actions: {total['invalid_actions']}\n")
                ftxt.write(f"Bids: {total['bids']}\n")


def analyze_memory_usage(results_list):
    """
    Analisa padrões de uso de memória ao longo de múltiplos jogos.
    """
    if not results_list:
        return

    print("=" * 60)
    print("ANÁLISE DETALHADA DE MEMÓRIA")
    print("=" * 60)

    # Coleta de todas as medições de memória
    all_measurements = []
    for result in results_list:
        all_measurements.extend(result['memory_stats']['memory_measurements'])

    if not all_measurements:
        print("Nenhuma medição de memória encontrada.")
        return

    # Estatísticas de crescimento de memória
    memory_growths = [m['memory_growth_mb'] for m in all_measurements]
    tracemalloc_peaks = [m['tracemalloc_peak_mb'] for m in all_measurements]

    print(f"Total de medições de memória: {len(all_measurements)}")
    print(f"Crescimento de memória por chamada RIS-MCTS:")
    print(f"  Média: {np.mean(memory_growths):.3f} MB")
    print(f"  Mediana: {np.median(memory_growths):.3f} MB")
    print(f"  Mínimo: {np.min(memory_growths):.3f} MB")
    print(f"  Máximo: {np.max(memory_growths):.3f} MB")
    print(f"  Desvio padrão: {np.std(memory_growths):.3f} MB")

    print(f"\nPico de memória Tracemalloc por chamada:")
    print(f"  Média: {np.mean(tracemalloc_peaks):.3f} MB")
    print(f"  Mediana: {np.median(tracemalloc_peaks):.3f} MB")
    print(f"  Mínimo: {np.min(tracemalloc_peaks):.3f} MB")
    print(f"  Máximo: {np.max(tracemalloc_peaks):.3f} MB")
    print(f"  Desvio padrão: {np.std(tracemalloc_peaks):.3f} MB")

    # Análise de correlação entre tempo e memória
    times = [m['time_elapsed'] for m in all_measurements]
    correlation = np.corrcoef(times, memory_growths)[0, 1]
    print(
        f"\nCorrelação entre tempo de execução e crescimento de memória: {correlation:.3f}")

    # Análise de outliers
    q75, q25 = np.percentile(memory_growths, [75, 25])
    iqr = q75 - q25
    upper_bound = q75 + (1.5 * iqr)
    lower_bound = q25 - (1.5 * iqr)
    outliers = [x for x in memory_growths if x >
                upper_bound or x < lower_bound]
    print(
        f"Outliers de crescimento de memória: {len(outliers)} ({len(outliers)/len(memory_growths)*100:.1f}%)")

    # Análise de tendência ao longo do tempo
    if len(all_measurements) > 10:
        # Dividir em quartis e comparar
        n = len(all_measurements)
        first_quarter = memory_growths[:n//4]
        last_quarter = memory_growths[-n//4:]

        print(f"\nAnálise de tendência:")
        print(f"  Primeiro quartil - média: {np.mean(first_quarter):.3f} MB")
        print(f"  Último quartil - média: {np.mean(last_quarter):.3f} MB")
        print(
            f"  Diferença: {np.mean(last_quarter) - np.mean(first_quarter):.3f} MB")

    print("=" * 60)


if __name__ == "__main__":
    n_games = [20]
    mcts_iterations = [1000]
    for games in n_games:
        for iterations in mcts_iterations:
            print(
                f"Running simulation with {games} games and {iterations} MCTS iterations...")
            run_tarot_simulation(n_games=games, n_log_games=10,
                                 verbose=False, mcts_iterations=iterations)
