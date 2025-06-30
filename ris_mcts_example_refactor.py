import os
import gc
import psutil
from time import time
import numpy as np
import pyspiel
import random
import tracemalloc
from tarot.game import TarotGameState
from tarot import Const, TarotSearch
from tarot.bids import Bid
from tarot.constants import Phase


class MemoryProfiler:
    """Classe para gerenciar profiling de memória do RIS-MCTS"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None
        self.gc_baseline = None

    def start_profiling(self):
        """Inicia o profiling de memória"""
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        self.gc_baseline = gc.get_stats()
        return self.baseline_memory

    def measure_before_mcts(self):
        """Medições antes da chamada RIS-MCTS"""
        gc.collect()  # Força coleta de lixo
        memory_before = self.process.memory_info().rss / 1024 / 1024
        gc_before = gc.get_stats()
        tracemalloc.start()
        return memory_before, gc_before

    def measure_after_mcts(self, memory_before, gc_before, start_time, game_length, mcts_iterations):
        """Medições após a chamada RIS-MCTS"""
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time()

        memory_after = self.process.memory_info().rss / 1024 / 1024
        gc_after = gc.get_stats()

        measurement = {
            'iteration': game_length,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_growth_mb': memory_after - memory_before,
            'tracemalloc_peak_mb': peak_mem / 1024 / 1024,
            'tracemalloc_current_mb': current_mem / 1024 / 1024,
            'time_elapsed': end_time - start_time,
            'mcts_iterations_attempted': mcts_iterations
        }

        gc_collections = {}
        for i, (before, after) in enumerate(zip(gc_before, gc_after)):
            gen_name = f'gen{i}'
            gc_collections[gen_name] = after['collections'] - \
                before['collections']

        return measurement, gc_collections

    def finalize_profiling(self):
        """Finaliza o profiling e retorna crescimento total"""
        if self.baseline_memory is None:
            return 0.0
        final_memory = self.process.memory_info().rss / 1024 / 1024
        return final_memory - self.baseline_memory


class GameResultsAnalyzer:
    """Classe para análise de resultados do jogo"""

    @staticmethod
    def print_game_summary(results, memory_baseline, memory_final, verbose=True):
        """Imprime resumo do jogo"""
        if not verbose:
            return

        memory_total_growth = memory_final - memory_baseline

        print("=" * 20)
        print(f"Bids: {results['bids']}")
        print(f"Game length: {results['game_length']}")
        print(f"Legal actions count: {results['legal_actions_count']}")
        print(f"Chance outcomes count: {results['chance_outcomes_count']}")
        print(
            f"Average legal actions per turn: {results['legal_actions_count'] / results['game_length']:.2f}")
        print(
            f"Average chance outcomes per turn: {results['chance_outcomes_count'] / results['game_length']:.2f}")
        print(f"Invalid actions: {results['invalid_action_count']}")
        print(f"RIS-MCTS position: {results['ris_mcts_position']}")
        print(f"RIS-MCTS taker win: {results['ris_taker_win']}")
        print(f"Win: {results['win']}")

        GameResultsAnalyzer.print_memory_analysis(
            results, memory_baseline, memory_final, memory_total_growth)

    @staticmethod
    def print_memory_analysis(results, memory_baseline, memory_final, memory_total_growth):
        """Imprime análise detalhada de memória"""
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
            measurements = results['memory_stats']['memory_measurements']
            avg_growth = sum(m['memory_growth_mb']
                             for m in measurements) / len(measurements)
            max_growth = max(m['memory_growth_mb'] for m in measurements)
            print(
                f"Average memory growth per RIS-MCTS call: {avg_growth:.2f} MB")
            print(
                f"Maximum memory growth per RIS-MCTS call: {max_growth:.2f} MB")


def ris_mcts_example(verbose=True, mcts_iterations=100):
    """Executa um jogo de Tarot com RIS-MCTS e profiling de memória"""
    game = pyspiel.load_game('french_tarot')
    state: TarotGameState = game.new_initial_state()

    # Configuração dos bots
    bots = ['random'] * Const.NUM_PLAYERS
    ris_mcts = TarotSearch()
    ris_mcts_position = random.randint(0, Const.NUM_PLAYERS - 1)
    bots[ris_mcts_position] = 'ris-mcts'

    # Inicialização dos resultados
    results = _initialize_results(ris_mcts_position)
    game_state = _initialize_game_state()

    # Inicialização do profiler de memória
    profiler = MemoryProfiler()
    memory_baseline = profiler.start_profiling()

    if verbose:
        print(f"Baseline memory usage: {memory_baseline:.2f} MB")
        print(TarotGameState.pretty_print(state))
        print("=" * 20)

    # Loop principal do jogo
    while not state.is_terminal() and game_state['iteration'] < 100:
        game_state['game_length'] += 1

        if verbose:
            print("--" * 10)
            print(f"Iteration: {game_state['iteration'] + 1}")
            print(f"Current player: {state.current}")
            print(f"Current phase: {state.phase}")

        # Processar nó de chance ou ação do jogador
        if state.is_chance_node():
            action = _handle_chance_node(
                state, bots, game_state['bids'], verbose)
        else:
            action = _handle_player_action(
                state, bots, ris_mcts, profiler, results,
                game_state, mcts_iterations, verbose
            )
            if action is None:  # Condição de término
                break

        if verbose and action is not None:
            print(f'Taking action {action} {state.action_to_string(action)}')

        state.apply_action(action)
        game_state['iteration'] += 1

    # Finalização e análise
    memory_final = memory_baseline + profiler.finalize_profiling()
    _finalize_results(results, game_state, state, ris_mcts_position,
                      memory_final - memory_baseline, verbose)

    GameResultsAnalyzer.print_game_summary(
        results, memory_baseline, memory_final, verbose)

    return results


def _initialize_results(ris_mcts_position):
    """Inicializa estrutura de resultados"""
    return {
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


def _initialize_game_state():
    """Inicializa estado do jogo"""
    return {
        'iteration': 0,
        'game_length': 0,
        'legal_actions_count': 0,
        'chance_outcomes_count': 0,
        'invalid_action_count': 0,
        'elapsed_time_mcts': 0,
        'bids': [{} for _ in range(Const.NUM_PLAYERS)]
    }


def _handle_chance_node(state, bots, bids, verbose):
    """Processa nós de chance (lances, declarações)"""
    if verbose:
        print(f"Chances outcome: {state.chance_outcomes()}")

    # Determinar ação baseada no tipo de bot
    if bots[state.current] != 'ris-mcts':
        action = Const.BID_PASS
    else:
        action = Const.BID_PETIT

    # Aplicar probabilidades de chance
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes)
    action = int(np.random.choice(actions, p=probs))

    # Registrar lances
    if state.phase == Phase.BIDDING:
        bids[state.current].setdefault(Bid.name(action), 0)
        bids[state.current][Bid.name(action)] += 1

    return action


def _handle_player_action(state, bots, ris_mcts, profiler, results, game_state, mcts_iterations, verbose):
    """Processa ações dos jogadores"""
    if verbose:
        # Verificar condições de término
        print(f"Legal actions: {state.legal_actions()}")
        if (not state.legal_actions() and state.phase == Phase.TRICK) or state.phase == Phase.END:
            return None

    legal_actions = state.legal_actions()
    action = random.choice(legal_actions)

    if bots[state.current] == 'random':
        action = random.choice(legal_actions)

    elif bots[state.current] == 'ris-mcts':
        action = _execute_ris_mcts(
            state, ris_mcts, profiler, results, game_state,
            mcts_iterations, legal_actions, verbose
        )

    game_state['legal_actions_count'] += len(legal_actions)
    return action


def _execute_ris_mcts(state, ris_mcts, profiler, results, game_state, mcts_iterations, legal_actions, verbose):
    """Executa RIS-MCTS com profiling de memória"""
    # Medições antes do RIS-MCTS
    memory_before, gc_before = profiler.measure_before_mcts()
    start_time = time()

    # Execução do RIS-MCTS
    mcts_action = ris_mcts.search(
        state, state.current, max_time=10, iterations=mcts_iterations, verbose=verbose
    )

    # Medições após o RIS-MCTS
    measurement, gc_collections = profiler.measure_after_mcts(
        memory_before, gc_before, start_time, game_state['game_length'], mcts_iterations
    )

    # Atualizar estatísticas
    _update_memory_stats(results, measurement, gc_collections)
    game_state['elapsed_time_mcts'] += measurement['time_elapsed']

    if verbose:
        print(f"[Memory] Before: {measurement['memory_before_mb']:.2f} MB, "
              f"After: {measurement['memory_after_mb']:.2f} MB, "
              f"Growth: {measurement['memory_growth_mb']:.2f} MB")
        print(f"[Memory] Tracemalloc Peak: {measurement['tracemalloc_peak_mb']:.2f} MB, "
              f"Current: {measurement['tracemalloc_current_mb']:.2f} MB")

    # Validar ação
    if mcts_action is not None:
        if mcts_action in legal_actions:
            return mcts_action
        else:
            game_state['invalid_action_count'] += 1
            return random.choice(legal_actions)

    return random.choice(legal_actions)


def _update_memory_stats(results, measurement, gc_collections):
    """Atualiza estatísticas de memória"""
    stats = results['memory_stats']

    # Atualizar picos de memória
    stats['tracemalloc_peak_mb'] = max(
        stats['tracemalloc_peak_mb'], measurement['tracemalloc_peak_mb'])
    stats['psutil_peak_mb'] = max(
        stats['psutil_peak_mb'], measurement['memory_after_mb'])

    # Atualizar crescimento acumulado
    stats['memory_growth_mb'] += measurement['memory_growth_mb']
    stats['tracemalloc_current_mb'] = measurement['tracemalloc_current_mb']
    stats['psutil_current_mb'] = measurement['memory_after_mb']

    # Atualizar coletas GC
    for gen_name, collections in gc_collections.items():
        stats['gc_collections'][gen_name] += collections

    # Adicionar medição
    stats['memory_measurements'].append(measurement)


def _finalize_results(results, game_state, state, ris_mcts_position, total_memory_growth, verbose=False):
    """Finaliza e processa resultados do jogo"""
    if verbose:
        print("--" * 10)
        print(f"Results: {state.returns()}")
        print("Game over")

    # Calcular resultados finais
    result = state.returns()
    win = result[ris_mcts_position] > 0
    ris_taker_win = False
    if state.taker == ris_mcts_position:
        ris_taker_win = result[ris_mcts_position] > 0

    # Atualizar resultados
    results.update({
        'win': win,
        'ris_taker_win': ris_taker_win,
        'legal_actions_count': game_state['legal_actions_count'],
        'chance_outcomes_count': game_state['chance_outcomes_count'],
        'game_length': game_state['game_length'],
        'bids': game_state['bids'],
        'invalid_action_count': game_state['invalid_action_count'],
        'elapsed_time_mcts': game_state['elapsed_time_mcts'],
        'ris_taker': state.taker == ris_mcts_position
    })

    results['memory_stats']['total_memory_growth_mb'] = total_memory_growth

    # Garantir valores padrão para estatísticas de memória
    if results['memory_stats']['tracemalloc_peak_mb'] == 0:
        results['memory_stats']['psutil_peak_mb'] = max(
            results['memory_stats']['psutil_peak_mb'], 0)
        results['memory_stats']['psutil_current_mb'] = max(
            results['memory_stats']['psutil_current_mb'], 0)


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


class SimulationRunner:
    """Classe para gerenciar simulações em lote do RIS-MCTS"""

    def __init__(self, n_games, mcts_iterations):
        self.n_games = n_games
        self.mcts_iterations = mcts_iterations
        self.bid_types = ['Pass', 'Petit', 'Garde']
        self.total_stats = self._initialize_total_stats()

    def _initialize_total_stats(self):
        """Inicializa estatísticas totais da simulação"""
        return {
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

    def run_simulation(self, n_log_games=10, verbose=False):
        """Executa a simulação completa"""
        self._create_csv_header()

        for i in range(self.n_games):
            start_time = time()
            results = ris_mcts_example(
                verbose=verbose, mcts_iterations=self.mcts_iterations)
            end_time = time()

            elapsed = end_time - start_time
            self._update_total_stats(results, elapsed)

            if (i + 1) % n_log_games == 0:
                self._log_intermediate_results(i + 1)

    def _create_csv_header(self):
        """Cria cabeçalho do arquivo CSV"""
        header = [
            "Game", "Win", "Legal Actions", "Chance Outcomes", "Game Length",
            "Invalid Actions", "TraceMallocPeakMB", "PSUtilPeakMB", "MemoryGrowthMB",
            "TotalMemoryGrowthMB", "ElapsedTimeMCTS", "GC_Gen0", "GC_Gen1", "GC_Gen2"
        ]

        # Adicionar colunas de posição RIS-MCTS
        header += [f"RIS_Position_{i}" for i in range(Const.NUM_PLAYERS)]

        # Adicionar colunas de lances por jogador e tipo
        for p in range(Const.NUM_PLAYERS):
            for bid in self.bid_types:
                header.append(f"Bids_{p}_{bid}")

        with open(f'ris_mcts_results_{self.n_games}_{self.mcts_iterations}.csv', 'w') as f:
            f.write(",".join(header) + "\n")

    def _update_total_stats(self, results, elapsed_time):
        """Atualiza estatísticas totais com resultado de um jogo"""
        total = self.total_stats

        # Estatísticas básicas
        total['time_sum'] += elapsed_time
        total['legal_actions_count'] += results['legal_actions_count']
        total['chance_outcomes_count'] += results['chance_outcomes_count']
        total['game_length'] += results['game_length']
        total['invalid_actions'] += results['invalid_action_count']
        total['ris_positions'][results['ris_mcts_position']] += 1
        total['elapsed_time_mcts_sum'] += results['elapsed_time_mcts']
        total['ris_was_taker'] += results['ris_taker']

        # Estatísticas de memória
        memory_stats = total['memory_stats']
        result_memory = results['memory_stats']

        memory_stats['tracemalloc_peak_sum'] += result_memory['tracemalloc_peak_mb']
        memory_stats['psutil_peak_sum'] += result_memory['psutil_peak_mb']
        memory_stats['memory_growth_sum'] += result_memory['memory_growth_mb']
        memory_stats['total_memory_growth_sum'] += result_memory.get(
            'total_memory_growth_mb', 0)
        memory_stats['memory_measurements_count'] += len(
            result_memory['memory_measurements'])

        # Estatísticas de garbage collection
        for gen in ['gen0', 'gen1', 'gen2']:
            memory_stats['gc_collections_sum'][gen] += result_memory['gc_collections'][gen]

        # Lances
        for key, value in results['bids'][results['ris_mcts_position']].items():
            total['bids'][results['ris_mcts_position']].setdefault(key, 0)
            total['bids'][results['ris_mcts_position']][key] += value

        # Vitórias
        if results['win']:
            total['won'] += 1
        if results['ris_taker_win']:
            total['ris_taker_wins'] += 1

    def _log_intermediate_results(self, games_completed):
        """Registra resultados intermediários em arquivos"""
        self._write_csv_row(games_completed)
        self._write_detailed_report(games_completed)

    def _write_csv_row(self, games_completed):
        """Escreve linha no arquivo CSV"""
        total = self.total_stats

        row = [
            str(games_completed),
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

        # Posições RIS-MCTS
        row += [str(pos) for pos in total['ris_positions']]

        # Lances por jogador e tipo
        for p in range(Const.NUM_PLAYERS):
            for bid in self.bid_types:
                row.append(str(total['bids'][p].get(bid, 0)))

        with open(f'ris_mcts_results_{self.n_games}_{self.mcts_iterations}.csv', 'a') as f:
            f.write(",".join(row) + "\n")

    def _write_detailed_report(self, games_completed):
        """Escreve relatório detalhado"""
        total = self.total_stats

        with open(f'ris_mcts_results_{self.n_games}_{self.mcts_iterations}_{games_completed}.txt', 'w') as f:
            f.write(
                f"Wins: {total['won']} out of {games_completed} games ({total['won'] / games_completed:.2%})\n")
            f.write(
                f"Taker Wins: {total['ris_taker_wins']} out of {games_completed} games ({total['ris_taker_wins'] / games_completed:.2%})\n")
            f.write(
                f"Average legal actions per game: {total['legal_actions_count'] / games_completed:.2f}\n")
            f.write(
                f"Average game length: {total['game_length'] / games_completed:.2f} turns\n")
            f.write(
                f"Average time per game: {total['time_sum'] / games_completed:.2f} seconds\n")
            f.write(
                f"Average invalid actions per game: {total['invalid_actions'] / games_completed:.2f}\n")
            f.write(
                f"Average chance outcomes per game: {total['chance_outcomes_count'] / games_completed:.2f}\n")
            f.write(
                f"Average elapsed time RIS-MCTS per game: {total['elapsed_time_mcts_sum'] / games_completed:.4f} s\n")
            f.write("=" * 50 + "\n")
            f.write("DETAILED MEMORY ANALYSIS:\n")

            memory_stats = total['memory_stats']
            f.write(
                f"Average Tracemalloc peak per game: {memory_stats['tracemalloc_peak_sum'] / games_completed:.2f} MB\n")
            f.write(
                f"Average PSUtil peak per game: {memory_stats['psutil_peak_sum'] / games_completed:.2f} MB\n")
            f.write(
                f"Average memory growth from RIS-MCTS per game: {memory_stats['memory_growth_sum'] / games_completed:.2f} MB\n")
            f.write(
                f"Average total memory growth per game: {memory_stats['total_memory_growth_sum'] / games_completed:.2f} MB\n")
            f.write(
                f"Total memory measurements taken: {memory_stats['memory_measurements_count']}\n")
            f.write(
                f"Average GC Gen0 collections per game: {memory_stats['gc_collections_sum']['gen0'] / games_completed:.1f}\n")
            f.write(
                f"Average GC Gen1 collections per game: {memory_stats['gc_collections_sum']['gen1'] / games_completed:.1f}\n")
            f.write(
                f"Average GC Gen2 collections per game: {memory_stats['gc_collections_sum']['gen2'] / games_completed:.1f}\n")
            f.write("=" * 50 + "\n")
            f.write(f"RIS-MCTS positions: {total['ris_positions']}\n")
            f.write(f"Invalid actions: {total['invalid_actions']}\n")
            f.write(f"Bids: {total['bids']}\n")


def run_tarot_simulation(n_games=100, n_log_games=10, verbose=False, mcts_iterations=100):
    """Função wrapper para manter compatibilidade"""
    runner = SimulationRunner(n_games, mcts_iterations)
    runner.run_simulation(n_log_games, verbose)


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
