"""
Report Generator for Tarot Game Metrics

This module creates human-readable reports summarizing simulation results
with key statistics, averages, and insights.
"""

import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional
from metrics import MetricsCollector


class MetricsReporter:
    """Generates readable reports from simulation metrics."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def generate_summary_report(self, output_dir: str = "results", filename: str = "simulation_report.txt"):
        """Generate a comprehensive summary report."""

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)

        # Get all strategies in the data
        all_strategies = list(
            set(game.strategy for game in self.collector.games))

        with open(output_file, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("TAROT GAME SIMULATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Games: {len(self.collector.games)}\n")
            f.write(f"Strategies Tested: {', '.join(all_strategies)}\n")
            f.write("\n")

            # Overall Statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            for strategy in all_strategies:
                games = self.collector.get_strategy_metrics(strategy)
                f.write(
                    f"{strategy.upper().replace('_', ' ')}: {len(games)} games\n")
            f.write("\n")

            # Win Rate Analysis
            f.write("WIN RATE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Strategy':<15} {'Taker Win %':<12} {'Defender Win %':<15} {'Games':<8}\n")
            f.write("-" * 50 + "\n")

            for strategy in all_strategies:
                win_rates = self.collector.get_win_rates(strategy)
                games_count = len(
                    self.collector.get_strategy_metrics(strategy))
                f.write(f"{strategy:<15} "
                        f"{win_rates['taker_win_rate']*100:>9.1f}% "
                        f"{win_rates['defender_win_rate']*100:>12.1f}% "
                        f"{games_count:>6}\n")
            f.write("\n")

            # Game Length Analysis
            f.write("GAME LENGTH ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Strategy':<15} {'Avg Turns':<10} {'Min':<6} {'Max':<6} {'Std Dev':<8}\n")
            f.write("-" * 45 + "\n")

            for strategy in all_strategies:
                games = self.collector.get_strategy_metrics(strategy)
                game_lengths = [len(game.legal_moves_history)
                                for game in games if game.legal_moves_history]

                if game_lengths:
                    avg_length = statistics.mean(game_lengths)
                    min_length = min(game_lengths)
                    max_length = max(game_lengths)
                    std_dev = statistics.stdev(game_lengths) if len(
                        game_lengths) > 1 else 0

                    f.write(f"{strategy:<15} "
                            f"{avg_length:>7.1f} "
                            f"{min_length:>6} "
                            f"{max_length:>6} "
                            f"{std_dev:>7.1f}\n")
            f.write("\n")

            # Legal Moves Analysis
            f.write("LEGAL MOVES ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Strategy':<15} {'Avg Start':<10} {'Avg Mid':<10} {'Avg End':<10}\n")
            f.write("-" * 45 + "\n")

            for strategy in all_strategies:
                legal_moves = self.collector.get_average_legal_moves(
                    strategy, smoothing_window=1)
                if legal_moves and len(legal_moves) > 6:
                    # Early game (first 20% of moves)
                    early_end = max(1, len(legal_moves) // 5)
                    avg_start = statistics.mean(legal_moves[:early_end])

                    # Mid game (middle 60% of moves)
                    mid_start = early_end
                    mid_end = len(legal_moves) - early_end
                    avg_mid = statistics.mean(
                        legal_moves[mid_start:mid_end]) if mid_end > mid_start else 0

                    # End game (last 20% of moves)
                    avg_end = statistics.mean(legal_moves[-early_end:])

                    f.write(f"{strategy:<15} "
                            f"{avg_start:>7.1f} "
                            f"{avg_mid:>7.1f} "
                            f"{avg_end:>7.1f}\n")
            f.write("\n")

            # MCTS Performance Analysis
            mcts_games = self.collector.get_strategy_metrics("ris_mcts")
            if mcts_games:
                f.write("RIS-MCTS PERFORMANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")

                # Decision time statistics
                all_decision_times = []
                for game in mcts_games:
                    # Filter out zero times (non-MCTS decisions)
                    mcts_times = [t for t in game.decision_times if t > 0.0]
                    all_decision_times.extend(mcts_times)

                if all_decision_times:
                    avg_time = statistics.mean(all_decision_times)
                    median_time = statistics.median(all_decision_times)
                    max_time = max(all_decision_times)
                    min_time = min(all_decision_times)
                    std_time = statistics.stdev(all_decision_times) if len(
                        all_decision_times) > 1 else 0

                    f.write(f"Decision Time Statistics:\n")
                    f.write(f"  Average: {avg_time:.4f} seconds\n")
                    f.write(f"  Median:  {median_time:.4f} seconds\n")
                    f.write(f"  Min:     {min_time:.4f} seconds\n")
                    f.write(f"  Max:     {max_time:.4f} seconds\n")
                    f.write(f"  Std Dev: {std_time:.4f} seconds\n")
                    f.write(f"  Total Decisions: {len(all_decision_times)}\n")
                else:
                    avg_time = 0  # Default value if no decision times

                # MCTS Configuration Analysis
                f.write(f"\nMCTS Configuration:\n")
                if mcts_games:
                    config = mcts_games[0].mcts_config
                    f.write(
                        f"  Iterations: {config.get('iterations', 'N/A')}\n")
                    f.write(
                        f"  Exploration Constant: {config.get('exploration_constant', 'N/A')}\n")
                    f.write(
                        f"  Progressive Widening Alpha: {config.get('pw_alpha', 'N/A')}\n")
                    f.write(
                        f"  Progressive Widening Constant: {config.get('pw_constant', 'N/A')}\n")

                # Multi-agent analysis
                agent_counts = [game.num_mcts_agents for game in mcts_games]
                if agent_counts:
                    f.write(f"\nMCTS Agent Distribution:\n")
                    for i in range(1, 6):  # 1-5 agents
                        count = agent_counts.count(i)
                        if count > 0:
                            percentage = (count / len(agent_counts)) * 100
                            f.write(
                                f"  {i} agent(s): {count} games ({percentage:.1f}%)\n")

                f.write("\n")
            else:
                avg_time = 0  # Default if no MCTS games

            # Strategy Comparison Summary
            f.write("STRATEGY COMPARISON SUMMARY\n")
            f.write("-" * 40 + "\n")

            # Find best performing strategies
            strategy_scores = {}
            for strategy in all_strategies:
                win_rates = self.collector.get_win_rates(strategy)
                # Simple scoring: average of taker and defender win rates
                score = (win_rates['taker_win_rate'] +
                         win_rates['defender_win_rate']) / 2
                strategy_scores[strategy] = score

            # Sort by performance
            sorted_strategies = sorted(
                strategy_scores.items(), key=lambda x: x[1], reverse=True)

            f.write("Performance Ranking (by combined win rate):\n")
            for i, (strategy, score) in enumerate(sorted_strategies, 1):
                f.write(
                    f"  {i}. {strategy.replace('_', ' ').title()}: {score*100:.1f}%\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        print(f"Report saved to: {output_file}")

    def generate_csv_summary(self, output_dir: str = "results", filename: str = "simulation_summary.csv"):
        """Generate a CSV summary for spreadsheet analysis."""

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)

        all_strategies = list(
            set(game.strategy for game in self.collector.games))

        with open(output_file, 'w') as f:
            # Header
            f.write(
                "Strategy,Games,Taker_Win_Rate,Defender_Win_Rate,Avg_Game_Length,")
            f.write(
                "Avg_Legal_Moves_Start,Avg_Legal_Moves_End,Avg_Decision_Time,Total_Decisions\n")

            for strategy in all_strategies:
                games = self.collector.get_strategy_metrics(strategy)
                win_rates = self.collector.get_win_rates(strategy)

                # Game length stats
                game_lengths = [len(game.legal_moves_history)
                                for game in games if game.legal_moves_history]
                avg_length = statistics.mean(
                    game_lengths) if game_lengths else 0

                # Legal moves stats
                legal_moves = self.collector.get_average_legal_moves(
                    strategy, smoothing_window=1)
                avg_start = legal_moves[0] if legal_moves else 0
                avg_end = legal_moves[-1] if legal_moves else 0

                # Decision time stats (for MCTS)
                all_times = []
                for game in games:
                    mcts_times = [t for t in game.decision_times if t > 0.0]
                    all_times.extend(mcts_times)
                avg_time = statistics.mean(all_times) if all_times else 0

                f.write(
                    f"{strategy},{len(games)},{win_rates['taker_win_rate']:.4f},")
                f.write(
                    f"{win_rates['defender_win_rate']:.4f},{avg_length:.1f},")
                f.write(
                    f"{avg_start:.2f},{avg_end:.2f},{avg_time:.4f},{len(all_times)}\n")

        print(f"CSV summary saved to: {output_file}")


def create_reports_from_metrics(metrics_file: str, output_dir: str = "results"):
    """Create both text and CSV reports from a metrics file."""

    collector = MetricsCollector.load_from_file(metrics_file)
    reporter = MetricsReporter(collector)

    # Generate reports with directory structure
    reporter.generate_summary_report(output_dir, "simulation_report.txt")
    reporter.generate_csv_summary(output_dir, "simulation_summary.csv")

    print(f"\nReports generated in directory: {output_dir}")
    print(
        f"  Text report: {os.path.join(output_dir, 'simulation_report.txt')}")
    print(
        f"  CSV summary: {os.path.join(output_dir, 'simulation_summary.csv')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <metrics_file.json>")
        print("Example: python report_generator.py simulation_metrics_seed_7264828.json")
    else:
        metrics_file = sys.argv[1]
        create_reports_from_metrics(metrics_file)
