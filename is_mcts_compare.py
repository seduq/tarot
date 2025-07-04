import time
import random
from tarot import Tarot, Const, Phase, TarotISMCTSAgent


def compare_tarot_strategies(num_games=3):
    """Compare different determinization strategies for French Tarot"""
    print("Comparing IS-MCTS Strategies for French Tarot")
    print("=" * 50)

    strategies = [
        ("Single Determinization", False, TarotISMCTSAgent.Strategy.PER_ACTION),
        ("Multiple Per-Action", True, TarotISMCTSAgent.Strategy.PER_ACTION),
        ("Multiple Per-Trick", True, TarotISMCTSAgent.Strategy.PER_TRICK)
    ]

    results = {}

    for strategy_name, use_multiple, det_strategy in strategies:
        print(f"\n=== Testing {strategy_name} ===")

        total_time = 0
        total_decisions = 0

        for game_num in range(num_games):
            print(f"Game {game_num + 1}/{num_games}")

            game = Tarot()
            agents = [TarotISMCTSAgent(i, iterations=200)
                      for i in range(Const.NUM_PLAYERS)]

            # Configure agents
            for agent in agents:
                agent.use_multiple_determinization = use_multiple
                agent.determinization_strategy = det_strategy
                agent.num_determinizations = 6 if use_multiple else 1

            game_start = time.time()
            decision_count = 0

            while not game.is_terminal():
                if game.phase == Phase.TRICK:
                    agent = agents[game.current]

                    decision_start = time.time()
                    action = agent.get_action(game)
                    decision_time = time.time() - decision_start

                    decision_count += 1
                    if decision_count <= 3:  # Show timing for first few decisions
                        print(
                            f"  Decision {decision_count}: {decision_time:.3f}s")

                elif game.phase == Phase.TRICK_FINISHED:
                    action = Const.TRICK_FINISHED
                else:
                    legal_actions = game.legal_actions()
                    action = random.choice(
                        legal_actions) if legal_actions else None

                if action is not None:
                    try:
                        game.apply_action(action)
                        game.next()
                    except ValueError as e:
                        print(f"  Error applying action {action}: {e}")
                        break
                else:
                    break

            game_time = time.time() - game_start
            total_time += game_time
            total_decisions += decision_count

            print(
                f"  Game time: {game_time:.2f}s, Decisions: {decision_count}")

        avg_time = total_time / num_games
        avg_decision_time = total_time / total_decisions if total_decisions > 0 else 0

        results[strategy_name] = {
            'avg_game_time': avg_time,
            'avg_decision_time': avg_decision_time,
            'decisions_per_second': total_decisions / total_time if total_time > 0 else 0
        }

        print(f"\n{strategy_name} Results:")
        print(f"  Average game time: {avg_time:.2f}s")
        print(f"  Average decision time: {avg_decision_time:.3f}s")
        print(
            f"  Decisions per second: {results[strategy_name]['decisions_per_second']:.1f}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")
    print(f"{'Strategy':<25} {'Game Time':<12} {'Decision Time':<15} {'Dec/sec':<10}")
    print("-" * 62)

    for strategy_name, data in results.items():
        print(f"{strategy_name:<25} {data['avg_game_time']:<12.2f} "
              f"{data['avg_decision_time']:<15.3f} {data['decisions_per_second']:<10.1f}")

    return results


if __name__ == "__main__":
    print("French Tarot IS-MCTS Implementation")
    print("=" * 60)

    # Compare strategies
    compare_tarot_strategies(2)

    print("\nFrench Tarot IS-MCTS analysis completed!")
