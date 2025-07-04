import time
import random
from tarot import Tarot, Const, Phase, TarotISMCTSAgent


def play_tarot_game_demo():
    """Play a demo game with IS-MCTS agents focusing on TRICK phase"""
    game = Tarot()

    # Create agents
    agents = [TarotISMCTSAgent(i, iterations=300)
              for i in range(Const.NUM_PLAYERS)]

    # Configure agents for multiple determinization
    for agent in agents:
        agent.use_multiple_determinization = True
        agent.num_determinizations = 6
        agent.determinization_strategy = "per_trick"

    print("Starting French Tarot game with IS-MCTS agents")
    print("=" * 50)

    move_count = 0
    trick_count = 0
    max_moves = 120  # Limit for demo

    while not game.is_terminal() and move_count < max_moves:
        print(f"\nPhase: {game.phase}, Current Player: {game.current}")

        if game.phase == Phase.TRICK:
            # Check if starting new trick
            current_trick_cards = [card for card in game.trick if card != -1]
            if len(current_trick_cards) == 0:
                trick_count += 1
                print(f"\n--- Trick {trick_count} ---")

            # Get action from IS-MCTS agent
            agent = agents[game.current]
            start_time = time.time()
            action = agent.get_action(game)
            decision_time = time.time() - start_time

            # Debug: Check if agent returns Const.TRICK_FINISHED when it shouldn't
            if action == Const.TRICK_FINISHED and len(game.hands[game.current]) > 0:
                print(
                    f"  WARNING: Player {game.current} has {len(game.hands[game.current])} cards but chose Const.TRICK_FINISHED")
                # Force to play a random card instead
                legal_actions = game.legal_actions()
                if legal_actions and legal_actions[0] != Const.TRICK_FINISHED:
                    action = random.choice(
                        [a for a in legal_actions if a != Const.TRICK_FINISHED])
                    print(f"  Forcing player to choose card {action} instead")

            print(
                f"Player {game.current} plays card {action} (took {decision_time:.3f}s)")
            move_count += 1

            # Stop after a few tricks for demo
            if trick_count >= 3:
                print("\nDemo stopping after 3 tricks")
                break

        elif game.phase == Phase.TRICK_FINISHED:
            print("Trick finished, applying Const.TRICK_FINISHED action")
            action = Const.TRICK_FINISHED

        else:
            # For other phases, use random actions
            legal_actions = game.legal_actions()
            action = random.choice(legal_actions) if legal_actions else None
            print(
                f"Phase {game.phase}: Player {game.current} takes action {action}")

        if action is not None:
            game.apply_action(action)
            game.next()
        else:
            break

    print(
        f"\nGame completed after {move_count} moves and {trick_count} tricks")
    print("Final scores:", game.returns()
          if hasattr(game, 'returns') else "N/A")


def compare_tarot_strategies(num_games=3):
    """Compare different determinization strategies for French Tarot"""
    print("Comparing IS-MCTS Strategies for French Tarot")
    print("=" * 50)

    strategies = [
        ("Single Determinization", False, "per_action"),
        ("Multiple Per-Action", True, "per_action"),
        ("Multiple Per-Trick", True, "per_trick")
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

                # Stop early for demo purposes
                if decision_count >= 1000:
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
    print("Focusing on TRICK phase with proper Const.TRICK_FINISHED handling")
    print("=" * 60)

    # Run demo game
    play_tarot_game_demo()

    print("\n" + "=" * 60)

    # Compare strategies
    compare_tarot_strategies(2)

    print("\nFrench Tarot IS-MCTS analysis completed!")
