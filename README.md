# RIS-MCTS Application in Imperfect Information Games

This project is an implementation of French Tarot using [Open Spiel](https://github.com/google-deepmind/open_spiel).
The goal is to create agents using MCTS (Monte Carlo Tree Search) variations focused on Imperfect Information such as "Re-determinization Information Set".

## French Tarot

The game was implemented based on the version described on [Pagat](https://www.pagat.com/tarot/frtarot.html), it is a trick-taking card game where each turn is a trick.

** Note: Only the 4-player version has been implemented.
### Deck

French Tarot is a card game with a 78-card deck, consisting of: fourteen cards among 4 **suits**; a group of **trump** cards from 1 to 21; plus the **Excuse**.
The suits are the same as a common deck: Hearts, Spades, Diamonds and Clubs. Similar to a common deck, they go from Ace to 10, plus Jack, Knight, Queen and King (in order).
Among the numbered cards, only three stand out: 1 _Le Petit_, 21 _Le Monde_, and 0 _Le Fou_ or _L'excuse_. The Excuse is often considered the card number 0.
These three cards are finishers, _oudlers_, and determine how the final score will be calculated.

### Game Flow

At the beginning, a person is chosen as _dealer_ to distribute the cards, 18 cards for each player and 6 face down, _chien_ or dog.
A round of bids is made, each contributing to the final score, bets are made counterclockwise, starting to the right of the _dealer_.
* _Petit_ (Small)
* _Garde_ (Guard)
* _Garde sans chien_ (Guard without the dog)
* _Garde contre chien_ (Guard against the dog)

#### Bids and Taker

After the bids are made, the highest bid wins and becomes the _taker_.
They will be the starting player and the other players are defenders who cannot let the taker win.
In both _petit_ and _garde_ the player turns the _chien_ face up, showing the 6 cards for everyone to see.
Then the taker can choose cards from the _chien_ to add to their hand and must discard the same number of cards.
The taker cannot discard numbered cards or Kings.
In _garde sans chien_ the taker does not turn the cards, but they count towards their final score.
Finally, in _garde contre chien_ the taker does not turn the cards, and they will be counted as if they belonged to the defending team.

#### _Poignée_ and _Chelem_

Before each player starts their first play they can declare two things:
* _Poignée_, handful, showing enough numbered cards to get final bonus
* _Chelem_, declaring that the team will win by taking all tricks

#### Tricks

The taker starts with a card, all other players must follow **suit** or **trump**.
In the case of suit, any card of the same suit can be used, but with trump it is mandatory to play a higher trump or discard any card if you have no more trumps in hand.
The trick is won by whoever has the highest suit value or trump number, that trick goes to the taker's pile or the defending team, depending on who wins.

#### The Excuse and Contracts

The excuse is an escape route in case you have no way to follow suit or trump, it doesn't count in the trick and stays with the player who used it in the pile, but the player must substitute it with a low-value card as replacement.
If the excuse is used in the final trick there are two possible rules: according to Pagat it stays with the team that won the trick, according to the French Tarot Federation rules, the trick goes to the opposing team.

Depending on the number of _oudlers_ the taker has, they need to reach a certain amount of total points:
* Zero _oudlers_: 56
* One _oudler_: 51
* Two _oudlers_: 41
* Three _oudlers_: 36

Cards are worth:
* Ace to 10: 0.5 points
* Jack: 1.5 points
* Knight: 2.5 points
* Queen: 3.5 points
* King: 4.5 points
* _Le Fou_ (0), _Le Petit_ (1) and _Le Monde_ (21): 4.5 points
* Other numbered cards: 0.5 points

### Scoring
The final formula is as follows:
`((25 + pt + pb) * mu) + pg + ch`

Where **pt** is if _le petit_ was used in the last trick, **pb** is the difference between the score needed to win and the score from tricks won.

The **mu** is the bid multiplier:
* _Petit_ 1x
* _Garde_ 2x
* _Garde sans chien_ 4x
* _Garde contre chien_ 6x

The bonuses **pg** and **ch** are the handful and the _chelem_. The handful if declared is worth:
* 10 trumps: 20 points
* 13 trumps: 30 points
* 15 trumps: 40 points
The _chelem_ is worth 400 points if declared **and** achieved or 200 points if not declared. If declared and not achieved, the declarer loses 200 points.

### Accounting
The game is *zero-sum* format, meaning the score sums to zero, points are divided equally between the defending team and the taker.
If the taker wins, they deduct points from the other players, if the taker loses, they lose points and the defenders gain equally.

## MCTS Comparison Tool

This project includes a comprehensive tool for comparing different MCTS configurations to optimize performance in Tarot gameplay.

### Basic Usage

```bash
# Run default parameter sweep comparison (saves plots only)
python mcts_comparison.py

# Run with custom number of games and display plots interactively
python mcts_comparison.py --games 100 --show

# Run with custom seed and output directory
python mcts_comparison.py --games 200 --seed 123 --output-dir my_results

# Show plots interactively and save to custom directory
python mcts_comparison.py --show --output-dir my_results
```

### Parameter Sweeps

The tool supports several types of parameter sweeps:

#### Iterations Sweep
```bash
python mcts_comparison.py --sweep-type iterations --iterations 50 100 200 500
```

#### Exploration Constant Sweep
```bash
python mcts_comparison.py --sweep-type exploration --exploration 0.5 1.0 1.4 2.0
```

#### Progressive Widening Alpha Sweep
```bash
python mcts_comparison.py --sweep-type pw_alpha --pw-alpha 0.2 0.5 0.8
```

#### Full Grid Search
```bash
python mcts_comparison.py --sweep-type grid --iterations 100 200 --exploration 1.0 1.4 --pw-alpha 0.5 0.7
```

### Output Options

- `--show`: Display plots interactively (default is save-only)
- `--verbose`: Enable detailed output during execution
- `--output-dir`: Specify custom output directory (default: `results`)

### Output Files

The tool generates three types of output:

1. **Visual plots** (`mcts_parameter_comparison.png`): Comparison charts showing win rates and performance metrics
2. **Text report** (`mcts_comparison_report.txt`): Detailed analysis of each configuration's performance
3. **Raw data** (`mcts_comparison_metrics.json`): Complete dataset for further analysis

### Configuration Parameters

- `--iterations`: Number of MCTS iterations per decision
- `--exploration`: UCB exploration constant (typically 1.0-2.0)
- `--pw-alpha`: Progressive widening alpha parameter (0.0-1.0)
- `--pw-constant`: Progressive widening constant (default: 2.0)
- `--games`: Number of games to simulate per configuration (default: 200)
- `--seed`: Random seed for reproducible results (default: 42)

Use `python mcts_comparison.py --help` for complete argument reference.

## Basic Plotting Tool

The project includes a plotting tool for visualizing game metrics and comparing strategy performance.

### Basic Usage

```bash
# Generate all plots from default metrics file (saves plots only)
python plot_basic.py

# Display plots interactively and use custom metrics file
python plot_basic.py --show --input results/my_metrics.json --output my_plots

# Generate only specific plot types with game count validation
python plot_basic.py --plot-types win_rates legal_moves --games 200

# Include only specific strategies and show plots interactively
python plot_basic.py --strategies random ris_mcts --games 100 --show

# Save plots to custom directory without displaying them
python plot_basic.py --output batch_plots --games 300
```

### Available Plot Types

- **`win_rates`**: Bar charts showing taker and defender win rates by strategy
- **`legal_moves`**: Line plot showing legal moves growth during games (smoothed)
- **`decision_times`**: RIS-MCTS decision time progression through games
- **`mcts_comparison`**: Comparison of different MCTS configurations

### Command Line Options

- `--input, -i`: Path to input metrics JSON file (default: `results/simulation_metrics.json`)
- `--output, -o`: Output directory for plots (default: `plots`)
- `--games`: Expected number of games per strategy (for validation and display)
- `--strategies`: Specific strategies to include (`random`, `max_card`, `min_card`, `ris_mcts`)
- `--plot-types`: Specific plot types to generate
- `--show`: Display plots interactively (default is save-only)
- `--verbose, -v`: Enable detailed output

### Output Files

All plots are saved as high-resolution PNG files:
- `win_rates.png`: Strategy comparison charts
- `legal_moves_growth.png`: Game progression analysis
- `decision_times.png`: MCTS performance metrics
- `mcts_comparison.png`: MCTS configuration comparison

Use `python plot_basic.py --help` for complete usage information.
