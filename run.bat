python simulation.py --seed 42 --games 200
python plot_basic.py --games 200
python .\mcts_comparison.py --games 200 --sweep-type grid --iterations 50 100 200 --exploration 1.0 1.4 2.0 --pw-alpha 0.3 0.5 0.7 --pw-constant 2