"""
Dashboard and summary visualization functions for Tarot strategy analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any
from plotting_base import get_strategy_colors, format_strategy_name, create_value_labels_on_bars


def create_summary_dashboard(results: Dict[str, Any], save_path: str = "plot/strategy_dashboard.png"):
    """Create a single comprehensive dashboard with all key metrics"""

    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    strategies = list(results.keys())
    colors = get_strategy_colors(strategies)

    # Main title
    fig.suptitle('IS-MCTS Strategy Performance Dashboard\n(IS-MCTS Strategies Comparison)',
                 fontsize=20, fontweight='bold', y=0.95)

    # 1. Time Performance (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    total_times = [results[strategy]['avg_total_time']
                   for strategy in strategies]
    bars1 = ax1.bar(range(len(strategies)), total_times,
                    color=colors, alpha=0.8)
    ax1.set_title('Average Game Duration', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)
    create_value_labels_on_bars(ax1, bars1, total_times, "{:.2f}s")

    # 2. Decision Speed (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    decisions_per_sec = [results[strategy]
                         ['avg_decisions_per_second'] for strategy in strategies]
    bars2 = ax2.bar(range(len(strategies)), decisions_per_sec,
                    color=colors, alpha=0.8)
    ax2.set_title('Decision Speed', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Decisions per Second')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)
    create_value_labels_on_bars(ax2, bars2, decisions_per_sec, "{:.1f}")

    # 3. Memory Usage (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    memory_usage = [results[strategy]['avg_peak_memory']
                    for strategy in strategies]
    bars3 = ax3.bar(range(len(strategies)), memory_usage,
                    color=colors, alpha=0.8)
    ax3.set_title('Peak Memory Usage', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Memory (MB)')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)
    create_value_labels_on_bars(ax3, bars3, memory_usage, "{:.1f}MB")

    # 4. MCTS Node Activity (Second Row Left)
    ax4 = fig.add_subplot(gs[1, 0])
    node_visits = [results[strategy]['avg_node_visits']
                   for strategy in strategies]
    from plotting_base import apply_log_scale_to_data
    log_node_visits = apply_log_scale_to_data(node_visits)
    ax4.bar(range(len(strategies)), log_node_visits, color=colors, alpha=0.8)
    ax4.set_title('MCTS Node Visits', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Node Visits (log10 scale)')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)

    # 5. MCTS Nodes Created (Second Row Center)
    ax5 = fig.add_subplot(gs[1, 1])
    nodes_created = [results[strategy]['avg_nodes_created']
                     for strategy in strategies]
    log_nodes_created = apply_log_scale_to_data(nodes_created)
    ax5.bar(range(len(strategies)), log_nodes_created, color=colors, alpha=0.8)
    ax5.set_title('MCTS Nodes Created', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Nodes Created (log10 scale)')
    ax5.set_xticks(range(len(strategies)))
    ax5.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)

    # 6. Trees Created (Second Row Right)
    ax6 = fig.add_subplot(gs[1, 2])
    trees_created = [results[strategy]['avg_trees_created']
                     for strategy in strategies]
    bars6 = ax6.bar(range(len(strategies)), trees_created,
                    color=colors, alpha=0.8)
    ax6.set_title('MCTS Trees Created', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Trees Created')
    ax6.set_xticks(range(len(strategies)))
    ax6.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)

    for bar, value in zip(bars6, trees_created):
        if value > 0:
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(trees_created)*0.01,
                     f'{value:.0f}', ha='center', va='bottom', fontsize=9)

    # 7. Cumulative Time Progression (Third Row Left)
    ax7 = fig.add_subplot(gs[2, 0])
    cumulative_times = np.cumsum(total_times)
    ax7.plot(range(len(strategies)), cumulative_times, 'o-',
             linewidth=3, markersize=8, color='darkblue')
    ax7.fill_between(range(len(strategies)), cumulative_times,
                     alpha=0.3, color='lightblue')
    ax7.set_title('Cumulative Game Time', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Cumulative Time (seconds)')
    ax7.set_xticks(range(len(strategies)))
    ax7.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)
    ax7.grid(True, alpha=0.3)

    # 8. Cumulative Memory Usage (Third Row Center)
    ax8 = fig.add_subplot(gs[2, 1])
    cumulative_memory = np.cumsum(memory_usage)
    ax8.plot(range(len(strategies)), cumulative_memory, 's-',
             linewidth=3, markersize=8, color='darkred')
    ax8.fill_between(range(len(strategies)), cumulative_memory,
                     alpha=0.3, color='lightcoral')
    ax8.set_title('Cumulative Memory Usage', fontweight='bold', fontsize=12)
    ax8.set_ylabel('Cumulative Memory (MB)')
    ax8.set_xticks(range(len(strategies)))
    ax8.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)
    ax8.grid(True, alpha=0.3)

    # 9. Cumulative MCTS Activity (Third Row Right)
    ax9 = fig.add_subplot(gs[2, 2])
    cumulative_nodes = np.cumsum(nodes_created)
    log_cumulative_nodes = apply_log_scale_to_data(cumulative_nodes.tolist())
    ax9.plot(range(len(strategies)), log_cumulative_nodes, '^-',
             linewidth=3, markersize=8, color='darkgreen')
    ax9.fill_between(range(len(strategies)),
                     log_cumulative_nodes, alpha=0.3, color='lightgreen')
    ax9.set_title('Cumulative Nodes Created', fontweight='bold', fontsize=12)
    ax9.set_ylabel('Cumulative Nodes (log10 scale)')
    ax9.set_xticks(range(len(strategies)))
    ax9.set_xticklabels([format_strategy_name(s, multiline=True) for s in strategies],
                        rotation=0, fontsize=10)
    ax9.grid(True, alpha=0.3)

    # 10. Performance Summary Table (Bottom span)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')

    # Create a summary table
    table_data = []
    headers = ['Strategy', 'Game Time (s)', 'Dec/Sec',
               'Memory (MB)', 'Nodes Created', 'Trees Created']

    for strategy in strategies:
        row = [
            format_strategy_name(strategy),
            f"{results[strategy]['avg_total_time']:.2f}",
            f"{results[strategy]['avg_decisions_per_second']:.1f}",
            f"{results[strategy]['avg_peak_memory']:.1f}",
            f"{results[strategy]['avg_nodes_created']:,.0f}" if results[strategy]['avg_nodes_created'] > 0 else "N/A",
            f"{results[strategy]['avg_trees_created']:,.0f}" if results[strategy]['avg_trees_created'] > 0 else "N/A"
        ]
        table_data.append(row)

    # Create table
    table = ax10.table(cellText=table_data, colLabels=headers,
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax10.set_title('Performance Summary Table',
                   fontweight='bold', fontsize=14, pad=20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive dashboard saved as {save_path}")
    return fig
