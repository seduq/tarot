import warnings
import matplotlib.font_manager as fm
import os
from typing import Dict, Any
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# Use a backend that doesn't require display (for server environments)
# matplotlib.use('Agg')

# Configure matplotlib to handle font issues and avoid font warnings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Suppress font warnings
warnings.filterwarnings("ignore", message=".*font.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="matplotlib.font_manager")

# Set style for better-looking plots
try:
    plt.style.use('default')
except:
    pass

sns.set_palette("husl")


def load_results_from_json(file_path: str = "simulation_results.json") -> Dict[str, Any]:
    """Load results from JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Results file {file_path} not found. Please run the simulation first.")

    with open(file_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded results for {len(results)} strategies")
    return results


def create_summary_dashboard(results: Dict[str, Any], save_path: str = "plot/strategy_dashboard.png"):
    """Create a single comprehensive dashboard with all key metrics"""

    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    strategies = list(results.keys())
    colors = sns.color_palette("Set2", len(strategies))

    # Main title
    fig.suptitle('Tarot Strategy Performance Dashboard\n(All Strategies vs Random Opponents)',
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
    ax1.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)

    # Add value labels on bars
    for bar, value in zip(bars1, total_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{value:.2f}s', ha='center', va='bottom', fontsize=9)

    # 2. Decision Speed (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    decisions_per_sec = [results[strategy]
                         ['avg_decisions_per_second'] for strategy in strategies]
    bars2 = ax2.bar(range(len(strategies)), decisions_per_sec,
                    color=colors, alpha=0.8)
    ax2.set_title('Decision Speed', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Decisions per Second')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)

    for bar, value in zip(bars2, decisions_per_sec):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    # 3. Memory Usage (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    memory_usage = [results[strategy]['avg_peak_memory']
                    for strategy in strategies]
    bars3 = ax3.bar(range(len(strategies)), memory_usage,
                    color=colors, alpha=0.8)
    ax3.set_title('Peak Memory Usage', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Memory (MB)')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)

    for bar, value in zip(bars3, memory_usage):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{value:.1f}MB', ha='center', va='bottom', fontsize=9)

    # 4. MCTS Node Activity (Second Row Left)
    ax4 = fig.add_subplot(gs[1, 0])
    node_visits = [results[strategy]['avg_node_visits']
                   for strategy in strategies]

    # Use log scale for better visualization
    log_node_visits = [np.log10(max(1, x)) for x in node_visits]
    ax4.bar(range(len(strategies)), log_node_visits, color=colors, alpha=0.8)
    ax4.set_title('MCTS Node Visits', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Node Visits (log10 scale)')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)

    # 5. MCTS Nodes Created (Second Row Center)
    ax5 = fig.add_subplot(gs[1, 1])
    nodes_created = [results[strategy]['avg_nodes_created']
                     for strategy in strategies]

    log_nodes_created = [np.log10(max(1, x)) for x in nodes_created]
    ax5.bar(range(len(strategies)), log_nodes_created, color=colors, alpha=0.8)
    ax5.set_title('MCTS Nodes Created', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Nodes Created (log10 scale)')
    ax5.set_xticks(range(len(strategies)))
    ax5.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)

    # 6. Trees Created (Second Row Right)
    ax6 = fig.add_subplot(gs[1, 2])
    trees_created = [results[strategy]['avg_trees_created']
                     for strategy in strategies]
    bars6 = ax6.bar(range(len(strategies)), trees_created,
                    color=colors, alpha=0.8)
    ax6.set_title('MCTS Trees Created', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Trees Created')
    ax6.set_xticks(range(len(strategies)))
    ax6.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)

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
    ax7.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)
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
    ax8.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)
    ax8.grid(True, alpha=0.3)

    # 9. Cumulative MCTS Activity (Third Row Right)
    ax9 = fig.add_subplot(gs[2, 2])
    cumulative_nodes = np.cumsum(nodes_created)
    log_cumulative_nodes = [np.log10(max(1, x)) for x in cumulative_nodes]
    ax9.plot(range(len(strategies)), log_cumulative_nodes, '^-',
             linewidth=3, markersize=8, color='darkgreen')
    ax9.fill_between(range(len(strategies)),
                     log_cumulative_nodes, alpha=0.3, color='lightgreen')
    ax9.set_title('Cumulative Nodes Created', fontweight='bold', fontsize=12)
    ax9.set_ylabel('Cumulative Nodes (log10 scale)')
    ax9.set_xticks(range(len(strategies)))
    ax9.set_xticklabels([s.replace('_', '\n')
                        for s in strategies], rotation=0, fontsize=10)
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
            strategy.replace('_', ' ').title(),
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


def create_comparison_analysis(results: Dict[str, Any], save_path: str = "plot/strategy_analysis.png"):
    """Create detailed comparison analysis charts"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Strategy Analysis Against Random Opponents',
                 fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    colors = sns.color_palette("viridis", len(strategies))

    # Find baseline (random strategy)
    baseline_strategy = 'random' if 'random' in strategies else strategies[0]
    baseline_idx = strategies.index(baseline_strategy)

    # 1. Efficiency Analysis (Time vs Speed)
    total_times = [results[strategy]['avg_total_time']
                   for strategy in strategies]
    decisions_per_sec = [results[strategy]
                         ['avg_decisions_per_second'] for strategy in strategies]

    scatter = axes[0, 0].scatter(
        total_times, decisions_per_sec, c=colors, s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Average Game Time (seconds)')
    axes[0, 0].set_ylabel('Decisions Per Second')
    axes[0, 0].set_title('Speed vs Duration Efficiency')
    axes[0, 0].grid(True, alpha=0.3)

    for i, strategy in enumerate(strategies):
        axes[0, 0].annotate(strategy.replace('_', '\n'), (total_times[i], decisions_per_sec[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 2. Resource Usage (Memory vs MCTS complexity)
    memory_usage = [results[strategy]['avg_peak_memory']
                    for strategy in strategies]
    nodes_created = [results[strategy]['avg_nodes_created']
                     for strategy in strategies]

    # Filter for strategies that create nodes
    mcts_mask = [x > 0 for x in nodes_created]
    mcts_memory = [memory_usage[i] for i, mask in enumerate(mcts_mask) if mask]
    mcts_nodes = [nodes_created[i] for i, mask in enumerate(mcts_mask) if mask]
    mcts_strategies = [strategies[i]
                       for i, mask in enumerate(mcts_mask) if mask]
    mcts_colors = [colors[i] for i, mask in enumerate(mcts_mask) if mask]

    if mcts_memory:
        log_mcts_nodes = [np.log10(max(1, x)) for x in mcts_nodes]
        axes[0, 1].scatter(mcts_memory, log_mcts_nodes,
                           c=mcts_colors, s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Peak Memory Usage (MB)')
        axes[0, 1].set_ylabel('Average Nodes Created (log10 scale)')
        axes[0, 1].set_title('Memory vs MCTS Complexity')
        axes[0, 1].grid(True, alpha=0.3)

        for i, strategy in enumerate(mcts_strategies):
            axes[0, 1].annotate(strategy.replace('_', '\n'), (mcts_memory[i], log_mcts_nodes[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, 'No MCTS strategies with\nsignificant node creation',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Memory vs MCTS Complexity')

    # 3. Performance Relative to Baseline
    metrics_for_comparison = ['avg_total_time',
                              'avg_decisions_per_second', 'avg_peak_memory']
    metric_labels = ['Game Time', 'Dec/Sec', 'Memory']

    baseline_values = [results[baseline_strategy][metric]
                       for metric in metrics_for_comparison]

    x_pos = np.arange(len(strategies))
    width = 0.25

    for j, (metric, label) in enumerate(zip(metrics_for_comparison, metric_labels)):
        values = [results[strategy][metric] for strategy in strategies]

        # Normalize relative to baseline
        if metric == 'avg_total_time' or metric == 'avg_peak_memory':  # Lower is better
            relative_values = [baseline_values[j] /
                               max(v, 0.001) for v in values]
        else:  # Higher is better
            relative_values = [v / max(baseline_values[j], 0.001)
                               for v in values]

        axes[0, 2].bar(x_pos + j*width, relative_values, width,
                       label=label, alpha=0.7, color=colors)

    axes[0, 2].set_xlabel('Strategy')
    axes[0, 2].set_ylabel('Performance Relative to ' +
                          baseline_strategy.title())
    axes[0, 2].set_title('Relative Performance (>1 is better)')
    axes[0, 2].set_xticks(x_pos + width)
    axes[0, 2].set_xticklabels([s.replace('_', '\n')
                               for s in strategies], rotation=0)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=1, color='red', linestyle='--',
                       alpha=0.7, label='Baseline')

    # 4. Decision Complexity
    legal_actions = [results[strategy]['avg_legal_actions']
                     for strategy in strategies]
    total_decisions = [results[strategy]['avg_total_decisions']
                       for strategy in strategies]

    axes[1, 0].bar(x_pos, legal_actions, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Strategy')
    axes[1, 0].set_ylabel('Average Legal Actions per Decision')
    axes[1, 0].set_title('Decision Space Complexity')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([s.replace('_', '\n')
                               for s in strategies], rotation=0)

    # 5. MCTS Tree Statistics
    trees_created = [results[strategy]['avg_trees_created']
                     for strategy in strategies]
    node_visits = [results[strategy]['avg_node_visits']
                   for strategy in strategies]

    # Create a grouped bar chart for MCTS metrics
    mcts_metrics = np.array([nodes_created, trees_created, [
                            v/1000 for v in node_visits]])  # Scale visits down
    mcts_labels = ['Nodes Created', 'Trees Created', 'Node Visits (K)']

    x_mcts = np.arange(len(strategies))
    width_mcts = 0.25

    for i, (metric_data, label) in enumerate(zip(mcts_metrics, mcts_labels)):
        log_metric_data = [np.log10(max(0.1, x)) for x in metric_data]
        axes[1, 1].bar(x_mcts + i*width_mcts, log_metric_data,
                       width_mcts, label=label, alpha=0.7)

    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('MCTS Activity (log10 scale)')
    axes[1, 1].set_title('MCTS Tree Statistics')
    axes[1, 1].set_xticks(x_mcts + width_mcts)
    axes[1, 1].set_xticklabels([s.replace('_', '\n')
                               for s in strategies], rotation=0)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Cumulative Resource Progression
    cumulative_time = np.cumsum(total_times)
    cumulative_memory = np.cumsum(memory_usage)
    cumulative_nodes = np.cumsum(nodes_created)

    axes[1, 2].plot(range(len(strategies)), cumulative_time,
                    'o-', linewidth=2, label='Time (s)', color='blue')
    axes[1, 2].plot(range(len(strategies)), cumulative_memory,
                    's-', linewidth=2, label='Memory (MB)', color='red')

    # Scale nodes to fit on same plot
    if max(cumulative_nodes) > 0:
        scale_factor = max(cumulative_time) / max(cumulative_nodes)
        scaled_nodes = [x * scale_factor for x in cumulative_nodes]
        axes[1, 2].plot(range(len(strategies)), scaled_nodes, '^-', linewidth=2,
                        label=f'Nodes (×{scale_factor:.0e})', color='green')

    axes[1, 2].set_xlabel('Strategy Order')
    axes[1, 2].set_ylabel('Cumulative Resource Usage')
    axes[1, 2].set_title('Cumulative Resource Progression')
    axes[1, 2].set_xticks(range(len(strategies)))
    axes[1, 2].set_xticklabels([s.replace('_', '\n')
                               for s in strategies], rotation=0)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved as {save_path}")
    return fig


def main():
    """Main function to create plots from existing results"""
    print("Tarot Strategy Metrics Visualization (from saved results)")
    print("=" * 60)

    os.makedirs("plot", exist_ok=True)  # Ensure plot directory exists

    try:
        # Try to load existing results
        results = load_results_from_json()

        print("\nCreating visualization plots...")

        # Create comprehensive dashboard
        create_summary_dashboard(results)

        # Create detailed analysis
        create_comparison_analysis(results)

        print("\nPlots created successfully!")
        print("Generated files:")
        print("- strategy_dashboard.png (Main dashboard)")
        print("- strategy_analysis.png (Detailed analysis)")

        # Show plots
        plt.show()

    except FileNotFoundError:
        print("No existing results found.")
        print("Please run the main simulation first with:")
        print("python plot_metrics.py")
        print("\nOr run the simulation to generate data:")
        from plot_metrics import main as run_simulation
        run_simulation()


if __name__ == "__main__":
    main()
