"""
Visualization and Analysis for ACO Pipeline Optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json


def plot_convergence(results, save_path=None):
    """
    Plot convergence curve showing best score over iterations.
    
    Args:
        results: Results dictionary from ACO optimization
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = range(1, len(results['iteration_best_scores']) + 1)
    scores = results['iteration_best_scores']
    
    ax.plot(iterations, scores, 'b-', linewidth=2, label='Best Score per Iteration', alpha=0.7)
    ax.fill_between(iterations, scores, alpha=0.3, color='blue')
    
    # Mark global best
    best_iter = np.argmax(scores) + 1
    best_score = max(scores)
    ax.scatter([best_iter], [best_score], c='red', s=200, marker='*', 
               zorder=5, label=f'Global Best (Iter {best_iter})', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_title('ACO Convergence Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved convergence plot to {save_path}")
    
    return fig


def plot_pheromone_heatmap(results, save_path=None):
    """
    Plot heatmap of final pheromone levels.
    
    Args:
        results: Results dictionary from ACO optimization
        save_path: Path to save figure (optional)
    """
    pheromone_data = results['final_pheromone']
    
    # Convert to matrix format
    steps = list(pheromone_data.keys())
    all_operators = []
    for step in steps:
        all_operators.extend(list(pheromone_data[step].keys()))
    all_operators = sorted(list(set(all_operators)))
    
    # Build matrix
    matrix = []
    row_labels = []
    for step in steps:
        for operator, pheromone in sorted(pheromone_data[step].items()):
            matrix.append([pheromone if operator in pheromone_data[s] else 0 
                          for s in steps])
            row_labels.append(f"{step}:{operator}")
    
    # Simpler approach: create one row per step-operator combination
    data_rows = []
    for step in steps:
        for operator in sorted(pheromone_data[step].keys()):
            data_rows.append({
                'Step-Operator': f"{step}:{operator}",
                'Pheromone': pheromone_data[step][operator]
            })
    
    df = pd.DataFrame(data_rows)
    
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Create bar plot instead of heatmap for clarity
    y_pos = np.arange(len(df))
    colors = plt.cm.viridis(df['Pheromone'] / df['Pheromone'].max())
    
    ax.barh(y_pos, df['Pheromone'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Step-Operator'], fontsize=8)
    ax.set_xlabel('Pheromone Level', fontsize=12)
    ax.set_title('Final Pheromone Levels', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved pheromone heatmap to {save_path}")
    
    return fig


def plot_operator_usage(results, save_path=None):
    """
    Plot operator usage frequency and performance.
    
    Args:
        results: Results dictionary from ACO optimization
        save_path: Path to save figure (optional)
    """
    if 'operator_statistics' not in results:
        print("No operator statistics available")
        return None
    
    stats_df = results['operator_statistics']
    
    if stats_df.empty:
        print("Operator statistics DataFrame is empty")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Usage frequency
    usage_data = stats_df.groupby('step')['usage_count'].sum().sort_values(ascending=False)
    ax1.bar(range(len(usage_data)), usage_data.values, color='skyblue', edgecolor='black')
    ax1.set_xticks(range(len(usage_data)))
    ax1.set_xticklabels(usage_data.index, rotation=45, ha='right')
    ax1.set_ylabel('Total Usage Count', fontsize=12)
    ax1.set_title('Operator Usage by Step', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average performance by step
    perf_data = stats_df.groupby('step')['avg_score'].mean().sort_values(ascending=False)
    ax2.bar(range(len(perf_data)), perf_data.values, color='lightgreen', edgecolor='black')
    ax2.set_xticks(range(len(perf_data)))
    ax2.set_xticklabels(perf_data.index, rotation=45, ha='right')
    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_title('Average Performance by Step', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved operator usage plot to {save_path}")
    
    return fig


def plot_multi_dataset_comparison(summary_df, save_path=None):
    """
    Plot comparison across multiple datasets.
    
    Args:
        summary_df: Summary DataFrame from batch optimization
        save_path: Path to save figure (optional)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Best scores by dataset
    datasets = summary_df['dataset_name'].values
    scores = summary_df['best_score'].values
    
    ax1.bar(range(len(datasets)), scores, color='steelblue', edgecolor='black')
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.set_ylabel('Best Score', fontsize=12)
    ax1.set_title('Best Score by Dataset', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {scores.mean():.3f}', linewidth=2)
    ax1.legend()
    
    # Plot 2: Optimization time
    times = summary_df['total_time'].values
    ax2.bar(range(len(datasets)), times, color='coral', edgecolor='black')
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Optimization Time by Dataset', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Score distribution
    ax3.hist(scores, bins=15, color='lightblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {scores.mean():.3f}', linewidth=2)
    ax3.axvline(x=np.median(scores), color='green', linestyle='--', 
                label=f'Median: {np.median(scores):.3f}', linewidth=2)
    ax3.set_xlabel('Best Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time vs Score scatter
    ax4.scatter(times, scores, s=100, c='purple', alpha=0.6, edgecolors='black')
    for i, name in enumerate(datasets):
        ax4.annotate(name, (times[i], scores[i]), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Optimization Time (s)', fontsize=12)
    ax4.set_ylabel('Best Score', fontsize=12)
    ax4.set_title('Time vs Performance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved multi-dataset comparison to {save_path}")
    
    return fig


def plot_pipeline_comparison(pipeline1, pipeline2, names=None, save_path=None):
    """
    Visualize comparison between two pipelines.
    
    Args:
        pipeline1: First pipeline dict
        pipeline2: Second pipeline dict
        names: Tuple of names for the pipelines
        save_path: Path to save figure (optional)
    """
    if names is None:
        names = ('Pipeline 1', 'Pipeline 2')
    
    steps = list(pipeline1.keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create comparison table
    table_data = []
    for step in steps:
        op1 = pipeline1.get(step, 'N/A')
        op2 = pipeline2.get(step, 'N/A')
        match = '✓' if op1 == op2 else '✗'
        table_data.append([step, op1, op2, match])
    
    df = pd.DataFrame(table_data, columns=['Step', names[0], names[1], 'Match'])
    
    # Plot table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.3, 0.3, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color matching rows
    for i, row in enumerate(df.values):
        if row[3] == '✓':
            for j in range(4):
                table[(i+1, j)].set_facecolor('#90EE90')
        else:
            for j in range(4):
                table[(i+1, j)].set_facecolor('#FFB6C6')
    
    # Color header
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Pipeline Comparison', fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved pipeline comparison to {save_path}")
    
    return fig


def generate_optimization_report(results, dataset_name, save_path=None):
    """
    Generate a comprehensive text report of optimization results.
    
    Args:
        results: Results dictionary from ACO optimization
        dataset_name: Name of the dataset
        save_path: Path to save report (optional)
    """
    report = []
    report.append("="*80)
    report.append(f"ACO PIPELINE OPTIMIZATION REPORT")
    report.append(f"Dataset: {dataset_name}")
    report.append("="*80)
    report.append("")
    
    # Summary
    report.append("OPTIMIZATION SUMMARY")
    report.append("-"*80)
    report.append(f"Best Score: {results['best_score']:.4f}")
    report.append(f"Total Time: {results['total_time']:.2f} seconds")
    report.append(f"Total Iterations: {len(results['iteration_best_scores'])}")
    report.append(f"Total Evaluations: {len(results['all_solutions'])}")
    report.append("")
    
    # Best Pipeline
    report.append("BEST PIPELINE")
    report.append("-"*80)
    for step, operator in results['best_pipeline'].items():
        report.append(f"  {step:30s}: {operator}")
    report.append("")
    
    # Convergence Statistics
    scores = results['iteration_best_scores']
    report.append("CONVERGENCE STATISTICS")
    report.append("-"*80)
    report.append(f"Initial Best Score: {scores[0]:.4f}")
    report.append(f"Final Best Score: {scores[-1]:.4f}")
    report.append(f"Improvement: {(scores[-1] - scores[0]):.4f}")
    report.append(f"Best Iteration: {np.argmax(scores) + 1}")
    report.append("")
    
    # Top Operators
    if 'operator_statistics' in results and not results['operator_statistics'].empty:
        report.append("TOP PERFORMING OPERATORS (by average score)")
        report.append("-"*80)
        top_ops = results['operator_statistics'].nlargest(10, 'avg_score')
        for _, row in top_ops.iterrows():
            report.append(f"  {row['step']:20s} : {row['operator']:20s} "
                         f"(avg={row['avg_score']:.4f}, uses={int(row['usage_count'])})")
        report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Saved report to {save_path}")
    
    return report_text


def create_full_visualization_suite(results, dataset_name, output_dir="."):
    """
    Create all visualizations for a single dataset optimization.
    
    Args:
        results: Results dictionary from ACO optimization
        dataset_name: Name of the dataset
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = f"{output_dir}/{dataset_name}"
    
    print(f"\nGenerating visualization suite for {dataset_name}...")
    
    # Convergence plot
    plot_convergence(results, save_path=f"{prefix}_convergence.png")
    
    # Pheromone heatmap
    plot_pheromone_heatmap(results, save_path=f"{prefix}_pheromone.png")
    
    # Operator usage
    plot_operator_usage(results, save_path=f"{prefix}_operator_usage.png")
    
    # Report
    generate_optimization_report(results, dataset_name, save_path=f"{prefix}_report.txt")
    
    print(f"✓ Visualization suite complete for {dataset_name}")


if __name__ == "__main__":
    print("ACO Visualization and Analysis Module")
    print("Import this module to use visualization functions")
