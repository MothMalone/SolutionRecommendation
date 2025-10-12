"""
Comprehensive Recommender Debugging and Visualization Script

This script provides deep analysis and visualization to understand:
- Why certain recommenders perform better/worse
- What patterns they learn
- Where they fail and succeed
- Concrete evidence with numbers and visualizations
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class RecommenderDebugger:
    """Deep debugging and analysis of recommender performance."""
    
    def __init__(self, comparison_csv, detailed_csv, test_ground_truth_csv, performance_matrix_csv, metafeatures_csv):
        """Load all necessary data for analysis."""
        print("Loading data for analysis...")
        
        self.summary = pd.read_csv(comparison_csv)
        self.detailed = pd.read_csv(detailed_csv)
        self.test_ground_truth = pd.read_csv(test_ground_truth_csv, index_col=0)
        self.performance_matrix = pd.read_csv(performance_matrix_csv, index_col=0)
        self.metafeatures = pd.read_csv(metafeatures_csv, index_col=0)
        
        print(f"✅ Loaded {len(self.summary)} recommenders")
        print(f"✅ Loaded {len(self.detailed)} detailed recommendations")
        print(f"✅ Test ground truth: {self.test_ground_truth.shape}")
        print(f"✅ Performance matrix: {self.performance_matrix.shape}")
        print(f"✅ Metafeatures: {self.metafeatures.shape}")
    
    def generate_all_analyses(self, output_dir='debug_output'):
        """Generate all debugging analyses and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE RECOMMENDER DEBUGGING")
        print("="*80)
        
        # 1. Pipeline Diversity Analysis
        print("\n📊 1. Analyzing pipeline diversity...")
        self.analyze_pipeline_diversity(output_dir)
        
        # 2. Ranking Distribution Analysis
        print("\n📊 2. Analyzing ranking distributions...")
        self.analyze_ranking_distributions(output_dir)
        
        # 3. Dataset Difficulty Analysis
        print("\n📊 3. Analyzing dataset difficulty...")
        self.analyze_dataset_difficulty(output_dir)
        
        # 4. Baseline Comparison Deep Dive
        print("\n📊 4. Analyzing baseline comparison...")
        self.analyze_baseline_comparison(output_dir)
        
        # 5. Recommender Agreement Analysis
        print("\n📊 5. Analyzing recommender agreement...")
        self.analyze_recommender_agreement(output_dir)
        
        # 6. Performance vs Metafeatures
        print("\n📊 6. Analyzing performance vs metafeatures...")
        self.analyze_performance_vs_metafeatures(output_dir)
        
        # 7. Error Pattern Analysis
        print("\n📊 7. Analyzing error patterns...")
        self.analyze_error_patterns(output_dir)
        
        # 8. Head-to-Head Comparison
        print("\n📊 8. Generating head-to-head comparisons...")
        self.generate_head_to_head_comparison(output_dir)
        
        # 9. Statistical Significance Tests
        print("\n📊 9. Running statistical tests...")
        self.run_statistical_tests(output_dir)
        
        # 10. Concrete Evidence Summary
        print("\n📊 10. Generating concrete evidence summary...")
        self.generate_concrete_evidence_summary(output_dir)
        
        print(f"\n✅ All analyses saved to: {output_dir}/")
    
    def analyze_pipeline_diversity(self, output_dir):
        """Analyze how diverse recommendations are from each recommender."""
        
        diversity_data = []
        
        for recommender in self.detailed['Recommender'].unique():
            rec_data = self.detailed[self.detailed['Recommender'] == recommender]
            
            # Calculate diversity metrics
            total_recommendations = len(rec_data)
            unique_pipelines = rec_data['Recommended Pipeline'].nunique()
            diversity_ratio = unique_pipelines / total_recommendations
            
            # Get most common pipeline
            pipeline_counts = rec_data['Recommended Pipeline'].value_counts()
            most_common_pipeline = pipeline_counts.index[0]
            most_common_count = pipeline_counts.values[0]
            most_common_pct = (most_common_count / total_recommendations) * 100
            
            # Calculate entropy (higher = more diverse)
            pipeline_probs = pipeline_counts / total_recommendations
            entropy = -np.sum(pipeline_probs * np.log2(pipeline_probs + 1e-10))
            
            diversity_data.append({
                'Recommender': recommender,
                'Total Recommendations': total_recommendations,
                'Unique Pipelines': unique_pipelines,
                'Diversity Ratio': diversity_ratio,
                'Most Common Pipeline': most_common_pipeline,
                'Most Common %': most_common_pct,
                'Entropy': entropy
            })
        
        df_diversity = pd.DataFrame(diversity_data).sort_values('Entropy', ascending=False)
        
        # Save results
        df_diversity.to_csv(f'{output_dir}/01_pipeline_diversity.csv', index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("PIPELINE DIVERSITY ANALYSIS")
        print("="*80)
        print("\nDiversity Rankings (Higher Entropy = More Diverse):")
        print(df_diversity[['Recommender', 'Unique Pipelines', 'Diversity Ratio', 'Most Common %', 'Entropy']].to_string(index=False))
        
        # Visualization 1: Diversity bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Unique pipelines
        ax1 = axes[0, 0]
        df_diversity.plot(x='Recommender', y='Unique Pipelines', kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Number of Unique Pipelines Recommended', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Unique Pipelines')
        ax1.set_xlabel('')
        ax1.axhline(y=12, color='red', linestyle='--', label='Max Possible (12)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Diversity ratio
        ax2 = axes[0, 1]
        df_diversity.plot(x='Recommender', y='Diversity Ratio', kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Diversity Ratio (Unique / Total)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Diversity Ratio')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Most common pipeline percentage
        ax3 = axes[1, 0]
        df_diversity.plot(x='Recommender', y='Most Common %', kind='bar', ax=ax3, color='lightgreen')
        ax3.set_title('Most Common Pipeline % (Lower = More Diverse)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Most Common Pipeline %')
        ax3.set_xlabel('')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Entropy
        ax4 = axes[1, 1]
        df_diversity.plot(x='Recommender', y='Entropy', kind='bar', ax=ax4, color='plum')
        ax4.set_title('Shannon Entropy (Higher = More Diverse)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Entropy (bits)')
        ax4.set_xlabel('')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_pipeline_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved: {output_dir}/01_pipeline_diversity.csv")
        print(f"✅ Saved: {output_dir}/01_pipeline_diversity.png")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        most_diverse = df_diversity.iloc[0]
        least_diverse = df_diversity.iloc[-1]
        print(f"   • Most diverse: {most_diverse['Recommender']} (Entropy: {most_diverse['Entropy']:.2f}, {most_diverse['Unique Pipelines']} unique pipelines)")
        print(f"   • Least diverse: {least_diverse['Recommender']} (Entropy: {least_diverse['Entropy']:.2f}, only {least_diverse['Unique Pipelines']} unique pipelines)")
        
        # Find "always same pipeline" recommenders
        always_same = df_diversity[df_diversity['Unique Pipelines'] == 1]
        if len(always_same) > 0:
            print(f"\n   ⚠️  WARNING: These recommenders ALWAYS predict the same pipeline:")
            for _, row in always_same.iterrows():
                print(f"      - {row['Recommender']}: always predicts '{row['Most Common Pipeline']}'")
    
    def analyze_ranking_distributions(self, output_dir):
        """Analyze what ranks each recommender typically achieves."""
        
        # Visualization: Heatmap of rank distributions
        recommenders = self.detailed['Recommender'].unique()
        rank_matrix = np.zeros((len(recommenders), 12))  # 12 possible ranks
        
        for i, recommender in enumerate(recommenders):
            rec_data = self.detailed[self.detailed['Recommender'] == recommender]
            rank_counts = rec_data['Pipeline Rank'].value_counts()
            
            for rank in range(1, 13):
                if rank in rank_counts.index:
                    rank_matrix[i, rank-1] = rank_counts[rank]
        
        # Convert to percentages
        rank_matrix_pct = (rank_matrix / rank_matrix.sum(axis=1, keepdims=True)) * 100
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(rank_matrix_pct, 
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn_r',
                   xticklabels=range(1, 13),
                   yticklabels=recommenders,
                   cbar_kws={'label': 'Percentage (%)'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title('Rank Distribution Heatmap (%)\nGreen = Often achieves this rank, Red = Rarely', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Pipeline Rank (1=Best, 12=Worst)', fontsize=12)
        ax.set_ylabel('Recommender', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_rank_distribution_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical summary
        rank_stats = []
        for recommender in recommenders:
            rec_data = self.detailed[self.detailed['Recommender'] == recommender]
            ranks = rec_data['Pipeline Rank'].values
            
            rank_stats.append({
                'Recommender': recommender,
                'Mean Rank': np.mean(ranks),
                'Median Rank': np.median(ranks),
                'Std Rank': np.std(ranks),
                'Min Rank': np.min(ranks),
                'Max Rank': np.max(ranks),
                'Rank 1 Count': np.sum(ranks == 1),
                'Top-3 Count': np.sum(ranks <= 3),
                'Bottom-3 Count': np.sum(ranks >= 10)
            })
        
        df_rank_stats = pd.DataFrame(rank_stats).sort_values('Mean Rank')
        df_rank_stats.to_csv(f'{output_dir}/02_rank_statistics.csv', index=False)
        
        print("\n" + "="*80)
        print("RANKING DISTRIBUTION ANALYSIS")
        print("="*80)
        print("\nRank Statistics:")
        print(df_rank_stats.to_string(index=False))
        
        print(f"\n✅ Saved: {output_dir}/02_rank_distribution_heatmap.png")
        print(f"✅ Saved: {output_dir}/02_rank_statistics.csv")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        best_avg = df_rank_stats.iloc[0]
        worst_avg = df_rank_stats.iloc[-1]
        print(f"   • Best average rank: {best_avg['Recommender']} (Mean: {best_avg['Mean Rank']:.2f}, {best_avg['Rank 1 Count']} perfect predictions)")
        print(f"   • Worst average rank: {worst_avg['Recommender']} (Mean: {worst_avg['Mean Rank']:.2f})")
        
        # Find consistent vs inconsistent recommenders
        most_consistent = df_rank_stats.loc[df_rank_stats['Std Rank'].idxmin()]
        most_inconsistent = df_rank_stats.loc[df_rank_stats['Std Rank'].idxmax()]
        print(f"   • Most consistent: {most_consistent['Recommender']} (Std: {most_consistent['Std Rank']:.2f})")
        print(f"   • Most inconsistent: {most_inconsistent['Recommender']} (Std: {most_inconsistent['Std Rank']:.2f})")
    
    def analyze_dataset_difficulty(self, output_dir):
        """Identify which datasets are hard/easy for recommenders."""
        
        dataset_stats = []
        
        for dataset_id in self.detailed['Dataset'].unique():
            dataset_data = self.detailed[self.detailed['Dataset'] == dataset_id]
            
            # Calculate statistics
            avg_rank = dataset_data['Pipeline Rank'].mean()
            std_rank = dataset_data['Pipeline Rank'].std()
            avg_regret = dataset_data['Regret'].mean()
            
            # Find best and worst recommenders
            best_rec_idx = dataset_data['Pipeline Rank'].idxmin()
            worst_rec_idx = dataset_data['Pipeline Rank'].idxmax()
            
            best_recommender = dataset_data.loc[best_rec_idx, 'Recommender']
            best_rank = dataset_data.loc[best_rec_idx, 'Pipeline Rank']
            
            worst_recommender = dataset_data.loc[worst_rec_idx, 'Recommender']
            worst_rank = dataset_data.loc[worst_rec_idx, 'Pipeline Rank']
            
            # Count how many recommenders get it right (rank 1)
            perfect_count = np.sum(dataset_data['Pipeline Rank'] == 1)
            top3_count = np.sum(dataset_data['Pipeline Rank'] <= 3)
            
            # Get ground truth info
            col_name = f'D_{dataset_id}'
            if col_name in self.test_ground_truth.columns:
                gt_performances = self.test_ground_truth[col_name].dropna()
                best_pipeline = gt_performances.idxmax()
                best_score = gt_performances.max()
                worst_score = gt_performances.min()
                score_range = best_score - worst_score
                baseline_score = gt_performances.get('baseline', np.nan)
                baseline_rank = sorted(gt_performances.values, reverse=True).index(baseline_score) + 1 if not np.isnan(baseline_score) else 999
            else:
                best_pipeline = 'N/A'
                best_score = np.nan
                score_range = np.nan
                baseline_rank = 999
            
            dataset_stats.append({
                'Dataset': dataset_id,
                'Avg Rank': avg_rank,
                'Std Rank': std_rank,
                'Avg Regret': avg_regret,
                'Perfect Count': perfect_count,
                'Top-3 Count': top3_count,
                'Best Recommender': best_recommender,
                'Best Rank': best_rank,
                'Worst Recommender': worst_recommender,
                'Worst Rank': worst_rank,
                'GT Best Pipeline': best_pipeline,
                'GT Best Score': best_score,
                'Score Range': score_range,
                'Baseline Rank': baseline_rank
            })
        
        df_difficulty = pd.DataFrame(dataset_stats).sort_values('Avg Rank', ascending=False)
        df_difficulty.to_csv(f'{output_dir}/03_dataset_difficulty.csv', index=False)
        
        print("\n" + "="*80)
        print("DATASET DIFFICULTY ANALYSIS")
        print("="*80)
        
        print("\n📉 HARDEST DATASETS (High Avg Rank = Hard):")
        print(df_difficulty.head(5)[['Dataset', 'Avg Rank', 'Perfect Count', 'Best Recommender', 'Baseline Rank']].to_string(index=False))
        
        print("\n📈 EASIEST DATASETS (Low Avg Rank = Easy):")
        print(df_difficulty.tail(5)[['Dataset', 'Avg Rank', 'Perfect Count', 'Best Recommender', 'Baseline Rank']].to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Dataset difficulty (avg rank)
        ax1 = axes[0, 0]
        df_difficulty.plot(x='Dataset', y='Avg Rank', kind='bar', ax=ax1, color='coral')
        ax1.set_title('Dataset Difficulty (Higher = Harder)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Rank Across All Recommenders')
        ax1.set_xlabel('Dataset ID')
        ax1.axhline(y=6.5, color='red', linestyle='--', label='Middle Rank (6.5)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Perfect prediction count
        ax2 = axes[0, 1]
        df_difficulty.plot(x='Dataset', y='Perfect Count', kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('# Recommenders Getting Perfect Prediction (Rank 1)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count of Perfect Predictions')
        ax2.set_xlabel('Dataset ID')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Score range
        ax3 = axes[1, 0]
        df_difficulty.plot(x='Dataset', y='Score Range', kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Performance Difference (Best - Worst Pipeline)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score Range')
        ax3.set_xlabel('Dataset ID')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Baseline rank
        ax4 = axes[1, 1]
        df_difficulty.plot(x='Dataset', y='Baseline Rank', kind='bar', ax=ax4, color='plum')
        ax4.set_title('Baseline Pipeline Rank (Lower = Baseline is good)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Baseline Rank')
        ax4.set_xlabel('Dataset ID')
        ax4.axhline(y=6.5, color='red', linestyle='--', label='Middle Rank')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_dataset_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved: {output_dir}/03_dataset_difficulty.csv")
        print(f"✅ Saved: {output_dir}/03_dataset_difficulty.png")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        hardest = df_difficulty.iloc[0]
        easiest = df_difficulty.iloc[-1]
        print(f"   • Hardest dataset: {hardest['Dataset']} (Avg Rank: {hardest['Avg Rank']:.2f}, only {hardest['Perfect Count']} perfect predictions)")
        print(f"   • Easiest dataset: {easiest['Dataset']} (Avg Rank: {easiest['Avg Rank']:.2f}, {easiest['Perfect Count']} perfect predictions)")
        
        # Baseline analysis
        baseline_good = len(df_difficulty[df_difficulty['Baseline Rank'] <= 3])
        baseline_bad = len(df_difficulty[df_difficulty['Baseline Rank'] >= 10])
        print(f"   • Baseline is in top-3 for {baseline_good}/{len(df_difficulty)} datasets ({baseline_good/len(df_difficulty)*100:.1f}%)")
        print(f"   • Baseline is in bottom-3 for {baseline_bad}/{len(df_difficulty)} datasets ({baseline_bad/len(df_difficulty)*100:.1f}%)")
    
    def analyze_baseline_comparison(self, output_dir):
        """Deep dive into how recommenders compare to baseline."""
        
        baseline_comparison = []
        
        for recommender in self.detailed['Recommender'].unique():
            rec_data = self.detailed[self.detailed['Recommender'] == recommender]
            
            better = 0
            equal = 0
            worse = 0
            
            for _, row in rec_data.iterrows():
                dataset_id = row['Dataset']
                col_name = f'D_{dataset_id}'
                
                if col_name not in self.test_ground_truth.columns:
                    continue
                
                gt_performances = self.test_ground_truth[col_name].dropna()
                
                if 'baseline' not in gt_performances.index or row['Recommended Pipeline'] not in gt_performances.index:
                    continue
                
                rec_score = gt_performances[row['Recommended Pipeline']]
                baseline_score = gt_performances['baseline']
                
                if rec_score > baseline_score:
                    better += 1
                elif rec_score == baseline_score:
                    equal += 1
                else:
                    worse += 1
            
            total = better + equal + worse
            if total > 0:
                baseline_comparison.append({
                    'Recommender': recommender,
                    'Better than Baseline': better,
                    'Equal to Baseline': equal,
                    'Worse than Baseline': worse,
                    'Total': total,
                    'Better %': (better / total) * 100,
                    'Equal %': (equal / total) * 100,
                    'Worse %': (worse / total) * 100
                })
        
        df_baseline = pd.DataFrame(baseline_comparison).sort_values('Better %', ascending=False)
        df_baseline.to_csv(f'{output_dir}/04_baseline_comparison.csv', index=False)
        
        print("\n" + "="*80)
        print("BASELINE COMPARISON ANALYSIS")
        print("="*80)
        print("\nHow often does each recommender beat baseline?")
        print(df_baseline[['Recommender', 'Better %', 'Equal %', 'Worse %']].to_string(index=False))
        
        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df_baseline))
        width = 0.8
        
        p1 = ax.bar(x, df_baseline['Better %'], width, label='Better than Baseline', color='green', alpha=0.8)
        p2 = ax.bar(x, df_baseline['Equal %'], width, bottom=df_baseline['Better %'], label='Equal to Baseline', color='gray', alpha=0.8)
        p3 = ax.bar(x, df_baseline['Worse %'], width, bottom=df_baseline['Better %'] + df_baseline['Equal %'], label='Worse than Baseline', color='red', alpha=0.8)
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Baseline Comparison: Better/Equal/Worse Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_baseline['Recommender'], rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved: {output_dir}/04_baseline_comparison.csv")
        print(f"✅ Saved: {output_dir}/04_baseline_comparison.png")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        best_vs_baseline = df_baseline.iloc[0]
        worst_vs_baseline = df_baseline.iloc[-1]
        print(f"   • Best vs baseline: {best_vs_baseline['Recommender']} ({best_vs_baseline['Better %']:.1f}% better)")
        print(f"   • Worst vs baseline: {worst_vs_baseline['Recommender']} ({worst_vs_baseline['Worse %']:.1f}% worse)")
        
        # Find recommenders that never beat baseline
        never_better = df_baseline[df_baseline['Better than Baseline'] == 0]
        if len(never_better) > 0:
            print(f"\n   ⚠️  WARNING: These recommenders NEVER beat baseline:")
            for _, row in never_better.iterrows():
                print(f"      - {row['Recommender']}")
    
    def analyze_recommender_agreement(self, output_dir):
        """Analyze how much recommenders agree with each other."""
        
        # Create agreement matrix
        recommenders = sorted(self.detailed['Recommender'].unique())
        n = len(recommenders)
        agreement_matrix = np.zeros((n, n))
        
        for i, rec1 in enumerate(recommenders):
            for j, rec2 in enumerate(recommenders):
                if i == j:
                    agreement_matrix[i, j] = 100.0
                    continue
                
                # Get recommendations from both
                rec1_data = self.detailed[self.detailed['Recommender'] == rec1].set_index('Dataset')['Recommended Pipeline']
                rec2_data = self.detailed[self.detailed['Recommender'] == rec2].set_index('Dataset')['Recommended Pipeline']
                
                # Find common datasets
                common_datasets = set(rec1_data.index) & set(rec2_data.index)
                
                if len(common_datasets) == 0:
                    continue
                
                # Calculate agreement
                agreements = sum(rec1_data[ds] == rec2_data[ds] for ds in common_datasets)
                agreement_pct = (agreements / len(common_datasets)) * 100
                agreement_matrix[i, j] = agreement_pct
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(agreement_matrix,
                   annot=True,
                   fmt='.1f',
                   cmap='RdYlGn',
                   xticklabels=recommenders,
                   yticklabels=recommenders,
                   cbar_kws={'label': 'Agreement (%)'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title('Recommender Agreement Matrix\nGreen = High Agreement, Red = Low Agreement',
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_recommender_agreement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save matrix to CSV
        df_agreement = pd.DataFrame(agreement_matrix, index=recommenders, columns=recommenders)
        df_agreement.to_csv(f'{output_dir}/05_recommender_agreement.csv')
        
        print("\n" + "="*80)
        print("RECOMMENDER AGREEMENT ANALYSIS")
        print("="*80)
        
        # Find most similar pairs
        print("\n🤝 Most Similar Recommender Pairs:")
        similar_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                similar_pairs.append({
                    'Recommender 1': recommenders[i],
                    'Recommender 2': recommenders[j],
                    'Agreement %': agreement_matrix[i, j]
                })
        
        df_similar = pd.DataFrame(similar_pairs).sort_values('Agreement %', ascending=False)
        print(df_similar.head(5).to_string(index=False))
        
        # Find most different pairs
        print("\n🔄 Most Different Recommender Pairs:")
        print(df_similar.tail(5).to_string(index=False))
        
        print(f"\n✅ Saved: {output_dir}/05_recommender_agreement.png")
        print(f"✅ Saved: {output_dir}/05_recommender_agreement.csv")
    
    def analyze_performance_vs_metafeatures(self, output_dir):
        """Analyze if dataset metafeatures correlate with recommender performance."""
        
        print("\n" + "="*80)
        print("PERFORMANCE VS METAFEATURES ANALYSIS")
        print("="*80)
        
        # For each recommender, check correlation between dataset metafeatures and performance
        correlations = []
        
        for recommender in self.detailed['Recommender'].unique():
            rec_data = self.detailed[self.detailed['Recommender'] == recommender]
            
            # Get dataset IDs
            dataset_ids = rec_data['Dataset'].values
            ranks = rec_data['Pipeline Rank'].values
            
            # Get metafeatures for these datasets
            valid_datasets = [ds_id for ds_id in dataset_ids if ds_id in self.metafeatures.index]
            
            if len(valid_datasets) < 5:
                continue
            
            # Calculate correlation with key metafeatures
            mf_subset = self.metafeatures.loc[valid_datasets]
            
            # Get ranks for valid datasets
            valid_ranks = rec_data[rec_data['Dataset'].isin(valid_datasets)]['Pipeline Rank'].values
            
            # Find metafeatures with highest correlation to rank
            correlations_per_feature = {}
            for col in mf_subset.columns:
                if mf_subset[col].notna().sum() >= 5:  # Need at least 5 non-NaN values
                    corr = np.corrcoef(mf_subset[col].fillna(0), valid_ranks)[0, 1]
                    if not np.isnan(corr):
                        correlations_per_feature[col] = abs(corr)
            
            # Get top correlations
            if correlations_per_feature:
                sorted_corrs = sorted(correlations_per_feature.items(), key=lambda x: x[1], reverse=True)
                top_feature, top_corr = sorted_corrs[0] if sorted_corrs else (None, 0)
                
                correlations.append({
                    'Recommender': recommender,
                    'Top Feature': top_feature,
                    'Correlation': top_corr,
                    'Avg Rank Std': np.std(valid_ranks)
                })
        
        if correlations:
            df_corr = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
            df_corr.to_csv(f'{output_dir}/06_metafeature_correlations.csv', index=False)
            
            print("\nMetafeature Correlations with Performance:")
            print(df_corr.to_string(index=False))
            
            print(f"\n✅ Saved: {output_dir}/06_metafeature_correlations.csv")
        else:
            print("\n⚠️  Not enough data for metafeature correlation analysis")
    
    def analyze_error_patterns(self, output_dir):
        """Analyze patterns in errors and failures."""
        
        print("\n" + "="*80)
        print("ERROR PATTERN ANALYSIS")
        print("="*80)
        
        # Analyze which datasets/pipelines cause errors
        error_data = []
        
        for recommender in self.detailed['Recommender'].unique():
            rec_data = self.detailed[self.detailed['Recommender'] == recommender]
            
            # Find worst predictions (highest rank)
            worst_predictions = rec_data.nlargest(5, 'Pipeline Rank')
            
            for _, row in worst_predictions.iterrows():
                error_data.append({
                    'Recommender': recommender,
                    'Dataset': row['Dataset'],
                    'Recommended': row['Recommended Pipeline'],
                    'Rank': row['Pipeline Rank'],
                    'Regret': row['Regret'],
                    'Ground Truth': row['Ground Truth Best']
                })
        
        df_errors = pd.DataFrame(error_data).sort_values('Rank', ascending=False)
        df_errors.to_csv(f'{output_dir}/07_worst_predictions.csv', index=False)
        
        print("\n📉 Worst Predictions (Highest Ranks):")
        print(df_errors.head(10).to_string(index=False))
        
        print(f"\n✅ Saved: {output_dir}/07_worst_predictions.csv")
    
    def generate_head_to_head_comparison(self, output_dir):
        """Generate head-to-head win/loss comparison between recommenders."""
        
        recommenders = sorted(self.detailed['Recommender'].unique())
        n = len(recommenders)
        
        # Win matrix: win_matrix[i, j] = how many times recommender i beats recommender j
        win_matrix = np.zeros((n, n))
        
        for dataset_id in self.detailed['Dataset'].unique():
            dataset_data = self.detailed[self.detailed['Dataset'] == dataset_id]
            
            for i, rec1 in enumerate(recommenders):
                rec1_data = dataset_data[dataset_data['Recommender'] == rec1]
                if len(rec1_data) == 0:
                    continue
                rec1_rank = rec1_data.iloc[0]['Pipeline Rank']
                
                for j, rec2 in enumerate(recommenders):
                    if i == j:
                        continue
                    
                    rec2_data = dataset_data[dataset_data['Recommender'] == rec2]
                    if len(rec2_data) == 0:
                        continue
                    rec2_rank = rec2_data.iloc[0]['Pipeline Rank']
                    
                    if rec1_rank < rec2_rank:  # Lower rank = better
                        win_matrix[i, j] += 1
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(win_matrix,
                   annot=True,
                   fmt='.0f',
                   cmap='RdYlGn',
                   xticklabels=recommenders,
                   yticklabels=recommenders,
                   cbar_kws={'label': 'Wins'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title('Head-to-Head Win Matrix\nRow beats Column (# of times)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Opponent', fontsize=12)
        ax.set_ylabel('Recommender', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_head_to_head.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save matrix
        df_wins = pd.DataFrame(win_matrix, index=recommenders, columns=recommenders)
        df_wins.to_csv(f'{output_dir}/08_head_to_head_wins.csv')
        
        # Calculate win rates
        total_matches = win_matrix.sum(axis=1)
        win_rates = []
        for i, rec in enumerate(recommenders):
            total = total_matches[i]
            win_rates.append({
                'Recommender': rec,
                'Total Wins': total,
                'Win Rate %': (total / (n-1) / len(self.detailed['Dataset'].unique())) * 100 if n > 1 else 0
            })
        
        df_win_rates = pd.DataFrame(win_rates).sort_values('Win Rate %', ascending=False)
        
        print("\n" + "="*80)
        print("HEAD-TO-HEAD COMPARISON")
        print("="*80)
        print("\nWin Rates (How often does each recommender beat others?):")
        print(df_win_rates.to_string(index=False))
        
        print(f"\n✅ Saved: {output_dir}/08_head_to_head.png")
        print(f"✅ Saved: {output_dir}/08_head_to_head_wins.csv")
    
    def run_statistical_tests(self, output_dir):
        """Run statistical significance tests between recommenders."""
        
        from scipy import stats
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        recommenders = self.detailed['Recommender'].unique()
        results = []
        
        # Pairwise t-tests on ranks
        for i, rec1 in enumerate(recommenders):
            rec1_ranks = self.detailed[self.detailed['Recommender'] == rec1]['Pipeline Rank'].values
            
            for rec2 in recommenders[i+1:]:
                rec2_ranks = self.detailed[self.detailed['Recommender'] == rec2]['Pipeline Rank'].values
                
                # Paired t-test (same datasets for both)
                if len(rec1_ranks) == len(rec2_ranks):
                    t_stat, p_value = stats.ttest_rel(rec1_ranks, rec2_ranks)
                    
                    # Determine significance
                    if p_value < 0.001:
                        significance = '***'
                    elif p_value < 0.01:
                        significance = '**'
                    elif p_value < 0.05:
                        significance = '*'
                    else:
                        significance = 'ns'
                    
                    # Determine winner
                    mean_diff = np.mean(rec1_ranks) - np.mean(rec2_ranks)
                    if mean_diff < 0:
                        winner = rec1
                    else:
                        winner = rec2
                    
                    results.append({
                        'Recommender 1': rec1,
                        'Recommender 2': rec2,
                        'Mean Rank 1': np.mean(rec1_ranks),
                        'Mean Rank 2': np.mean(rec2_ranks),
                        'Difference': abs(mean_diff),
                        'T-statistic': t_stat,
                        'P-value': p_value,
                        'Significance': significance,
                        'Better Recommender': winner
                    })
        
        df_stats = pd.DataFrame(results).sort_values('P-value')
        df_stats.to_csv(f'{output_dir}/09_statistical_tests.csv', index=False)
        
        print("\nPairwise T-Tests (Lower rank = better):")
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("\nTop 10 most significant differences:")
        print(df_stats.head(10)[['Recommender 1', 'Recommender 2', 'Difference', 'P-value', 'Significance', 'Better Recommender']].to_string(index=False))
        
        print(f"\n✅ Saved: {output_dir}/09_statistical_tests.csv")
    
    def generate_concrete_evidence_summary(self, output_dir):
        """Generate a concrete evidence summary with key numbers."""
        
        summary_file = f'{output_dir}/10_CONCRETE_EVIDENCE_SUMMARY.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("CONCRETE EVIDENCE SUMMARY: WHY DO RECOMMENDERS PERFORM DIFFERENTLY?\n")
            f.write("="*100 + "\n\n")
            
            # 1. Overall Performance Rankings
            f.write("1. OVERALL PERFORMANCE RANKINGS\n")
            f.write("-"*100 + "\n")
            
            summary_sorted = self.summary.sort_values('Avg Rank')
            for i, row in summary_sorted.iterrows():
                if row['Training Success'] == '✅':
                    f.write(f"#{i+1}. {row['Recommender']}\n")
                    f.write(f"    • Average Rank: {row['Avg Rank']:.2f}\n")
                    f.write(f"    • Accuracy: {row['Accuracy (%)']}%\n")
                    f.write(f"    • Avg Degradation: {row['Avg Degradation (%)']}%\n")
                    f.write(f"    • Better than Baseline: {row['Better than Baseline (%)']}%\n")
                    f.write(f"    • Training Time: {row['Training Time (s)']}s\n\n")
            
            # 2. Key Findings
            f.write("\n2. KEY FINDINGS\n")
            f.write("-"*100 + "\n\n")
            
            # Diversity finding
            diversity_df = pd.read_csv(f'{output_dir}/01_pipeline_diversity.csv')
            most_diverse = diversity_df.iloc[0]
            least_diverse = diversity_df.iloc[-1]
            
            f.write("A. Pipeline Diversity:\n")
            f.write(f"   • Most diverse: {most_diverse['Recommender']} ({most_diverse['Unique Pipelines']} unique pipelines, Entropy: {most_diverse['Entropy']:.2f})\n")
            f.write(f"   • Least diverse: {least_diverse['Recommender']} ({least_diverse['Unique Pipelines']} unique pipeline(s), Entropy: {least_diverse['Entropy']:.2f})\n")
            f.write(f"   → Low diversity = recommender is not learning dataset-specific patterns!\n\n")
            
            # Dataset difficulty
            difficulty_df = pd.read_csv(f'{output_dir}/03_dataset_difficulty.csv')
            hardest = difficulty_df.iloc[0]
            easiest = difficulty_df.iloc[-1]
            
            f.write("B. Dataset Difficulty:\n")
            f.write(f"   • Hardest: Dataset {hardest['Dataset']} (Avg Rank: {hardest['Avg Rank']:.2f}, only {hardest['Perfect Count']} perfect predictions)\n")
            f.write(f"   • Easiest: Dataset {easiest['Dataset']} (Avg Rank: {easiest['Avg Rank']:.2f}, {easiest['Perfect Count']} perfect predictions)\n")
            f.write(f"   → Some datasets are universally hard/easy for all recommenders!\n\n")
            
            # Baseline comparison
            baseline_df = pd.read_csv(f'{output_dir}/04_baseline_comparison.csv')
            best_vs_baseline = baseline_df.iloc[0]
            worst_vs_baseline = baseline_df.iloc[-1]
            
            f.write("C. Baseline Comparison:\n")
            f.write(f"   • Best vs baseline: {best_vs_baseline['Recommender']} ({best_vs_baseline['Better %']:.1f}% better, {best_vs_baseline['Worse %']:.1f}% worse)\n")
            f.write(f"   • Worst vs baseline: {worst_vs_baseline['Recommender']} ({worst_vs_baseline['Better %']:.1f}% better, {worst_vs_baseline['Worse %']:.1f}% worse)\n")
            f.write(f"   → Beating baseline is HARD! Even best recommender only wins {best_vs_baseline['Better %']:.1f}% of the time.\n\n")
            
            # 3. Root Cause Analysis
            f.write("\n3. ROOT CAUSE ANALYSIS: WHY DO SOME RECOMMENDERS FAIL?\n")
            f.write("-"*100 + "\n\n")
            
            # Analyze each recommender's failure mode
            for recommender in diversity_df['Recommender'].unique():
                rec_diversity = diversity_df[diversity_df['Recommender'] == recommender].iloc[0]
                rec_baseline = baseline_df[baseline_df['Recommender'] == recommender]
                
                if len(rec_baseline) == 0:
                    continue
                
                rec_baseline = rec_baseline.iloc[0]
                
                f.write(f"{recommender}:\n")
                
                # Check for "always same pipeline" issue
                if rec_diversity['Unique Pipelines'] == 1:
                    f.write(f"   ❌ FAILURE MODE: Always predicts '{rec_diversity['Most Common Pipeline']}'\n")
                    f.write(f"      → Model is NOT learning dataset-specific patterns\n")
                    f.write(f"      → Likely cause: Undertrained, collapsed to most common class\n")
                
                elif rec_diversity['Unique Pipelines'] <= 3:
                    f.write(f"   ⚠️  WARNING: Low diversity ({rec_diversity['Unique Pipelines']} unique pipelines)\n")
                    f.write(f"      → Model is learning very little variation\n")
                    f.write(f"      → Likely cause: Insufficient model capacity or poor training\n")
                
                elif rec_baseline['Worse %'] > 50:
                    f.write(f"   ⚠️  WARNING: Worse than baseline {rec_baseline['Worse %']:.1f}% of the time\n")
                    f.write(f"      → Model is making suboptimal recommendations\n")
                    f.write(f"      → Likely cause: Overfitting or wrong inductive bias\n")
                
                else:
                    f.write(f"   ✅ WORKING: Diverse predictions ({rec_diversity['Unique Pipelines']} unique pipelines)\n")
                    f.write(f"      → Better than baseline {rec_baseline['Better %']:.1f}% of the time\n")
                
                f.write("\n")
        
        print("\n" + "="*100)
        print("CONCRETE EVIDENCE SUMMARY")
        print("="*100)
        print(f"\n✅ Comprehensive evidence summary saved to: {summary_file}")
        print("\nPlease review this file for detailed explanations of each recommender's performance!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug and analyze recommender performance')
    parser.add_argument('--comparison', type=str, default='recommender_comparison_report.csv',
                       help='Path to comparison CSV file')
    parser.add_argument('--detailed', type=str, default='recommender_comparison_report_detailed_analysis.csv',
                       help='Path to detailed analysis CSV file')
    parser.add_argument('--test-gt', type=str, default='test_ground_truth_performance.csv',
                       help='Path to test ground truth CSV file')
    parser.add_argument('--perf-matrix', type=str, default='preprocessed_performance.csv',
                       help='Path to performance matrix CSV file')
    parser.add_argument('--metafeatures', type=str, default='dataset_feats.csv',
                       help='Path to metafeatures CSV file')
    parser.add_argument('--output-dir', type=str, default='debug_output',
                       help='Output directory for debug files')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.comparison, args.detailed, args.test_gt, args.perf_matrix, args.metafeatures]:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found: {file_path}")
            return 1
    
    # Create debugger
    debugger = RecommenderDebugger(
        comparison_csv=args.comparison,
        detailed_csv=args.detailed,
        test_ground_truth_csv=args.test_gt,
        performance_matrix_csv=args.perf_matrix,
        metafeatures_csv=args.metafeatures
    )
    
    # Generate all analyses
    debugger.generate_all_analyses(output_dir=args.output_dir)
    
    print("\n" + "="*100)
    print("✅ DEBUGGING COMPLETE!")
    print("="*100)
    print(f"\nAll analyses and visualizations saved to: {args.output_dir}/")
    print("\n📖 Start with: 10_CONCRETE_EVIDENCE_SUMMARY.txt")
    print("   This file contains the most important findings with concrete numbers!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
