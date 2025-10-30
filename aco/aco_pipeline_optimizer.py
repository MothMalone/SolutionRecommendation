import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")


class ACOPipelineOptimizer:
    """
    Ant Colony Optimization for finding optimal preprocessing pipelines.
    
    Each ant constructs a complete pipeline by selecting one operator from each
    preprocessing category (imputation, scaling, encoding, etc.).
    """
    
    def __init__(
        self,
        options: Dict[str, List[str]],
        n_ants: int = 20,
        n_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.9,
        initial_pheromone: float = 0.1,
        min_pheromone: float = 0.01,
        max_pheromone: float = 10.0,
        elite_weight: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize ACO optimizer.
        
        Args:
            options: Dict mapping preprocessing step to list of available operators
            n_ants: Number of ants per iteration
            n_iterations: Number of iterations to run
            alpha: Pheromone importance (default 1.0)
            beta: Heuristic importance (default 2.0)
            rho: Pheromone evaporation rate (0-1, default 0.1)
            q0: Exploitation vs exploration parameter (0-1, default 0.9)
            initial_pheromone: Initial pheromone level
            min_pheromone: Minimum pheromone level (prevents stagnation)
            max_pheromone: Maximum pheromone level (prevents premature convergence)
            elite_weight: Weight for best solution pheromone update
            verbose: Print progress
        """
        self.options = options
        self.steps = list(options.keys())
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.elite_weight = elite_weight
        self.verbose = verbose
        
        # Initialize pheromone matrix
        # pheromone[step][operator] = pheromone level
        self.pheromone = {}
        for step, operators in options.items():
            self.pheromone[step] = {op: initial_pheromone for op in operators}
        
        # Heuristic information (will be updated based on performance history)
        self.heuristic = {}
        for step, operators in options.items():
            # Start with uniform heuristic
            self.heuristic[step] = {op: 1.0 for op in operators}
        
        # History tracking
        self.best_pipeline = None
        self.best_score = -float('inf')
        self.iteration_best_scores = []
        self.all_solutions = []
        
        # Performance history for each operator
        self.operator_performance = defaultdict(lambda: {'scores': [], 'count': 0})
        
    def _calculate_probabilities(self, step: str, available_ops: List[str]) -> Dict[str, float]:
        """
        Calculate selection probabilities for operators in a given step.
        
        Uses the ACO probability formula:
        P(operator) = (pheromone^alpha * heuristic^beta) / sum(all)
        """
        probabilities = {}
        total = 0.0
        
        for op in available_ops:
            pheromone = self.pheromone[step][op]
            heuristic = self.heuristic[step][op]
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities[op] = prob
            total += prob
        
        # Normalize
        if total > 0:
            for op in probabilities:
                probabilities[op] /= total
        else:
            # Uniform if all zero
            uniform = 1.0 / len(available_ops)
            probabilities = {op: uniform for op in available_ops}
        
        return probabilities
    
    def _select_operator(self, step: str) -> str:
        """
        Select an operator for a given step using ACO selection strategy.
        
        Uses pseudo-random proportional rule:
        - With probability q0: select best operator (exploitation)
        - With probability 1-q0: select probabilistically (exploration)
        """
        available_ops = self.options[step]
        
        if np.random.random() < self.q0:
            # Exploitation: select best operator
            probabilities = self._calculate_probabilities(step, available_ops)
            best_op = max(probabilities, key=probabilities.get)
            return best_op
        else:
            # Exploration: probabilistic selection
            probabilities = self._calculate_probabilities(step, available_ops)
            ops = list(probabilities.keys())
            probs = list(probabilities.values())
            
            # Handle edge case
            if sum(probs) == 0:
                return np.random.choice(ops)
            
            return np.random.choice(ops, p=probs)
    
    def _construct_pipeline(self) -> Dict[str, str]:
        """
        Construct a complete pipeline by selecting one operator from each step.
        """
        pipeline = {}
        for step in self.steps:
            pipeline[step] = self._select_operator(step)
        return pipeline
    
    def _update_heuristic(self):
        """
        Update heuristic information based on historical performance.
        
        Operators that have performed well in the past get higher heuristic values.
        """
        for step, operators in self.options.items():
            for op in operators:
                key = f"{step}:{op}"
                if key in self.operator_performance and self.operator_performance[key]['count'] > 0:
                    # Average performance of this operator
                    avg_score = np.mean(self.operator_performance[key]['scores'])
                    # Normalize to [0.5, 2.0] range to avoid extreme values
                    self.heuristic[step][op] = 0.5 + 1.5 * (avg_score + 1) / 2  # assuming scores in [-1, 1]
                else:
                    self.heuristic[step][op] = 1.0
    
    def _evaporate_pheromone(self):
        """
        Apply pheromone evaporation to all edges.
        """
        for step in self.steps:
            for op in self.options[step]:
                # Evaporate
                self.pheromone[step][op] *= (1 - self.rho)
                # Enforce bounds
                self.pheromone[step][op] = max(self.min_pheromone, 
                                              min(self.max_pheromone, self.pheromone[step][op]))
    
    def _deposit_pheromone(self, pipeline: Dict[str, str], score: float, weight: float = 1.0):
        """
        Deposit pheromone on the path taken by an ant.
        
        Args:
            pipeline: The pipeline configuration
            score: Performance score achieved
            weight: Multiplier for pheromone deposit (for elite solutions)
        """
        # Amount of pheromone to deposit (proportional to quality)
        # Normalize score to positive value
        deposit = weight * max(0, score)
        
        for step, operator in pipeline.items():
            self.pheromone[step][operator] += deposit
            # Enforce max bound
            self.pheromone[step][operator] = min(self.max_pheromone, self.pheromone[step][operator])
    
    def _record_operator_performance(self, pipeline: Dict[str, str], score: float):
        """
        Record performance for each operator used in the pipeline.
        """
        for step, operator in pipeline.items():
            key = f"{step}:{operator}"
            self.operator_performance[key]['scores'].append(score)
            self.operator_performance[key]['count'] += 1
    
    def optimize(self, dataset: Dict, evaluate_func, evaluate_kwargs: Dict = None) -> Dict:
        """
        Run ACO optimization to find the best pipeline for a dataset.
        
        Args:
            dataset: Dataset dictionary with 'X', 'y', 'name', 'id'
            evaluate_func: Function to evaluate pipeline (returns score)
            evaluate_kwargs: Additional kwargs for evaluate_func
            
        Returns:
            Dictionary with best pipeline and optimization results
        """
        if evaluate_kwargs is None:
            evaluate_kwargs = {}
        
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ACO Pipeline Optimization for {dataset.get('name', 'Dataset')}")
            print(f"{'='*80}")
            print(f"Configuration:")
            print(f"  Ants: {self.n_ants}, Iterations: {self.n_iterations}")
            print(f"  Alpha: {self.alpha}, Beta: {self.beta}, Rho: {self.rho}, Q0: {self.q0}")
            print(f"{'='*80}\n")
        
        # Use tqdm if available
        if TQDM_AVAILABLE and self.verbose:
            iteration_range = tqdm(range(self.n_iterations), desc="ACO Optimization", 
                                  unit="iter", position=0, leave=True)
        else:
            iteration_range = range(self.n_iterations)
        
        for iteration in iteration_range:
            iter_start = time.time()
            
            # Store solutions and scores for this iteration
            iteration_solutions = []
            iteration_scores = []
            
            # Use tqdm for ants if available - simpler without nested bar
            ant_range = range(self.n_ants)
            
            # Each ant constructs and evaluates a pipeline
            for ant_id in ant_range:
                # Construct pipeline
                pipeline = self._construct_pipeline()
                
                # Evaluate pipeline
                try:
                    score = evaluate_func(dataset, pipeline, **evaluate_kwargs)
                    
                    # Handle NaN scores
                    if np.isnan(score):
                        score = 0.0
                    
                except Exception as e:
                    if self.verbose:
                        print(f"  Ant {ant_id+1} evaluation failed: {e}")
                    score = 0.0
                
                iteration_solutions.append(pipeline)
                iteration_scores.append(score)
                
                # Record operator performance
                self._record_operator_performance(pipeline, score)
                
                # Track all solutions
                self.all_solutions.append({
                    'iteration': iteration,
                    'ant': ant_id,
                    'pipeline': pipeline.copy(),
                    'score': score
                })
            
            # Find best solution in this iteration
            best_idx = np.argmax(iteration_scores)
            iteration_best_score = iteration_scores[best_idx]
            iteration_best_pipeline = iteration_solutions[best_idx]
            
            # Update global best
            if iteration_best_score > self.best_score:
                self.best_score = iteration_best_score
                self.best_pipeline = iteration_best_pipeline.copy()
            
            self.iteration_best_scores.append(iteration_best_score)
            
            # Pheromone evaporation
            self._evaporate_pheromone()
            
            # Pheromone deposit from all ants (weighted by performance)
            for pipeline, score in zip(iteration_solutions, iteration_scores):
                if score > 0:  # Only deposit for successful solutions
                    self._deposit_pheromone(pipeline, score)
            
            # Extra pheromone for iteration best (elitist strategy)
            if iteration_best_score > 0:
                self._deposit_pheromone(iteration_best_pipeline, iteration_best_score, 
                                       weight=self.elite_weight)
            
            # Extra pheromone for global best
            if self.best_score > 0:
                self._deposit_pheromone(self.best_pipeline, self.best_score, 
                                       weight=self.elite_weight)
            
            # Update heuristic information
            if iteration % 5 == 0:  # Update every 5 iterations
                self._update_heuristic()
            
            # Progress reporting
            iter_time = time.time() - iter_start
            avg_score = np.mean([s for s in iteration_scores if s > 0] or [0])
            
            if self.verbose and not TQDM_AVAILABLE:
                print(f"Iter {iteration+1:3d}/{self.n_iterations}: "
                      f"Best={iteration_best_score:.4f}, "
                      f"Avg={avg_score:.4f}, "
                      f"Global Best={self.best_score:.4f}, "
                      f"Time={iter_time:.2f}s")
            elif TQDM_AVAILABLE and self.verbose:
                # Update tqdm postfix
                if isinstance(iteration_range, tqdm):
                    iteration_range.set_postfix({
                        'Best': f'{iteration_best_score:.4f}',
                        'Avg': f'{avg_score:.4f}',
                        'Global': f'{self.best_score:.4f}'
                    })
                
                # Show best pipeline every 10 iterations
                if (iteration + 1) % 10 == 0:
                    print(f"  Current best pipeline: {self._pipeline_to_string(self.best_pipeline)}")
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Optimization Complete!")
            print(f"{'='*80}")
            print(f"Best Score: {self.best_score:.4f}")
            print(f"Best Pipeline: {self._pipeline_to_string(self.best_pipeline)}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"{'='*80}\n")
        
        return {
            'best_pipeline': self.best_pipeline,
            'best_score': self.best_score,
            'iteration_best_scores': self.iteration_best_scores,
            'all_solutions': self.all_solutions,
            'total_time': total_time,
            'final_pheromone': self.pheromone.copy(),
            'final_heuristic': self.heuristic.copy()
        }
    
    def _pipeline_to_string(self, pipeline: Dict[str, str]) -> str:
        """Convert pipeline dict to readable string."""
        parts = [f"{step}={op}" for step, op in pipeline.items()]
        return ", ".join(parts)
    
    def get_pheromone_summary(self) -> pd.DataFrame:
        """
        Get a summary of pheromone levels for all operators.
        
        Returns:
            DataFrame with pheromone levels
        """
        data = []
        for step, operators in self.pheromone.items():
            for op, pheromone in operators.items():
                data.append({
                    'step': step,
                    'operator': op,
                    'pheromone': pheromone,
                    'heuristic': self.heuristic[step][op]
                })
        return pd.DataFrame(data)
    
    def get_operator_statistics(self) -> pd.DataFrame:
        """
        Get performance statistics for each operator.
        
        Returns:
            DataFrame with operator usage and performance stats
        """
        data = []
        for key, stats in self.operator_performance.items():
            step, operator = key.split(':')
            if stats['count'] > 0:
                data.append({
                    'step': step,
                    'operator': operator,
                    'usage_count': stats['count'],
                    'avg_score': np.mean(stats['scores']),
                    'max_score': np.max(stats['scores']),
                    'min_score': np.min(stats['scores']),
                    'std_score': np.std(stats['scores'])
                })
        return pd.DataFrame(data)


def pipeline_dict_to_config(pipeline: Dict[str, str], config_name: str = "aco_optimized") -> Dict:
    """
    Convert ACO pipeline dictionary to the format expected by apply_preprocessing.
    
    Args:
        pipeline: Dictionary mapping step names to operator names
        config_name: Name for the configuration
        
    Returns:
        Configuration dictionary compatible with apply_preprocessing
    """
    config = {'name': config_name}
    config.update(pipeline)
    return config


if __name__ == "__main__":
    # Example usage
    print("ACO Pipeline Optimizer Module")
    print("Import this module and use ACOPipelineOptimizer class")
