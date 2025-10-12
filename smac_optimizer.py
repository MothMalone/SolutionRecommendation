"""
SMAC-based hyperparameter optimization for pipeline selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import os
import time
import json
import logging

# Try to import SMAC
try:
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.scenario.scenario import Scenario
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
    )
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
    print("Warning: SMAC optimization not available. Install SMAC3 for this feature.")

class SMACOptimizer:
    """
    Uses SMAC (Sequential Model-based Algorithm Configuration) to optimize 
    pipeline selection and hyperparameters.
    """
    
    def __init__(self, pipeline_configs, time_limit=300, n_trials=50, random_seed=42):
        """
        Initialize the SMAC optimizer.
        
        Args:
            pipeline_configs: List of pipeline configurations
            time_limit: Time limit in seconds (default: 300s = 5min)
            n_trials: Maximum number of trials (default: 50)
            random_seed: Random seed for reproducibility
        """
        self.pipeline_configs = pipeline_configs
        self.time_limit = time_limit
        self.n_trials = n_trials
        self.random_seed = random_seed
        
        if not SMAC_AVAILABLE:
            raise ImportError("SMAC is not available. Please install SMAC3.")
    
    def create_configspace(self, include_hyperparams=True):
        """
        Create the configuration space for SMAC.
        
        Args:
            include_hyperparams: Whether to include hyperparameters beyond pipeline selection
            
        Returns:
            ConfigurationSpace object
        """
        cs = ConfigurationSpace()
        
        # Add pipeline selection as a hyperparameter
        pipeline_names = [config['name'] for config in self.pipeline_configs]
        pipeline_param = CategoricalHyperparameter("pipeline", pipeline_names)
        cs.add_hyperparameter(pipeline_param)
        
        if include_hyperparams:
            # Add hyperparameters for imputation strategies
            imputation_param = CategoricalHyperparameter(
                "imputation", ["mean", "median", "most_frequent", "none"]
            )
            cs.add_hyperparameter(imputation_param)
            
            # Add hyperparameters for scaling strategies
            scaling_param = CategoricalHyperparameter(
                "scaling", ["standard", "minmax", "robust", "none"]
            )
            cs.add_hyperparameter(scaling_param)
            
            # Add hyperparameters for feature selection
            feature_selection_param = CategoricalHyperparameter(
                "feature_selection", ["none", "k_best", "variance_threshold", "mutual_info"]
            )
            cs.add_hyperparameter(feature_selection_param)
            
            # Add k parameter for k_best feature selection
            k_best_param = UniformIntegerHyperparameter(
                "k_best", lower=1, upper=100, default_value=10
            )
            cs.add_hyperparameter(k_best_param)
            
            # Add variance threshold parameter
            variance_threshold_param = UniformFloatHyperparameter(
                "variance_threshold", lower=0.0, upper=0.2, default_value=0.0
            )
            cs.add_hyperparameter(variance_threshold_param)
            
            # Add dimensionality reduction hyperparameters
            dim_reduction_param = CategoricalHyperparameter(
                "dimensionality_reduction", ["none", "pca"]
            )
            cs.add_hyperparameter(dim_reduction_param)
            
            # Add PCA components parameter
            pca_components_param = UniformFloatHyperparameter(
                "pca_components", lower=0.5, upper=0.99, default_value=0.95
            )
            cs.add_hyperparameter(pca_components_param)
        
        return cs
    
    def objective_function(self, config, dataset):
        """
        Objective function for SMAC to optimize.
        
        Args:
            config: Configuration to evaluate
            dataset: Dataset to evaluate on
            
        Returns:
            Negative performance score (to minimize)
        """
        # Get pipeline configuration
        pipeline_name = config["pipeline"]
        pipeline_config = next(pc for pc in self.pipeline_configs if pc['name'] == pipeline_name)
        
        # Override pipeline configuration with SMAC parameters if needed
        if "imputation" in config:
            pipeline_config["imputation"] = config["imputation"]
        
        if "scaling" in config:
            pipeline_config["scaling"] = config["scaling"]
            
        if "feature_selection" in config:
            pipeline_config["feature_selection"] = config["feature_selection"]
            
        if "dimensionality_reduction" in config:
            pipeline_config["dimensionality_reduction"] = config["dimensionality_reduction"]
        
        # Apply additional parameters
        custom_params = {}
        
        if "k_best" in config and pipeline_config["feature_selection"] == "k_best":
            custom_params["k_best"] = int(config["k_best"])
            
        if "variance_threshold" in config and pipeline_config["feature_selection"] == "variance_threshold":
            custom_params["variance_threshold"] = config["variance_threshold"]
            
        if "pca_components" in config and pipeline_config["dimensionality_reduction"] == "pca":
            custom_params["pca_components"] = config["pca_components"]
        
        # Evaluate pipeline
        try:
            # Import from solurec.py
            from solurec import create_pipeline, evaluate_pipeline_with_cv
            
            # Create pipeline
            pipeline = create_pipeline(pipeline_config, custom_params)
            
            # Evaluate pipeline
            score = evaluate_pipeline_with_cv(dataset, pipeline)
            
            # Return negative score (SMAC minimizes the objective)
            return -score
            
        except Exception as e:
            print(f"Error in pipeline evaluation: {e}")
            return 0.0  # Return worst score on error
    
    def optimize(self, dataset):
        """
        Optimize pipeline selection for a dataset.
        
        Args:
            dataset: Dataset to optimize for
            
        Returns:
            Tuple of (best_config, best_score, all_results)
        """
        try:
            # Create configuration space
            cs = self.create_configspace()
            
            # Create SMAC scenario
            scenario = Scenario({
                "run_obj": "quality",  # Optimize for quality/performance
                "runcount-limit": self.n_trials,  # Maximum number of trials
                "wallclock-limit": self.time_limit,  # Maximum time in seconds
                "cs": cs,  # Configuration space
                "deterministic": True,
                "output_dir": None,
                "abort_on_first_run_crash": False
            })
            
            # Create wrapper for objective function
            def wrapped_objective(config):
                return self.objective_function(config, dataset)
            
            # Initialize SMAC
            smac = SMAC4HPO(
                scenario=scenario,
                rng=np.random.RandomState(self.random_seed),
                tae_runner=wrapped_objective
            )
            
            # Run optimization
            print(f"Starting SMAC optimization for dataset {dataset.get('name', 'unknown')}")
            print(f"Time limit: {self.time_limit} seconds, Maximum trials: {self.n_trials}")
            
            start_time = time.time()
            incumbent = smac.optimize()
            end_time = time.time()
            
            # Get optimal configuration
            opt_config = incumbent.get_dictionary()
            
            # Evaluate best configuration (convert from negative)
            best_score = -self.objective_function(opt_config, dataset)
            
            # Get all evaluated configurations
            runhistory = smac.get_runhistory()
            all_configs = []
            
            for run_key in runhistory.data:
                config_id = run_key.config_id
                config = runhistory.ids_config[config_id]
                cost = runhistory.get_cost(config)
                
                all_configs.append({
                    'config': config.get_dictionary(),
                    'score': -cost  # Convert back to positive score
                })
            
            # Sort by score
            all_configs.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"SMAC optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Best configuration: {opt_config}")
            print(f"Best score: {best_score:.4f}")
            
            return opt_config, best_score, all_configs
            
        except Exception as e:
            print(f"Error in SMAC optimization: {e}")
            return None, 0.0, []
    
    def recommend_pipeline(self, dataset):
        """
        Recommend a pipeline for a dataset using SMAC optimization.
        
        Args:
            dataset: Dataset to recommend for
            
        Returns:
            dict with pipeline recommendation
        """
        if not SMAC_AVAILABLE:
            print("SMAC not available. Cannot perform optimization.")
            return None
        
        try:
            # Run optimization
            best_config, best_score, all_results = self.optimize(dataset)
            
            if best_config is None:
                return None
            
            # Get pipeline configuration
            pipeline_name = best_config["pipeline"]
            pipeline_config = next(pc for pc in self.pipeline_configs if pc['name'] == pipeline_name)
            
            # Create combined configuration with optimization results
            optimized_config = pipeline_config.copy()
            
            # Override with optimized parameters
            for param, value in best_config.items():
                if param != "pipeline":
                    optimized_config[param] = value
            
            # Create recommendation
            recommendation = {
                'pipeline_config': optimized_config,
                'pipeline_name': pipeline_name,
                'expected_performance': best_score,
                'confidence': 'high',
                'method': 'SMAC',
                'optimization_results': {
                    'best_config': best_config,
                    'best_score': best_score,
                    'total_trials': len(all_results),
                    'top_configurations': all_results[:5]
                }
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error in SMAC pipeline recommendation: {e}")
            return None


def recommend_with_smac(dataset, pipeline_configs, time_limit=300, n_trials=50):
    """
    Recommend a pipeline for a dataset using SMAC optimization.
    
    Args:
        dataset: Dataset to recommend for
        pipeline_configs: List of pipeline configurations
        time_limit: Time limit in seconds
        n_trials: Maximum number of trials
        
    Returns:
        dict with pipeline recommendation
    """
    if not SMAC_AVAILABLE:
        print("SMAC not available. Cannot perform optimization.")
        return None
    
    try:
        # Create optimizer
        optimizer = SMACOptimizer(
            pipeline_configs=pipeline_configs,
            time_limit=time_limit,
            n_trials=n_trials
        )
        
        # Get recommendation
        recommendation = optimizer.recommend_pipeline(dataset)
        
        return recommendation
        
    except Exception as e:
        print(f"Error in SMAC recommendation: {e}")
        return None