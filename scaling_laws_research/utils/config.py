"""Configuration management for Random Forest scaling experiments."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest parameters."""
    n_estimators: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200, 500, 1000])
    max_depth: List[Optional[int]] = field(default_factory=lambda: [3, 5, 10, 15, 20, None])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 4])
    random_state: int = 42
    n_jobs: int = -1
    n_random_seeds: int = 3
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    base_samples: int = 1000
    base_features: int = 20
    sample_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    feature_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    sample_sizes: List[int] = field(default_factory=lambda: [500, 1000, 2000, 5000, 10000, 20000, 38000])
    n_classes: int = 2
    task_type: str = "classification"  # "classification" or "regression"
    test_size: float = 0.2
    random_state: int = 42
    dataset_name: str = "adult"  # "adult" or "synthetic"


@dataclass
class ExperimentConfig:
    """Main configuration for scaling experiments."""
    name: str = "rf_scaling_experiment"
    description: str = "Random Forest scaling law analysis"
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "results"
    save_models: bool = False
    save_predictions: bool = False
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'random_forest': {
                'n_estimators': self.random_forest.n_estimators,
                'max_depth': self.random_forest.max_depth,
                'min_samples_split': self.random_forest.min_samples_split,
                'min_samples_leaf': self.random_forest.min_samples_leaf,
                'random_state': self.random_forest.random_state,
                'n_jobs': self.random_forest.n_jobs,
                'n_random_seeds': self.random_forest.n_random_seeds,
                'random_seeds': self.random_forest.random_seeds
            },
            'data': {
                'base_samples': self.data.base_samples,
                'base_features': self.data.base_features,
                'sample_scales': self.data.sample_scales,
                'feature_scales': self.data.feature_scales,
                'sample_sizes': self.data.sample_sizes,
                'n_classes': self.data.n_classes,
                'task_type': self.data.task_type,
                'test_size': self.data.test_size,
                'random_state': self.data.random_state,
                'dataset_name': self.data.dataset_name
            },
            'output_dir': self.output_dir,
            'save_models': self.save_models,
            'save_predictions': self.save_predictions,
            'verbose': self.verbose
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create nested configs
        rf_config = RandomForestConfig(**data['random_forest'])
        data_config = DataConfig(**data['data'])
        
        # Remove nested configs from main data
        config_data = {k: v for k, v in data.items() 
                      if k not in ['random_forest', 'data']}
        
        return cls(
            random_forest=rf_config,
            data=data_config,
            **config_data
        )
    
    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Generate parameter grid for hyperparameter combinations.
        
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Get all parameter combinations
        param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        param_values = [
            self.random_forest.n_estimators,
            self.random_forest.max_depth,
            self.random_forest.min_samples_split,
            self.random_forest.min_samples_leaf
        ]
        
        param_grid = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            param_dict['random_state'] = self.random_forest.random_state
            param_dict['n_jobs'] = self.random_forest.n_jobs
            param_grid.append(param_dict)
        
        return param_grid