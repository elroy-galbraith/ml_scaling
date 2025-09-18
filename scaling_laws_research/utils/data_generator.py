"""Data generation utilities for Random Forest scaling experiments."""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from typing import Tuple, Dict, Any, Optional


class DataGenerator:
    """Generates synthetic datasets for Random Forest scaling experiments."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_state: Random state for reproducible data generation
        """
        self.random_state = random_state
        
    def generate_classification_dataset(
        self,
        n_samples: int,
        n_features: int,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        n_classes: int = 2,
        class_sep: float = 1.0,
        flip_y: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features (default: n_features // 2)
            n_redundant: Number of redundant features (default: n_features // 4)
            n_classes: Number of classes
            class_sep: Factor multiplying the hypercube size
            flip_y: Fraction of samples whose class is randomly flipped
            
        Returns:
            Tuple of (features, labels)
        """
        if n_informative is None:
            n_informative = max(1, n_features // 2)
        if n_redundant is None:
            n_redundant = max(0, n_features // 4)
            
        # Ensure we don't exceed n_features
        n_informative = min(n_informative, n_features)
        n_redundant = min(n_redundant, n_features - n_informative)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=self.random_state
        )
        
        return X, y
    
    def generate_regression_dataset(
        self,
        n_samples: int,
        n_features: int,
        n_informative: Optional[int] = None,
        noise: float = 0.1,
        bias: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic regression dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features (default: n_features // 2)
            noise: Standard deviation of the gaussian noise
            bias: Bias term in the underlying linear model
            
        Returns:
            Tuple of (features, targets)
        """
        if n_informative is None:
            n_informative = max(1, n_features // 2)
            
        n_informative = min(n_informative, n_features)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            bias=bias,
            random_state=self.random_state
        )
        
        return X, y
    
    def generate_scaling_datasets(
        self,
        base_samples: int,
        base_features: int,
        sample_scales: list = [1, 2, 4, 8, 16],
        feature_scales: list = [1, 2, 4, 8],
        task_type: str = "classification"
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate multiple datasets for scaling experiments.
        
        Args:
            base_samples: Base number of samples
            base_features: Base number of features
            sample_scales: Scaling factors for sample counts
            feature_scales: Scaling factors for feature counts
            task_type: Either "classification" or "regression"
            
        Returns:
            Dictionary mapping scale descriptions to (X, y) tuples
        """
        datasets = {}
        
        for s_scale in sample_scales:
            for f_scale in feature_scales:
                n_samples = base_samples * s_scale
                n_features = base_features * f_scale
                
                key = f"samples_{n_samples}_features_{n_features}"
                
                if task_type == "classification":
                    X, y = self.generate_classification_dataset(n_samples, n_features)
                elif task_type == "regression":
                    X, y = self.generate_regression_dataset(n_samples, n_features)
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
                    
                datasets[key] = (X, y)
                
        return datasets