"""Performance tracking utilities for Random Forest scaling experiments."""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    training_time: float
    prediction_time: float
    peak_memory_mb: float
    avg_cpu_percent: float
    accuracy_score: Optional[float] = None
    mse_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_cpu_percent': self.avg_cpu_percent,
            'accuracy_score': self.accuracy_score,
            'mse_score': self.mse_score
        }


class PerformanceTracker:
    """Tracks performance metrics during Random Forest training and prediction."""
    
    def __init__(self, monitor_interval: float = 0.1):
        """
        Initialize the performance tracker.
        
        Args:
            monitor_interval: Interval in seconds for monitoring system resources
        """
        self.monitor_interval = monitor_interval
        self.reset()
        
    def reset(self):
        """Reset all tracking variables."""
        self._start_time = None
        self._end_time = None
        self._monitoring = False
        self._memory_samples = []
        self._cpu_samples = []
        self._monitor_thread = None
        
    def _monitor_resources(self):
        """Monitor CPU and memory usage in a separate thread."""
        process = psutil.Process()
        
        while self._monitoring:
            try:
                # Get memory usage in MB
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self._memory_samples.append(memory_mb)
                
                # Get CPU usage percentage
                cpu_percent = process.cpu_percent()
                self._cpu_samples.append(cpu_percent)
                
                time.sleep(self.monitor_interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
                
    def start_tracking(self):
        """Start tracking performance metrics."""
        self.reset()
        self._start_time = time.time()
        self._monitoring = True
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_tracking(self) -> float:
        """
        Stop tracking and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self._end_time = time.time()
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
        return self._end_time - self._start_time
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self._memory_samples:
            return 0.0
        return max(self._memory_samples)
    
    def get_avg_cpu(self) -> float:
        """Get average CPU usage percentage."""
        if not self._cpu_samples:
            return 0.0
        return np.mean(self._cpu_samples)
    
    def track_training(self, model, X, y) -> float:
        """
        Track performance during model training.
        
        Args:
            model: The model to train
            X: Training features
            y: Training labels
            
        Returns:
            Training time in seconds
        """
        self.start_tracking()
        model.fit(X, y)
        return self.stop_tracking()
    
    def track_prediction(self, model, X) -> tuple:
        """
        Track performance during model prediction.
        
        Args:
            model: The trained model
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, prediction_time)
        """
        self.start_tracking()
        predictions = model.predict(X)
        pred_time = self.stop_tracking()
        return predictions, pred_time
    
    def create_metrics(
        self,
        training_time: float,
        prediction_time: float,
        accuracy_score: Optional[float] = None,
        mse_score: Optional[float] = None
    ) -> PerformanceMetrics:
        """
        Create a PerformanceMetrics object with current measurements.
        
        Args:
            training_time: Time spent training
            prediction_time: Time spent predicting
            accuracy_score: Classification accuracy (optional)
            mse_score: Regression MSE (optional)
            
        Returns:
            PerformanceMetrics object
        """
        return PerformanceMetrics(
            training_time=training_time,
            prediction_time=prediction_time,
            peak_memory_mb=self.get_peak_memory(),
            avg_cpu_percent=self.get_avg_cpu(),
            accuracy_score=accuracy_score,
            mse_score=mse_score
        )