"""Simple test to verify the Random Forest scaling framework works."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_basic_functionality():
    """Test basic functionality of the scaling framework."""
    print("Testing Random Forest Scaling Framework...")
    
    try:
        # Test imports
        from scaling_laws_research.utils.config import ExperimentConfig
        from scaling_laws_research.utils.data_generator import DataGenerator
        from scaling_laws_research.utils.performance_tracker import PerformanceTracker
        from scaling_laws_research.experiments.scaling_experiments import ScalingExperiment
        print("✓ All imports successful")
        
        # Test data generation
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_classification_dataset(100, 10)
        assert X.shape == (100, 10)
        assert len(y) == 100
        print("✓ Data generation works")
        
        # Test configuration
        config = ExperimentConfig()
        config.data.base_samples = 100
        config.data.sample_scales = [1, 2]
        config.data.feature_scales = [1, 2]
        config.random_forest.n_estimators = [10, 20]
        config.random_forest.max_depth = [None, 5]
        config.random_forest.min_samples_split = [2]
        config.random_forest.min_samples_leaf = [1]
        print("✓ Configuration setup works")
        
        # Test experiment (small scale)
        config.output_dir = "/tmp/test_results"
        config.verbose = False
        experiment = ScalingExperiment(config)
        
        # Test data scaling with small dataset
        data_results = experiment.run_data_scaling_experiment()
        assert len(data_results) == 4  # 2 sample scales × 2 feature scales
        print("✓ Data scaling experiment works")
        
        # Test parameter scaling with small dataset
        param_results = experiment.run_parameter_scaling_experiment()
        assert len(param_results) == 4  # 2 estimators × 2 depths × 1 split × 1 leaf
        print("✓ Parameter scaling experiment works")
        
        print("\n✅ All tests passed! Framework is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_basic_functionality()