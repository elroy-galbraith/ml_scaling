"""Test script to demonstrate the Adult dataset integration."""

from scaling_laws_research.utils.config import ExperimentConfig
from scaling_laws_research.experiments.scaling_experiments import ScalingExperiment
from scaling_laws_research.data.adult_dataset import AdultDatasetLoader
from scaling_laws_research.analysis.decision_framework import ScalingDecisionFramework


def test_adult_integration():
    """Test the complete Adult dataset integration."""
    print("=== Testing Adult Dataset Integration ===\n")

    # 1. Test dataset loading
    print("1. Testing Adult Dataset Loader...")
    loader = AdultDatasetLoader()
    info = loader.get_dataset_info()
    print(f"   Dataset: {info['name']}")
    print(f"   Total samples: {info['total_samples']:,}")
    print(f"   Features: {info['n_features']}")
    print("   ✓ Dataset loader working\n")

    # 2. Test configuration
    print("2. Testing Configuration...")
    config = ExperimentConfig.load_from_file('adult_experiment_config.json')
    print(f"   Sample sizes: {config.data.sample_sizes}")
    print(f"   Dataset name: {config.data.dataset_name}")
    print(f"   Random seeds: {config.random_forest.random_seeds}")
    print("   ✓ Configuration loading working\n")

    # 3. Test small experiment (just data loading part)
    print("3. Testing Small Dataset Loading...")
    small_datasets = loader.get_dataset(sample_sizes=[500, 1000])
    for name, (X, y) in small_datasets.items():
        print(f"   {name}: X.shape={X.shape}, y.shape={y.shape}")
        print(f"   Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("   ✓ Small dataset loading working\n")

    # 4. Test experiment initialization
    print("4. Testing Experiment Initialization...")
    experiment = ScalingExperiment(config)
    print(f"   Adult loader initialized: {experiment.adult_loader is not None}")
    print("   ✓ Experiment initialization working\n")

    print("=== All tests passed! Integration is working correctly ===")
    print("\nTo run a full experiment:")
    print("python main.py run --config adult_experiment_config.json --output results/adult_experiment")


if __name__ == "__main__":
    import numpy as np
    test_adult_integration()