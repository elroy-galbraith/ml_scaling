"""Main CLI interface for running Random Forest scaling experiments."""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scaling_laws_research.utils.config import ExperimentConfig
from scaling_laws_research.experiments.scaling_experiments import ScalingExperiment
from scaling_laws_research.analysis.scaling_laws import ScalingLawAnalyzer
from scaling_laws_research.visualizations.scaling_plots import ScalingPlotter


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        name="default_rf_scaling",
        description="Default Random Forest scaling experiment"
    )


def run_experiment(config_path: str = None, output_dir: str = "results"):
    """Run a complete scaling experiment."""
    
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        config = ExperimentConfig.load_from_file(config_path)
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override output directory if specified
    if output_dir != "results":
        config.output_dir = output_dir
    
    print(f"Starting experiment: {config.name}")
    print(f"Output directory: {config.output_dir}")
    
    # Run experiment
    experiment = ScalingExperiment(config)
    results_df = experiment.run_full_experiment()
    
    print("\nGenerating analysis and visualizations...")
    
    # Analyze results
    analyzer = ScalingLawAnalyzer()
    data_analysis = analyzer.analyze_data_scaling(results_df)
    param_analysis = analyzer.analyze_parameter_scaling(results_df)
    
    # Generate report
    report = analyzer.generate_summary_report(data_analysis, param_analysis)
    
    # Save report
    report_path = os.path.join(config.output_dir, "scaling_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Analysis report saved to: {report_path}")
    
    # Generate visualizations
    plotter = ScalingPlotter()
    
    # Data scaling plots
    data_plot = plotter.plot_data_scaling(results_df, metric="training_time")
    data_plot.savefig(os.path.join(config.output_dir, "data_scaling_training_time.png"))
    
    # Parameter scaling plots
    param_plot = plotter.plot_parameter_scaling(results_df, metric="training_time")
    param_plot.savefig(os.path.join(config.output_dir, "parameter_scaling_training_time.png"))
    
    # Scaling laws with fits
    scaling_laws_plot = plotter.plot_scaling_laws(results_df)
    scaling_laws_plot.savefig(os.path.join(config.output_dir, "scaling_laws_analysis.png"))
    
    # Performance comparison
    performance_plot = plotter.plot_performance_comparison(results_df)
    performance_plot.savefig(os.path.join(config.output_dir, "performance_comparison.png"))
    
    print("Visualizations saved to output directory")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total experiments conducted: {len(results_df)}")
    print(f"Data scaling experiments: {len(results_df[results_df['scaling_type'] == 'data'])}")
    print(f"Parameter scaling experiments: {len(results_df[results_df['scaling_type'] == 'parameters'])}")
    print(f"Results saved to: {config.output_dir}")
    print("\nGenerated files:")
    print(f"  • scaling_results.csv - Raw experimental results")
    print(f"  • experiment_config.json - Experiment configuration")
    print(f"  • scaling_analysis_report.txt - Analysis summary")
    print(f"  • *.png - Visualization plots")
    
    return results_df


def create_sample_config(output_path: str = "sample_config.json"):
    """Create a sample configuration file."""
    config = ExperimentConfig(
        name="sample_rf_scaling_experiment",
        description="Sample Random Forest scaling experiment configuration"
    )
    
    config.save_to_file(output_path)
    print(f"Sample configuration saved to: {output_path}")
    print("You can modify this file and use it with --config option")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Random Forest Scaling Laws Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python main.py run
  
  # Run with custom configuration
  python main.py run --config my_config.json --output my_results
  
  # Create a sample configuration file
  python main.py create-config --output my_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run experiment command
    run_parser = subparsers.add_parser('run', help='Run scaling experiment')
    run_parser.add_argument('--config', '-c', type=str, 
                           help='Path to configuration JSON file')
    run_parser.add_argument('--output', '-o', type=str, default='results',
                           help='Output directory for results (default: results)')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', 
                                         help='Create sample configuration file')
    config_parser.add_argument('--output', '-o', type=str, default='sample_config.json',
                              help='Output path for configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_experiment(config_path=args.config, output_dir=args.output)
    elif args.command == 'create-config':
        create_sample_config(output_path=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()