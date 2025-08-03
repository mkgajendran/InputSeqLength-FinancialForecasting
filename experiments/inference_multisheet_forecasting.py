# ==============================================================
# Multisheet Forecasting Inference Script
# ==============================================================

import sys
import yaml
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from visualization.forecasting_plots import ForecastingPlotter


class MultisheetInference:
    """Handles inference and plotting from saved training results."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize inference with configuration."""
        self.config = self._load_config(config_path)
        self.exp_config = self.config['experiments']['multisheet_forecasting']
        
        # Initialize plotter
        self.plotter = ForecastingPlotter(self.config)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def generate_plots(self):
        """Generate plots from saved training results."""
        print("=" * 60)
        print("Multisheet Forecasting Inference - Plot Generation")
        print("=" * 60)
        
        # Check for saved training results
        output_dir = Path(self.exp_config['output_dir'])
        results_file = output_dir / "training_results" / "training_results.pkl"
        
        if not results_file.exists():
            print(f"❌ Training results not found: {results_file}")
            print("Please run training first: python experiments/train_multisheet_forecasting.py")
            return
        
        print(f"📊 Loading training results from: {results_file}")
        
        # Generate plots
        fig = self.plotter.plot_from_saved_results(results_file)
        
        print("\n🎯 Plot generation completed successfully!")
        print("📈 You can now modify plot settings in visualization/forecasting_plots.py")
        print("   and rerun this script without retraining!")
    
    def customize_plots(self, **plot_kwargs):
        """Generate customized plots with specific parameters."""
        print("=" * 60)
        print("Generating Customized Plots")
        print("=" * 60)
        
        # Load training results
        output_dir = Path(self.exp_config['output_dir'])
        results_file = output_dir / "training_results" / "training_results.pkl"
        
        if not results_file.exists():
            print(f"❌ Training results not found: {results_file}")
            return
        
        import pickle
        with open(results_file, 'rb') as f:
            training_results = pickle.load(f)
        
        # Apply custom plotting parameters
        original_plotter = self.plotter
        
        # You can modify plotting parameters here
        # For example:
        if 'figsize' in plot_kwargs:
            # Custom figure size
            pass
        
        sheets = list(training_results.keys())
        fig = self.plotter.plot_multisheet_forecasts(training_results, sheets)
        
        print("🎨 Custom plots generated!")


def main():
    """Main function for inference."""
    inference = MultisheetInference()
    inference.generate_plots()


if __name__ == "__main__":
    main() 