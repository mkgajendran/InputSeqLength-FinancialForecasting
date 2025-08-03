# ==============================================================
# Multisheet Forecasting Inference Script with Zoom
# ==============================================================

import sys
import yaml
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from visualization.forecasting_plots_withzoom import ForecastingPlotterWithZoom


class MultisheetInferenceWithZoom:
    """Handles inference and plotting from saved training results with zoom insets."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize inference with configuration."""
        self.config = self._load_config(config_path)
        self.exp_config = self.config['experiments']['multisheet_forecasting']
        
        self.plotter = ForecastingPlotterWithZoom(self.config)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_plots(self):
        """Generate plots from saved training results."""
        print("============================================================")
        print("Multisheet Forecasting Inference with Zoom - Plot Generation")
        print("============================================================")
        
        # Load saved training results
        results_dir = Path(self.exp_config['output_dir']) / "training_results"
        results_file = results_dir / "training_results.pkl"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Training results not found: {results_file}")
        
        print(f"📊 Loading training results from: {results_file}")
        
        # Generate plots
        fig = self.plotter.plot_from_saved_results(results_file)
        
        return fig
    
    def customize_plots(self, **kwargs):
        """Customize plot appearance."""
        self.plotter.customize_inset(**kwargs)


def main():
    """Main function to generate zoomed plots."""
    inference = MultisheetInferenceWithZoom()
    inference.generate_plots()
    
    print("\n🎯 Zoomed plot generation completed successfully!")
    print("📈 You can now modify plot settings in visualization/forecasting_plots_withzoom.py")
    print("   and rerun this script without retraining!")


if __name__ == "__main__":
    main() 