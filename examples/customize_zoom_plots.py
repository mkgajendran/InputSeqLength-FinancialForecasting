# ==============================================================
# Example: Customizing Zoom Inset Plots
# ==============================================================

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from experiments.inference_multisheet_forecasting_withzoom import MultisheetInferenceWithZoom


def main():
    """Examples of customizing the zoom inset plots."""
    
    # Initialize inference with zoom
    inference = MultisheetInferenceWithZoom()
    
    print("🔍 Customizing Zoom Inset Plots")
    print("=" * 50)
    
    # Example 1: Large inset in upper right
    print("\n📊 Example 1: Large inset in upper right")
    inference.plotter.customize_inset(
        width='50%',
        height='50%', 
        location='upper right',
        connection_color='red',
        connection_linewidth=1.2
    )
    inference.generate_plots()
    
    print("\n" + "="*50)
    
    # Example 2: Small inset without connections
    print("\n📊 Example 2: Small inset, no connections")
    inference.plotter.customize_inset(
        width='35%',
        height='35%',
        location='upper left', 
        show_connections=False
    )
    inference.generate_plots()
    
    print("\n" + "="*50)
    
    # Example 3: Medium inset in lower left with custom styling
    print("\n📊 Example 3: Medium inset, lower left, custom connections")
    inference.plotter.customize_inset(
        width='40%',
        height='40%',
        location='lower left',
        show_connections=True,
        connection_color='blue',
        connection_linewidth=0.5
    )
    inference.generate_plots()
    
    print("\n" + "="*50)
    
    # Example 4: Back to default
    print("\n📊 Example 4: Back to default settings")
    inference.plotter.customize_inset(
        width='42%',
        height='42%',
        location='lower right',
        show_connections=True,
        connection_color='0.4',
        connection_linewidth=0.8
    )
    inference.generate_plots()
    
    print("\n🎯 Zoom plot customization examples completed!")
    print("💡 Edit visualization/forecasting_plots_withzoom.py for permanent changes")
    print("📁 All plots saved to results/plots/multisheet_forecasts_withzoom.pdf")


if __name__ == "__main__":
    main() 