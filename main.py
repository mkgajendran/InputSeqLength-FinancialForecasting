# ==============================================================
# InputSeqLength-FinancialForecasting - Main Entry Point
# ==============================================================

import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.downloader import download_indices_data, load_config
from visualization.indices_trend import plot_indices_trends
from experiments.descriptive_statistics_closing import analyze_descriptive_statistics
from experiments.train_multisheet_forecasting import MultisheetTrainer


# ==============================================================
# Main Execution
# ==============================================================

def main():
    """Main function to run the project."""
    print("=" * 60)
    print("Financial Forecasting - Input Sequence Length Analysis")
    print("=" * 60)
    
    # Download financial indices data
    download_indices_data()
    
    # Generate trend visualization
    print("\n" + "="*60)
    print("Generating Indices Trend Visualization...")
    print("="*60)
    plot_indices_trends()
    
    # Generate descriptive statistics
    print("\n" + "="*60)
    print("Generating Descriptive Statistics...")
    print("="*60)
    analyze_descriptive_statistics()
    
    # Run multisheet forecasting training
    print("\n" + "="*60)
    print("Running Multisheet Forecasting Training...")
    print("="*60)
    trainer = MultisheetTrainer()
    trainer.train_all_models()
    
    print("\n🎯 Project execution completed successfully!")


if __name__ == "__main__":
    main()
