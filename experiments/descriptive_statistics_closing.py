# ==============================================================
# Descriptive Statistics Analysis
# ==============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


# ==============================================================
# Configuration
# ==============================================================

def load_config():
    """Load configuration from YAML file."""
    import yaml
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load configuration
config = load_config()
EXCEL_FILE = config['output']['excel_filename']
OUTPUT_DIR = config['analysis']['descriptive_statistics']['output_dir']
OUTPUT_FILE = config['analysis']['descriptive_statistics']['output_filename']


# ==============================================================
# Analysis Functions
# ==============================================================

def analyze_descriptive_statistics():
    """Analyze descriptive statistics for closing prices of all indices."""
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Read Excel file and get all sheet names
    excel_file = pd.ExcelFile(EXCEL_FILE)
    sheet_names = excel_file.sheet_names
    
    print(f"Found {len(sheet_names)} indices: {sheet_names}")
    
    # Initialize list to store statistics
    stats_data = []
    
    # Analyze each index
    for sheet_name in sheet_names:
        # Read data for this sheet
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
        
        # Get closing prices
        close_prices = df['Close'].dropna()
        
        # Calculate descriptive statistics
        stats = {
            'Index': sheet_name,
            'Count': len(close_prices),
            'Mean': round(close_prices.mean(), 2),
            'Std. Dev.': round(close_prices.std(), 2),
            'Min': round(close_prices.min(), 2),
            '25%': round(close_prices.quantile(0.25), 2),
            '50%': round(close_prices.quantile(0.50), 2),
            '75%': round(close_prices.quantile(0.75), 2),
            'Max': round(close_prices.max(), 2)
        }
        
        stats_data.append(stats)
        
        print(f"  ✓ Analyzed {sheet_name}: {len(close_prices)} data points")
    
    # Create DataFrame from statistics
    stats_df = pd.DataFrame(stats_data)
    
    # Save to Excel
    output_file_path = output_path / OUTPUT_FILE
    stats_df.to_excel(output_file_path, index=False, sheet_name='Descriptive_Statistics')
    
    print(f"\n✅ Saved descriptive statistics to: {output_file_path}")
    print(f"📊 Total indices analyzed: {len(sheet_names)}")
    
    # Display summary
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))
    
    return stats_df


# ==============================================================
# Main Execution
# ==============================================================

def main():
    """Main function to generate descriptive statistics analysis."""
    print("=" * 60)
    print("Descriptive Statistics Analysis")
    print("=" * 60)
    
    try:
        analyze_descriptive_statistics()
        print("\n🎯 Descriptive statistics analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Excel file '{EXCEL_FILE}' not found.")
        print("Please run the data downloader first.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main() 