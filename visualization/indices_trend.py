# ==============================================================
# Indices Trend Visualization
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math
from datetime import datetime


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
OUTPUT_DIR = config['visualization']['indices_trend']['output_dir']
OUTPUT_FILE = config['visualization']['indices_trend']['output_filename']


# ==============================================================
# Visualization Functions
# ==============================================================

def calculate_subplot_layout(num_sheets):
    """Calculate optimal rows and columns for subplots."""
    if num_sheets <= 2:
        return 1, 2
    elif num_sheets <= 4:
        return 2, 2
    elif num_sheets <= 6:
        return 2, 3
    elif num_sheets <= 8:
        return 2, 4  # Perfect for 8 indices
    elif num_sheets <= 9:
        return 3, 3
    elif num_sheets <= 12:
        return 3, 4
    else:
        # For more than 12, use square-ish layout
        cols = math.ceil(math.sqrt(num_sheets))
        rows = math.ceil(num_sheets / cols)
        return rows, cols


def plot_indices_trends():
    """Plot close price trends for all indices and save as PDF."""
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Read Excel file and get all sheet names
    excel_file = pd.ExcelFile(EXCEL_FILE)
    sheet_names = excel_file.sheet_names
    
    print(f"Found {len(sheet_names)} indices: {sheet_names}")
    
    # Calculate subplot layout
    rows, cols = calculate_subplot_layout(len(sheet_names))
    print(f"Using {rows}x{cols} subplot layout")
    
    # Create figure with proper aspect ratio: width = 2 * height for each subplot
    # For 2x4 layout: total width should be 4*2 = 8 units, total height should be 2*1 = 2 units
    subplot_width = 4  # Each subplot width
    subplot_height = 2  # Each subplot height (width = 2 * height)
    fig_width = cols * subplot_width
    fig_height = rows * subplot_height
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Flatten axes array for easier indexing
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each index
    for i, sheet_name in enumerate(sheet_names):
        # Read data for this sheet
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
        
        # Convert Date column to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to ensure proper plotting
        df = df.sort_values('Date')
        
        # Plot on the corresponding subplot
        ax = axes[i]
        ax.plot(df['Date'], df['Close'], linewidth=1.0, color='#1f77b4', alpha=0.9, label=sheet_name)
        
        # Calculate row and column position
        row = i // cols
        col = i % cols
        
        # Only show x-label on bottom row
        if row == rows - 1:
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        
        # Only show y-label on leftmost column
        if col == 0:
            ax.set_ylabel('Close', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend box inside the subplot
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                 fontsize=9, framealpha=0.9, edgecolor='black', facecolor='white',
                 handlelength=0, handletextpad=0)
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Format y-axis to show values clearly
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        print(f"  ✓ Plotted {sheet_name}: {len(df)} data points")
    
    # Hide any unused subplots
    for j in range(len(sheet_names), rows * cols):
        axes[j].set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=0.5)
    
    # Save as PDF
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Saved trend analysis to: {output_path}")
    print(f"📊 Total indices plotted: {len(sheet_names)}")
    print(f"📐 Figure size: {fig_width} x {fig_height} (each subplot: {subplot_width} x {subplot_height})")
    
    # Close the figure to free memory
    plt.close()


# ==============================================================
# Main Execution
# ==============================================================

def main():
    """Main function to generate indices trend plots."""
    print("=" * 60)
    print("Indices Trend Visualization")
    print("=" * 60)
    
    try:
        plot_indices_trends()
        print("\n🎯 Trend analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Excel file '{EXCEL_FILE}' not found.")
        print("Please run the data downloader first.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
