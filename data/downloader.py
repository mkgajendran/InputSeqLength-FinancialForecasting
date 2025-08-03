"""
Data Downloader Module
Downloads financial index data from Yahoo Finance and saves to Excel.
Reads configuration from config.yaml file.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Dict, Optional
import os

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_indices_data(config: Optional[Dict] = None, 
                         config_path: str = "config.yaml",
                         save_to_excel: bool = True,
                         excel_filename: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Download financial indices data from Yahoo Finance.
    
    Args:
        config: Configuration dictionary. If None, loads from config_path
        config_path: Path to config.yaml file
        save_to_excel: Whether to save data to Excel file
        excel_filename: Custom Excel filename. If None, uses config default
    
    Returns:
        Dictionary with index names as keys and DataFrames as values
    """
    
    # Load configuration if not provided
    if config is None:
        config = load_config(config_path)
    
    # Extract configuration parameters
    indices = config['indices']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    # Use current date if end_date is None
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Use config filename if not specified
    if excel_filename is None:
        excel_filename = config['output']['excel_filename']
    
    print(f"Downloading data from {start_date} to {end_date}")
    print(f"Indices to download: {list(indices.keys())}")
    
    # Dictionary to store individual DataFrames
    index_data_dict = {}
    
    # Download and store data
    for index_name, ticker in indices.items():
        print(f"Downloading data for {index_name} ({ticker})...")
        try:
            # Download single ticker
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                print(f"  Original columns: {list(data.columns)}")
                print(f"  Shape: {data.shape}")
                
                # Flatten multi-level columns if they exist
                if isinstance(data.columns, pd.MultiIndex):
                    # For single ticker, take the first level (the actual column names)
                    data.columns = [col[0] for col in data.columns]
                
                print(f"  Flattened columns: {list(data.columns)}")
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Remove any rows with all NaN values
                data = data.dropna(how='all')
                
                # Forward fill any remaining NaN values in OHLCV columns
                ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                existing_columns = [col for col in ohlcv_columns if col in data.columns]
                if existing_columns:
                    data[existing_columns] = data[existing_columns].ffill()
                    data[existing_columns] = data[existing_columns].bfill()
                
                # Drop any rows that still have NaN values in critical columns
                if 'Date' in data.columns and 'Close' in data.columns:
                    data = data.dropna(subset=['Date', 'Close'])
                
                if not data.empty:
                    # Reorder columns to Date, OHLCV format
                    desired_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    existing_cols = [col for col in desired_order if col in data.columns]
                    data = data[existing_cols]
                    
                    # Store individual data for Excel sheets
                    index_data_dict[index_name] = data.copy()
                    print(f"  ✓ Downloaded {len(data)} rows for {index_name}")
                else:
                    print(f"  ✗ No valid data for {index_name}")
            else:
                print(f"  ✗ No data downloaded for {index_name}")
                
        except Exception as e:
            print(f"  ✗ Error downloading {index_name}: {str(e)}")
    
    # Save individual indices to separate Excel sheets
    if index_data_dict and save_to_excel:
        print(f"\nSaving individual indices to Excel file...")
        
        # Create Excel writer object
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            for index_name, data in index_data_dict.items():
                # Clean sheet name (Excel sheet names have restrictions)
                sheet_name = index_name.replace('/', '_').replace('\\', '_')[:31]  # Max 31 chars
                data.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ Saved {index_name} to sheet '{sheet_name}' ({len(data)} rows)")
        
        print(f"\nDownload complete!")
        print(f"Individual indices saved to '{excel_filename}'")
        
        # Show summary statistics
        print(f"\nData summary by index:")
        total_rows = 0
        for index_name, data in index_data_dict.items():
            total_rows += len(data)
            print(f"  {index_name}: {len(data)} rows, Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        print(f"\nTotal rows across all indices: {total_rows}")
        
        # Check for any remaining NaN values across all indices
        total_nan = 0
        for index_name, data in index_data_dict.items():
            nan_count = data.isnull().sum().sum()
            total_nan += nan_count
        
        if total_nan > 0:
            print(f"Total NaN values across all indices: {total_nan}")
        else:
            print(f"✓ No NaN values remaining in any dataset")
            
    elif not index_data_dict:
        print("No data was successfully downloaded.")
    
    return index_data_dict

def download_single_index(index_name: str, 
                         config: Optional[Dict] = None,
                         config_path: str = "config.yaml") -> Optional[pd.DataFrame]:
    """
    Download data for a single index.
    
    Args:
        index_name: Name of the index to download
        config: Configuration dictionary. If None, loads from config_path
        config_path: Path to config.yaml file
    
    Returns:
        DataFrame with the index data or None if failed
    """
    
    # Load configuration if not provided
    if config is None:
        config = load_config(config_path)
    
    # Check if index exists in config
    if index_name not in config['indices']:
        print(f"Index '{index_name}' not found in configuration")
        return None
    
    # Download all indices and return the requested one
    all_data = download_indices_data(config, save_to_excel=False)
    return all_data.get(index_name)

if __name__ == "__main__":
    # Example usage
    print("Downloading all indices...")
    data_dict = download_indices_data()
    print(f"Downloaded {len(data_dict)} indices successfully!")
