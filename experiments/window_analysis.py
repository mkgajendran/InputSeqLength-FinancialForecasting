# ==============================================================
# Window Analysis Script - PyTorch Implementation
# ==============================================================
# This script trains LSTM & GRU models across multiple look-back
# windows spanning 1 day to a full trading year:
#   1, 3, 7, 21, 30, 60, 90, 120, 180, 252 days.
# It then:
#   • saves **one PDF per window** (all indices plotted together)
#   • writes **one Excel workbook** with a metrics sheet per window.
#
# Usage:
# 1. Ensure your Excel data file exists at the configured path
# 2. Run: python window_analysis.py
# 3. Find outputs in plots_windowed/ & multisheet_test_metrics_windowed.xlsx
# ==============================================================

import os
import sys
import random
import math
import warnings
import datetime
import pickle
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.lstm_model import LSTMModelPyTorch
from models.gru_model import GRUModelPyTorch
from utils.device_utils import DeviceManager


class WindowAnalyzer:
    """Handles window-based analysis of forecasting models."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.exp_config = self.config['experiments']['multisheet_forecasting']
        self.output_config = self.config['output']
        
        # Window analysis specific settings
        self.windows = [1, 3, 7, 21, 30, 60, 90, 120, 180, 252]
        self.plot_dir = "results/plots_windowed"
        self.metrics_file = "results/metrics/multisheet_test_metrics_windowed.xlsx"
        
        # Create output directories
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(Path(self.metrics_file).parent, exist_ok=True)
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Set up device manager
        device_config = self.exp_config.get('device_optimization', {})
        self.device_manager = DeviceManager(device_config)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_reproducibility(self):
        """Set up global random seed for reproducibility."""
        seed = self.exp_config['seed']
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def make_sequences(self, arr, past_steps, target_idx):
        """Create sequences for time series prediction.
        
        Args:
            arr: Input array with shape [timesteps, features]
            past_steps: Number of past steps to use as input
            target_idx: Index of target feature
            
        Returns:
            X: Input sequences [samples, past_steps, features]
            y: Target values [samples, 1]
        """
        if len(arr) <= past_steps:
            return np.empty((0, past_steps, arr.shape[1]), dtype=np.float32), \
                   np.empty((0, 1), dtype=np.float32)
        
        # Create sliding windows
        X = []
        y = []
        
        for i in range(past_steps, len(arr)):
            X.append(arr[i-past_steps:i])
            y.append(arr[i, target_idx])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)
    
    def train_model(self, model, train_loader, val_loader, epochs=50, patience=10):
        """Train PyTorch model with early stopping."""
        optimizer = model.get_optimizer()
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        model.train()
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device_manager.device)
                batch_y = batch_y.to(self.device_manager.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    # Move data to device
                    batch_X = batch_X.to(self.device_manager.device)
                    batch_y = batch_y.to(self.device_manager.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
            model.train()
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def evaluate_model(self, model, X_test, close_series, target_scaler):
        """Evaluate model and calculate metrics."""
        model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device_manager.device)
            pred_scaled = model(X_test_tensor).cpu().numpy()
        
        # Inverse transform predictions
        pred_returns = target_scaler.inverse_transform(pred_scaled).flatten()
        
        # Convert log returns back to prices
        close_seed = close_series.iloc[-len(pred_returns) - 1:].values
        pred_prices = close_seed[:-1] * np.exp(pred_returns)
        gt_prices = close_seed[1:]
        
        return {
            "MAE": mean_absolute_error(gt_prices, pred_prices),
            "RMSE": math.sqrt(mean_squared_error(gt_prices, pred_prices)),
            "MAPE": mean_absolute_percentage_error(gt_prices, pred_prices),
            "R2": r2_score(gt_prices, pred_prices),
            "pred_prices": pred_prices,
            "gt_prices": gt_prices,
        }
    
    def run_window_analysis(self):
        """Run complete window analysis across all window sizes."""
        print("=" * 60)
        print("Window Analysis - Financial Forecasting")
        print("=" * 60)
        
        # Load Excel file
        excel_file = self.output_config['excel_filename']
        print(f"Loading workbook: {excel_file}")
        
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
        
        xl = pd.ExcelFile(excel_file)
        sheets = xl.sheet_names[:8]  # Limit to 8 sheets
        
        # Create Excel writer for metrics
        writer = pd.ExcelWriter(self.metrics_file, engine="openpyxl")
        
        # Process each window size
        for window_size in self.windows:
            print(f"\n########  WINDOW = {window_size} days  ########")
            
            # Set up plotting
            n_rows, n_cols = 4, 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24))
            axes = axes.flatten()
            
            metrics_rows = []
            
            # Process each sheet/index
            for idx, sheet in enumerate(sheets):
                ax = axes[idx]
                print(f"\n=== {sheet} (window {window_size}) ===")
                
                # Load and prepare data
                df = (
                    pd.read_excel(excel_file, sheet_name=sheet)
                    .dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
                    .sort_values("Date")
                    .set_index("Date")
                )
                
                # Feature engineering
                df["log_ret"] = np.log(df["Close"]).diff()
                df.dropna(inplace=True)
                
                if len(df) <= window_size:
                    print(f"⚠️  Insufficient data for window {window_size} → {sheet}. Skipping...")
                    continue
                
                # Prepare features and target
                feat_cols = ["Open", "High", "Low", "Close", "Volume"]
                feature_scaler = StandardScaler()
                feat_scaled = feature_scaler.fit_transform(df[feat_cols])
                
                # Combine features with log returns
                data = np.hstack([feat_scaled, df[["log_ret"]].values])
                target_idx = len(feat_cols)  # log_ret is the last column
                
                # Create train/val/test splits
                train_size = int(len(data) * self.exp_config['train_split'])
                val_size = int(len(data) * self.exp_config['val_split'])
                
                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]
                
                # Create sequences
                X_train, y_train_raw = self.make_sequences(train_data, window_size, target_idx)
                X_val, y_val_raw = self.make_sequences(val_data, window_size, target_idx)
                X_test, y_test_raw = self.make_sequences(test_data, window_size, target_idx)
                
                if min(len(X_train), len(X_val), len(X_test)) == 0:
                    print(f"⚠️  Insufficient samples for window {window_size} → {sheet}. Skipping...")
                    continue
                
                # Scale targets
                target_scaler = StandardScaler()
                y_train = target_scaler.fit_transform(y_train_raw)
                y_val = target_scaler.transform(y_val_raw)
                
                print(f"First window shape: {X_train[0].shape}; target = {y_train_raw[0,0]:.3e}")
                
                # Create data loaders
                batch_size = min(256, len(X_train) // 4)  # Adaptive batch size
                train_loader = self.device_manager.create_dataloader(
                    X_train, y_train, batch_size, shuffle=True
                )
                val_loader = self.device_manager.create_dataloader(
                    X_val, y_val, batch_size, shuffle=False
                )
                
                # Store predictions for plotting
                predictions = {}
                
                # Train both models
                for model_type in ["LSTM", "GRU"]:
                    print(f"  Training {model_type}...")
                    
                    # Create model with dynamic window size
                    if model_type == "LSTM":
                        model = LSTMModelPyTorch().build_model(
                            X_train.shape[2], self.device_manager.device, past_steps=window_size
                        )
                    else:
                        model = GRUModelPyTorch().build_model(
                            X_train.shape[2], self.device_manager.device, past_steps=window_size
                        )
                    
                    # Train model
                    epochs = min(50, 100)  # Adaptive epochs based on window size
                    model = self.train_model(
                        model, train_loader, val_loader, epochs, 
                        patience=self.exp_config['early_stopping_patience']
                    )
                    
                    # Evaluate model
                    results = self.evaluate_model(
                        model, X_test, df["Close"], target_scaler
                    )
                    results.update({
                        "Sheet": sheet, 
                        "Model": model_type, 
                        "Window": window_size
                    })
                    metrics_rows.append(results)
                    
                    print(f"  {model_type} - MAE: {results['MAE']:.2f}, "
                          f"RMSE: {results['RMSE']:.2f}, R²: {results['R2']:.4f}")
                    
                    # Store predictions for plotting
                    predictions[model_type] = pd.Series(
                        results["pred_prices"], 
                        index=df.index[-len(results["pred_prices"]):]
                    )
                
                # Plot results
                self._plot_sheet_results(ax, df, predictions, train_size, val_size, 
                                       sheet, idx, n_rows, n_cols)
            
            # Finalize and save plot
            self._finalize_plot(fig, axes, sheets, n_rows, n_cols, window_size)
            
            # Save metrics for this window
            if metrics_rows:
                metrics_df = (
                    pd.DataFrame(metrics_rows)
                    .drop(columns=["pred_prices", "gt_prices"])
                    .set_index(["Sheet", "Model"])
                )
                metrics_df.to_excel(writer, sheet_name=f"win_{window_size}")
                print(f"Metrics → sheet win_{window_size}")
        
        # Close Excel writer
        writer.close()
        print(f"\nAll metrics saved to {self.metrics_file}")
        print("✓ Window Analysis Finished @", 
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    def _plot_sheet_results(self, ax, df, predictions, train_size, val_size, 
                           sheet, idx, n_rows, n_cols):
        """Plot results for a single sheet."""
        actual = df["Close"]
        dates = df.index
        test_start = train_size + val_size
        
        # Add background shading for train/val/test periods
        if idx == 0:  # Only add labels for first subplot
            ax.axvspan(dates[0], dates[train_size-1], 
                      facecolor="#98fb98", alpha=0.25, hatch="///", 
                      ec="#2e8b57", lw=0, label="Train")
            ax.axvspan(dates[train_size], dates[test_start-1], 
                      facecolor="#fff8dc", alpha=0.35, hatch="\\\\\\", 
                      ec="#daa520", lw=0, label="Val")
            ax.axvspan(dates[test_start], dates[-1], 
                      facecolor="#d3d3d3", alpha=0.40, hatch="...", 
                      ec="#696969", lw=0, label="Test")
        else:
            ax.axvspan(dates[0], dates[train_size-1], 
                      facecolor="#98fb98", alpha=0.25, hatch="///", 
                      ec="#2e8b57", lw=0)
            ax.axvspan(dates[train_size], dates[test_start-1], 
                      facecolor="#fff8dc", alpha=0.35, hatch="\\\\\\", 
                      ec="#daa520", lw=0)
            ax.axvspan(dates[test_start], dates[-1], 
                      facecolor="#d3d3d3", alpha=0.40, hatch="...", 
                      ec="#696969", lw=0)
        
        # Plot actual and predicted prices
        ax.plot(dates, actual, lw=1.0, label="Actual", color='black')
        ax.plot(predictions["LSTM"].index, predictions["LSTM"].values, 
               lw=1.2, label="LSTM", color='blue')
        ax.plot(predictions["GRU"].index, predictions["GRU"].values, 
               lw=1.2, label="GRU", color='red')
        
        ax.set_title(sheet, fontsize=12, fontweight='bold')
        ax.grid(ls=":", alpha=0.6)
        
        # Set axis labels
        if idx % n_cols == 0:
            ax.set_ylabel("Price", fontsize=10)
        if idx >= (n_rows-1) * n_cols:
            ax.set_xlabel("Date", fontsize=10)
    
    def _finalize_plot(self, fig, axes, sheets, n_rows, n_cols, window_size):
        """Finalize and save the plot."""
        # Remove unused subplots
        for j in range(len(sheets), n_rows * n_cols):
            fig.delaxes(axes[j])
        
        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10)
        
        # Adjust layout and save
        fig.suptitle(f"Financial Forecasting - Window Size: {window_size} days", 
                    fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        
        output_path = os.path.join(self.plot_dir, f"multisheet_forecasts_win{window_size}.pdf")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved plots → {output_path}")


def main():
    """Main function to run window analysis."""
    analyzer = WindowAnalyzer()
    analyzer.run_window_analysis()


if __name__ == "__main__":
    main()
