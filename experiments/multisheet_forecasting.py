# ==============================================================
# Multisheet Forecasting Experiment (PyTorch CPU Version)
# ==============================================================

import os
import sys
import random
import math
import warnings
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


class MultisheetForecastingPyTorch:
    """Multisheet forecasting experiment using PyTorch (CPU optimized)."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize experiment with configuration."""
        self.config = self._load_config(config_path)
        self.exp_config = self.config['experiments']['multisheet_forecasting']
        self.data_config = self.config['data']
        self.output_config = self.config['output']
        
        # Set up device and optimization
        self.device = self._setup_device()
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Set up device-specific optimization
        self._setup_device_optimization()
        
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
    
    def _setup_device(self):
        """Set up the best available device (GPU or CPU)."""
        device_config = self.exp_config.get('device_optimization', {})
        
        if not device_config.get('auto_detect_device', True):
            return torch.device('cpu')
        
        # Check for NVIDIA GPU (CUDA)
        if torch.cuda.is_available() and device_config.get('prefer_gpu', True):
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using NVIDIA GPU: {gpu_name}")
            return device
        
        # Check for Apple Silicon GPU (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and device_config.get('prefer_gpu', True):
            device = torch.device('mps')
            print(f"🚀 Using Apple Silicon GPU (MPS)")
            return device
        
        # Fallback to CPU
        device = torch.device('cpu')
        print(f"💻 Using CPU (no GPU available or GPU disabled)")
        return device
    
    def _setup_device_optimization(self):
        """Set up device-specific optimization."""
        device_config = self.exp_config.get('device_optimization', {})
        
        if self.device.type == 'cuda':
            # NVIDIA GPU optimization
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
        elif self.device.type == 'mps':
            # Apple Silicon GPU optimization
            print("MPS GPU optimization enabled")
            
        elif self.device.type == 'cpu':
            # CPU optimization
            total_cores = os.cpu_count()
            cores_to_reserve = device_config.get('cpu_cores_to_reserve', 1)
            cores_to_use = max(1, total_cores - cores_to_reserve)
            
            # Set PyTorch thread count
            torch.set_num_threads(cores_to_use)
            
            # Set OpenMP threads (used by PyTorch internally)
            os.environ["OMP_NUM_THREADS"] = str(cores_to_use)
            os.environ["MKL_NUM_THREADS"] = str(cores_to_use)
            os.environ["NUMEXPR_NUM_THREADS"] = str(cores_to_use)
            
            print(f"CPU Optimization: Using {cores_to_use}/{total_cores} cores")
        
        print(f"Device: {self.device}")
        if self.device.type == 'cpu':
            print(f"PyTorch threads: {torch.get_num_threads()}")
    
    def make_sequences(self, arr, past_steps, target_index):
        """Build windowed sequences for time series."""
        X, y = [], []
        for i in range(past_steps, len(arr)):
            X.append(arr[i - past_steps : i, :])
            y.append(arr[i, target_index])
        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32).reshape(-1, 1),
        )
    
    def create_dataloader(self, X, y, batch_size, shuffle=False):
        """Create PyTorch DataLoader with device-specific optimization."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Device-specific optimization
        if self.device.type == 'cuda':
            # NVIDIA GPU optimization
            num_workers = min(4, os.cpu_count() // 2)
            pin_memory = True
        elif self.device.type == 'mps':
            # Apple Silicon optimization
            num_workers = 0  # MPS works better with num_workers=0
            pin_memory = False
        else:
            # CPU optimization
            num_workers = min(4, max(1, torch.get_num_threads() // 2))
            pin_memory = False
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
    
    def train_model(self, model, train_loader, val_loader, epochs, patience=10):
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
                print(f"  Early stopping at epoch {epoch+1}")
                break
                
            model.train()
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def eval_model(self, model, X_te, df_close, ret_scaler):
        """Evaluate model and calculate metrics."""
        model.eval()
        
        with torch.no_grad():
            X_te_tensor = torch.FloatTensor(X_te).to(self.device)
            pred_scaled = model(X_te_tensor).cpu().numpy()
        
        pred_rets = ret_scaler.inverse_transform(pred_scaled).flatten()
        
        # Use actual close_{t-1} to reconstruct predicted close_{t}
        close_seed = df_close.iloc[-len(pred_rets) - 1 :].values
        pred_prices = close_seed[:-1] * np.exp(pred_rets)
        gt_prices = close_seed[1:]
        
        return {
            "MAE": mean_absolute_error(gt_prices, pred_prices),
            "RMSE": math.sqrt(mean_squared_error(gt_prices, pred_prices)),
            "MAPE": mean_absolute_percentage_error(gt_prices, pred_prices),
            "R2": r2_score(gt_prices, pred_prices),
            "pred_prices": pred_prices,
            "gt_prices": gt_prices,
        }
    
    def run_experiment(self):
        """Run the multisheet forecasting experiment."""
        print("=" * 60)
        print("Multisheet Forecasting Experiment (PyTorch)")
        print("=" * 60)
        
        # Load Excel file
        excel_file = self.output_config['excel_filename']
        print(f"Loading workbook: {excel_file}")
        
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
        
        xl = pd.ExcelFile(excel_file)
        sheets = xl.sheet_names[:8]  # Limit to 8 sheets
        
        # Set up plotting
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24), sharex=False)
        axes = axes.flatten()
        
        metrics_rows = []
        
        for idx, sheet in enumerate(sheets):
            ax = axes[idx]
            print(f"\n=== {sheet} ===")
            
            # Load and engineer data
            df = (
                pd.read_excel(excel_file, sheet_name=sheet)
                .dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
                .sort_values("Date")
                .set_index("Date")
            )
            df["log_ret"] = np.log(df["Close"]).diff()
            df.dropna(inplace=True)
            
            # Scale OHLCV features per-sheet
            feat_cols = self.exp_config['feature_columns']
            f_scaler = StandardScaler().fit(df[feat_cols])
            feat_scaled = f_scaler.transform(df[feat_cols])
            
            # Assemble feature matrix: scaled OHLCV + raw log_ret
            data = np.hstack([feat_scaled, df[["log_ret"]].values])
            RET_IDX = len(feat_cols)  # Index of log_ret in feature matrix
            
            # Split data
            train_sz = int(len(data) * self.exp_config['train_split'])
            val_sz = int(len(data) * self.exp_config['val_split'])
            
            train = data[:train_sz]
            val = data[train_sz : train_sz + val_sz]
            test = data[train_sz + val_sz :]
            
            # Create sequences
            past_steps = 60  # From config
            X_tr, y_tr_raw = self.make_sequences(train, past_steps, RET_IDX)
            X_val, y_val_raw = self.make_sequences(val, past_steps, RET_IDX)
            X_te, y_te_raw = self.make_sequences(test, past_steps, RET_IDX)
            
            # Scale the target (per-sheet)
            ret_scaler = StandardScaler().fit(y_tr_raw)
            y_tr = ret_scaler.transform(y_tr_raw)
            y_val = ret_scaler.transform(y_val_raw)
            y_te = ret_scaler.transform(y_te_raw)
            
            # Sanity check
            print(f"First window features shape: {X_tr[0].shape}; target (raw log-ret) = {y_tr_raw[0,0]:.4e}")
            
            n_features = X_tr.shape[2]
            
            # Create data loaders with device-optimized batch size
            if self.device.type == 'cuda':
                # Larger batch sizes for GPU
                batch_size = 256
            elif self.device.type == 'mps':
                # Medium batch sizes for Apple Silicon
                batch_size = 128
            else:
                # CPU - scale with thread count
                base_batch_size = 64
                batch_size = min(256, base_batch_size * max(1, torch.get_num_threads() // 4))
            train_loader = self.create_dataloader(X_tr, y_tr, batch_size, shuffle=True)
            val_loader = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)
            
            # Train and evaluate models
            for model_type in ["LSTM", "GRU"]:
                print(f"  Training {model_type}...")
                
                if model_type == "LSTM":
                    model = LSTMModelPyTorch().build_model(n_features, self.device)
                else:
                    model = GRUModelPyTorch().build_model(n_features, self.device)
                
                # Train model
                epochs = min(20, model.get_config()['epochs'])  # Reduced epochs for CPU
                model = self.train_model(
                    model, train_loader, val_loader, epochs, 
                    patience=self.exp_config['early_stopping_patience']
                )
                
                # Evaluate model
                res = self.eval_model(model, X_te, df["Close"], ret_scaler)
                res.update({"Sheet": sheet, "Model": model_type})
                metrics_rows.append(res)
                
                print(f"  {model_type} - MAE: {res['MAE']:.2f}, RMSE: {res['RMSE']:.2f}, R²: {res['R2']:.4f}")
                
                # Save model if configured
                if self.exp_config.get('save_models', False):
                    self._save_model(model, sheet, model_type)
                
                # Store predictions for plotting
                if model_type == "LSTM":
                    pred_lstm = pd.Series(res["pred_prices"], index=df.index[-len(res["pred_prices"]):])
                else:
                    pred_gru = pd.Series(res["pred_prices"], index=df.index[-len(res["pred_prices"]):])
            
            # Plot results
            self._plot_sheet_results(ax, df, pred_lstm, pred_gru, train_sz, val_sz, sheet, idx, n_cols, n_rows)
        
        # Finalize plot
        self._finalize_plot(fig, axes, sheets, n_rows, n_cols)
        
        # Save metrics
        self._save_metrics(metrics_rows)
        
        print("\n🎯 Multisheet forecasting experiment completed successfully!")
    
    def _plot_sheet_results(self, ax, df, pred_lstm, pred_gru, train_sz, val_sz, sheet, idx, n_cols, n_rows):
        """Plot results for a single sheet."""
        actual_close = df["Close"]
        plot_dates = df.index
        test_start = train_sz + val_sz
        
        # Add background shading
        ax.axvspan(plot_dates[0], plot_dates[train_sz-1], 
                  facecolor="#98fb98", alpha=0.25, hatch="///", ec="#2e8b57", lw=0, 
                  label="Train" if idx == 0 else None)
        ax.axvspan(plot_dates[train_sz], plot_dates[test_start-1], 
                  facecolor="#fff8dc", alpha=0.35, hatch="\\\\\\", ec="#daa520", lw=0, 
                  label="Val" if idx == 0 else None)
        ax.axvspan(plot_dates[test_start], plot_dates[-1], 
                  facecolor="#d3d3d3", alpha=0.40, hatch="...", ec="#696969", lw=0, 
                  label="Test" if idx == 0 else None)
        
        # Plot lines
        ax.plot(plot_dates, actual_close, lw=1.0, label="Actual")
        ax.plot(pred_lstm.index, pred_lstm.values, lw=1.2, label="LSTM")
        ax.plot(pred_gru.index, pred_gru.values, lw=1.2, label="GRU")
        
        ax.set_title(sheet, fontsize=16)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if idx % n_cols == 0:
            ax.set_ylabel("Price")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Date")
    
    def _finalize_plot(self, fig, axes, sheets, n_rows, n_cols):
        """Finalize and save the plot."""
        # Global legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=11)
        
        # Remove unused subplots
        for j in range(len(sheets), n_rows * n_cols):
            fig.delaxes(axes[j])
        
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        
        # Save plot in plots directory
        output_dir = Path(self.exp_config['output_dir'])
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / self.exp_config['plot_filename']
        
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def _save_model(self, model, sheet, model_type):
        """Save trained model."""
        models_dir = Path(self.exp_config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{sheet}_{model_type}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")
    
    def _save_metrics(self, metrics_rows):
        """Save metrics to CSV file."""
        metrics_df = (
            pd.DataFrame(metrics_rows)
            .drop(columns=["pred_prices", "gt_prices"])
            .set_index(["Sheet", "Model"])
            .sort_index()
        )
        
        print("\n=== Test-set metrics (lower is better, except R²) ===")
        print(metrics_df)
        
        # Save to CSV in metrics directory
        output_dir = Path(self.exp_config['output_dir'])
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = metrics_dir / self.exp_config['metrics_filename']
        metrics_df.to_csv(metrics_path, index=True)
        print(f"Metrics saved to: {metrics_path}")
        
        # Also save summary metrics
        summary_metrics = metrics_df.groupby('Model').agg({
            'MAE': ['mean', 'std'],
            'RMSE': ['mean', 'std'], 
            'MAPE': ['mean', 'std'],
            'R2': ['mean', 'std']
        }).round(4)
        
        summary_path = metrics_dir / "model_performance_summary.csv"
        summary_metrics.to_csv(summary_path)
        print(f"Summary metrics saved to: {summary_path}")


def main():
    """Main function to run multisheet forecasting experiment."""
    experiment = MultisheetForecastingPyTorch()
    experiment.run_experiment()


if __name__ == "__main__":
    main() 