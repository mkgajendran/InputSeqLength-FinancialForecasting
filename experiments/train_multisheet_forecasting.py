# ==============================================================
# Multisheet Forecasting Training Script
# ==============================================================

import os
import sys
import random
import math
import warnings
import pickle
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.lstm_model import LSTMModelPyTorch
from models.gru_model import GRUModelPyTorch
from utils.data_preprocessing import DataPreprocessor
from utils.device_utils import DeviceManager


class MultisheetTrainer:
    """Handles training of multisheet forecasting models."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.exp_config = self.config['experiments']['multisheet_forecasting']
        self.data_config = self.config['data']
        self.output_config = self.config['output']
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Set up device manager
        device_config = self.exp_config.get('device_optimization', {})
        self.device_manager = DeviceManager(device_config)
        
        # Set up data preprocessor
        self.preprocessor = DataPreprocessor(
            feature_columns=self.exp_config['feature_columns'],
            past_steps=60
        )
        
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
    
    def evaluate_model(self, model, X_te, df_close, ret_scaler):
        """Evaluate model and calculate metrics."""
        model.eval()
        
        with torch.no_grad():
            X_te_tensor = torch.FloatTensor(X_te).to(self.device_manager.device)
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
    
    def train_all_models(self):
        """Train models for all sheets and save results."""
        print("=" * 60)
        print("Multisheet Forecasting Training")
        print("=" * 60)
        
        # Load Excel file
        excel_file = self.output_config['excel_filename']
        print(f"Loading workbook: {excel_file}")
        
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
        
        xl = pd.ExcelFile(excel_file)
        sheets = xl.sheet_names[:8]  # Limit to 8 sheets
        
        metrics_rows = []
        training_results = {}
        
        for sheet in sheets:
            print(f"\n=== Training {sheet} ===")
            
            # Load and prepare data
            df = pd.read_excel(excel_file, sheet_name=sheet).dropna(
                subset=["Date", "Open", "High", "Low", "Close", "Volume"]
            )
            
            # Prepare model data
            data_dict = self.preprocessor.prepare_model_data(
                df, 
                train_split=self.exp_config['train_split'],
                val_split=self.exp_config['val_split']
            )
            
            print(f"First window features shape: {data_dict['X_train'][0].shape}; "
                  f"target (raw log-ret) = {data_dict['y_train_raw'][0,0]:.4e}")
            
            # Create data loaders
            batch_size = self.device_manager.get_optimal_batch_size()
            train_loader = self.device_manager.create_dataloader(
                data_dict['X_train'], data_dict['y_train'], batch_size, shuffle=True
            )
            val_loader = self.device_manager.create_dataloader(
                data_dict['X_val'], data_dict['y_val'], batch_size, shuffle=False
            )
            
            sheet_results = {
                'dataframe': self.preprocessor.prepare_data(df),
                'train_size': data_dict['train_size'],
                'val_size': data_dict['val_size'],
                'feature_scaler': data_dict['feature_scaler'],
                'target_scaler': data_dict['target_scaler']
            }
            
            # Train both models
            for model_type in ["LSTM", "GRU"]:
                print(f"  Training {model_type}...")
                
                if model_type == "LSTM":
                    model = LSTMModelPyTorch().build_model(data_dict['n_features'], self.device_manager.device)
                else:
                    model = GRUModelPyTorch().build_model(data_dict['n_features'], self.device_manager.device)
                
                # Train model
                epochs = min(20, model.get_config()['epochs'])
                model = self.train_model(
                    model, train_loader, val_loader, epochs, 
                    patience=self.exp_config['early_stopping_patience']
                )
                
                # Evaluate model
                res = self.evaluate_model(
                    model, data_dict['X_test'], sheet_results['dataframe']["Close"], 
                    data_dict['target_scaler']
                )
                res.update({"Sheet": sheet, "Model": model_type})
                metrics_rows.append(res)
                
                print(f"  {model_type} - MAE: {res['MAE']:.2f}, RMSE: {res['RMSE']:.2f}, R²: {res['R2']:.4f}")
                
                # Save model if configured
                if self.exp_config.get('save_models', False):
                    self._save_model(model, sheet, model_type)
                
                # Store predictions for plotting
                pred_series = pd.Series(
                    res["pred_prices"], 
                    index=sheet_results['dataframe'].index[-len(res["pred_prices"]):]
                )
                sheet_results[f'pred_{model_type.lower()}'] = pred_series
            
            training_results[sheet] = sheet_results
        
        # Save training results for inference
        self._save_training_results(training_results)
        
        # Save metrics
        self._save_metrics(metrics_rows)
        
        print("\n🎯 Training completed successfully!")
        print("📊 Results saved for inference and plotting")
        
        return training_results, metrics_rows
    
    def _save_model(self, model, sheet, model_type):
        """Save trained model."""
        models_dir = Path(self.exp_config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{sheet}_{model_type}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")
    
    def _save_training_results(self, training_results):
        """Save training results for inference."""
        output_dir = Path(self.exp_config['output_dir'])
        results_dir = output_dir / "training_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / "training_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(training_results, f)
        
        print(f"Training results saved to: {results_path}")
    
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
    """Main function to run training."""
    trainer = MultisheetTrainer()
    trainer.train_all_models()


if __name__ == "__main__":
    main() 