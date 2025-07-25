# ==============================================================
# Data Preprocessing Utilities
# ==============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles data preprocessing for financial time series."""
    
    def __init__(self, feature_columns, past_steps=60):
        """Initialize preprocessor with configuration."""
        self.feature_columns = feature_columns
        self.past_steps = past_steps
        
    def prepare_data(self, df):
        """Prepare data for training/inference."""
        # Sort by date and set as index
        df = df.sort_values("Date").set_index("Date")
        
        # Add log returns
        df["log_ret"] = np.log(df["Close"]).diff()
        df.dropna(inplace=True)
        
        return df
    
    def scale_features(self, df, scaler=None):
        """Scale OHLCV features."""
        if scaler is None:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(df[self.feature_columns])
        else:
            feat_scaled = scaler.transform(df[self.feature_columns])
            
        return feat_scaled, scaler
    
    def make_sequences(self, arr, target_index):
        """Build windowed sequences for time series."""
        X, y = [], []
        for i in range(self.past_steps, len(arr)):
            X.append(arr[i - self.past_steps : i, :])
            y.append(arr[i, target_index])
        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32).reshape(-1, 1),
        )
    
    def split_data(self, data, train_split=0.70, val_split=0.15):
        """Split data into train/val/test sets."""
        train_sz = int(len(data) * train_split)
        val_sz = int(len(data) * val_split)
        
        train = data[:train_sz]
        val = data[train_sz : train_sz + val_sz]
        test = data[train_sz + val_sz :]
        
        return train, val, test, train_sz, val_sz
    
    def prepare_model_data(self, df, train_split=0.70, val_split=0.15):
        """Complete data preparation pipeline."""
        # Prepare basic data
        df = self.prepare_data(df)
        
        # Scale features
        feat_scaled, f_scaler = self.scale_features(df)
        
        # Assemble feature matrix: scaled OHLCV + raw log_ret
        data = np.hstack([feat_scaled, df[["log_ret"]].values])
        RET_IDX = len(self.feature_columns)  # Index of log_ret in feature matrix
        
        # Split data
        train, val, test, train_sz, val_sz = self.split_data(data, train_split, val_split)
        
        # Create sequences
        X_tr, y_tr_raw = self.make_sequences(train, RET_IDX)
        X_val, y_val_raw = self.make_sequences(val, RET_IDX)
        X_te, y_te_raw = self.make_sequences(test, RET_IDX)
        
        # Scale targets
        ret_scaler = StandardScaler().fit(y_tr_raw)
        y_tr = ret_scaler.transform(y_tr_raw)
        y_val = ret_scaler.transform(y_val_raw)
        y_te = ret_scaler.transform(y_te_raw)
        
        return {
            'X_train': X_tr, 'y_train': y_tr, 'y_train_raw': y_tr_raw,
            'X_val': X_val, 'y_val': y_val, 'y_val_raw': y_val_raw,
            'X_test': X_te, 'y_test': y_te, 'y_test_raw': y_te_raw,
            'feature_scaler': f_scaler,
            'target_scaler': ret_scaler,
            'train_size': train_sz,
            'val_size': val_sz,
            'n_features': X_tr.shape[2]
        } 