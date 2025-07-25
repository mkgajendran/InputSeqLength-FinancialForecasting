# ==============================================================
# LSTM Model (PyTorch CPU Version)
# ==============================================================

import yaml
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModelPyTorch(nn.Module):
    """PyTorch LSTM model for financial forecasting (CPU optimized)."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize LSTM model with configuration."""
        super(LSTMModelPyTorch, self).__init__()
        self.config = self._load_config(config_path)
        self.model_config = self.config['models']['lstm']
        
        # Device will be set by the experiment class
        self.device = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def build_model(self, n_features, device=None):
        """Build LSTM model architecture."""
        self.n_features = n_features
        self.past_steps = self.model_config['past_steps']
        
        # Set device
        if device is not None:
            self.device = device
        
        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.model_config['lstm_units_1'],
            batch_first=True,
            bidirectional=True,
            dropout=self.model_config['dropout'] if self.model_config['dropout'] > 0 else 0
        )
        
        # Layer normalization after first LSTM
        self.layernorm1 = nn.LayerNorm(self.model_config['lstm_units_1'] * 2)  # *2 for bidirectional
        
        # Dropout after first LSTM
        self.dropout1 = nn.Dropout(self.model_config['dropout'])
        
        # Second LSTM layer (unidirectional)
        self.lstm2 = nn.LSTM(
            input_size=self.model_config['lstm_units_1'] * 2,
            hidden_size=self.model_config['lstm_units_2'],
            batch_first=True,
            bidirectional=False,
            dropout=self.model_config['dropout'] if self.model_config['dropout'] > 0 else 0
        )
        
        # Layer normalization after second LSTM
        self.layernorm2 = nn.LayerNorm(self.model_config['lstm_units_2'])
        
        # Dropout after second LSTM
        self.dropout2 = nn.Dropout(self.model_config['dropout'])
        
        # Dense layers
        self.dense1 = nn.Linear(self.model_config['lstm_units_2'], self.model_config['dense_units'])
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.model_config['dense_units'], 1)
        
        # Move to device
        if self.device is not None:
            self.to(self.device)
        
        return self
    
    def forward(self, x):
        """Forward pass through the model."""
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.layernorm1(lstm1_out)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer (take only the last output)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = lstm2_out[:, -1, :]  # Take last time step
        lstm2_out = self.layernorm2(lstm2_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Dense layers
        dense1_out = self.relu(self.dense1(lstm2_out))
        output = self.dense2(dense1_out)
        
        return output
    
    def get_optimizer(self):
        """Get optimizer with configured learning rate."""
        return optim.Adam(self.parameters(), lr=self.model_config['learning_rate'])
    
    def get_config(self):
        """Get model configuration."""
        return self.model_config 