# ==============================================================
# GRU Model (PyTorch CPU Version)
# ==============================================================

import yaml
import torch
import torch.nn as nn
import torch.optim as optim


class GRUModelPyTorch(nn.Module):
    """PyTorch GRU model for financial forecasting (CPU optimized)."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize GRU model with configuration."""
        super(GRUModelPyTorch, self).__init__()
        self.config = self._load_config(config_path)
        self.model_config = self.config['models']['gru']
        
        # Device will be set by the experiment class
        self.device = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def build_model(self, n_features, device=None):
        """Build GRU model architecture."""
        self.n_features = n_features
        self.past_steps = self.model_config['past_steps']
        
        # Set device
        if device is not None:
            self.device = device
        
        # First bidirectional GRU layer
        self.gru1 = nn.GRU(
            input_size=n_features,
            hidden_size=self.model_config['gru_units_1'],
            batch_first=True,
            bidirectional=True,
            dropout=self.model_config['dropout'] if self.model_config['dropout'] > 0 else 0
        )
        
        # Layer normalization after first GRU
        self.layernorm1 = nn.LayerNorm(self.model_config['gru_units_1'] * 2)  # *2 for bidirectional
        
        # Dropout after first GRU
        self.dropout1 = nn.Dropout(self.model_config['dropout'])
        
        # Second GRU layer (unidirectional)
        self.gru2 = nn.GRU(
            input_size=self.model_config['gru_units_1'] * 2,
            hidden_size=self.model_config['gru_units_2'],
            batch_first=True,
            bidirectional=False,
            dropout=self.model_config['dropout'] if self.model_config['dropout'] > 0 else 0
        )
        
        # Layer normalization after second GRU
        self.layernorm2 = nn.LayerNorm(self.model_config['gru_units_2'])
        
        # Dropout after second GRU
        self.dropout2 = nn.Dropout(self.model_config['dropout'])
        
        # Dense layers
        self.dense1 = nn.Linear(self.model_config['gru_units_2'], self.model_config['dense_units'])
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.model_config['dense_units'], 1)
        
        # Move to device
        if self.device is not None:
            self.to(self.device)
        
        return self
    
    def forward(self, x):
        """Forward pass through the model."""
        # First GRU layer
        gru1_out, _ = self.gru1(x)
        gru1_out = self.layernorm1(gru1_out)
        gru1_out = self.dropout1(gru1_out)
        
        # Second GRU layer (take only the last output)
        gru2_out, _ = self.gru2(gru1_out)
        gru2_out = gru2_out[:, -1, :]  # Take last time step
        gru2_out = self.layernorm2(gru2_out)
        gru2_out = self.dropout2(gru2_out)
        
        # Dense layers
        dense1_out = self.relu(self.dense1(gru2_out))
        output = self.dense2(dense1_out)
        
        return output
    
    def get_optimizer(self):
        """Get optimizer with configured learning rate."""
        return optim.Adam(self.parameters(), lr=self.model_config['learning_rate'])
    
    def get_config(self):
        """Get model configuration."""
        return self.model_config 