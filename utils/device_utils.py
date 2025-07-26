# ==============================================================
# Device Detection and Optimization Utilities
# ==============================================================

import os
import torch
from torch.utils.data import DataLoader, TensorDataset


class DeviceManager:
    """Manages device detection and optimization."""
    
    def __init__(self, device_config=None):
        """Initialize device manager with configuration."""
        self.device_config = device_config or {}
        self.device = self._setup_device()
        self._setup_device_optimization()
    
    def _setup_device(self):
        """Set up the best available device (GPU or CPU)."""
        if not self.device_config.get('auto_detect_device', True):
            return torch.device('cpu')
        
        # Check for NVIDIA GPU (CUDA)
        if torch.cuda.is_available() and self.device_config.get('prefer_gpu', True):
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using NVIDIA GPU: {gpu_name}")
            return device
        
        # Check for Apple Silicon GPU (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.device_config.get('prefer_gpu', True):
            device = torch.device('mps')
            print(f"🚀 Using Apple Silicon GPU (MPS)")
            return device
        
        # Fallback to CPU
        device = torch.device('cpu')
        print(f"💻 Using CPU (no GPU available or GPU disabled)")
        return device
    
    def _setup_device_optimization(self):
        """Set up device-specific optimization."""
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
            cores_to_reserve = self.device_config.get('cpu_cores_to_reserve', 1)
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
    
    def get_optimal_batch_size(self):
        """Get optimal batch size for the device."""
        if self.device.type == 'cuda':
            return 256  # Larger batch sizes for GPU
        elif self.device.type == 'mps':
            return 128  # Medium batch sizes for Apple Silicon
        else:
            # CPU - scale with thread count
            base_batch_size = 64
            return min(256, base_batch_size * max(1, torch.get_num_threads() // 4))
    
    def create_dataloader(self, X, y, batch_size=None, shuffle=False):
        """Create PyTorch DataLoader with device-specific optimization."""
        if batch_size is None:
            batch_size = self.get_optimal_batch_size()
        
        # Device-specific optimization
        if self.device.type == 'cuda':
            # NVIDIA GPU optimization - keep tensors on CPU for pin_memory
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            num_workers = min(4, os.cpu_count() // 2)
            pin_memory = True
        elif self.device.type == 'mps':
            # Apple Silicon optimization
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            num_workers = 0  # MPS works better with num_workers=0
            pin_memory = False
        else:
            # CPU optimization
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            num_workers = min(4, max(1, torch.get_num_threads() // 2))
            pin_memory = False
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        ) 