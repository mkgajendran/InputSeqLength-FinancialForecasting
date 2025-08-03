# InputSeqLength-FinancialForecasting

## 📊 Financial Time Series Forecasting with Deep Learning

A comprehensive Python project analyzing the effects of input sequence length and window size on financial forecasting using LSTM and GRU models across multiple global indices.

## 🎯 Project Overview

This project implements and compares deep learning models (LSTM and GRU) for financial time series forecasting, specifically focusing on:
- **Input sequence length optimization**
- **Multi-index comparative analysis**
- **Advanced visualization with zoomed insets**
- **Comprehensive performance metrics**

## 🌟 Key Features

### 📈 Data & Models
- **Dynamic data download** from Yahoo Finance for 7 global indices
- **LSTM and GRU models** with configurable architectures
- **Smart device optimization** (NVIDIA CUDA, Apple Silicon MPS, optimized CPU)
- **Reproducible experiments** with fixed random seeds

### 🎨 Advanced Visualizations
- **Multi-sheet forecasting plots** with train/validation/test demarcation
- **Zoomed inset views** of prediction regions with intelligent positioning
- **Smart legends** that avoid data overlap
- **Professional styling** with uniform fonts and transparency effects

### ⚙️ Configuration-Driven
- **YAML-based configuration** for easy parameter tuning
- **Modular architecture** with separate training and inference
- **Flexible model parameters** and experiment settings

## 📊 Results

### Global Indices Trend Analysis
The project analyzes 7 major global financial indices:
- **BSESensex** (India)
- **Nifty50** (India) 
- **Nikkei** (Japan)
- **DAX** (Germany)
- **Shanghai** (China)
- **NASDAQ** (USA)
- **NYSE** (USA)

### Sample Visualizations

#### 📈 Indices Trend Analysis
![Indices Trend Analysis](results/plots/indices_trend_analysis.pdf)

#### 🔍 Multisheet Forecasting Results
![Forecasting Results](results/plots/multisheet_forecasts.pdf)

#### 🔎 Enhanced Forecasting with Zoom Insets
![Enhanced Forecasting with Zoom](results/plots/multisheet_forecasts_withzoom.pdf)

### 📊 Performance Metrics
The project generates comprehensive metrics including:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (R-squared Score)

Results are saved in `results/metrics/multisheet_test_metrics.csv`

## 🚀 Quick Start

### Prerequisites
```bash
# Create virtual environment
python -m venv financial_forecasting_env
source financial_forecasting_env/bin/activate  # On Windows: financial_forecasting_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run complete pipeline
python main.py

# Generate only plots (after training)
python experiments/inference_multisheet_forecasting.py

# Generate enhanced plots with zoom insets
python experiments/inference_multisheet_forecasting_withzoom.py
```

## 📁 Project Structure

```
InputSeqLength-FinancialForecasting/
├── 📊 data/
│   └── downloader.py              # Dynamic data download from Yahoo Finance
├── 🧠 models/
│   ├── lstm_model.py             # PyTorch LSTM implementation
│   └── gru_model.py              # PyTorch GRU implementation
├── 🔬 experiments/
│   ├── train_multisheet_forecasting.py           # Training script
│   ├── inference_multisheet_forecasting.py       # Standard inference
│   ├── inference_multisheet_forecasting_withzoom.py  # Enhanced inference
│   └── descriptive_statistics_closing.py         # Statistical analysis
├── 🔧 utils/
│   ├── data_preprocessing.py     # Data preparation utilities
│   └── device_utils.py           # Device optimization
├── 📈 visualization/
│   ├── indices_trend.py          # Trend analysis plots
│   ├── forecasting_plots.py      # Standard forecasting plots
│   └── forecasting_plots_withzoom.py  # Enhanced plots with zoom insets
├── 📊 results/
│   ├── plots/                    # Generated visualizations
│   ├── metrics/                  # Performance metrics
│   ├── models/                   # Saved model weights
│   └── training_results/         # Training artifacts
├── 📄 config.yaml               # Configuration file
├── 📄 main.py                   # Main execution script
└── 📄 requirements.txt          # Dependencies
```

## ⚙️ Configuration

The project uses `config.yaml` for centralized configuration:

```yaml
# Data Configuration
data:
  start_date: "2017-04-01"
  end_date: "2025-03-31"

# Model Parameters
models:
  lstm:
    past_steps: 60
    units_1: 128
    units_2: 64
    dropout: 0.2
    learning_rate: 0.001
  
# Experiment Settings
experiments:
  multisheet_forecasting:
    train_split: 0.70
    val_split: 0.15
    test_split: 0.15
    device_optimization:
      auto_detect_device: true
      prefer_gpu: true
      cpu_cores_to_reserve: 1
```

## 🎨 Visualization Features

### Enhanced Plotting Capabilities
- **Background shading** for train/validation/lookback/test regions
- **Intelligent legend positioning** to avoid data overlap
- **Zoomed inset views** of prediction regions
- **Professional styling** with uniform fonts and transparency
- **Connection lines** linking main plots to zoom insets

### Customization Options
```python
# Customize zoom insets
plotter.customize_inset(
    width='50%',
    height='50%',
    location='lower left',
    show_connections=True
)
```

## 📊 Model Performance

The project implements comprehensive evaluation across multiple metrics:
- **Time series accuracy** with proper train/validation/test splits
- **Cross-index comparison** to identify model strengths
- **Statistical significance** testing
- **Visual performance assessment** with detailed plots

## 🔧 Advanced Features

### Device Optimization
- **Automatic GPU detection** (NVIDIA CUDA, Apple Silicon MPS)
- **CPU optimization** with core reservation
- **Memory management** for large datasets

### Reproducibility
- **Fixed random seeds** across all libraries
- **Deterministic operations** for consistent results
- **Version-controlled configurations**

## 📈 Future Enhancements

- [ ] Additional model architectures (Transformer, CNN-LSTM)
- [ ] Hyperparameter optimization with Optuna
- [ ] Real-time prediction capabilities
- [ ] Interactive web dashboard
- [ ] Extended technical indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing financial data
- **PyTorch** team for the deep learning framework
- **Matplotlib** for visualization capabilities
- **Scikit-learn** for preprocessing utilities

---

**⭐ If you find this project useful, please consider giving it a star!** 