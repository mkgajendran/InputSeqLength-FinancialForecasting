# ==============================================================
# Forecasting Visualization Utilities
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13
})

from pathlib import Path


class ForecastingPlotter:
    """Handles plotting for forecasting experiments."""
    
    def __init__(self, config):
        """Initialize plotter with configuration."""
        self.config = config
        self.exp_config = config['experiments']['multisheet_forecasting']
        
    def plot_multisheet_forecasts(self, results_data, sheets):
        """Create multisheet forecasting plots."""
        # Set up plotting
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24), sharex=False)
        axes = axes.flatten()
        
        for idx, sheet in enumerate(sheets):
            ax = axes[idx]
            sheet_data = results_data[sheet]
            
            self._plot_single_sheet(
                ax, sheet_data, sheet, idx, n_cols, n_rows
            )
        
        # Finalize plot
        self._finalize_plot(fig, axes, sheets, n_rows, n_cols)
        
        return fig
    

    
    def _plot_single_sheet(self, ax, sheet_data, sheet, idx, n_cols, n_rows):
        """Plot results for a single sheet."""
        df = sheet_data['dataframe']
        pred_lstm = sheet_data['pred_lstm']
        pred_gru = sheet_data['pred_gru']
        train_sz = sheet_data['train_size']
        val_sz = sheet_data['val_size']
        
        actual_close = df["Close"]
        plot_dates = df.index
        test_start = train_sz + val_sz
        
        # Get the actual prediction start point (where predictions begin)
        # Predictions start past_steps into the test data due to sequence generation
        past_steps = 60  # This should match the model's past_steps
        prediction_start = min(len(plot_dates) - 1, test_start + past_steps)
        
        # Add background shading with intuitive colors and patterns
        # Train: Blue (learning phase)
        ax.axvspan(plot_dates[0], plot_dates[train_sz-1], 
                  facecolor="#e3f2fd", alpha=0.4, hatch="|||", ec="#1976d2", lw=0.5, 
                  label="Train" if idx == 0 else None)
        
        # Validation: Orange (tuning phase)  
        ax.axvspan(plot_dates[train_sz], plot_dates[test_start-1], 
                  facecolor="#fff3e0", alpha=0.4, hatch="---", ec="#f57c00", lw=0.5, 
                  label="Val" if idx == 0 else None)
        
        # Lookback: Light gray (preparation phase - data used but not predicted)
        if test_start < prediction_start:
            ax.axvspan(plot_dates[test_start], plot_dates[prediction_start-1], 
                      facecolor="#f5f5f5", alpha=0.5, hatch="...", ec="#757575", lw=0.5, 
                      label="Lookback" if idx == 0 else None)
        
        # Test: Red (evaluation phase - actual predictions)
        if prediction_start < len(plot_dates):
            ax.axvspan(plot_dates[prediction_start], plot_dates[-1], 
                      facecolor="#ffebee", alpha=0.4, hatch="xxx", ec="#d32f2f", lw=0.5, 
                      label="Test" if idx == 0 else None)
        
        # Plot lines
        ax.plot(plot_dates, actual_close, lw=1.0, label="Actual")
        ax.plot(pred_lstm.index, pred_lstm.values, lw=1.2, label="LSTM")
        ax.plot(pred_gru.index, pred_gru.values, lw=1.2, label="GRU")
        
        # Draw shadow (gray, slightly offset, no box fill)
        ax.text(
            0.024, 0.984,  # Slightly offset from main text
            sheet,
            transform=ax.transAxes,
            fontweight='bold',
            va='top', ha='left',
            color='gray',
            alpha=0.4,
            bbox=dict(
                facecolor='none',
                edgecolor='none',
                boxstyle='round,pad=0.4',
                linewidth=0
            )
        )
        # Draw main label (white box, black border)
        ax.text(
            0.02, 0.98,
            sheet,
            transform=ax.transAxes,
            fontweight='bold',
            va='top', ha='left',
            color='black',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.4',
                alpha=0.9,
                linewidth=1.2
            )
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        
        if idx % n_cols == 0:
            ax.set_ylabel("Price", fontweight='bold')
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Date", fontweight='bold')
    

    
    def _finalize_plot(self, fig, axes, sheets, n_rows, n_cols):
        """Finalize and save the plot."""
        # Global legend at top in single row
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles, labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.96),
            prop={'weight': 'bold', 'size': 15},
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95
        )
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_facecolor('white')
        
        # Remove unused subplots
        for j in range(len(sheets), n_rows * n_cols):
            fig.delaxes(axes[j])
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # Save plot in plots directory
        output_dir = Path(self.exp_config['output_dir'])
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / self.exp_config['plot_filename']
        
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_from_saved_results(self, results_file):
        """Create plots from saved training results."""
        import pickle
        
        # Load saved results
        with open(results_file, 'rb') as f:
            saved_results = pickle.load(f)
        
        sheets = list(saved_results.keys())
        
        # Create plots
        fig = self.plot_multisheet_forecasts(saved_results, sheets)
        
        return fig 