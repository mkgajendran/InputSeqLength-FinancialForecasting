# ==============================================================
# Enhanced Forecasting Visualization with Zoom Insets
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


class ForecastingPlotterWithZoom:
    """Handles plotting for forecasting experiments with zoom insets."""
    
    def __init__(self, config):
        """Initialize plotter with configuration."""
        self.config = config
        self.exp_config = config['experiments']['multisheet_forecasting']
        
        # Inset configuration (can be customized)
        self.inset_config = {
            'width': '42%',
            'height': '42%', 
            'location': 'lower right',
            'padding': 0.05,
            'show_connections': True,
            'background_color': None,  # No background for cleaner look
            'connection_color': '0.4',
            'connection_linewidth': 0.8
        }
        
    def plot_multisheet_forecasts(self, results_data, sheets):
        """Create multisheet forecasting plots with zoom insets."""
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24), sharex=False)
        axes = axes.flatten()
        
        for idx, sheet in enumerate(sheets):
            sheet_data = results_data[sheet]
            self._plot_single_sheet_with_zoom(
                axes[idx], sheet_data, sheet, idx, n_cols, n_rows
            )
        
        # Finalize plot
        self._finalize_plot(fig, axes, sheets, n_rows, n_cols)
        
        return fig

    def _plot_single_sheet_with_zoom(self, ax, sheet_data, sheet, idx, n_cols, n_rows):
        """Plot results for a single sheet with zoom inset."""
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
        
        # Plot lines with transparency for LSTM and GRU
        ax.plot(plot_dates, actual_close, lw=1.0, label="Actual")
        ax.plot(pred_lstm.index, pred_lstm.values, lw=1.2, label="LSTM", alpha=0.5)
        ax.plot(pred_gru.index, pred_gru.values, lw=1.2, label="GRU", alpha=0.5)
        
        # Add sheet name inside the subplot (top-left)
        ax.text(
            0.02, 0.98,
            sheet,
            transform=ax.transAxes,
            fontsize=20,
            fontweight='bold',
            va='top', ha='left',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle='round,pad=0.4',
                alpha=0.9,
                linewidth=1.2
            )
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # Set uniform tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Adjust y-axis limits to extend 45% lower
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        new_y_min = y_min - (y_range * 0.45)
        ax.set_ylim(new_y_min, y_max)
        
        # Create zoomed inset for test region
        self._add_test_zoom_inset(ax, df, pred_lstm, pred_gru, prediction_start)
        
        if idx % n_cols == 0:
            ax.set_ylabel("Price", fontsize=20, fontweight='bold')
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Date", fontsize=20, fontweight='bold')

    def _add_test_zoom_inset(self, ax, df, pred_lstm, pred_gru, prediction_start):
        """Add zoomed inset view of the test region."""
        # Create inset axes using configuration
        padding = self.inset_config['padding']
        inset = inset_axes(ax, 
                          width=self.inset_config['width'], 
                          height=self.inset_config['height'], 
                          loc=self.inset_config['location'], 
                          bbox_to_anchor=(padding, padding, 1, 1), 
                          bbox_transform=ax.transAxes)
        
        # Set inset to be on top layer with white background
        inset.set_zorder(10)
        inset.set_facecolor('white')
        inset.patch.set_alpha(1.0)  # Fully opaque background
        
        # Get test region data
        actual_close = df["Close"]
        plot_dates = df.index
        
        # Use prediction start as the beginning of the zoom region
        test_dates = plot_dates[prediction_start:]
        test_actual = actual_close[prediction_start:]
        
        # Plot test region data in inset with transparency for LSTM and GRU
        inset.plot(test_dates, test_actual, lw=1.0, color='#1f77b4', label="Actual")
        inset.plot(pred_lstm.index, pred_lstm.values, lw=1.2, color='#d62728', label="LSTM", alpha=0.5)
        inset.plot(pred_gru.index, pred_gru.values, lw=1.2, color='#2ca02c', label="GRU", alpha=0.5)
        
        # Combine actual + predictions for y-limits
        y_pred = pd.concat([pred_lstm.dropna(), pred_gru.dropna()])
        y_min = min(test_actual.min(), y_pred.min())
        y_max = max(test_actual.max(), y_pred.max())
        y_padding = 0.05 * (y_max - y_min)
        
        # Set inset limits
        inset.set_xlim(test_dates[0], test_dates[-1])
        inset.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Remove tick labels for cleaner look
        inset.set_xticks([])
        inset.set_yticks([])
        
        # Add smart legend inside the inset
        self._add_smart_legend(inset, test_dates, test_actual, pred_lstm, pred_gru)
        
        # Add connection lines from main plot to inset (if enabled)
        if self.inset_config['show_connections']:
            # Add only top 2 connection lines - cleaner look
            mark_inset(ax, inset, loc1=1, loc2=2,  # top-left to top-right
                      fc="none", 
                      ec=self.inset_config['connection_color'], 
                      lw=self.inset_config['connection_linewidth'])

    def _add_smart_legend(self, inset, test_dates, test_actual, pred_lstm, pred_gru):
        """Add legend with intelligent positioning to avoid plot intersection."""
        import numpy as np
        
        # Get data ranges for each corner
        y_min, y_max = test_actual.min(), test_actual.max()
        x_min, x_max = test_dates[0], test_dates[-1]
        
        # Calculate data density in each corner (normalized 0-1)
        x_range = len(test_dates)
        y_range = y_max - y_min
        
        # Define corner regions (first 25% and last 25% of each axis)
        x_split = x_range // 4
        y_split = y_range * 0.25
        
        # Get data points in each corner region
        corners_data = {
            'upper left': [],
            'upper right': [],
            'lower left': [],
            'lower right': []
        }
        
        # Combine all y-values for analysis
        all_y_values = []
        all_x_indices = []
        
        # Add actual data points
        for i, (date, value) in enumerate(zip(test_dates, test_actual)):
            all_y_values.append(value)
            all_x_indices.append(i)
        
        # Add LSTM predictions
        for date, value in zip(pred_lstm.index, pred_lstm.values):
            if date in test_dates:
                idx = list(test_dates).index(date)
                all_y_values.append(value)
                all_x_indices.append(idx)
        
        # Add GRU predictions  
        for date, value in zip(pred_gru.index, pred_gru.values):
            if date in test_dates:
                idx = list(test_dates).index(date)
                all_y_values.append(value)
                all_x_indices.append(idx)
        
        # Count data points in each corner
        for i, (x_idx, y_val) in enumerate(zip(all_x_indices, all_y_values)):
            if x_idx < x_split:  # Left side
                if y_val > y_max - y_split:  # Upper
                    corners_data['upper left'].append((x_idx, y_val))
                elif y_val < y_min + y_split:  # Lower
                    corners_data['lower left'].append((x_idx, y_val))
            elif x_idx > x_range - x_split:  # Right side
                if y_val > y_max - y_split:  # Upper
                    corners_data['upper right'].append((x_idx, y_val))
                elif y_val < y_min + y_split:  # Lower
                    corners_data['lower right'].append((x_idx, y_val))
        
        # Find corner with least data points
        corner_counts = {corner: len(points) for corner, points in corners_data.items()}
        best_corner = min(corner_counts, key=corner_counts.get)
        
        # Fallback order if the best corner still has too many points
        corner_priority = [best_corner, 'upper left', 'upper right', 'lower left', 'lower right']
        
        # Choose the first corner from priority list
        chosen_corner = corner_priority[0]
        
        # Add legend with chosen position
        inset.legend(['Actual', 'LSTM', 'GRU'], 
                    loc=chosen_corner, 
                    fontsize=9, 
                    frameon=True, 
                    fancybox=True, 
                    shadow=False,
                    framealpha=0.9,
                    edgecolor='gray',
                    facecolor='white')

    def _finalize_plot(self, fig, axes, sheets, n_rows, n_cols):
        """Finalize and save the plot."""
        # Global legend at top in single row
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="upper center", ncol=len(handles), 
                           bbox_to_anchor=(0.5, 0.96), prop={'weight': 'bold', 'size': 20})
        

        
        # Remove unused subplots
        for j in range(len(sheets), n_rows * n_cols):
            fig.delaxes(axes[j])
        
        fig.tight_layout(rect=[0, 0.02, 1, 0.94])
        
        # Add larger gap between columns
        fig.subplots_adjust(wspace=0.25)
        
        # Save plot in plots directory with zoom suffix
        output_dir = Path(self.exp_config['output_dir'])
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create zoom version filename
        original_filename = self.exp_config['plot_filename']
        name_parts = original_filename.split('.')
        zoom_filename = f"{name_parts[0]}_withzoom.{name_parts[1]}"
        plot_path = plots_dir / zoom_filename
        
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Zoomed plot saved to: {plot_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def customize_inset(self, **kwargs):
        """Customize inset appearance and behavior.
        
        Args:
            width (str): Width of inset (e.g., '42%', '50%')
            height (str): Height of inset (e.g., '42%', '50%') 
            location (str): Location ('upper right', 'upper left', 'lower right', 'lower left')
            padding (float): Padding from edges (0.05 = 5%)
            show_connections (bool): Show connection lines to main plot
            connection_color (str): Color of connection lines
            connection_linewidth (float): Width of connection lines
        """
        self.inset_config.update(kwargs)
        print(f"📊 Inset configuration updated: {kwargs}")
        print("   Rerun inference script to see changes!")
    
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