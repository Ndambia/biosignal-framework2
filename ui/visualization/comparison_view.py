from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QCheckBox
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
import numpy as np
from scipy import signal

class ComparisonView(QWidget):
    """Split view for comparing original and processed signals."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout(self)
        
        # Create left panel (original signal)
        left_panel = QGroupBox("Original Signal")
        left_layout = QVBoxLayout(left_panel)
        
        self.original_plot = pg.PlotWidget(title="Time Domain")
        self.original_freq_plot = pg.PlotWidget(title="Frequency Domain")
        self.original_metrics = QLabel("Signal Metrics:\nNot available")
        
        left_layout.addWidget(self.original_plot)
        left_layout.addWidget(self.original_freq_plot)
        left_layout.addWidget(self.original_metrics)
        
        # Create right panel (processed signal)
        right_panel = QGroupBox("Processed Signal")
        right_layout = QVBoxLayout(right_panel)
        
        self.processed_plot = pg.PlotWidget(title="Time Domain")
        self.processed_freq_plot = pg.PlotWidget(title="Frequency Domain")
        self.processed_metrics = QLabel("Signal Metrics:\nNot available")
        
        right_layout.addWidget(self.processed_plot)
        right_layout.addWidget(self.processed_freq_plot)
        right_layout.addWidget(self.processed_metrics)
        
        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        # Add control panel at bottom
        control_panel = QGroupBox("Visualization Controls")
        control_layout = QHBoxLayout(control_panel)
        
        # Add view options
        self.sync_zoom = QCheckBox("Sync Zoom/Pan")
        self.sync_zoom.setChecked(True)
        self.show_grid = QCheckBox("Show Grid")
        self.show_grid.setChecked(True)
        
        # Add plot type selector
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "Time Domain",
            "Frequency Domain",
            "Time-Frequency",
            "Combined"
        ])
        
        control_layout.addWidget(QLabel("Plot Type:"))
        control_layout.addWidget(self.plot_type)
        control_layout.addWidget(self.sync_zoom)
        control_layout.addWidget(self.show_grid)
        
        # Add export button
        self.export_btn = QPushButton("Export Comparison")
        control_layout.addWidget(self.export_btn)
        
        layout.addWidget(control_panel)
        
        # Connect signals
        self.sync_zoom.toggled.connect(self._sync_views)
        self.show_grid.toggled.connect(self._toggle_grid)
        self.plot_type.currentTextChanged.connect(self._update_plot_type)
        self.export_btn.clicked.connect(self._export_comparison)
        
        # Initialize plots
        self._setup_plots()
        
    def _setup_plots(self):
        """Configure plot settings."""
        # Time domain plots
        self.original_plot.setLabel('left', 'Amplitude')
        self.original_plot.setLabel('bottom', 'Time (s)')
        self.processed_plot.setLabel('left', 'Amplitude')
        self.processed_plot.setLabel('bottom', 'Time (s)')
        
        # Frequency domain plots
        self.original_freq_plot.setLabel('left', 'Magnitude')
        self.original_freq_plot.setLabel('bottom', 'Frequency (Hz)')
        self.processed_freq_plot.setLabel('left', 'Magnitude')
        self.processed_freq_plot.setLabel('bottom', 'Frequency (Hz)')
        
        # Enable grid by default
        self._toggle_grid(True)
        
        # Link views if sync is enabled
        self._sync_views(True)
        
    def _sync_views(self, enabled: bool):
        """Synchronize view regions between plots."""
        if enabled:
            self.processed_plot.setXLink(self.original_plot)
            self.processed_plot.setYLink(self.original_plot)
            self.processed_freq_plot.setXLink(self.original_freq_plot)
            self.processed_freq_plot.setYLink(self.original_freq_plot)
        else:
            self.processed_plot.setXLink(None)
            self.processed_plot.setYLink(None)
            self.processed_freq_plot.setXLink(None)
            self.processed_freq_plot.setYLink(None)
            
    def _toggle_grid(self, show: bool):
        """Toggle grid visibility."""
        self.original_plot.showGrid(x=show, y=show)
        self.processed_plot.showGrid(x=show, y=show)
        self.original_freq_plot.showGrid(x=show, y=show)
        self.processed_freq_plot.showGrid(x=show, y=show)
        
    def _update_plot_type(self, plot_type: str):
        """Update plot type display."""
        if plot_type == "Time Domain":
            self.original_freq_plot.hide()
            self.processed_freq_plot.hide()
            self.original_plot.show()
            self.processed_plot.show()
        elif plot_type == "Frequency Domain":
            self.original_plot.hide()
            self.processed_plot.hide()
            self.original_freq_plot.show()
            self.processed_freq_plot.show()
        elif plot_type == "Combined":
            self.original_plot.show()
            self.processed_plot.show()
            self.original_freq_plot.show()
            self.processed_freq_plot.show()
            
    def _export_comparison(self):
        """Export comparison plots."""
        # Implementation depends on export format requirements
        pass
        
    def _calculate_metrics(self, data: np.ndarray) -> str:
        """Calculate signal quality metrics."""
        if len(data) == 0:
            return "Not available"
            
        metrics = []
        
        # Basic statistics
        metrics.append(f"Mean: {np.mean(data):.3f}")
        metrics.append(f"Std Dev: {np.std(data):.3f}")
        metrics.append(f"RMS: {np.sqrt(np.mean(data**2)):.3f}")
        
        # Signal-to-noise ratio (estimated)
        signal_power = np.mean(data**2)
        noise_power = np.var(data - np.mean(data))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        metrics.append(f"SNR: {snr:.1f} dB")
        
        return "\n".join(metrics)
        
    def update_original(self, data: np.ndarray, fs: float = 1000.0):
        """Update original signal display."""
        if len(data) == 0:
            return
            
        # Time domain plot
        time = np.arange(len(data)) / fs
        self.original_plot.clear()
        self.original_plot.plot(time, data)
        
        # Frequency domain plot
        freqs, psd = signal.welch(data, fs=fs)
        self.original_freq_plot.clear()
        self.original_freq_plot.plot(freqs, 10 * np.log10(psd))
        
        # Update metrics
        self.original_metrics.setText(
            "Signal Metrics:\n" + self._calculate_metrics(data)
        )
        
    def update_processed(self, data: np.ndarray, fs: float = 1000.0):
        """Update processed signal display."""
        if len(data) == 0:
            return
            
        # Time domain plot
        time = np.arange(len(data)) / fs
        self.processed_plot.clear()
        self.processed_plot.plot(time, data)
        
        # Frequency domain plot
        freqs, psd = signal.welch(data, fs=fs)
        self.processed_freq_plot.clear()
        self.processed_freq_plot.plot(freqs, 10 * np.log10(psd))
        
        # Update metrics
        self.processed_metrics.setText(
            "Signal Metrics:\n" + self._calculate_metrics(data)
        )