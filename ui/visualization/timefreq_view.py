from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
import pyqtgraph as pg
import numpy as np
from scipy import signal
from .base_view import BaseVisualizationView

class TimeFrequencyView(BaseVisualizationView):
    """View for time-frequency analysis visualization."""
    
    # Window functions
    WINDOWS = {
        'Hann': signal.windows.hann,
        'Hamming': signal.windows.hamming,
        'Blackman': signal.windows.blackman,
        'Kaiser': signal.windows.kaiser,
        'Rectangular': signal.windows.boxcar
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_timefreq_plot()
        
    def _setup_timefreq_plot(self):
        """Set up time-frequency plotting."""
        # Add time-frequency specific toolbar controls
        self.toolbar.addSeparator()
        
        # Window type selection
        self.toolbar.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(list(self.WINDOWS.keys()))
        self.window_combo.currentTextChanged.connect(self._update_spectrogram)
        self.toolbar.addWidget(self.window_combo)
        
        # Window size control
        self.toolbar.addWidget(QLabel("Size:"))
        self.window_size = QSpinBox()
        self.window_size.setRange(32, 4096)
        self.window_size.setValue(256)
        self.window_size.setSingleStep(32)
        self.window_size.valueChanged.connect(self._update_spectrogram)
        self.toolbar.addWidget(self.window_size)
        
        # Overlap control
        self.toolbar.addWidget(QLabel("Overlap:"))
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 90)
        self.overlap.setValue(50)
        self.overlap.setSuffix("%")
        self.overlap.valueChanged.connect(self._update_spectrogram)
        self.toolbar.addWidget(self.overlap)
        
        # Color map selection
        self.toolbar.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'magma', 'inferno'])
        self.colormap_combo.currentTextChanged.connect(self._update_colormap)
        self.toolbar.addWidget(self.colormap_combo)
        
        # Scale selection
        self.toolbar.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['Linear', 'Log'])
        self.scale_combo.currentTextChanged.connect(self._update_scale)
        self.toolbar.addWidget(self.scale_combo)
        
        # Set up plot
        self.plot.setLabel('bottom', 'Time', 's')
        self.plot.setLabel('left', 'Frequency', 'Hz')
        
        # Create image item for spectrogram
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        
        # Add colorbar
        self.colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap='viridis',
            label='Power (dB)'
        )
        self.colorbar.setImageItem(self.image_item)
        
        # Store data
        self.signal_data = None
        self.sampling_rate = None
        
    def set_data(self, data: np.ndarray, sampling_rate: float = 1000.0):
        """Set signal data and compute spectrogram."""
        self.signal_data = data
        self.sampling_rate = sampling_rate
        self._update_spectrogram()
        
    def _update_spectrogram(self):
        """Update spectrogram with current parameters."""
        if self.signal_data is None:
            return
            
        # Get parameters
        window_type = self.window_combo.currentText()
        window_size = self.window_size.value()
        overlap = self.overlap.value() / 100.0
        
        # Create window
        if window_type == 'Kaiser':
            window = self.WINDOWS[window_type](window_size, beta=8.6)
        else:
            window = self.WINDOWS[window_type](window_size)
            
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            self.signal_data,
            fs=self.sampling_rate,
            window=window,
            nperseg=window_size,
            noverlap=int(window_size * overlap),
            scaling='density'
        )
        
        # Convert to dB
        Sxx = 10 * np.log10(Sxx + 1e-10)
        
        # Update image
        self.image_item.setImage(Sxx)
        
        # Set proper scaling
        self.image_item.scale(
            times[-1] / Sxx.shape[1],
            frequencies[-1] / Sxx.shape[0]
        )
        
        # Update colorbar
        self.colorbar.setLevels((np.min(Sxx), np.max(Sxx)))
        
        # Auto range
        self.plot_widget.getViewBox().autoRange()
        
        # Update status
        self.update_status(
            f"Window: {window_type}, Size: {window_size}, "
            f"Overlap: {overlap:.0%}, Freq range: [0, {frequencies[-1]:.0f}] Hz"
        )
        
    def _update_colormap(self, colormap: str):
        """Update colormap."""
        self.colorbar.setColorMap(colormap)
        
    def _update_scale(self, scale: str):
        """Update frequency scale."""
        if scale == 'Log':
            self.plot.setLogMode(y=True)
        else:
            self.plot.setLogMode(y=False)
            
    def clear(self):
        """Clear all data."""
        super().clear()
        self.signal_data = None
        self.sampling_rate = None
        self.image_item.clear()
        
    def _export_view(self):
        """Export spectrogram as image."""
        # Get current view
        view_box = self.plot_widget.getViewBox()
        view_rect = view_box.viewRect()
        
        # Create exporter
        exporter = pg.exporters.ImageExporter(self.plot_widget)
        exporter.parameters()['width'] = 1000
        
        # Export
        exporter.export('spectrogram.png')
        
    def _copy_view(self):
        """Copy spectrogram to clipboard."""
        # Get current view
        view_box = self.plot_widget.getViewBox()
        view_rect = view_box.viewRect()
        
        # Create exporter
        exporter = pg.exporters.ImageExporter(self.plot_widget)
        exporter.parameters()['width'] = 1000
        
        # Export to clipboard
        exporter.export(copy=True)