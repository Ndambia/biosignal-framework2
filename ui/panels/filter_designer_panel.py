from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
import numpy as np
from scipy import signal

from .base_panel import BaseControlPanel, NumericParameter, EnumParameter
from preprocessing_bio import SignalDenoising

class FilterDesignerPanel(BaseControlPanel):
    """Interactive filter design panel with real-time visualization."""
    
    filter_changed = pyqtSignal(dict)  # Emitted when filter parameters change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.denoiser = SignalDenoising()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        super()._init_ui()
        
        # Create main layout sections
        self.filter_controls = self.add_parameter_group("Filter Configuration")
        self.visualization = self.add_parameter_group("Filter Response")
        
        # Add filter type selector
        self.filter_type = EnumParameter("filter_type", [
            "Bandpass Filter",
            "Notch Filter",
            "Wavelet Denoising"
        ])
        self.add_parameter(self.filter_controls, "Filter Type:", self.filter_type)
        
        # Initialize parameter groups for each filter type
        self._init_bandpass_params()
        self._init_notch_params()
        self._init_wavelet_params()
        
        # Add visualization widgets
        self._init_visualization()
        
        # Add preview button
        self.preview_btn = QPushButton("Preview Filter Response")
        self.preview_btn.clicked.connect(self._update_visualization)
        self.layout.addWidget(self.preview_btn)
        
        # Connect signals
        self.filter_type.value_changed.connect(self._on_filter_type_changed)
        self.parameters_changed.connect(self._on_params_changed)
        
        # Show initial filter type
        self._on_filter_type_changed("filter_type", "Bandpass Filter")
        
    def _init_bandpass_params(self):
        """Initialize bandpass filter parameters."""
        self.bandpass_group = QGroupBox()
        layout = QVBoxLayout(self.bandpass_group)
        
        # Add parameters
        self.lowcut = NumericParameter("lowcut", 0.1, 500, 0.1, 1)
        self.highcut = NumericParameter("highcut", 0.1, 500, 0.1, 1)
        self.order = NumericParameter("order", 1, 10, 1, 0)
        
        self.add_parameter(self.bandpass_group, "Low Cutoff (Hz):", self.lowcut)
        self.add_parameter(self.bandpass_group, "High Cutoff (Hz):", self.highcut)
        self.add_parameter(self.bandpass_group, "Filter Order:", self.order)
        
        self.filter_controls.layout().addWidget(self.bandpass_group)
        
    def _init_notch_params(self):
        """Initialize notch filter parameters."""
        self.notch_group = QGroupBox()
        layout = QVBoxLayout(self.notch_group)
        
        # Add parameters
        self.center_freq = NumericParameter("center_freq", 45, 65, 1, 1)
        self.q_factor = NumericParameter("q_factor", 1, 100, 1, 1)
        
        self.add_parameter(self.notch_group, "Center Frequency (Hz):", self.center_freq)
        self.add_parameter(self.notch_group, "Q Factor:", self.q_factor)
        
        self.filter_controls.layout().addWidget(self.notch_group)
        self.notch_group.hide()
        
    def _init_wavelet_params(self):
        """Initialize wavelet denoising parameters."""
        self.wavelet_group = QGroupBox()
        layout = QVBoxLayout(self.wavelet_group)
        
        # Add parameters
        self.wavelet_type = EnumParameter("wavelet_type", ['db4', 'db6', 'sym4', 'coif3'])
        self.decomp_level = NumericParameter("decomp_level", 1, 10, 1, 0)
        
        self.add_parameter(self.wavelet_group, "Wavelet Type:", self.wavelet_type)
        self.add_parameter(self.wavelet_group, "Decomposition Level:", self.decomp_level)
        
        self.filter_controls.layout().addWidget(self.wavelet_group)
        self.wavelet_group.hide()
        
    def _init_visualization(self):
        """Initialize the filter response visualization."""
        plot_layout = QVBoxLayout()
        
        # Create frequency response plot
        self.freq_plot = pg.PlotWidget(title="Frequency Response")
        self.freq_plot.setLabel('left', 'Magnitude (dB)')
        self.freq_plot.setLabel('bottom', 'Frequency (Hz)')
        self.freq_plot.showGrid(x=True, y=True)
        
        # Create phase response plot
        self.phase_plot = pg.PlotWidget(title="Phase Response")
        self.phase_plot.setLabel('left', 'Phase (degrees)')
        self.phase_plot.setLabel('bottom', 'Frequency (Hz)')
        self.phase_plot.showGrid(x=True, y=True)
        
        plot_layout.addWidget(self.freq_plot)
        plot_layout.addWidget(self.phase_plot)
        self.visualization.setLayout(plot_layout)
        
    def _on_filter_type_changed(self, name: str, value: str):
        """Handle filter type changes."""
        # Hide all parameter groups
        self.bandpass_group.hide()
        self.notch_group.hide()
        self.wavelet_group.hide()
        
        # Show selected group
        if value == "Bandpass Filter":
            self.bandpass_group.show()
        elif value == "Notch Filter":
            self.notch_group.show()
        elif value == "Wavelet Denoising":
            self.wavelet_group.show()
            
        self._update_visualization()
        
    def _on_params_changed(self, params: dict):
        """Handle parameter changes."""
        self._update_visualization()
        self.filter_changed.emit(params)
        
    def _update_visualization(self):
        """Update the filter response visualization."""
        fs = 1000  # Sampling frequency
        nyq = fs / 2
        freqs = np.logspace(0, np.log10(nyq), 1000)
        
        filter_type = self.filter_type.get_value()
        
        if filter_type == "Bandpass Filter":
            # Calculate bandpass filter response
            low = self.lowcut.get_value()
            high = self.highcut.get_value()
            order = self.order.get_value()
            
            b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
            w, h = signal.freqz(b, a, worN=freqs, fs=fs)
            
            # Update plots
            self._plot_response(w, h)
            
        elif filter_type == "Notch Filter":
            # Calculate notch filter response
            freq = self.center_freq.get_value()
            q = self.q_factor.get_value()
            
            b, a = signal.iirnotch(freq/nyq, q)
            w, h = signal.freqz(b, a, worN=freqs, fs=fs)
            
            # Update plots
            self._plot_response(w, h)
            
        elif filter_type == "Wavelet Denoising":
            # For wavelet denoising, show wavelet shape instead
            wavelet = self.wavelet_type.get_value()
            level = self.decomp_level.get_value()
            
            # Clear previous plots
            self.freq_plot.clear()
            self.phase_plot.clear()
            
            # Show wavelet shape
            wavelet_func = signal.ricker if wavelet == 'mexh' else None
            if wavelet_func:
                points = np.linspace(-4, 4, 100)
                wavelet_shape = wavelet_func(points)
                self.freq_plot.plot(points, wavelet_shape, clear=True)
                self.freq_plot.setTitle(f"Wavelet Shape: {wavelet}")
            
    def _plot_response(self, w, h):
        """Plot frequency and phase response."""
        # Magnitude response
        self.freq_plot.clear()
        db = 20 * np.log10(np.abs(h))
        self.freq_plot.plot(w, db, clear=True)
        self.freq_plot.setLogMode(x=True, y=False)
        
        # Phase response
        self.phase_plot.clear()
        phase = np.unwrap(np.angle(h))
        phase = np.degrees(phase)
        self.phase_plot.plot(w, phase, clear=True)
        self.phase_plot.setLogMode(x=True, y=False)
        
    def get_filter_config(self) -> dict:
        """Get current filter configuration."""
        params = self.get_parameters()
        return {
            'type': params['filter_type'],
            'parameters': params
        }
        
    def reset_parameters(self):
        """Reset parameters to defaults."""
        if self.filter_type.get_value() == "Bandpass Filter":
            self.lowcut.set_value(20)
            self.highcut.set_value(450)
            self.order.set_value(4)
        elif self.filter_type.get_value() == "Notch Filter":
            self.center_freq.set_value(50)
            self.q_factor.set_value(30)
        else:  # Wavelet
            self.wavelet_type.set_value('db4')
            self.decomp_level.set_value(3)