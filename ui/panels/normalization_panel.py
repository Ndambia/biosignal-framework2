from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
import numpy as np

from .base_panel import BaseControlPanel, NumericParameter, EnumParameter, BoolParameter
from preprocessing_bio import SignalNormalization

class NormalizationPanel(BaseControlPanel):
    """Panel for signal normalization with statistics display."""
    
    normalization_changed = pyqtSignal(dict)  # Emitted when normalization parameters change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.normalizer = SignalNormalization()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        super()._init_ui()
        
        # Create main layout sections
        self.norm_controls = self.add_parameter_group("Normalization Settings")
        self.stats_display = self.add_parameter_group("Signal Statistics")
        
        # Add normalization method selector
        self.norm_method = EnumParameter("norm_method", [
            "Z-score",
            "Min-Max",
            "Robust"
        ])
        self.add_parameter(self.norm_controls, "Method:", self.norm_method)
        
        # Initialize parameter groups for each method
        self._init_zscore_params()
        self._init_minmax_params()
        self._init_robust_params()
        
        # Add statistics visualization
        self._init_statistics()
        
        # Add preview button
        self.preview_btn = QPushButton("Preview Normalization")
        self.preview_btn.clicked.connect(self._update_preview)
        self.layout.addWidget(self.preview_btn)
        
        # Connect signals
        self.norm_method.value_changed.connect(self._on_method_changed)
        self.parameters_changed.connect(self._on_params_changed)
        
        # Show initial method
        self._on_method_changed("norm_method", "Z-score")
        
    def _init_zscore_params(self):
        """Initialize z-score normalization parameters."""
        self.zscore_group = QGroupBox()
        layout = QVBoxLayout(self.zscore_group)
        
        # Add parameters
        self.zscore_robust = BoolParameter("zscore_robust", "Use robust statistics")
        self.add_parameter(self.zscore_group, "", self.zscore_robust)
        
        self.filter_controls.layout().addWidget(self.zscore_group)
        
    def _init_minmax_params(self):
        """Initialize min-max scaling parameters."""
        self.minmax_group = QGroupBox()
        layout = QVBoxLayout(self.minmax_group)
        
        # Add parameters
        self.feature_min = NumericParameter("feature_min", -10, 10, 0.1, 2)
        self.feature_max = NumericParameter("feature_max", -10, 10, 0.1, 2)
        
        self.add_parameter(self.minmax_group, "Min Value:", self.feature_min)
        self.add_parameter(self.minmax_group, "Max Value:", self.feature_max)
        
        self.filter_controls.layout().addWidget(self.minmax_group)
        self.minmax_group.hide()
        
    def _init_robust_params(self):
        """Initialize robust scaling parameters."""
        self.robust_group = QGroupBox()
        layout = QVBoxLayout(self.robust_group)
        
        # Add parameters
        self.quantile_range = NumericParameter("quantile_range", 0, 100, 1, 0)
        self.add_parameter(self.robust_group, "Quantile Range (%):", self.quantile_range)
        
        self.filter_controls.layout().addWidget(self.robust_group)
        self.robust_group.hide()
        
    def _init_statistics(self):
        """Initialize the statistics display."""
        layout = QVBoxLayout()
        
        # Create statistics display
        self.stats_plot = pg.PlotWidget(title="Value Distribution")
        self.stats_plot.setLabel('left', 'Frequency')
        self.stats_plot.setLabel('bottom', 'Value')
        self.stats_plot.showGrid(x=True, y=True)
        
        # Create statistics labels
        self.stats_layout = QVBoxLayout()
        self.mean_label = QLabel("Mean: N/A")
        self.std_label = QLabel("Std Dev: N/A")
        self.min_label = QLabel("Min: N/A")
        self.max_label = QLabel("Max: N/A")
        self.median_label = QLabel("Median: N/A")
        self.iqr_label = QLabel("IQR: N/A")
        
        # Add labels to layout
        stats_widget = QWidget()
        stats_widget.setLayout(self.stats_layout)
        self.stats_layout.addWidget(self.mean_label)
        self.stats_layout.addWidget(self.std_label)
        self.stats_layout.addWidget(self.min_label)
        self.stats_layout.addWidget(self.max_label)
        self.stats_layout.addWidget(self.median_label)
        self.stats_layout.addWidget(self.iqr_label)
        
        layout.addWidget(self.stats_plot)
        layout.addWidget(stats_widget)
        self.stats_display.setLayout(layout)
        
    def _on_method_changed(self, name: str, value: str):
        """Handle normalization method changes."""
        # Hide all parameter groups
        self.zscore_group.hide()
        self.minmax_group.hide()
        self.robust_group.hide()
        
        # Show selected group
        if value == "Z-score":
            self.zscore_group.show()
        elif value == "Min-Max":
            self.minmax_group.show()
        elif value == "Robust":
            self.robust_group.show()
            
        self._update_preview()
        
    def _on_params_changed(self, params: dict):
        """Handle parameter changes."""
        self._update_preview()
        self.normalization_changed.emit(params)
        
    def _update_preview(self):
        """Update the statistics preview."""
        # This would be connected to actual signal data in practice
        # For now, generate sample data for visualization
        data = np.random.normal(0, 1, 1000)
        
        # Update histogram
        self.stats_plot.clear()
        y, x = np.histogram(data, bins=50)
        self.stats_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0,0,255,150))
        
        # Update statistics
        self.mean_label.setText(f"Mean: {np.mean(data):.3f}")
        self.std_label.setText(f"Std Dev: {np.std(data):.3f}")
        self.min_label.setText(f"Min: {np.min(data):.3f}")
        self.max_label.setText(f"Max: {np.max(data):.3f}")
        self.median_label.setText(f"Median: {np.median(data):.3f}")
        q75, q25 = np.percentile(data, [75, 25])
        self.iqr_label.setText(f"IQR: {q75 - q25:.3f}")
        
    def get_normalization_config(self) -> dict:
        """Get current normalization configuration."""
        params = self.get_parameters()
        return {
            'method': params['norm_method'],
            'parameters': params
        }
        
    def reset_parameters(self):
        """Reset parameters to defaults."""
        if self.norm_method.get_value() == "Z-score":
            self.zscore_robust.set_value(False)
        elif self.norm_method.get_value() == "Min-Max":
            self.feature_min.set_value(0)
            self.feature_max.set_value(1)
        else:  # Robust
            self.quantile_range.set_value(50)
            
    def update_statistics(self, data: np.ndarray):
        """Update statistics display with actual signal data."""
        if data is None or len(data) == 0:
            return
            
        # Update histogram
        self.stats_plot.clear()
        y, x = np.histogram(data, bins=50)
        self.stats_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0,0,255,150))
        
        # Update statistics
        self.mean_label.setText(f"Mean: {np.mean(data):.3f}")
        self.std_label.setText(f"Std Dev: {np.std(data):.3f}")
        self.min_label.setText(f"Min: {np.min(data):.3f}")
        self.max_label.setText(f"Max: {np.max(data):.3f}")
        self.median_label.setText(f"Median: {np.median(data):.3f}")
        q75, q25 = np.percentile(data, [75, 25])
        self.iqr_label.setText(f"IQR: {q75 - q25:.3f}")