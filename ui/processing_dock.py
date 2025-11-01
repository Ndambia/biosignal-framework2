from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QGroupBox, QFormLayout, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
from .data_manager import DataManager

class ProcessingDock(QDockWidget):
    """Dock widget for configuring the signal processing pipeline."""
    
    # Signals for pipeline configuration changes
    filter_changed = pyqtSignal(dict)
    features_changed = pyqtSignal(dict)
    model_changed = pyqtSignal(dict)
    process_requested = pyqtSignal()
    export_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Signal Processing", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                            Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Create sub-sections
        self._create_filter_section()
        self._create_features_section()
        self._create_model_section()
        self._create_control_section()
        self._create_export_section()
        
        self.setWidget(self.main_widget)
        
    def _create_filter_section(self):
        """Create the filter configuration section."""
        filter_group = QGroupBox("Signal Filtering")
        filter_layout = QFormLayout()
        
        # Filter type selection
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Bandpass", "Notch", "Wavelet"])
        filter_layout.addRow("Filter Type:", self.filter_type)
        
        # Filter parameters
        self.low_freq = QDoubleSpinBox()
        self.low_freq.setRange(0.1, 500.0)
        self.low_freq.setValue(20.0)
        filter_layout.addRow("Low Cutoff (Hz):", self.low_freq)
        
        self.high_freq = QDoubleSpinBox()
        self.high_freq.setRange(0.1, 500.0)
        self.high_freq.setValue(200.0)
        filter_layout.addRow("High Cutoff (Hz):", self.high_freq)
        
        self.filter_order = QSpinBox()
        self.filter_order.setRange(1, 10)
        self.filter_order.setValue(4)
        filter_layout.addRow("Filter Order:", self.filter_order)
        
        # Connect signals
        self.filter_type.currentTextChanged.connect(self._update_filter_config)
        self.low_freq.valueChanged.connect(self._update_filter_config)
        self.high_freq.valueChanged.connect(self._update_filter_config)
        self.filter_order.valueChanged.connect(self._update_filter_config)
        
        filter_group.setLayout(filter_layout)
        self.main_layout.addWidget(filter_group)
        
    def _create_features_section(self):
        """Create the feature extraction configuration section."""
        features_group = QGroupBox("Feature Extraction")
        features_layout = QFormLayout()
        
        # Feature type selection
        self.feature_type = QComboBox()
        self.feature_type.addItems(["Time Domain", "Frequency Domain", "Nonlinear"])
        features_layout.addRow("Feature Type:", self.feature_type)
        
        # Feature parameters
        self.window_size = QDoubleSpinBox()
        self.window_size.setRange(0.1, 10.0)
        self.window_size.setValue(1.0)
        features_layout.addRow("Window Size (s):", self.window_size)
        
        self.overlap = QDoubleSpinBox()
        self.overlap.setRange(0.0, 0.99)
        self.overlap.setValue(0.5)
        features_layout.addRow("Overlap:", self.overlap)
        
        # Connect signals
        self.feature_type.currentTextChanged.connect(self._update_features_config)
        self.window_size.valueChanged.connect(self._update_features_config)
        self.overlap.valueChanged.connect(self._update_features_config)
        
        features_group.setLayout(features_layout)
        self.main_layout.addWidget(features_group)
        
    def _create_model_section(self):
        """Create the model configuration section."""
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        # Model type selection
        self.model_type = QComboBox()
        self.model_type.addItems(["SVM", "Random Forest", "CNN", "LSTM"])
        model_layout.addRow("Model Type:", self.model_type)
        
        # Model parameters (basic for now, can be expanded)
        self.model_config = QComboBox()
        self.model_config.addItems(["Default", "Custom"])
        model_layout.addRow("Configuration:", self.model_config)
        
        # Connect signals
        self.model_type.currentTextChanged.connect(self._update_model_config)
        self.model_config.currentTextChanged.connect(self._update_model_config)
        
        model_group.setLayout(model_layout)
        self.main_layout.addWidget(model_group)
        
    def _create_control_section(self):
        """Create the processing control section."""
        control_group = QGroupBox("Processing Controls")
        control_layout = QHBoxLayout()
        
        # Process button
        self.process_btn = QPushButton("Process Signal")
        self.process_btn.clicked.connect(self.process_requested.emit)
        
        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_configuration)
        
        # Clear cache button
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.clear_cache_btn)
        
        control_group.setLayout(control_layout)
        self.main_layout.addWidget(control_group)
        
    def _create_export_section(self):
        """Create the export controls section."""
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout()
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)  # Enable after processing
        
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        self.main_layout.addWidget(export_group)
        
    def _clear_cache(self):
        """Clear the processing cache."""
        self.data_manager.clear_cache()
        
    def _export_results(self):
        """Handle export button click."""
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if export_dir:
            self.export_requested.emit()
            
    def enable_export(self, enabled: bool = True):
        """Enable or disable export functionality."""
        self.export_btn.setEnabled(enabled)
        
    def _update_filter_config(self):
        """Update and emit filter configuration."""
        config = {
            'type': self.filter_type.currentText(),
            'low_freq': self.low_freq.value(),
            'high_freq': self.high_freq.value(),
            'order': self.filter_order.value()
        }
        self.filter_changed.emit(config)
        
    def _update_features_config(self):
        """Update and emit features configuration."""
        config = {
            'type': self.feature_type.currentText(),
            'window_size': self.window_size.value(),
            'overlap': self.overlap.value()
        }
        self.features_changed.emit(config)
        
    def _update_model_config(self):
        """Update and emit model configuration."""
        config = {
            'type': self.model_type.currentText(),
            'config_type': self.model_config.currentText()
        }
        self.model_changed.emit(config)
        
    def _reset_configuration(self):
        """Reset all configurations to default values."""
        self.filter_type.setCurrentText("Bandpass")
        self.low_freq.setValue(20.0)
        self.high_freq.setValue(200.0)
        self.filter_order.setValue(4)
        
        self.feature_type.setCurrentText("Time Domain")
        self.window_size.setValue(1.0)
        self.overlap.setValue(0.5)
        
        self.model_type.setCurrentText("SVM")
        self.model_config.setCurrentText("Default")
        
        # Emit updated configurations
        self._update_filter_config()
        self._update_features_config()
        self._update_model_config()