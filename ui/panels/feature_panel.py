from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QCheckBox, QProgressBar
)
from PyQt6.QtCore import pyqtSignal
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .base_panel import BaseControlPanel, NumericParameter, EnumParameter

class BaseFeaturePanel(BaseControlPanel):
    """Base class for all feature extraction panels."""
    
    # Signals
    feature_computed = pyqtSignal(str, np.ndarray)  # feature_name, values
    computation_progress = pyqtSignal(int)  # progress percentage
    computation_finished = pyqtSignal(pd.DataFrame)  # all features
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.features = {}  # Dictionary to store feature computation functions
        self.selected_features = set()  # Set of currently selected features
        self.cached_results = {}  # Cache for computed features
        
    def _init_ui(self):
        """Initialize the base UI components."""
        super()._init_ui()
        
        # Feature selection group
        self.feature_group = self.add_parameter_group("Features")
        self.feature_checkboxes = {}
        
        # Control buttons
        self.button_layout = QHBoxLayout()
        
        self.compute_btn = QPushButton("Compute Features")
        self.compute_btn.clicked.connect(self.compute_selected_features)
        self.button_layout.addWidget(self.compute_btn)
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        self.button_layout.addWidget(self.clear_cache_btn)

        self.export_btn = QPushButton("Export Features")
        self.export_btn.clicked.connect(self.export_features)
        self.button_layout.addWidget(self.export_btn)
        
        self.layout.addLayout(self.button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        
    def add_feature(self, name: str, compute_func, parameters: Dict[str, Any] = None):
        """Add a new feature computation function with optional parameters."""
        self.features[name] = {
            'function': compute_func,
            'parameters': parameters or {}
        }
        
        # Add checkbox for feature selection
        checkbox = QCheckBox(name)
        checkbox.toggled.connect(lambda checked: self._on_feature_toggled(name, checked))
        self.feature_checkboxes[name] = checkbox
        self.feature_group.layout().addWidget(checkbox)
        
        # Add parameter widgets if any
        if parameters:
            param_group = self.add_parameter_group(f"{name} Parameters")
            for param_name, param_config in parameters.items():
                widget = self._create_parameter_widget(param_name, param_config)
                if widget:
                    self.add_parameter(param_group, param_name, widget)
                    
    def _create_parameter_widget(self, name: str, config: Dict):
        """Create appropriate parameter widget based on configuration."""
        widget_type = config.get('type', 'numeric')
        
        if widget_type == 'numeric':
            return NumericParameter(
                name,
                config.get('min', 0),
                config.get('max', 100),
                config.get('step', 1),
                config.get('decimals', 2)
            )
        elif widget_type == 'enum':
            return EnumParameter(name, config.get('options', []))
        
        return None
        
    def _on_feature_toggled(self, name: str, checked: bool):
        """Handle feature selection/deselection."""
        if checked:
            self.selected_features.add(name)
        else:
            self.selected_features.discard(name)
            
    def compute_selected_features(self):
        """Compute all selected features."""
        if not self.selected_features:
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.compute_btn.setEnabled(False)
        
        # TODO: Implement actual computation in worker thread
        # For now, just simulate progress
        self.progress_bar.setValue(100)
        self.compute_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def clear_cache(self):
        """Clear the cached feature computation results."""
        self.cached_results.clear()
        
    def get_feature_parameters(self, feature_name: str) -> Dict[str, Any]:
        """Get current parameter values for a feature."""
        if feature_name not in self.features:
            return {}
            
        params = {}
        for param_name, param_config in self.features[feature_name]['parameters'].items():
            if param_name in self.parameters:
                params[param_name] = self.parameters[param_name].get_value()
        return params
        
    def get_selected_features(self) -> List[str]:
        """Get list of currently selected features."""
        return list(self.selected_features)

    def export_features(self):
        """Export computed features to a CSV file."""
        if not self.cached_results:
            print("No features computed to export.")
            return

        try:
            df = pd.DataFrame(self.cached_results)
            # TODO: Implement a QFileDialog to let the user choose the save path
            file_path = "exported_features.csv" # Placeholder
            df.to_csv(file_path, index=False)
            print(f"Features exported to {file_path}")
        except Exception as e:
            print(f"Error exporting features: {e}")
        
    def reset_parameters(self):
        """Reset all parameters to their default values."""
        super().reset_parameters()
        self.clear_cache()