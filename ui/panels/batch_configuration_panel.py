from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QFileDialog, QLabel, QSpinBox,
    QComboBox, QCheckBox, QFormLayout, QListWidget,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any, List, Optional
import os

from .base_panel import BaseControlPanel, ParameterWidget, NumericParameter, EnumParameter
from ..error_handling import ErrorHandler, ErrorCategory, ErrorSeverity

class BatchConfigurationPanel(BaseControlPanel):
    """Panel for configuring batch processing settings."""
    
    configuration_complete = pyqtSignal()  # Emitted when configuration is complete
    
    def __init__(self, error_handler: ErrorHandler, parent=None):
        super().__init__(parent)
        self.error_handler = error_handler
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Dataset configuration
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QVBoxLayout(dataset_group)
        self.dataset_config = DatasetConfig("dataset")
        dataset_layout.addWidget(self.dataset_config)
        layout.addWidget(dataset_group)
        
        # Add configuration complete button
        self.complete_button = QPushButton("Complete Configuration")
        self.complete_button.clicked.connect(self._on_configuration_complete)
        layout.addWidget(self.complete_button)
        
    def _on_configuration_complete(self):
        """Handle configuration complete button click."""
        # Here you would typically validate the configuration
        # For now, just emit the signal
        self.configuration_complete.emit()

class DatasetConfig(ParameterWidget):
    """Widget for configuring dataset paths and options."""
    
    def __init__(self, name: str, parent=None):
        super().__init__(name, parent)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Dataset path selection
        path_layout = QHBoxLayout()
        self.path_label = QLabel("No files selected")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_files)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        
        # Selected files list
        self.files_list = QListWidget()
        layout.addWidget(self.files_list)
        
        self.paths: List[str] = []
        
    def _browse_files(self):
        """Open file dialog for selecting dataset files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Dataset Files",
            "",
            "All Files (*);;HDF5 Files (*.h5);;CSV Files (*.csv)"
        )
        
        if files:
            self.paths = files
            self.files_list.clear()
            for file in files:
                item = QListWidgetItem(os.path.basename(file))
                item.setToolTip(file)
                self.files_list.addItem(item)
            self.path_label.setText(f"{len(files)} files selected")
            self.value_changed.emit(self.name, self.paths)
            
    def get_value(self) -> List[str]:
        return self.paths
        
    def set_value(self, value: List[str]):
        self.paths = value
        self.files_list.clear()
        for file in value:
            item = QListWidgetItem(os.path.basename(file))
            item.setToolTip(file)
            self.files_list.addItem(item)
        self.path_label.setText(f"{len(value)} files selected")

class BatchConfigurationPanel(BaseControlPanel):
    """Panel for configuring batch processing parameters."""
    
    config_changed = pyqtSignal(dict)  # Emitted when configuration changes
    
    def __init__(self, error_handler: ErrorHandler, parent=None):
        self.error_handler = error_handler
        super().__init__(parent)
        
    def _init_ui(self):
        """Initialize the configuration interface."""
        super()._init_ui()
        
        # Dataset Configuration
        dataset_group = self.add_parameter_group("Dataset Configuration")
        self.dataset_config = DatasetConfig("dataset_paths")
        self.add_parameter(dataset_group, "Input Files", self.dataset_config)
        
        # Model Configuration
        model_group = self.add_parameter_group("Model Configuration")
        
        # Model selection
        self.model_type = EnumParameter("model_type", [
            "CNN", "LSTM", "Transformer", "Hybrid"
        ])
        self.add_parameter(model_group, "Model Architecture", self.model_type)
        
        # Training parameters
        self.batch_size = NumericParameter("batch_size", 1, 512, 1, 0)
        self.epochs = NumericParameter("epochs", 1, 1000, 1, 0)
        self.learning_rate = NumericParameter("learning_rate", 0.0001, 1.0, 0.0001, 4)
        
        self.add_parameter(model_group, "Batch Size", self.batch_size)
        self.add_parameter(model_group, "Epochs", self.epochs)
        self.add_parameter(model_group, "Learning Rate", self.learning_rate)
        
        # Preprocessing Configuration
        preprocess_group = self.add_parameter_group("Preprocessing")
        
        # Preprocessing options
        self.normalize = EnumParameter("normalize", [
            "None", "Z-Score", "Min-Max", "Robust"
        ])
        self.window_size = NumericParameter("window_size", 32, 4096, 32, 0)
        self.overlap = NumericParameter("overlap", 0, 100, 1, 0)
        
        self.add_parameter(preprocess_group, "Normalization", self.normalize)
        self.add_parameter(preprocess_group, "Window Size", self.window_size)
        self.add_parameter(preprocess_group, "Overlap %", self.overlap)
        
        # Cross-validation Configuration
        cv_group = self.add_parameter_group("Cross Validation")
        
        self.cv_folds = NumericParameter("cv_folds", 2, 10, 1, 0)
        self.cv_strategy = EnumParameter("cv_strategy", [
            "K-Fold", "Stratified K-Fold", "Leave One Out"
        ])
        
        self.add_parameter(cv_group, "Number of Folds", self.cv_folds)
        self.add_parameter(cv_group, "CV Strategy", self.cv_strategy)
        
        # Set default values
        self._set_defaults()
        
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "normalize": "Z-Score",
            "window_size": 256,
            "overlap": 50,
            "cv_folds": 5,
            "cv_strategy": "K-Fold",
            "model_type": "CNN"
        }
        self.set_parameters(defaults)
        
    def get_configuration(self) -> Dict[str, Any]:
        """Get complete batch configuration."""
        config = self.get_parameters()
        return config
        
    def validate_configuration(self) -> bool:
        """Validate the current configuration."""
        try:
            config = self.get_configuration()
            
            # Check dataset paths
            if not config["dataset_paths"]:
                raise ValueError("No dataset files selected")
                
            # Validate numeric parameters
            if config["batch_size"] <= 0:
                raise ValueError("Batch size must be positive")
            if config["epochs"] <= 0:
                raise ValueError("Number of epochs must be positive")
            if config["learning_rate"] <= 0:
                raise ValueError("Learning rate must be positive")
                
            # Validate window parameters
            if config["window_size"] <= 0:
                raise ValueError("Window size must be positive")
            if not 0 <= config["overlap"] <= 100:
                raise ValueError("Overlap must be between 0 and 100")
                
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.VALIDATION,
                ["Check parameter values", "Ensure dataset is selected"]
            )
            return False
            
    def reset_configuration(self):
        """Reset configuration to defaults."""
        self._set_defaults()