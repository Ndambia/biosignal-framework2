from typing import Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QScrollArea, QFrame, QListWidget,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_panel import (
    BaseControlPanel, NumericParameter, EnumParameter,
    BoolParameter, SliderParameter
)

class NoiseLayerWidget(QWidget):
    """Widget for configuring a single noise or artifact layer."""
    
    removed = pyqtSignal()
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, layer_type: str, layer_id: int, parent=None):
        super().__init__(parent)
        self.layer_type = layer_type
        self.layer_id = layer_id
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Create header
        header_layout = QHBoxLayout()
        
        # Add enable checkbox
        self.enable = BoolParameter("enabled", "")
        self.enable.setChecked(True)
        header_layout.addWidget(self.enable)
        
        # Add type label
        header_layout.addWidget(QLabel(self.layer_type.replace('_', ' ').title()))
        
        # Add remove button
        remove_btn = QPushButton("×")
        remove_btn.setMaximumWidth(20)
        remove_btn.clicked.connect(self.removed.emit)
        header_layout.addWidget(remove_btn)
        
        layout.addLayout(header_layout)
        
        # Add parameters based on type
        self.parameters = {}
        
        if self.layer_type in NOISE_TYPES:
            self._add_noise_parameters()
        else:
            self._add_artifact_parameters()
            
    def _add_noise_parameters(self):
        """Add parameters for noise layer."""
        param_group = QGroupBox()
        param_layout = QVBoxLayout(param_group)
        
        if self.layer_type == 'gaussian':
            self.parameters['std'] = SliderParameter(
                f"std_{self.layer_id}", 0.0, 1.0, 0.01
            )
            param_layout.addWidget(self.parameters['std'])
            
        elif self.layer_type == 'pink':
            self.parameters['amplitude'] = SliderParameter(
                f"pink_amp_{self.layer_id}", 0.0, 1.0, 0.01
            )
            param_layout.addWidget(self.parameters['amplitude'])
            
        elif self.layer_type == 'powerline':
            self.parameters['frequency'] = EnumParameter(
                f"powerline_freq_{self.layer_id}",
                ["50 Hz", "60 Hz"]
            )
            param_layout.addWidget(self.parameters['frequency'])
            
            self.parameters['amplitude'] = SliderParameter(
                f"powerline_amp_{self.layer_id}", 0.0, 1.0, 0.01
            )
            param_layout.addWidget(self.parameters['amplitude'])
            
        elif self.layer_type == 'baseline_wander':
            self.parameters['frequency'] = NumericParameter(
                f"baseline_freq_{self.layer_id}", 0.1, 1.0, 0.1
            )
            param_layout.addWidget(self.parameters['frequency'])
            
            self.parameters['amplitude'] = SliderParameter(
                f"baseline_amp_{self.layer_id}", 0.0, 1.0, 0.01
            )
            param_layout.addWidget(self.parameters['amplitude'])
            
        self.layout().addWidget(param_group)
        
        # Connect parameter changes
        for param in self.parameters.values():
            param.value_changed.connect(self._on_parameter_changed)
            
    def _add_artifact_parameters(self):
        """Add parameters for artifact layer."""
        param_group = QGroupBox()
        param_layout = QVBoxLayout(param_group)
        
        if self.layer_type == 'electrode_movement':
            self.parameters['amplitude'] = SliderParameter(
                f"movement_amp_{self.layer_id}", 0.0, 2.0, 0.1
            )
            param_layout.addWidget(self.parameters['amplitude'])
            
            self.parameters['duration'] = NumericParameter(
                f"movement_dur_{self.layer_id}", 0.1, 1.0, 0.1
            )
            param_layout.addWidget(self.parameters['duration'])
            
        elif self.layer_type == 'poor_contact':
            self.parameters['severity'] = SliderParameter(
                f"contact_sev_{self.layer_id}", 0.0, 1.0, 0.1
            )
            param_layout.addWidget(self.parameters['severity'])
            
            self.parameters['duration'] = NumericParameter(
                f"contact_dur_{self.layer_id}", 0.5, 5.0, 0.5
            )
            param_layout.addWidget(self.parameters['duration'])
            
        elif self.layer_type == 'emg_crosstalk':
            self.parameters['amplitude'] = SliderParameter(
                f"crosstalk_amp_{self.layer_id}", 0.0, 1.0, 0.1
            )
            param_layout.addWidget(self.parameters['amplitude'])
            
            self.parameters['frequency'] = NumericParameter(
                f"crosstalk_freq_{self.layer_id}", 20, 500, 10
            )
            param_layout.addWidget(self.parameters['frequency'])
            
        self.layout().addWidget(param_group)
        
        # Connect parameter changes
        for param in self.parameters.values():
            param.value_changed.connect(self._on_parameter_changed)
            
    def _on_parameter_changed(self, name: str, value: Any):
        """Handle parameter changes."""
        params = {
            'type': self.layer_type,
            'enabled': self.enable.get_value(),
            'parameters': {
                name: param.get_value() 
                for name, param in self.parameters.items()
            }
        }
        self.parameters_changed.emit(params)
        
    def get_parameters(self) -> dict:
        """Get layer parameters."""
        return {
            'type': self.layer_type,
            'enabled': self.enable.get_value(),
            'parameters': {
                name: param.get_value() 
                for name, param in self.parameters.items()
            }
        }

# Available noise and artifact types
NOISE_TYPES = [
    'gaussian',
    'pink',
    'brown',
    'powerline',
    'baseline_wander',
    'high_frequency'
]

ARTIFACT_TYPES = [
    'electrode_movement',
    'cable_motion',
    'subject_movement',
    'baseline_shift',
    'poor_contact',
    'electrode_pop',
    'impedance_change',
    'dc_offset',
    'emg_crosstalk',
    'ecg_interference',
    'environmental',
    'device_artifact'
]

class NoiseArtifactPanel(BaseControlPanel):
    """Control panel for noise and artifact generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.next_layer_id = 0
        self.layers = {}  # layer_id -> NoiseLayerWidget
        
    def _init_ui(self):
        """Initialize the UI layout."""
        super()._init_ui()
        
        # Create scroll area for layers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.layout.addWidget(scroll)
        
        # Create container for layers
        self.layer_container = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_container)
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_layout.addStretch()
        
        scroll.setWidget(self.layer_container)
        
        # Create buttons for adding noise/artifacts
        button_layout = QHBoxLayout()
        
        add_noise_btn = QPushButton("Add Noise")
        add_noise_btn.clicked.connect(self._show_noise_menu)
        button_layout.addWidget(add_noise_btn)
        
        add_artifact_btn = QPushButton("Add Artifact")
        add_artifact_btn.clicked.connect(self._show_artifact_menu)
        button_layout.addWidget(add_artifact_btn)
        
        self.layout.addLayout(button_layout)
        
    def _show_noise_menu(self):
        """Show menu for selecting noise type."""
        menu = QMenu(self)
        
        for noise_type in NOISE_TYPES:
            action = menu.addAction(noise_type.replace('_', ' ').title())
            action.triggered.connect(
                lambda checked, t=noise_type: self._add_layer(t)
            )
            
        menu.exec(QCursor.pos())
        
    def _show_artifact_menu(self):
        """Show menu for selecting artifact type."""
        menu = QMenu(self)
        
        for artifact_type in ARTIFACT_TYPES:
            action = menu.addAction(artifact_type.replace('_', ' ').title())
            action.triggered.connect(
                lambda checked, t=artifact_type: self._add_layer(t)
            )
            
        menu.exec(QCursor.pos())
        
    def _add_layer(self, layer_type: str):
        """Add a new noise/artifact layer."""
        layer = NoiseLayerWidget(layer_type, self.next_layer_id)
        
        # Connect signals
        layer.removed.connect(lambda: self._remove_layer(layer.layer_id))
        layer.parameters_changed.connect(
            lambda p: self._on_layer_changed(layer.layer_id, p)
        )
        
        # Add to layout (before stretch)
        self.layer_layout.insertWidget(self.layer_layout.count() - 1, layer)
        
        # Store reference
        self.layers[self.next_layer_id] = layer
        self.next_layer_id += 1
        
        # Emit parameters changed
        self.parameters_changed.emit(self.get_parameters())
        
    def _remove_layer(self, layer_id: int):
        """Remove a noise/artifact layer."""
        if layer_id in self.layers:
            layer = self.layers.pop(layer_id)
            self.layer_layout.removeWidget(layer)
            layer.deleteLater()
            
            # Emit parameters changed
            self.parameters_changed.emit(self.get_parameters())
            
    def _on_layer_changed(self, layer_id: int, parameters: dict):
        """Handle layer parameter changes."""
        self.parameters_changed.emit(self.get_parameters())
        
    def get_parameters(self) -> dict:
        """Get all layer parameters."""
        return {
            layer_id: layer.get_parameters()
            for layer_id, layer in self.layers.items()
        }
        
    def set_parameters(self, params: dict):
        """Set layer parameters."""
        # Clear existing layers
        for layer_id in list(self.layers.keys()):
            self._remove_layer(layer_id)
            
        # Add new layers
        for layer_id, layer_params in params.items():
            layer = self._add_layer(layer_params['type'])
            layer.enable.set_value(layer_params['enabled'])
            for name, value in layer_params['parameters'].items():
                if name in layer.parameters:
                    layer.parameters[name].set_value(value)
                    
    def reset_parameters(self):
        """Reset to default state (no layers)."""
        for layer_id in list(self.layers.keys()):
            self._remove_layer(layer_id)