from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QSlider, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any, Optional

class ParameterWidget(QWidget):
    """Base widget for parameter controls."""
    
    value_changed = pyqtSignal(str, object)  # parameter_name, new_value
    
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def get_value(self):
        """Get current parameter value. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def set_value(self, value):
        """Set parameter value. Must be implemented by subclasses."""
        raise NotImplementedError

class NumericParameter(ParameterWidget):
    """Widget for numeric parameter input."""
    
    def __init__(self, name: str, min_val: float, max_val: float, 
                 step: float = 1.0, decimals: int = 2, parent=None):
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals
        super().__init__(name, parent)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if self.decimals == 0:
            self.spin = QSpinBox()
            self.spin.setRange(int(self.min_val), int(self.max_val))
            self.spin.setSingleStep(int(self.step))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(self.min_val, self.max_val)
            self.spin.setSingleStep(self.step)
            self.spin.setDecimals(self.decimals)
            
        self.spin.valueChanged.connect(
            lambda v: self.value_changed.emit(self.name, v)
        )
        layout.addWidget(self.spin)
        
    def get_value(self):
        return self.spin.value()
        
    def set_value(self, value):
        self.spin.setValue(value)

class EnumParameter(ParameterWidget):
    """Widget for enumerated parameter selection."""
    
    def __init__(self, name: str, options: list, parent=None):
        self.options = options
        super().__init__(name, parent)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.combo = QComboBox()
        self.combo.addItems(self.options)
        self.combo.currentTextChanged.connect(
            lambda v: self.value_changed.emit(self.name, v)
        )
        layout.addWidget(self.combo)
        
    def get_value(self):
        return self.combo.currentText()
        
    def set_value(self, value):
        index = self.combo.findText(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)

class BoolParameter(ParameterWidget):
    """Widget for boolean parameter input."""
    
    def __init__(self, name: str, label: str, parent=None):
        self.label = label
        super().__init__(name, parent)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.checkbox = QCheckBox(self.label)
        self.checkbox.toggled.connect(
            lambda v: self.value_changed.emit(self.name, v)
        )
        layout.addWidget(self.checkbox)
        
    def get_value(self):
        return self.checkbox.isChecked()
        
    def set_value(self, value):
        self.checkbox.setChecked(value)

class SliderParameter(ParameterWidget):
    """Widget for slider-based parameter input."""
    
    def __init__(self, name: str, min_val: float, max_val: float,
                 step: float = 0.1, decimals: int = 2, parent=None):
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals
        self.scale = 10 ** decimals
        super().__init__(name, parent)
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            int(self.min_val * self.scale),
            int(self.max_val * self.scale)
        )
        self.slider.setSingleStep(int(self.step * self.scale))
        
        # Create value label
        self.value_label = QLabel(f"{self.min_val:.{self.decimals}f}")
        
        # Add widgets
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        
    def _on_slider_changed(self, value):
        """Handle slider value changes."""
        actual_value = value / self.scale
        self.value_label.setText(f"{actual_value:.{self.decimals}f}")
        self.value_changed.emit(self.name, actual_value)
        
    def get_value(self):
        return self.slider.value() / self.scale
        
    def set_value(self, value):
        self.slider.setValue(int(value * self.scale))

class BaseControlPanel(QWidget):
    """Base class for signal control panels."""
    
    parameters_changed = pyqtSignal(dict)  # All parameters
    parameter_changed = pyqtSignal(str, object)  # Single parameter
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameters = {}
        
    def _init_ui(self):
        """Initialize the UI. Must be implemented by subclasses."""
        pass
        
    def add_parameter_group(self, title: str) -> QGroupBox:
        """Add a new parameter group."""
        group = QGroupBox(title)
        form = QFormLayout(group)
        self.layout.addWidget(group)
        return group
        
    def add_parameter(self, group: QGroupBox, label: str, widget: ParameterWidget):
        """Add a parameter widget to a group."""
        form = group.layout()
        form.addRow(label, widget)
        self.parameters[widget.name] = widget
        widget.value_changed.connect(self._on_parameter_changed)
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameter values."""
        return {name: widget.get_value() 
                for name, widget in self.parameters.items()}
        
    def set_parameters(self, params: Dict[str, Any]):
        """Set parameter values."""
        for name, value in params.items():
            if name in self.parameters:
                self.parameters[name].set_value(value)
                
    def _on_parameter_changed(self, name: str, value: Any):
        """Handle parameter value changes."""
        self.parameter_changed.emit(name, value)
        self.parameters_changed.emit(self.get_parameters())
        
    def reset_parameters(self):
        """Reset all parameters to defaults."""
        raise NotImplementedError