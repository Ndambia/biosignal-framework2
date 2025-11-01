from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QPushButton, QButtonGroup, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_panel import (
    BaseControlPanel, NumericParameter, EnumParameter,
    BoolParameter, SliderParameter
)

class EMGControlPanel(BaseControlPanel):
    """Control panel for EMG signal generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_pattern_controls()
        
    def _init_ui(self):
        """Initialize the UI layout."""
        super()._init_ui()
        
        # Create pattern selection
        pattern_group = self.add_parameter_group("Pattern Selection")
        self.pattern_selector = EnumParameter(
            "pattern_type",
            ["Isometric", "Dynamic", "Repetitive", "Complex"]
        )
        self.add_parameter(pattern_group, "Type:", self.pattern_selector)
        
        # Create stacked widget for pattern-specific controls
        self.pattern_stack = QStackedWidget()
        self.layout.addWidget(self.pattern_stack)
        
        # Connect pattern selector
        self.pattern_selector.value_changed.connect(self._on_pattern_changed)
        
    def _setup_pattern_controls(self):
        """Set up controls for each pattern type."""
        # Isometric controls
        self.isometric_widget = QWidget()
        isometric_layout = QVBoxLayout(self.isometric_widget)
        
        iso_params = self.add_parameter_group("Isometric Parameters")
        iso_params.setLayout(QVBoxLayout())
        
        # Add isometric parameters
        self.intensity = SliderParameter("intensity", 0.0, 1.0, 0.01)
        self.add_parameter(iso_params, "Intensity:", self.intensity)
        
        self.duration = NumericParameter("duration", 0.1, 60.0, 0.1)
        self.add_parameter(iso_params, "Duration (s):", self.duration)
        
        self.fatigue = SliderParameter("fatigue_rate", 0.0, 1.0, 0.01)
        self.add_parameter(iso_params, "Fatigue Rate:", self.fatigue)
        
        isometric_layout.addWidget(iso_params)
        self.pattern_stack.addWidget(self.isometric_widget)
        
        # Dynamic controls
        self.dynamic_widget = QWidget()
        dynamic_layout = QVBoxLayout(self.dynamic_widget)
        
        dyn_params = self.add_parameter_group("Dynamic Parameters")
        dyn_params.setLayout(QVBoxLayout())
        
        # Add dynamic parameters
        self.ramp_type = EnumParameter(
            "ramp_type",
            ["Linear", "Exponential", "Step"]
        )
        self.add_parameter(dyn_params, "Ramp Type:", self.ramp_type)
        
        self.max_intensity = SliderParameter("max_intensity", 0.0, 1.0, 0.01)
        self.add_parameter(dyn_params, "Max Intensity:", self.max_intensity)
        
        self.ramp_duration = NumericParameter("ramp_duration", 0.1, 60.0, 0.1)
        self.add_parameter(dyn_params, "Duration (s):", self.ramp_duration)
        
        dynamic_layout.addWidget(dyn_params)
        self.pattern_stack.addWidget(self.dynamic_widget)
        
        # Repetitive controls
        self.repetitive_widget = QWidget()
        repetitive_layout = QVBoxLayout(self.repetitive_widget)
        
        rep_params = self.add_parameter_group("Repetitive Parameters")
        rep_params.setLayout(QVBoxLayout())
        
        # Add repetitive parameters
        self.frequency = NumericParameter("frequency", 0.1, 5.0, 0.1)
        self.add_parameter(rep_params, "Frequency (Hz):", self.frequency)
        
        self.duty_cycle = SliderParameter("duty_cycle", 0.0, 1.0, 0.01)
        self.add_parameter(rep_params, "Duty Cycle:", self.duty_cycle)
        
        self.burst_intensity = SliderParameter("burst_intensity", 0.0, 1.0, 0.01)
        self.add_parameter(rep_params, "Burst Intensity:", self.burst_intensity)
        
        repetitive_layout.addWidget(rep_params)
        self.pattern_stack.addWidget(self.repetitive_widget)
        
        # Complex pattern controls
        self.complex_widget = QWidget()
        complex_layout = QVBoxLayout(self.complex_widget)
        
        # Movement sequence group
        sequence_group = self.add_parameter_group("Movement Sequence")
        sequence_group.setLayout(QVBoxLayout())
        
        # Add sequence controls
        self.movements = []
        self.durations = []
        self.intensities = []
        
        # Add buttons for sequence management
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Movement")
        add_btn.clicked.connect(self._add_movement)
        button_layout.addWidget(add_btn)
        
        clear_btn = QPushButton("Clear Sequence")
        clear_btn.clicked.connect(self._clear_sequence)
        button_layout.addWidget(clear_btn)
        
        sequence_group.layout().addLayout(button_layout)
        
        # Add overlap option
        self.overlap = BoolParameter("overlap", "Allow Movement Overlap")
        sequence_group.layout().addWidget(self.overlap)
        
        complex_layout.addWidget(sequence_group)
        self.pattern_stack.addWidget(self.complex_widget)
        
    def _on_pattern_changed(self, pattern: str):
        """Handle pattern type changes."""
        index = ["Isometric", "Dynamic", "Repetitive", "Complex"].index(pattern)
        self.pattern_stack.setCurrentIndex(index)
        
    def _add_movement(self):
        """Add a new movement to the complex sequence."""
        # Create movement controls
        movement = EnumParameter(
            f"movement_{len(self.movements)}",
            ["Isometric", "Dynamic", "Rest"]
        )
        duration = NumericParameter(
            f"duration_{len(self.durations)}",
            0.1, 60.0, 0.1
        )
        intensity = SliderParameter(
            f"intensity_{len(self.intensities)}",
            0.0, 1.0, 0.01
        )
        
        # Add to lists
        self.movements.append(movement)
        self.durations.append(duration)
        self.intensities.append(intensity)
        
        # Add to UI
        sequence_group = self.pattern_stack.currentWidget().findChild(
            QGroupBox, "Movement Sequence"
        )
        
        movement_group = QGroupBox(f"Movement {len(self.movements)}")
        movement_layout = QVBoxLayout(movement_group)
        
        self.add_parameter(movement_group, "Type:", movement)
        self.add_parameter(movement_group, "Duration (s):", duration)
        self.add_parameter(movement_group, "Intensity:", intensity)
        
        sequence_group.layout().insertWidget(
            len(self.movements) - 1,
            movement_group
        )
        
    def _clear_sequence(self):
        """Clear the complex movement sequence."""
        sequence_group = self.pattern_stack.currentWidget().findChild(
            QGroupBox, "Movement Sequence"
        )
        
        # Remove movement groups
        for movement in self.movements:
            movement.parent().deleteLater()
            
        self.movements.clear()
        self.durations.clear()
        self.intensities.clear()
        
    def get_parameters(self) -> dict:
        """Get current parameter values."""
        params = super().get_parameters()
        
        # Add complex sequence parameters if active
        if params.get('pattern_type') == 'Complex':
            params['movements'] = [m.get_value() for m in self.movements]
            params['durations'] = [d.get_value() for d in self.durations]
            params['intensities'] = [i.get_value() for i in self.intensities]
            
        return params
        
    def reset_parameters(self):
        """Reset parameters to defaults."""
        defaults = {
            'pattern_type': 'Isometric',
            'intensity': 0.5,
            'duration': 5.0,
            'fatigue_rate': 0.0,
            'ramp_type': 'Linear',
            'max_intensity': 0.8,
            'ramp_duration': 2.0,
            'frequency': 1.0,
            'duty_cycle': 0.5,
            'burst_intensity': 0.7,
            'overlap': False
        }
        
        self.set_parameters(defaults)
        self._clear_sequence()