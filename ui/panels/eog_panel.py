from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QPushButton, QButtonGroup, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_panel import (
    BaseControlPanel, NumericParameter, EnumParameter,
    BoolParameter, SliderParameter
)

class EOGControlPanel(BaseControlPanel):
    """Control panel for EOG signal generation."""
    
    # Movement types with descriptions
    MOVEMENTS = {
        'saccade': 'Rapid eye movement between fixation points',
        'smooth_pursuit': 'Smooth tracking of moving target',
        'fixation': 'Stable gaze with microsaccades',
        'blink': 'Eye blink movement',
        'combined': 'Complex movement sequence'
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_movement_controls()
        
    def _init_ui(self):
        """Initialize the UI layout."""
        super()._init_ui()
        
        # Basic parameters group
        basic_group = self.add_parameter_group("Basic Parameters")
        
        # Movement type selection
        self.movement_type = EnumParameter(
            "movement_type",
            list(self.MOVEMENTS.keys())
        )
        self.add_parameter(basic_group, "Type:", self.movement_type)
        
        # Add movement description label
        self.movement_desc = QLabel(self.MOVEMENTS['saccade'])
        self.movement_desc.setWordWrap(True)
        basic_group.layout().addWidget(self.movement_desc)
        
        # Connect movement change
        self.movement_type.value_changed.connect(self._on_movement_changed)
        
        # Create stacked widget for movement-specific controls
        self.movement_stack = QStackedWidget()
        self.layout.addWidget(self.movement_stack)
        
    def _setup_movement_controls(self):
        """Set up controls for each movement type."""
        # Saccade controls
        self.saccade_widget = QWidget()
        saccade_layout = QVBoxLayout(self.saccade_widget)
        
        saccade_group = self.add_parameter_group("Saccade Parameters")
        saccade_group.setLayout(QVBoxLayout())
        
        # Add saccade parameters
        self.amplitude = NumericParameter("amplitude", 20, 500, 10, 0)
        self.add_parameter(saccade_group, "Amplitude (μV):", self.amplitude)
        
        self.direction = EnumParameter(
            "direction",
            ["Horizontal", "Vertical", "Oblique"]
        )
        self.add_parameter(saccade_group, "Direction:", self.direction)
        
        self.frequency = NumericParameter("frequency", 0.1, 5.0, 0.1, 1)
        self.add_parameter(saccade_group, "Frequency (Hz):", self.frequency)
        
        saccade_layout.addWidget(saccade_group)
        self.movement_stack.addWidget(self.saccade_widget)
        
        # Smooth pursuit controls
        self.pursuit_widget = QWidget()
        pursuit_layout = QVBoxLayout(self.pursuit_widget)
        
        pursuit_group = self.add_parameter_group("Smooth Pursuit Parameters")
        pursuit_group.setLayout(QVBoxLayout())
        
        # Add pursuit parameters
        self.velocity = NumericParameter("velocity", 5, 100, 5, 0)
        self.add_parameter(pursuit_group, "Velocity (deg/s):", self.velocity)
        
        self.target_freq = NumericParameter("target_frequency", 0.1, 2.0, 0.1, 1)
        self.add_parameter(pursuit_group, "Target Freq (Hz):", self.target_freq)
        
        self.pattern = EnumParameter(
            "pattern",
            ["Linear", "Sinusoidal", "Circular", "Custom"]
        )
        self.add_parameter(pursuit_group, "Pattern:", self.pattern)
        
        pursuit_layout.addWidget(pursuit_group)
        self.movement_stack.addWidget(self.pursuit_widget)
        
        # Fixation controls
        self.fixation_widget = QWidget()
        fixation_layout = QVBoxLayout(self.fixation_widget)
        
        fixation_group = self.add_parameter_group("Fixation Parameters")
        fixation_group.setLayout(QVBoxLayout())
        
        # Add fixation parameters
        self.duration = NumericParameter("duration", 0.5, 10.0, 0.5, 1)
        self.add_parameter(fixation_group, "Duration (s):", self.duration)
        
        self.micro_amp = NumericParameter("microsaccade_amplitude", 1, 50, 1, 0)
        self.add_parameter(fixation_group, "Microsaccade Amp (μV):", self.micro_amp)
        
        self.micro_freq = NumericParameter("microsaccade_frequency", 0.1, 5.0, 0.1, 1)
        self.add_parameter(fixation_group, "Microsaccade Freq (Hz):", self.micro_freq)
        
        fixation_layout.addWidget(fixation_group)
        self.movement_stack.addWidget(self.fixation_widget)
        
        # Blink controls
        self.blink_widget = QWidget()
        blink_layout = QVBoxLayout(self.blink_widget)
        
        blink_group = self.add_parameter_group("Blink Parameters")
        blink_group.setLayout(QVBoxLayout())
        
        # Add blink parameters
        self.blink_amp = NumericParameter("blink_amplitude", 50, 500, 10, 0)
        self.add_parameter(blink_group, "Amplitude (μV):", self.blink_amp)
        
        self.blink_rate = NumericParameter("blink_rate", 0.1, 2.0, 0.1, 1)
        self.add_parameter(blink_group, "Rate (Hz):", self.blink_rate)
        
        self.blink_duration = NumericParameter("blink_duration", 0.1, 1.0, 0.1, 1)
        self.add_parameter(blink_group, "Duration (s):", self.blink_duration)
        
        blink_layout.addWidget(blink_group)
        self.movement_stack.addWidget(self.blink_widget)
        
        # Combined movement controls
        self.combined_widget = QWidget()
        combined_layout = QVBoxLayout(self.combined_widget)
        
        sequence_group = self.add_parameter_group("Movement Sequence")
        sequence_group.setLayout(QVBoxLayout())
        
        # Add sequence controls
        self.movements = []
        self.durations = []
        
        # Add buttons for sequence management
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Movement")
        add_btn.clicked.connect(self._add_movement)
        button_layout.addWidget(add_btn)
        
        clear_btn = QPushButton("Clear Sequence")
        clear_btn.clicked.connect(self._clear_sequence)
        button_layout.addWidget(clear_btn)
        
        sequence_group.layout().addLayout(button_layout)
        
        combined_layout.addWidget(sequence_group)
        self.movement_stack.addWidget(self.combined_widget)
        
    def _on_movement_changed(self, movement: str):
        """Handle movement type changes."""
        # Update description
        self.movement_desc.setText(self.MOVEMENTS[movement])
        
        # Update stack widget
        index = list(self.MOVEMENTS.keys()).index(movement)
        self.movement_stack.setCurrentIndex(index)
        
    def _add_movement(self):
        """Add a new movement to the sequence."""
        # Create movement controls
        movement = EnumParameter(
            f"movement_{len(self.movements)}",
            ["Saccade", "Smooth Pursuit", "Fixation", "Blink"]
        )
        duration = NumericParameter(
            f"duration_{len(self.durations)}",
            0.1, 10.0, 0.1, 1
        )
        
        # Add to lists
        self.movements.append(movement)
        self.durations.append(duration)
        
        # Add to UI
        sequence_group = self.movement_stack.currentWidget().findChild(
            QGroupBox, "Movement Sequence"
        )
        
        movement_group = QGroupBox(f"Movement {len(self.movements)}")
        movement_layout = QVBoxLayout(movement_group)
        
        self.add_parameter(movement_group, "Type:", movement)
        self.add_parameter(movement_group, "Duration (s):", duration)
        
        sequence_group.layout().insertWidget(
            len(self.movements) - 1,
            movement_group
        )
        
    def _clear_sequence(self):
        """Clear the movement sequence."""
        sequence_group = self.movement_stack.currentWidget().findChild(
            QGroupBox, "Movement Sequence"
        )
        
        # Remove movement groups
        for movement in self.movements:
            movement.parent().deleteLater()
            
        self.movements.clear()
        self.durations.clear()
        
    def get_parameters(self) -> dict:
        """Get current parameter values."""
        params = super().get_parameters()
        
        # Add sequence parameters if in combined mode
        if params.get('movement_type') == 'combined':
            params['movements'] = [m.get_value() for m in self.movements]
            params['durations'] = [d.get_value() for d in self.durations]
            
        return params
        
    def reset_parameters(self):
        """Reset parameters to defaults."""
        defaults = {
            'movement_type': 'saccade',
            'amplitude': 100,
            'direction': 'Horizontal',
            'frequency': 1.0,
            'velocity': 30,
            'target_frequency': 0.5,
            'pattern': 'Sinusoidal',
            'duration': 2.0,
            'microsaccade_amplitude': 10,
            'microsaccade_frequency': 1.0,
            'blink_amplitude': 200,
            'blink_rate': 0.25,
            'blink_duration': 0.2
        }
        
        self.set_parameters(defaults)
        self._clear_sequence()