from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QLineEdit, QToolTip
)
from PyQt6.QtCore import pyqtSignal, Qt
from typing import Dict, Any, Optional
from .validation import ParameterValidator, ValidationError
from .error_handling import ErrorHandler

class ParameterWidget(QWidget):
    """Base class for parameter widgets"""
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.validator = ParameterValidator()
        self.error_handler = ErrorHandler()
        self._init_common_ui()
        
    def _init_common_ui(self):
        self.layout = QVBoxLayout(self)
        
        # Common parameters group
        common_group = QGroupBox("Common Parameters")
        common_layout = QFormLayout()
        
        # Get validation limits
        sampling_limits = self.validator.get_parameter_limits('signal', 'sampling_rate')
        duration_limits = self.validator.get_parameter_limits('signal', 'duration')
        
        self.sampling_rate = QDoubleSpinBox()
        self.sampling_rate.setRange(sampling_limits['min'], sampling_limits['max'])
        self.sampling_rate.setValue(1000)
        self.sampling_rate.setSuffix(" Hz")
        self.sampling_rate.setToolTip("Sampling rate in Hz (1 - 100000)")
        
        self.duration = QDoubleSpinBox()
        self.duration.setRange(duration_limits['min'], duration_limits['max'])
        self.duration.setValue(10)
        self.duration.setSuffix(" s")
        self.duration.setToolTip("Signal duration in seconds (0.1 - 3600)")
        
        common_layout.addRow("Sampling Rate:", self.sampling_rate)
        common_layout.addRow("Duration:", self.duration)
        common_group.setLayout(common_layout)
        self.layout.addWidget(common_group)
        
        # Connect signals
        self.sampling_rate.valueChanged.connect(self._on_parameter_changed)
        self.duration.valueChanged.connect(self._on_parameter_changed)
    
    def _on_parameter_changed(self):
        """Emit updated parameters when values change"""
        try:
            params = self.get_parameters()
            self.validator.validate_parameters('signal', {
                'sampling_rate': params['sampling_rate'],
                'duration': params['duration']
            })
            self.parameters_changed.emit(params)
        except ValidationError as e:
            self.error_handler.handle_error(e)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        return {
            'sampling_rate': self.sampling_rate.value(),
            'duration': self.duration.value()
        }

class EMGParameterWidget(ParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_specific_ui()
    
    def _init_specific_ui(self):
        group = QGroupBox("EMG Parameters")
        form = QFormLayout()
        
        # Get validation rules
        activation_limits = self.validator.get_parameter_limits('emg', 'activation_level')
        contraction_options = self.validator.get_parameter_options('emg', 'contraction_type')
        
        self.activation = QDoubleSpinBox()
        self.activation.setRange(activation_limits['min'], activation_limits['max'])
        self.activation.setSingleStep(0.1)
        self.activation.setValue(0.5)
        self.activation.setToolTip("Muscle activation level (0-1)")
        
        self.contraction = QComboBox()
        self.contraction.addItems(contraction_options)
        self.contraction.setToolTip("Type of muscle contraction")
        
        self.pattern = QLineEdit()
        
        form.addRow("Activation Level:", self.activation)
        form.addRow("Contraction Type:", self.contraction)
        form.addRow("Movement Pattern:", self.pattern)
        
        group.setLayout(form)
        self.layout.addWidget(group)
        
        # Connect signals
        self.activation.valueChanged.connect(self._on_parameter_changed)
        self.contraction.currentTextChanged.connect(self._on_parameter_changed)
        self.pattern.textChanged.connect(self._on_parameter_changed)
    
    def get_parameters(self) -> Dict[str, Any]:
        try:
            params = super().get_parameters()
            emg_params = {
                'activation_level': self.activation.value(),
                'contraction_type': self.contraction.currentText(),
                'movement_pattern': self.pattern.text() or None
            }
            self.validator.validate_parameters('emg', emg_params)
            params['emg_params'] = emg_params
            return params
        except ValidationError as e:
            self.error_handler.handle_error(e)
            return {}

class ECGParameterWidget(ParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_specific_ui()
    
    def _init_specific_ui(self):
        group = QGroupBox("ECG Parameters")
        form = QFormLayout()
        
        # Get validation limits
        hr_limits = self.validator.get_parameter_limits('ecg', 'heart_rate')
        
        self.heart_rate = QDoubleSpinBox()
        self.heart_rate.setRange(hr_limits['min'], hr_limits['max'])
        self.heart_rate.setValue(60)
        self.heart_rate.setSuffix(" bpm")
        self.heart_rate.setToolTip("Heart rate in beats per minute (20-250)")
        
        self.condition = QLineEdit()
        
        form.addRow("Heart Rate:", self.heart_rate)
        form.addRow("Condition:", self.condition)
        
        group.setLayout(form)
        self.layout.addWidget(group)
        
        # Connect signals
        self.heart_rate.valueChanged.connect(self._on_parameter_changed)
        self.condition.textChanged.connect(self._on_parameter_changed)
    
    def get_parameters(self) -> Dict[str, Any]:
        try:
            params = super().get_parameters()
            ecg_params = {
                'heart_rate': self.heart_rate.value(),
                'condition': self.condition.text() or None,
                'abnormalities': []  # TODO: Add abnormalities selection
            }
            self.validator.validate_parameters('ecg', ecg_params)
            params['ecg_params'] = ecg_params
            return params
        except ValidationError as e:
            self.error_handler.handle_error(e)
            return {}

class EOGParameterWidget(ParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_specific_ui()
    
    def _init_specific_ui(self):
        group = QGroupBox("EOG Parameters")
        form = QFormLayout()
        
        self.movement = QComboBox()
        self.movement.addItems(['saccade', 'pursuit', 'fixation'])
        
        # Get validation limits
        amp_limits = self.validator.get_parameter_limits('eog', 'amplitude')
        freq_limits = self.validator.get_parameter_limits('eog', 'frequency')
        
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(amp_limits['min'], amp_limits['max'])
        self.amplitude.setValue(10)
        self.amplitude.setSuffix(" μV")
        self.amplitude.setToolTip("Signal amplitude in microvolts (0-5000)")
        
        self.frequency = QDoubleSpinBox()
        self.frequency.setRange(freq_limits['min'], freq_limits['max'])
        self.frequency.setValue(1)
        self.frequency.setSuffix(" Hz")
        self.frequency.setToolTip("Signal frequency in Hz (0.1-100)")
        
        form.addRow("Movement Type:", self.movement)
        form.addRow("Amplitude:", self.amplitude)
        form.addRow("Frequency:", self.frequency)
        
        group.setLayout(form)
        self.layout.addWidget(group)
        
        # Connect signals
        self.movement.currentTextChanged.connect(self._on_parameter_changed)
        self.amplitude.valueChanged.connect(self._on_parameter_changed)
        self.frequency.valueChanged.connect(self._on_parameter_changed)
    
    def get_parameters(self) -> Dict[str, Any]:
        try:
            params = super().get_parameters()
            eog_params = {
                'movement_type': self.movement.currentText(),
                'amplitude': self.amplitude.value(),
                'frequency': self.frequency.value()
            }
            self.validator.validate_parameters('eog', eog_params)
            params['eog_params'] = eog_params
            return params
        except ValidationError as e:
            self.error_handler.handle_error(e)
            return {}