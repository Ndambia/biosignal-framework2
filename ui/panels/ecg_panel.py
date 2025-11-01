from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QButtonGroup, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_panel import (
    BaseControlPanel, NumericParameter, EnumParameter,
    BoolParameter, SliderParameter
)

class ECGControlPanel(BaseControlPanel):
    """Control panel for ECG signal generation."""
    
    # Standard ECG leads
    LEADS = [
        'I', 'II', 'III',        # Limb leads
        'aVR', 'aVL', 'aVF',     # Augmented limb leads
        'V1', 'V2', 'V3',        # Chest leads
        'V4', 'V5', 'V6'         # More chest leads
    ]
    
    # Cardiac conditions with descriptions
    CONDITIONS = {
        'normal_sinus_rhythm': 'Normal heart rhythm',
        'premature_ventricular_contraction': 'Early heartbeat from ventricles',
        'atrial_fibrillation': 'Irregular atrial rhythm',
        'ventricular_tachycardia': 'Fast rhythm from ventricles',
        'stemi': 'ST elevation myocardial infarction',
        'nstemi': 'Non-ST elevation myocardial infarction',
        'left_bundle_branch_block': 'Left bundle conduction block',
        'right_bundle_branch_block': 'Right bundle conduction block',
        'atrial_flutter': 'Regular rapid atrial rhythm',
        'heart_block_first': 'First degree AV block',
        'heart_block_second': 'Second degree AV block',
        'heart_block_third': 'Complete heart block',
        'wpw_syndrome': 'Wolff-Parkinson-White syndrome'
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_wave_controls()
        
    def _init_ui(self):
        """Initialize the UI layout."""
        super()._init_ui()
        
        # Basic parameters group
        basic_group = self.add_parameter_group("Basic Parameters")
        
        # Heart rate control
        self.heart_rate = NumericParameter("heart_rate", 30, 200, 1, 0)
        self.add_parameter(basic_group, "Heart Rate (bpm):", self.heart_rate)
        
        # Lead selection
        self.lead = EnumParameter("lead", self.LEADS)
        self.add_parameter(basic_group, "Lead:", self.lead)
        
        # Condition selection
        condition_group = self.add_parameter_group("Cardiac Condition")
        self.condition = EnumParameter("condition", list(self.CONDITIONS.keys()))
        self.add_parameter(condition_group, "Type:", self.condition)
        
        # Add condition description label
        self.condition_desc = QLabel(self.CONDITIONS['normal_sinus_rhythm'])
        self.condition_desc.setWordWrap(True)
        condition_group.layout().addWidget(self.condition_desc)
        
        # Connect condition change
        self.condition.value_changed.connect(self._on_condition_changed)
        
        # Create severity controls
        severity_group = self.add_parameter_group("Condition Severity")
        self.severity = SliderParameter("severity", 0.0, 1.0, 0.01)
        self.add_parameter(severity_group, "Severity:", self.severity)
        
        # Add variability controls
        var_group = self.add_parameter_group("Heart Rate Variability")
        
        self.hrv_enabled = BoolParameter("hrv_enabled", "Enable HRV")
        self.add_parameter(var_group, "", self.hrv_enabled)
        
        self.hrv_amount = SliderParameter("hrv_amount", 0.0, 1.0, 0.01)
        self.add_parameter(var_group, "Amount:", self.hrv_amount)
        
    def _setup_wave_controls(self):
        """Set up controls for individual wave parameters."""
        wave_group = self.add_parameter_group("Wave Parameters")
        
        # P wave parameters
        self.p_amplitude = SliderParameter("p_amplitude", 0.0, 0.5, 0.01)
        self.add_parameter(wave_group, "P Amplitude (mV):", self.p_amplitude)
        
        self.pr_interval = NumericParameter("pr_interval", 0.12, 0.20, 0.01, 2)
        self.add_parameter(wave_group, "PR Interval (s):", self.pr_interval)
        
        # QRS parameters
        self.qrs_amplitude = SliderParameter("qrs_amplitude", 0.5, 3.0, 0.1)
        self.add_parameter(wave_group, "QRS Amplitude (mV):", self.qrs_amplitude)
        
        self.qrs_duration = NumericParameter("qrs_duration", 0.06, 0.12, 0.01, 2)
        self.add_parameter(wave_group, "QRS Duration (s):", self.qrs_duration)
        
        # T wave parameters
        self.t_amplitude = SliderParameter("t_amplitude", 0.0, 1.0, 0.05)
        self.add_parameter(wave_group, "T Amplitude (mV):", self.t_amplitude)
        
        self.qt_interval = NumericParameter("qt_interval", 0.35, 0.45, 0.01, 2)
        self.add_parameter(wave_group, "QT Interval (s):", self.qt_interval)
        
        # ST segment parameters
        self.st_level = SliderParameter("st_level", -0.5, 0.5, 0.01)
        self.add_parameter(wave_group, "ST Level (mV):", self.st_level)
        
    def _on_condition_changed(self, condition: str):
        """Handle cardiac condition changes."""
        # Update description
        self.condition_desc.setText(self.CONDITIONS[condition])
        
        # Update wave parameters based on condition
        if condition == 'normal_sinus_rhythm':
            self._set_normal_parameters()
        elif condition == 'stemi':
            self._set_stemi_parameters()
        elif condition == 'left_bundle_branch_block':
            self._set_lbbb_parameters()
        elif condition == 'atrial_fibrillation':
            self._set_afib_parameters()
        # Add more condition-specific parameter sets
        
    def _set_normal_parameters(self):
        """Set parameters for normal sinus rhythm."""
        params = {
            'p_amplitude': 0.15,
            'pr_interval': 0.16,
            'qrs_amplitude': 1.0,
            'qrs_duration': 0.08,
            't_amplitude': 0.3,
            'qt_interval': 0.40,
            'st_level': 0.0
        }
        self.set_parameters(params)
        
    def _set_stemi_parameters(self):
        """Set parameters for STEMI."""
        params = {
            'p_amplitude': 0.15,
            'pr_interval': 0.16,
            'qrs_amplitude': 1.2,
            'qrs_duration': 0.10,
            't_amplitude': 0.5,
            'qt_interval': 0.42,
            'st_level': 0.3  # ST elevation
        }
        self.set_parameters(params)
        
    def _set_lbbb_parameters(self):
        """Set parameters for left bundle branch block."""
        params = {
            'p_amplitude': 0.15,
            'pr_interval': 0.16,
            'qrs_amplitude': 1.5,
            'qrs_duration': 0.12,  # Wide QRS
            't_amplitude': -0.2,   # T wave inversion
            'qt_interval': 0.44,
            'st_level': -0.1
        }
        self.set_parameters(params)
        
    def _set_afib_parameters(self):
        """Set parameters for atrial fibrillation."""
        params = {
            'p_amplitude': 0.0,    # No P waves
            'pr_interval': 0.16,
            'qrs_amplitude': 1.0,
            'qrs_duration': 0.08,
            't_amplitude': 0.3,
            'qt_interval': 0.40,
            'st_level': 0.0,
            'hrv_enabled': True,   # Enable HRV
            'hrv_amount': 0.8      # High variability
        }
        self.set_parameters(params)
        
    def reset_parameters(self):
        """Reset parameters to defaults."""
        defaults = {
            'heart_rate': 75,
            'lead': 'II',
            'condition': 'normal_sinus_rhythm',
            'severity': 0.0,
            'hrv_enabled': False,
            'hrv_amount': 0.2,
            'p_amplitude': 0.15,
            'pr_interval': 0.16,
            'qrs_amplitude': 1.0,
            'qrs_duration': 0.08,
            't_amplitude': 0.3,
            'qt_interval': 0.40,
            'st_level': 0.0
        }
        self.set_parameters(defaults)