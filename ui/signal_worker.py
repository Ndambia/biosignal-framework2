from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from simulation import EMGSimulator, ECGSimulator, EOGSimulator
from typing import Dict, Any

class SignalWorker(QThread):
    """Worker thread for signal generation"""
    data_ready = pyqtSignal(np.ndarray, np.ndarray)  # (signal, time)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._setup_simulators()
        self.parameters = None
        self.signal_type = "EMG"
        
    def _setup_simulators(self):
        """Initialize simulators"""
        self.simulators = {
            "EMG": EMGSimulator(sampling_rate=1000.0, duration=10.0),
            "ECG": ECGSimulator(sampling_rate=1000.0, duration=10.0),
            "EOG": EOGSimulator(sampling_rate=1000.0, duration=10.0)
        }
    
    def update_parameters(self, signal_type: str, parameters: Dict[str, Any]):
        """Update simulation parameters"""
        self.signal_type = signal_type
        self.parameters = parameters
        self.start()  # Start the thread
    
    def run(self):
        """Generate signal data in separate thread"""
        try:
            simulator = self.simulators[self.signal_type]
            
            # Extract signal-specific parameters
            params = {}
            if self.signal_type == "EMG":
                emg_params = self.parameters.get('emg_params', {})
                params.update({
                    'activation_level': emg_params.get('activation_level', 0.5),
                    'contraction_type': emg_params.get('contraction_type', 'isometric'),
                    'movement_pattern': emg_params.get('movement_pattern', None)
                })
            elif self.signal_type == "ECG":
                ecg_params = self.parameters.get('ecg_params', {})
                params.update({
                    'heart_rate': ecg_params.get('heart_rate', 60),
                    'condition': ecg_params.get('condition', None)
                })
            elif self.signal_type == "EOG":
                eog_params = self.parameters.get('eog_params', {})
                params.update({
                    'movement_type': eog_params.get('movement_type', 'saccades'),
                    'amplitude': eog_params.get('amplitude', 10),
                    'frequency': eog_params.get('frequency', 1.0)
                })
            
            # Update common parameters
            params.update({
                'sampling_rate': self.parameters.get('sampling_rate', 1000.0),
                'duration': self.parameters.get('duration', 10.0)
            })
            
            # Generate signal
            signal = simulator.generate(**params)
            time = np.linspace(0, params['duration'], len(signal))
            
            # Emit result
            self.data_ready.emit(signal, time)
            
        except Exception as e:
            self.error.emit(str(e))