from PyQt6.QtCore import pyqtSignal
import numpy as np
from typing import Dict, Any, Optional
from .base_worker import BaseWorker, ConfigurationError, OperationError
from simulation import EMGSimulator, ECGSimulator, EOGSimulator

class SignalWorker(BaseWorker):
    """Worker thread for signal generation."""
    
    # Additional signals specific to signal generation
    signal_ready = pyqtSignal(np.ndarray, np.ndarray)  # signal, time
    parameters_updated = pyqtSignal(dict)  # parameters
    
    def __init__(self):
        super().__init__(operation_type="signal_generation")
        self._setup_simulators()
        self.parameters = None
        self.signal_type = "EMG"
        
    def _setup_simulators(self):
        """Initialize signal simulators."""
        try:
            self.simulators = {
                "EMG": EMGSimulator(sampling_rate=1000.0, duration=10.0),
                "ECG": ECGSimulator(sampling_rate=1000.0, duration=10.0),
                "EOG": EOGSimulator(sampling_rate=1000.0, duration=10.0)
            }
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize simulators: {str(e)}")
        
    def configure(self, signal_type: str, parameters: Dict[str, Any]):
        """Configure signal generation parameters."""
        if signal_type not in self.simulators:
            raise ConfigurationError(f"Unsupported signal type: {signal_type}")
            
        self.signal_type = signal_type
        self.parameters = parameters
        self.parameters_updated.emit(parameters)
        
    def _execute(self) -> Dict[str, Any]:
        """Execute signal generation."""
        try:
            # Get simulator
            simulator = self.simulators[self.signal_type]
            
            # Extract parameters
            params = self._extract_parameters()
            
            # Report status
            self.report_status(f"Generating {self.signal_type} signal...")
            self.report_progress(0, "Starting generation")
            
            # Generate signal
            signal = simulator.generate(**params)
            
            # Create time array
            time = np.linspace(0, params['duration'], len(signal))
            
            # Report progress
            self.report_progress(50, "Signal generated")
            
            # Apply noise and artifacts if specified
            if 'noise' in self.parameters:
                signal = self._apply_noise(signal, self.parameters['noise'])
                self.report_progress(75, "Noise applied")
                
            if 'artifacts' in self.parameters:
                signal = self._apply_artifacts(signal, self.parameters['artifacts'])
                self.report_progress(90, "Artifacts applied")
            
            # Emit signal
            self.signal_ready.emit(signal, time)
            
            # Return results
            return {
                'signal': signal,
                'time': time,
                'parameters': self.parameters,
                'signal_type': self.signal_type
            }
            
        except Exception as e:
            raise OperationError(f"Signal generation failed: {str(e)}")
            
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract simulator-specific parameters."""
        params = {}
        
        # Common parameters
        params.update({
            'sampling_rate': self.parameters.get('sampling_rate', 1000.0),
            'duration': self.parameters.get('duration', 10.0)
        })
        
        # Signal-specific parameters
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
                'condition': ecg_params.get('condition', 'normal'),
                'lead': ecg_params.get('lead', 'II')
            })
            
        elif self.signal_type == "EOG":
            eog_params = self.parameters.get('eog_params', {})
            params.update({
                'movement_type': eog_params.get('movement_type', 'saccade'),
                'amplitude': eog_params.get('amplitude', 100),
                'frequency': eog_params.get('frequency', 1.0),
                'direction': eog_params.get('direction', 'horizontal')
            })
            
        return params
        
    def _apply_noise(self, signal: np.ndarray, noise_params: Dict[str, Any]) -> np.ndarray:
        """Apply noise to signal."""
        try:
            # Apply each noise type
            for noise_type, params in noise_params.items():
                if not params.get('enabled', True):
                    continue
                    
                self.report_status(f"Applying {noise_type} noise...")
                
                if noise_type == 'gaussian':
                    std = params.get('std', 0.1)
                    signal += np.random.normal(0, std, len(signal))
                    
                elif noise_type == 'powerline':
                    freq = params.get('frequency', 50)
                    amplitude = params.get('amplitude', 0.1)
                    t = np.linspace(0, self.parameters['duration'], len(signal))
                    signal += amplitude * np.sin(2 * np.pi * freq * t)
                    
                # Add more noise types as needed
                
            return signal
            
        except Exception as e:
            raise OperationError(f"Failed to apply noise: {str(e)}")
            
    def _apply_artifacts(self, signal: np.ndarray, artifact_params: Dict[str, Any]) -> np.ndarray:
        """Apply artifacts to signal."""
        try:
            # Apply each artifact type
            for artifact_type, params in artifact_params.items():
                if not params.get('enabled', True):
                    continue
                    
                self.report_status(f"Applying {artifact_type} artifact...")
                
                if artifact_type == 'movement':
                    # Simulate movement artifact
                    amplitude = params.get('amplitude', 0.5)
                    duration = params.get('duration', 0.2)
                    samples = int(duration * self.parameters['sampling_rate'])
                    start = params.get('start', 0)
                    start_idx = int(start * self.parameters['sampling_rate'])
                    
                    artifact = amplitude * np.random.randn(samples)
                    signal[start_idx:start_idx + samples] += artifact
                    
                elif artifact_type == 'electrode':
                    # Simulate electrode pop
                    amplitude = params.get('amplitude', 2.0)
                    position = params.get('position', 0.5)
                    pos_idx = int(position * len(signal))
                    
                    signal[pos_idx] += amplitude
                    
                # Add more artifact types as needed
                
            return signal
            
        except Exception as e:
            raise OperationError(f"Failed to apply artifacts: {str(e)}")