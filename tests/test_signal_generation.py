import sys
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

from ui.panels import EMGControlPanel, ECGControlPanel, EOGControlPanel, NoiseArtifactPanel
from ui.visualization import SignalPlotView
from ui.data_manager import DataManager

class TestSignalGeneration(unittest.TestCase):
    """Test signal generation functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Create application for testing."""
        cls.app = QApplication(sys.argv)
        cls.data_manager = DataManager()
        
    def setUp(self):
        """Set up test case."""
        # Create panels
        self.emg_panel = EMGControlPanel()
        self.ecg_panel = ECGControlPanel()
        self.eog_panel = EOGControlPanel()
        self.noise_panel = NoiseArtifactPanel()
        
        # Create visualization
        self.signal_view = SignalPlotView()
        
        # Connect signals
        self.data_manager.signals.signal_generated.connect(
            self.signal_view.set_data
        )
        
    def test_emg_isometric(self):
        """Test isometric EMG generation."""
        # Set parameters
        params = {
            'pattern_type': 'Isometric',
            'intensity': 0.7,
            'duration': 5.0,
            'fatigue_rate': 0.3
        }
        self.emg_panel.set_parameters(params)
        
        # Get generated signal
        signal = self._get_signal_data()
        
        # Verify signal properties
        self.assertIsNotNone(signal)
        self.assertEqual(len(signal), 5000)  # 5s at 1kHz
        self.assertTrue(np.all(np.abs(signal) <= 1.0))  # Check amplitude
        
    def test_emg_dynamic(self):
        """Test dynamic EMG generation."""
        params = {
            'pattern_type': 'Dynamic',
            'ramp_type': 'Linear',
            'max_intensity': 0.8,
            'ramp_duration': 2.0
        }
        self.emg_panel.set_parameters(params)
        
        signal = self._get_signal_data()
        self.assertIsNotNone(signal)
        self.assertEqual(len(signal), 2000)
        
        # Verify ramp profile
        peak_idx = np.argmax(np.abs(signal))
        self.assertGreater(peak_idx, len(signal) // 2)  # Peak in second half
        
    def test_ecg_normal(self):
        """Test normal sinus rhythm ECG."""
        params = {
            'heart_rate': 75,
            'condition': 'normal_sinus_rhythm',
            'lead': 'II'
        }
        self.ecg_panel.set_parameters(params)
        
        signal = self._get_signal_data()
        self.assertIsNotNone(signal)
        
        # Verify heart rate
        peaks = self._find_peaks(signal)
        ibi = np.diff(peaks) / 1000  # Convert to seconds
        hr = 60 / np.mean(ibi)
        self.assertAlmostEqual(hr, 75, delta=5)
        
    def test_ecg_afib(self):
        """Test atrial fibrillation ECG."""
        params = {
            'condition': 'atrial_fibrillation',
            'heart_rate': 120,
            'lead': 'II'
        }
        self.ecg_panel.set_parameters(params)
        
        signal = self._get_signal_data()
        self.assertIsNotNone(signal)
        
        # Verify irregular rhythm
        peaks = self._find_peaks(signal)
        ibi = np.diff(peaks) / 1000
        self.assertGreater(np.std(ibi), 0.1)  # High variability
        
    def test_eog_saccade(self):
        """Test saccadic eye movements."""
        params = {
            'movement_type': 'saccade',
            'amplitude': 100,
            'direction': 'Horizontal',
            'frequency': 1.0
        }
        self.eog_panel.set_parameters(params)
        
        signal = self._get_signal_data()
        self.assertIsNotNone(signal)
        
        # Verify saccade properties
        peaks = self._find_peaks(np.abs(signal))
        self.assertGreaterEqual(len(peaks), 1)
        self.assertLessEqual(np.max(np.abs(signal)), 100)
        
    def test_noise_addition(self):
        """Test noise addition to signals."""
        # Generate base signal
        self.emg_panel.set_parameters({
            'pattern_type': 'Isometric',
            'intensity': 0.5,
            'duration': 1.0
        })
        clean_signal = self._get_signal_data()
        
        # Add noise
        noise_params = {
            'gaussian': {
                'enabled': True,
                'std': 0.1
            }
        }
        self.noise_panel.set_parameters(noise_params)
        noisy_signal = self._get_signal_data()
        
        # Verify noise addition
        self.assertGreater(
            np.std(noisy_signal - clean_signal),
            0.05
        )
        
    def _get_signal_data(self) -> np.ndarray:
        """Get generated signal data."""
        signal_data = []
        
        def collect_signal(data, *args):
            signal_data.append(data)
            
        self.signal_view.set_data = collect_signal
        QTest.qWait(100)  # Wait for signal generation
        
        return signal_data[-1] if signal_data else None
        
    def _find_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Find signal peaks."""
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i-1] < signal[i] > signal[i+1]:
                peaks.append(i)
        return np.array(peaks)

if __name__ == '__main__':
    unittest.main()