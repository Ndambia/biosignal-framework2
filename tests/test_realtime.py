import sys
import time
import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt, QTimer

from ui.panels import EMGControlPanel
from ui.visualization import SignalPlotView
from ui.workers import SignalWorker
from ui.data_manager import DataManager

class TestRealtimeOperations(unittest.TestCase):
    """Test real-time signal generation and UI responsiveness."""
    
    @classmethod
    def setUpClass(cls):
        """Create application for testing."""
        cls.app = QApplication(sys.argv)
        cls.data_manager = DataManager()
        
    def setUp(self):
        """Set up test case."""
        # Create components
        self.emg_panel = EMGControlPanel()
        self.signal_view = SignalPlotView()
        self.worker = SignalWorker()
        
        # Connect signals
        self.worker.data_ready.connect(self.signal_view.set_data)
        self.emg_panel.parameters_changed.connect(self._on_params_changed)
        
        # Store signals for testing
        self.signals = []
        self.signal_times = []
        self.last_update = time.time()
        
    def _on_params_changed(self, params):
        """Handle parameter changes."""
        self.worker.update_parameters('EMG', params)
        
    def _on_data_ready(self, signal, time_values):
        """Handle new signal data."""
        self.signals.append(signal)
        self.signal_times.append(time.time() - self.last_update)
        self.last_update = time.time()
        
    def test_update_rate(self):
        """Test signal update rate."""
        # Connect to data ready signal
        self.worker.data_ready.connect(self._on_data_ready)
        
        # Set up continuous generation
        params = {
            'pattern_type': 'Isometric',
            'intensity': 0.7,
            'duration': 0.1  # Short duration for faster updates
        }
        self.emg_panel.set_parameters(params)
        
        # Run for 2 seconds
        QTest.qWait(2000)
        
        # Verify update rate
        self.assertGreater(len(self.signals), 10)  # At least 10 updates
        
        # Check timing
        update_intervals = np.diff(self.signal_times)
        mean_interval = np.mean(update_intervals)
        self.assertLess(mean_interval, 0.2)  # Updates faster than 200ms
        
    def test_ui_responsiveness(self):
        """Test UI responsiveness during signal generation."""
        # Start continuous generation
        params = {
            'pattern_type': 'Dynamic',
            'ramp_type': 'Linear',
            'max_intensity': 0.8,
            'ramp_duration': 5.0  # Longer duration
        }
        self.emg_panel.set_parameters(params)
        
        # Create event timestamps
        events = []
        
        def record_event():
            events.append(time.time())
            
        # Set up periodic UI interaction
        timer = QTimer()
        timer.timeout.connect(record_event)
        timer.start(100)  # Try to interact every 100ms
        
        # Run for 2 seconds
        QTest.qWait(2000)
        timer.stop()
        
        # Verify event timing
        intervals = np.diff(events)
        max_delay = np.max(intervals)
        self.assertLess(max_delay, 0.2)  # No UI freeze > 200ms
        
    def test_concurrent_operations(self):
        """Test multiple operations running concurrently."""
        completed_ops = []
        
        def operation_complete(op_id):
            completed_ops.append(op_id)
            
        # Start multiple operations
        workers = []
        for i in range(3):
            worker = SignalWorker()
            worker.completed.connect(
                lambda id=i: operation_complete(id)
            )
            workers.append(worker)
            
            # Configure different signals
            params = {
                'pattern_type': 'Isometric',
                'intensity': 0.5 + i*0.2,
                'duration': 1.0
            }
            worker.update_parameters('EMG', params)
            
        # Wait for completion
        QTest.qWait(2000)
        
        # Verify all operations completed
        self.assertEqual(len(completed_ops), 3)
        
    def test_parameter_updates(self):
        """Test real-time parameter updates."""
        update_received = []
        
        def on_update(signal, time):
            update_received.append(True)
            
        self.worker.data_ready.connect(on_update)
        
        # Start with initial parameters
        params = {
            'pattern_type': 'Isometric',
            'intensity': 0.5,
            'duration': 1.0
        }
        self.emg_panel.set_parameters(params)
        
        # Wait for first update
        QTest.qWait(100)
        self.assertTrue(update_received)
        
        # Change parameters
        update_received.clear()
        params['intensity'] = 0.8
        self.emg_panel.set_parameters(params)
        
        # Wait for update with new parameters
        QTest.qWait(100)
        self.assertTrue(update_received)
        
    def test_visualization_performance(self):
        """Test visualization update performance."""
        frame_times = []
        last_update = time.time()
        
        def on_data_updated():
            nonlocal last_update
            current = time.time()
            frame_times.append(current - last_update)
            last_update = current
            
        self.signal_view.data_updated.connect(on_data_updated)
        
        # Generate rapid updates
        params = {
            'pattern_type': 'Isometric',
            'intensity': 0.7,
            'duration': 0.05  # Very short duration
        }
        self.emg_panel.set_parameters(params)
        
        # Run for 1 second
        QTest.qWait(1000)
        
        # Check frame timing
        self.assertGreater(len(frame_times), 10)
        mean_frame_time = np.mean(frame_times)
        self.assertLess(mean_frame_time, 0.1)  # Updates faster than 100ms

if __name__ == '__main__':
    unittest.main()