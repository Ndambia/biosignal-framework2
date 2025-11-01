import sys
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from unittest.mock import MagicMock, patch
from datetime import datetime

from ui.panels.batch_processing_panel import BatchProcessingPanel
from ui.visualization.batch_comparison_view import BatchComparisonView
from ui.data_manager import DataManager
from ui.error_handling import ErrorHandler, ErrorInfo, ErrorSeverity, ErrorCategory

@pytest.fixture
def app():
    """Create QApplication instance for tests."""
    return QApplication(sys.argv)

@pytest.fixture
def error_handler():
    """Create ErrorHandler instance for tests."""
    return ErrorHandler()

@pytest.fixture
def data_manager(tmp_path):
    """Create DataManager instance with temporary cache directory."""
    return DataManager(cache_dir=str(tmp_path / "cache"))

@pytest.fixture
def batch_panel(app, error_handler, data_manager):
    """Create BatchProcessingPanel instance for tests."""
    return BatchProcessingPanel(error_handler, data_manager)

@pytest.fixture
def comparison_view(app):
    """Create BatchComparisonView instance for tests."""
    return BatchComparisonView()

def generate_test_data(num_samples: int = 1000) -> np.ndarray:
    """Generate test signal data."""
    t = np.linspace(0, 10, num_samples)
    return np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(num_samples)

def test_batch_processing_workflow(batch_panel, data_manager):
    """Test complete batch processing workflow integration."""
    # Setup test data
    test_data = generate_test_data()
    config = {
        "dataset_paths": ["/test/data.h5"],
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001
    }
    
    # Mock signals for tracking
    started_signal = MagicMock()
    completed_signal = MagicMock()
    error_signal = MagicMock()
    progress_signal = MagicMock()
    
    batch_panel.batch_started.connect(started_signal)
    batch_panel.batch_completed.connect(completed_signal)
    batch_panel.batch_error.connect(error_signal)
    batch_panel.progress_updated.connect(progress_signal)
    
    # Set configuration
    batch_panel.config_panel.set_configuration(config)
    
    # Start batch processing
    batch_panel._start_batch()
    
    # Verify signals
    assert started_signal.called
    assert batch_panel.pause_button.isEnabled()
    assert batch_panel.stop_button.isEnabled()
    
    # Simulate progress updates
    for progress in range(0, 101, 20):
        batch_panel._update_progress(
            "Processing batch",
            progress,
            {"metric": progress / 100},
            test_data
        )
        
    # Verify progress tracking
    assert progress_signal.call_count >= 5
    assert batch_panel.progress_bar.value() == 100
    
    # Complete batch
    metrics = {
        "accuracy": 0.95,
        "loss": 0.05,
        "data": test_data
    }
    batch_panel._on_task_completed("test_task", metrics)
    
    # Verify completion
    assert completed_signal.called
    assert not error_signal.called
    assert not batch_panel.pause_button.isEnabled()
    assert not batch_panel.stop_button.isEnabled()

def test_visualization_integration(batch_panel, data_manager):
    """Test integration between batch processing and visualization."""
    # Setup test data
    test_data = generate_test_data()
    batch_id = "test_batch"
    
    # Track visualization updates
    update_count = 0
    def count_updates():
        nonlocal update_count
        update_count += 1
    
    batch_panel.comparison_view.data_updated.connect(count_updates)
    
    # Set auto-update settings
    batch_panel.auto_update_check.setChecked(True)
    batch_panel.threshold_spin.setValue(10)
    batch_panel.buffer_spin.setValue(100)
    
    # Simulate batch updates
    for i in range(5):
        batch_data = test_data[i*200:(i+1)*200]
        batch_panel._update_progress(
            f"Batch {i+1}",
            (i+1) * 20,
            {"batch": i},
            batch_data
        )
    
    # Verify visualization updates
    assert update_count > 0
    assert len(batch_panel.comparison_view.batch_data) > 0
    assert batch_panel.comparison_view.current_batch is not None

def test_error_handling_integration(batch_panel, data_manager):
    """Test error handling integration across components."""
    # Setup error tracking
    error_signal = MagicMock()
    batch_panel.batch_error.connect(error_signal)
    
    # Simulate configuration error
    batch_panel.config_panel.set_configuration({})  # Empty config
    batch_panel._start_batch()
    
    # Verify error handling
    assert error_signal.called
    error_info = error_signal.call_args[0][0]
    assert isinstance(error_info, ErrorInfo)
    assert error_info.severity == ErrorSeverity.ERROR
    
    # Verify UI state after error
    assert batch_panel.start_button.isEnabled()
    assert not batch_panel.pause_button.isEnabled()
    assert not batch_panel.stop_button.isEnabled()

def test_cleanup_integration(batch_panel, data_manager):
    """Test cleanup integration across components."""
    # Setup test data
    test_data = generate_test_data()
    
    # Start batch processing
    batch_panel._start_batch()
    batch_panel._update_progress("Test", 50, {}, test_data)
    
    # Perform cleanup
    batch_panel.cleanup()
    
    # Verify cleanup effects
    assert len(batch_panel.workers) == 0
    assert batch_panel.current_batch is None
    assert len(batch_panel.comparison_view.batch_data) == 0
    assert not batch_panel.comparison_view.update_timer.isActive()
    
    # Verify UI state
    assert batch_panel.start_button.isEnabled()
    assert not batch_panel.pause_button.isEnabled()
    assert not batch_panel.stop_button.isEnabled()

def test_data_caching_integration(batch_panel, data_manager):
    """Test data caching integration."""
    # Setup test data
    test_data = generate_test_data()
    batch_id = f"batch_{int(datetime.now().timestamp())}"
    
    # Cache test data
    data_manager.cache_result(
        test_data,
        {"param": "value"},
        {"result": "success"},
        batch_id
    )
    
    # Verify data retrieval
    cached_result = data_manager.get_cached_result(
        test_data,
        {"param": "value"},
        batch_id
    )
    assert cached_result is not None
    assert cached_result["result"] == "success"
    
    # Verify cache cleanup
    data_manager.clear_cache(batch_id)
    assert data_manager.get_cached_result(
        test_data,
        {"param": "value"},
        batch_id
    ) is None

def test_real_time_updates_integration(batch_panel, data_manager):
    """Test real-time updates integration."""
    # Setup test data
    test_data = generate_test_data()
    update_rate = 10  # Hz
    
    # Configure real-time settings
    batch_panel.comparison_view.update_rate = update_rate
    batch_panel.comparison_view.start_real_time_updates()
    
    # Track updates
    updates = []
    def track_update():
        updates.append(datetime.now())
    batch_panel.comparison_view.data_updated.connect(track_update)
    
    # Simulate real-time data
    for i in range(5):
        batch_data = test_data[i*100:(i+1)*100]
        batch_panel._update_progress(
            "Real-time update",
            i * 20,
            {},
            batch_data
        )
        QTimer.singleShot(100, lambda: None)  # Wait for update
    
    # Verify update timing
    if len(updates) >= 2:
        intervals = [(updates[i+1] - updates[i]).total_seconds() 
                    for i in range(len(updates)-1)]
        avg_interval = sum(intervals) / len(intervals)
        assert abs(avg_interval - (1.0 / update_rate)) < 0.1

if __name__ == '__main__':
    pytest.main([__file__])