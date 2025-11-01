import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from unittest.mock import MagicMock, patch
import numpy as np

from ui.panels.batch_processing_panel import BatchProcessingPanel
from ui.panels.batch_configuration_panel import BatchConfigurationPanel
from ui.panels.batch_monitor_panel import BatchMonitorPanel, TaskProgressWidget
from ui.panels.result_comparison_panel import ResultComparisonPanel
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
def batch_panel(app, error_handler):
    """Create BatchProcessingPanel instance for tests."""
    return BatchProcessingPanel(error_handler)

@pytest.fixture
def config_panel(app, error_handler):
    """Create BatchConfigurationPanel instance for tests."""
    return BatchConfigurationPanel(error_handler)

@pytest.fixture
def monitor_panel(app, error_handler):
    """Create BatchMonitorPanel instance for tests."""
    return BatchMonitorPanel(error_handler)

@pytest.fixture
def results_panel(app, error_handler):
    """Create ResultComparisonPanel instance for tests."""
    return ResultComparisonPanel(error_handler)

def test_batch_panel_initialization(batch_panel):
    """Test BatchProcessingPanel initializes correctly."""
    assert batch_panel.config_panel is not None
    assert batch_panel.monitor_panel is not None
    assert batch_panel.results_panel is not None
    assert batch_panel.start_button.isEnabled()
    assert not batch_panel.pause_button.isEnabled()
    assert not batch_panel.stop_button.isEnabled()

def test_config_panel_validation(config_panel):
    """Test configuration validation."""
    # Test invalid configuration
    config_panel.dataset_config.set_value([])
    assert not config_panel.validate_configuration()
    
    # Test valid configuration
    config_panel.dataset_config.set_value(["/path/to/data.h5"])
    config_panel.set_parameters({
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "window_size": 256,
        "overlap": 50
    })
    assert config_panel.validate_configuration()

def test_monitor_panel_task_tracking(monitor_panel):
    """Test task monitoring functionality."""
    # Add task
    task_id = "task1"
    monitor_panel.add_task(task_id, "Test Task")
    assert task_id in monitor_panel.task_widgets
    
    # Update progress
    monitor_panel.update_task_progress(
        task_id, 
        50, 
        "Processing", 
        {"memory_usage": 1024 * 1024}  # 1MB
    )
    widget = monitor_panel.task_widgets[task_id]
    assert widget.progress_bar.value() == 50
    assert widget.status_label.text() == "Processing"
    
    # Test error handling
    error_info = ErrorInfo(
        message="Test error",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.PROCESSING
    )
    monitor_panel.set_task_error(task_id, error_info)
    assert widget.status_label.text() == "Error"
    
    # Remove task
    monitor_panel.remove_task(task_id)
    assert task_id not in monitor_panel.task_widgets

def test_results_panel_updates(results_panel):
    """Test results visualization and metrics updates."""
    test_results = {
        "metrics": {
            "accuracy": [0.85, 0.87, 0.86],
            "f1_score": [0.84, 0.86, 0.85],
            "loss": [0.15, 0.13, 0.14]
        },
        "labels": ["Model A", "Model B", "Model C"]
    }
    
    results_panel.update_results(test_results)
    
    # Check metrics table
    assert results_panel.metrics_table.rowCount() == 3
    assert results_panel.metric_combo.count() == 3
    
    # Verify best values are highlighted
    accuracy_best = results_panel.metrics_table.item(0, 1)
    assert float(accuracy_best.text()) == 0.87
    assert accuracy_best.background().color().name() == "#c8ffc8"

def test_batch_processing_workflow(batch_panel, error_handler):
    """Test complete batch processing workflow."""
    # Mock configuration
    config = {
        "dataset_paths": ["/path/to/data.h5"],
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "window_size": 256,
        "overlap": 50
    }
    batch_panel.config_panel.set_parameters(config)
    
    # Setup signal spies
    batch_started_spy = MagicMock()
    batch_completed_spy = MagicMock()
    batch_error_spy = MagicMock()
    
    batch_panel.batch_started.connect(batch_started_spy)
    batch_panel.batch_completed.connect(batch_completed_spy)
    batch_panel.batch_error.connect(batch_error_spy)
    
    # Start batch processing
    batch_panel._start_batch()
    assert batch_started_spy.called
    assert batch_panel.pause_button.isEnabled()
    assert batch_panel.stop_button.isEnabled()
    
    # Simulate task progress
    task_id = "test_task"
    batch_panel.monitor_panel.add_task(task_id, "Test Task")
    batch_panel._update_progress("Test Task", 50)
    
    # Simulate task completion
    metrics = {
        "accuracy": 0.85,
        "f1_score": 0.84,
        "loss": 0.15
    }
    batch_panel._on_task_completed(task_id, metrics)
    assert task_id not in batch_panel.monitor_panel.task_widgets
    
    # Stop batch processing
    batch_panel._stop_batch()
    assert not batch_panel.pause_button.isEnabled()
    assert not batch_panel.stop_button.isEnabled()
    assert batch_panel.start_button.isEnabled()

def test_error_handling(batch_panel, error_handler):
    """Test error handling across panels."""
    # Mock error handler methods
    error_handler.handle_error = MagicMock()
    error_handler.show_error_dialog = MagicMock()
    
    # Trigger configuration error
    batch_panel.config_panel.dataset_config.set_value([])
    batch_panel._start_batch()
    
    # Verify error was handled
    assert error_handler.handle_error.called
    args = error_handler.handle_error.call_args[0]
    assert isinstance(args[0], Exception)
    assert args[1] == ErrorSeverity.ERROR
    
    # Test task error handling
    error_info = ErrorInfo(
        message="Task failed",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.PROCESSING
    )
    batch_panel._on_task_error("test_task", error_info)
    
    # Verify error was propagated
    assert error_handler.show_error_dialog.called

if __name__ == '__main__':
    pytest.main([__file__])