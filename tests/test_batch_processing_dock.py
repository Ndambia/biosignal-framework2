import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.docks.batch_processing_dock import BatchProcessingDock

@pytest.fixture
def app():
    """Create QApplication instance for tests."""
    return QApplication([])

@pytest.fixture
def dock(app):
    """Create BatchProcessingDock instance for tests."""
    return BatchProcessingDock()

def test_dock_initialization(dock):
    """Test dock is properly initialized."""
    assert dock.windowTitle() == "Batch Processing"
    assert dock._current_state == "config"
    assert dock.isVisible()

def test_dock_panels(dock):
    """Test all required panels are initialized."""
    assert hasattr(dock, 'config_panel')
    assert hasattr(dock, 'monitor_panel')
    assert hasattr(dock, 'process_panel')
    assert dock.stack.count() == 3

def test_state_persistence(dock):
    """Test state saving and restoration."""
    # Set up initial state
    dock.show_monitor()
    dock.setVisible(False)
    
    # Save state
    state = dock.save_state()
    
    # Verify saved state
    assert state["current_state"] == "monitor"
    assert state["visible"] is False
    assert "config_state" in state
    assert "monitor_state" in state
    assert "process_state" in state
    
    # Create new dock and restore state
    new_dock = BatchProcessingDock()
    new_dock.restore_state(state)
    
    # Verify restored state
    assert new_dock._current_state == "monitor"
    assert not new_dock.isVisible()

def test_panel_switching(dock):
    """Test panel switching functionality."""
    # Test config -> monitor -> process -> config flow
    assert dock._current_state == "config"
    
    dock.show_monitor()
    assert dock._current_state == "monitor"
    assert dock.stack.currentWidget() == dock.monitor_panel
    
    dock.show_processing()
    assert dock._current_state == "process"
    assert dock.stack.currentWidget() == dock.process_panel
    
    dock.show_config()
    assert dock._current_state == "config"
    assert dock.stack.currentWidget() == dock.config_panel

def test_error_handling(dock):
    """Test error handling in dock."""
    # Simulate error in monitor panel
    dock.show_monitor()
    dock._on_task_error("task1", "Test error")
    
    # Should switch back to config panel on error
    assert dock._current_state == "config"
    assert dock.stack.currentWidget() == dock.config_panel

def test_batch_completion_flow(dock):
    """Test batch processing completion flow."""
    # Start in config
    assert dock._current_state == "config"
    
    # Move to processing
    dock.show_processing()
    assert dock._current_state == "process"
    
    # Simulate batch completion
    dock._on_batch_completed()
    
    # Should return to config
    assert dock._current_state == "config"
    assert dock.stack.currentWidget() == dock.config_panel

def test_state_signal_emission(dock, qtbot):
    """Test state change signals are properly emitted."""
    # Set up signal tracking
    with qtbot.waitSignal(dock.state_changed, timeout=1000) as blocker:
        dock.show_monitor()
    
    # Verify signal was emitted with correct state
    assert blocker.args == ["monitor"]