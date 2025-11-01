import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np
import json
import os
from datetime import datetime

from ui.visualization.comparison_view import ComparisonView
from ui.visualization.history_view import (
    HistoryView, ProcessingStep, ProcessingHistory
)

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication(sys.argv)

@pytest.fixture
def comparison_view(app):
    """Create a ComparisonView instance."""
    return ComparisonView()

@pytest.fixture
def history_view(app):
    """Create a HistoryView instance."""
    return HistoryView()

def test_comparison_view_initial_state(comparison_view):
    """Test initial state of comparison view."""
    assert comparison_view.sync_zoom.isChecked()
    assert comparison_view.show_grid.isChecked()
    assert comparison_view.plot_type.currentText() == "Time Domain"

def test_comparison_view_plot_updates(comparison_view):
    """Test plot updates with signal data."""
    # Create test signal
    t = np.linspace(0, 1, 1000)
    data = np.sin(2 * np.pi * 10 * t)
    
    # Update original signal
    comparison_view.update_original(data)
    
    # Update processed signal
    processed_data = data * 2  # Simple amplification
    comparison_view.update_processed(processed_data)
    
    # Verify metrics were updated
    assert "Signal Metrics:" in comparison_view.original_metrics.text()
    assert "Signal Metrics:" in comparison_view.processed_metrics.text()

def test_comparison_view_plot_type_switching(comparison_view):
    """Test switching between plot types."""
    # Time Domain
    comparison_view.plot_type.setCurrentText("Time Domain")
    assert comparison_view.original_plot.isVisible()
    assert comparison_view.processed_plot.isVisible()
    assert not comparison_view.original_freq_plot.isVisible()
    assert not comparison_view.processed_freq_plot.isVisible()
    
    # Frequency Domain
    comparison_view.plot_type.setCurrentText("Frequency Domain")
    assert not comparison_view.original_plot.isVisible()
    assert not comparison_view.processed_plot.isVisible()
    assert comparison_view.original_freq_plot.isVisible()
    assert comparison_view.processed_freq_plot.isVisible()
    
    # Combined
    comparison_view.plot_type.setCurrentText("Combined")
    assert comparison_view.original_plot.isVisible()
    assert comparison_view.processed_plot.isVisible()
    assert comparison_view.original_freq_plot.isVisible()
    assert comparison_view.processed_freq_plot.isVisible()

def test_comparison_view_sync_control(comparison_view):
    """Test view synchronization control."""
    # Disable sync
    comparison_view.sync_zoom.setChecked(False)
    assert comparison_view.processed_plot.getViewBox().linkedViews() == []
    
    # Enable sync
    comparison_view.sync_zoom.setChecked(True)
    assert comparison_view.processed_plot.getViewBox().linkedViews() != []

def test_comparison_view_grid_control(comparison_view):
    """Test grid visibility control."""
    # Hide grid
    comparison_view.show_grid.setChecked(False)
    # Grid visibility would need to be checked through internal plot state
    
    # Show grid
    comparison_view.show_grid.setChecked(True)
    # Grid visibility would need to be checked through internal plot state

def test_history_view_initial_state(history_view):
    """Test initial state of history view."""
    assert history_view.tree.topLevelItemCount() == 0
    assert len(history_view.history.steps) == 0

def test_history_view_add_step(history_view):
    """Test adding processing steps."""
    # Add a step
    step = history_view.add_step("Filter", {
        'type': 'bandpass',
        'lowcut': 10,
        'highcut': 100
    })
    
    assert len(history_view.history.steps) == 1
    assert history_view.tree.topLevelItemCount() == 1
    
    # Verify tree item
    item = history_view.tree.topLevelItem(0)
    assert item.text(0) == "Filter"
    assert item.text(2) == "Completed"

def test_history_view_clear(history_view):
    """Test clearing history."""
    # Add some steps
    history_view.add_step("Filter", {'type': 'bandpass'})
    history_view.add_step("Normalize", {'method': 'zscore'})
    
    # Clear history
    history_view._clear_history()
    
    assert len(history_view.history.steps) == 0
    assert history_view.tree.topLevelItemCount() == 0

def test_processing_step_serialization():
    """Test ProcessingStep serialization."""
    # Create step
    step = ProcessingStep("Filter", {
        'type': 'bandpass',
        'lowcut': 10,
        'highcut': 100
    })
    
    # Add metrics
    step.metrics = {
        'snr_before': 10.5,
        'snr_after': 15.2
    }
    
    # Convert to dict
    data = step.to_dict()
    
    # Create new step from dict
    new_step = ProcessingStep.from_dict(data)
    
    assert new_step.operation == step.operation
    assert new_step.parameters == step.parameters
    assert new_step.metrics == step.metrics

def test_processing_history_serialization(tmp_path):
    """Test ProcessingHistory serialization."""
    history = ProcessingHistory()
    
    # Add steps
    history.add_step("Filter", {'type': 'bandpass'})
    history.add_step("Normalize", {'method': 'zscore'})
    
    # Save to file
    filename = tmp_path / "test_history.json"
    history.save(str(filename))
    
    # Load from file
    loaded_history = ProcessingHistory.load(str(filename))
    
    assert len(loaded_history.steps) == len(history.steps)
    assert loaded_history.steps[0].operation == history.steps[0].operation
    assert loaded_history.steps[1].operation == history.steps[1].operation

def test_history_view_step_metrics(history_view):
    """Test updating step metrics."""
    # Add step
    step = history_view.add_step("Filter", {'type': 'bandpass'})
    
    # Update metrics
    metrics = {
        'snr_before': 10.5,
        'snr_after': 15.2
    }
    history_view.update_step_metrics(step, metrics)
    
    # Verify metrics were saved
    assert step.metrics == metrics
    
    # Verify tree item status
    item = history_view.tree.topLevelItem(0)
    assert item.text(2) == "Completed"