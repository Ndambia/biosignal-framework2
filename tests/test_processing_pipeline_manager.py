import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np
import json
import os

from ui.processing_pipeline_manager import (
    ProcessingPipelineManager,
    PipelineStep,
    PipelineTemplate
)

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication(sys.argv)

@pytest.fixture
def pipeline_manager(app):
    """Create a ProcessingPipelineManager instance."""
    return ProcessingPipelineManager()

@pytest.fixture
def sample_template():
    """Create a sample pipeline template."""
    return PipelineTemplate(
        name="Test Template",
        description="Test pipeline template",
        steps=[
            PipelineStep(
                type="filter",
                config={
                    'type': 'Bandpass Filter',
                    'parameters': {
                        'lowcut': 20,
                        'highcut': 450,
                        'order': 4
                    }
                }
            ),
            PipelineStep(
                type="normalize",
                config={
                    'method': 'Z-score',
                    'parameters': {
                        'zscore_robust': False
                    }
                }
            )
        ]
    )

def test_initial_state(pipeline_manager):
    """Test initial state of pipeline manager."""
    assert len(pipeline_manager.pipeline) == 0
    assert pipeline_manager.worker is None
    assert pipeline_manager.template_combo.currentText() == "Custom Pipeline"
    assert not pipeline_manager.stop_btn.isEnabled()
    assert pipeline_manager.run_btn.isEnabled()

def test_add_steps(pipeline_manager):
    """Test adding pipeline steps."""
    # Add filter step
    pipeline_manager._add_step("filter")
    assert len(pipeline_manager.pipeline) == 1
    assert pipeline_manager.pipeline[0].type == "filter"
    assert pipeline_manager.pipeline[0].enabled
    
    # Add normalization step
    pipeline_manager._add_step("normalize")
    assert len(pipeline_manager.pipeline) == 2
    assert pipeline_manager.pipeline[1].type == "normalize"
    
    # Add segmentation step
    pipeline_manager._add_step("segment")
    assert len(pipeline_manager.pipeline) == 3
    assert pipeline_manager.pipeline[2].type == "segment"

def test_remove_step(pipeline_manager):
    """Test removing pipeline steps."""
    # Add steps
    pipeline_manager._add_step("filter")
    pipeline_manager._add_step("normalize")
    assert len(pipeline_manager.pipeline) == 2
    
    # Remove first step
    pipeline_manager._remove_step(0)
    assert len(pipeline_manager.pipeline) == 1
    assert pipeline_manager.pipeline[0].type == "normalize"
    
    # Remove remaining step
    pipeline_manager._remove_step(0)
    assert len(pipeline_manager.pipeline) == 0

def test_move_step(pipeline_manager):
    """Test moving pipeline steps."""
    # Add steps
    pipeline_manager._add_step("filter")
    pipeline_manager._add_step("normalize")
    pipeline_manager._add_step("segment")
    
    # Move step up
    pipeline_manager._move_step(2, 1)
    assert pipeline_manager.pipeline[1].type == "segment"
    assert pipeline_manager.pipeline[2].type == "normalize"
    
    # Move step down
    pipeline_manager._move_step(0, 1)
    assert pipeline_manager.pipeline[0].type == "segment"
    assert pipeline_manager.pipeline[1].type == "filter"

def test_toggle_step(pipeline_manager):
    """Test toggling step enabled state."""
    pipeline_manager._add_step("filter")
    step = pipeline_manager.pipeline[0]
    
    # Initially enabled
    assert step.enabled
    
    # Toggle off
    pipeline_manager._toggle_step(step)
    assert not step.enabled
    
    # Toggle on
    pipeline_manager._toggle_step(step)
    assert step.enabled

def test_save_load_template(pipeline_manager, sample_template, tmp_path):
    """Test saving and loading pipeline templates."""
    # Save template
    template_path = tmp_path / "test_template.json"
    with open(template_path, 'w') as f:
        json.dump(sample_template.to_dict(), f)
    
    # Load template
    pipeline_manager._load_template(str(template_path))
    
    # Verify loaded pipeline
    assert len(pipeline_manager.pipeline) == 2
    assert pipeline_manager.pipeline[0].type == "filter"
    assert pipeline_manager.pipeline[1].type == "normalize"
    
    # Verify step configurations
    filter_step = pipeline_manager.pipeline[0]
    assert filter_step.config['type'] == "Bandpass Filter"
    assert filter_step.config['parameters']['lowcut'] == 20
    assert filter_step.config['parameters']['highcut'] == 450
    
    norm_step = pipeline_manager.pipeline[1]
    assert norm_step.config['method'] == "Z-score"
    assert not norm_step.config['parameters']['zscore_robust']

def test_run_pipeline(pipeline_manager):
    """Test running the processing pipeline."""
    # Add steps
    pipeline_manager._add_step("filter")
    pipeline_manager._add_step("normalize")
    
    # Create test data
    data = np.random.randn(1000)
    
    # Start processing
    pipeline_manager.run_pipeline(data)
    
    # Verify state
    assert not pipeline_manager.run_btn.isEnabled()
    assert pipeline_manager.stop_btn.isEnabled()
    assert pipeline_manager.status_label.text() == "Processing..."
    assert pipeline_manager.progress_bar.value() == 0
    
    # Stop processing
    pipeline_manager.stop_pipeline()
    
    # Verify state after stopping
    assert pipeline_manager.run_btn.isEnabled()
    assert not pipeline_manager.stop_btn.isEnabled()
    assert pipeline_manager.status_label.text() == "Processing stopped"

def test_pipeline_progress(pipeline_manager):
    """Test pipeline progress updates."""
    pipeline_manager._update_progress(50)
    assert pipeline_manager.progress_bar.value() == 50
    
    # Time remaining should be shown
    assert "Time remaining:" in pipeline_manager.time_label.text()

def test_error_handling(pipeline_manager):
    """Test error handling during processing."""
    pipeline_manager._processing_error("Test error")
    
    assert pipeline_manager.status_label.text() == "Error"
    assert pipeline_manager.run_btn.isEnabled()
    assert not pipeline_manager.stop_btn.isEnabled()

def test_processing_completion(pipeline_manager):
    """Test processing completion handling."""
    pipeline_manager._processing_finished()
    
    assert pipeline_manager.status_label.text() == "Processing complete"
    assert pipeline_manager.time_label.text() == "Time remaining: --:--"
    assert pipeline_manager.run_btn.isEnabled()
    assert not pipeline_manager.stop_btn.isEnabled()

def test_empty_pipeline_validation(pipeline_manager):
    """Test validation of empty pipeline."""
    data = np.random.randn(1000)
    pipeline_manager.run_pipeline(data)
    
    # Should not start processing
    assert pipeline_manager.worker is None
    assert pipeline_manager.run_btn.isEnabled()
    assert not pipeline_manager.stop_btn.isEnabled()

def test_concurrent_run_prevention(pipeline_manager):
    """Test prevention of concurrent pipeline runs."""
    # Add a step
    pipeline_manager._add_step("filter")
    
    # Start first run
    data = np.random.randn(1000)
    pipeline_manager.run_pipeline(data)
    
    # Try to start second run
    pipeline_manager.run_pipeline(data)
    
    # Should still be in first run state
    assert not pipeline_manager.run_btn.isEnabled()
    assert pipeline_manager.stop_btn.isEnabled()
    assert pipeline_manager.status_label.text() == "Processing..."