import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

from ui.panels.filter_designer_panel import FilterDesignerPanel
from ui.panels.normalization_panel import NormalizationPanel
from ui.panels.segmentation_panel import SegmentationPanel
from ui.processing_pipeline_manager import ProcessingPipelineManager
from ui.visualization.comparison_view import ComparisonView
from ui.visualization.history_view import HistoryView
from ui.presets.preset_manager import PresetManager
from ui.error_handling import ErrorHandler

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication([])

@pytest.fixture
def error_handler():
    """Create an ErrorHandler instance."""
    return ErrorHandler()

@pytest.fixture
def preset_manager(error_handler):
    """Create a PresetManager instance."""
    return PresetManager(error_handler)

@pytest.fixture
def pipeline_manager(error_handler):
    """Create a ProcessingPipelineManager instance."""
    return ProcessingPipelineManager()

@pytest.fixture
def sample_data():
    """Create sample signal data."""
    t = np.linspace(0, 1, 1000)
    clean = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    noise = np.random.normal(0, 0.1, len(t))
    return clean + noise

def test_filter_to_visualization_flow(app, error_handler, sample_data):
    """Test data flow from filter design to visualization."""
    # Create components
    filter_panel = FilterDesignerPanel()
    comparison_view = ComparisonView()
    
    # Configure filter
    filter_panel.filter_type.set_value("Bandpass Filter")
    filter_panel.lowcut.set_value(5)
    filter_panel.highcut.set_value(15)
    filter_panel.order.set_value(4)
    
    # Update visualization
    comparison_view.update_original(sample_data)
    config = filter_panel.get_filter_config()
    
    # Process data through pipeline manager
    pipeline = ProcessingPipelineManager()
    result = pipeline.apply_filter(sample_data, config)
    
    # Update visualization with result
    comparison_view.update_processed(result.data)
    
    # Verify metrics were updated
    assert "Signal Metrics:" in comparison_view.original_metrics.text()
    assert "Signal Metrics:" in comparison_view.processed_metrics.text()

def test_normalization_to_history_flow(app, error_handler, sample_data):
    """Test data flow from normalization to history tracking."""
    # Create components
    norm_panel = NormalizationPanel()
    history_view = HistoryView()
    
    # Configure normalization
    norm_panel.norm_method.set_value("Z-score")
    config = norm_panel.get_normalization_config()
    
    # Process data
    pipeline = ProcessingPipelineManager()
    result = pipeline.apply_normalization(sample_data, config)
    
    # Add to history
    step = history_view.add_step("Normalize", config)
    history_view.update_step_metrics(step, result.metrics)
    
    # Verify history was updated
    assert history_view.tree.topLevelItemCount() > 0

def test_segmentation_with_preset_flow(app, error_handler, preset_manager, sample_data):
    """Test segmentation with preset application."""
    # Create components
    seg_panel = SegmentationPanel()
    
    # Create preset
    preset = preset_manager.import_preset("tests/data/segmentation_preset.json")
    assert preset is not None
    
    # Apply preset configuration
    seg_panel.seg_method.set_value(preset.parameters['method'])
    if preset.parameters['method'] == "Fixed Window":
        seg_panel.window_size.set_value(preset.parameters['window_size'])
    elif preset.parameters['method'] == "Overlapping Window":
        seg_panel.overlap_size.set_value(preset.parameters['window_size'])
        seg_panel.overlap_percent.set_value(preset.parameters['overlap'])
    
    # Process data
    config = seg_panel.get_segmentation_config()
    pipeline = ProcessingPipelineManager()
    result = pipeline.apply_segmentation(sample_data, config)
    
    # Verify result
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0

def test_complete_processing_pipeline(app, error_handler, sample_data):
    """Test complete processing pipeline flow."""
    # Create pipeline manager
    pipeline = ProcessingPipelineManager()
    
    # Create processing steps
    filter_config = {
        'type': 'Bandpass Filter',
        'parameters': {
            'lowcut': 5,
            'highcut': 15,
            'order': 4
        }
    }
    
    norm_config = {
        'method': 'Z-score',
        'parameters': {}
    }
    
    seg_config = {
        'method': 'Fixed Window',
        'parameters': {
            'window_size': 100
        }
    }
    
    # Process data through pipeline
    filtered = pipeline.apply_filter(sample_data, filter_config)
    assert filtered.success
    
    normalized = pipeline.apply_normalization(filtered.data, norm_config)
    assert normalized.success
    
    segmented = pipeline.apply_segmentation(normalized.data, seg_config)
    assert segmented.success
    
    # Verify final result
    assert isinstance(segmented.data, np.ndarray)
    assert len(segmented.metrics) > 0

def test_error_handling_integration(app, error_handler, sample_data):
    """Test error handling across components."""
    # Create components
    filter_panel = FilterDesignerPanel()
    pipeline = ProcessingPipelineManager()
    
    # Configure invalid filter
    filter_panel.filter_type.set_value("Bandpass Filter")
    filter_panel.lowcut.set_value(15)  # Higher than highcut
    filter_panel.highcut.set_value(5)
    filter_panel.order.set_value(4)
    
    # Process data
    config = filter_panel.get_filter_config()
    result = pipeline.apply_filter(sample_data, config)
    
    # Verify error was handled
    assert not result.success
    assert result.error is not None

def test_preset_pipeline_integration(app, error_handler, preset_manager, sample_data):
    """Test preset application in pipeline."""
    # Create pipeline
    pipeline = ProcessingPipelineManager()
    
    # Load preset
    preset = preset_manager.import_preset("tests/data/pipeline_preset.json")
    assert preset is not None
    
    # Apply each step from preset
    for step in preset.parameters['steps']:
        if step['type'] == 'filter':
            result = pipeline.apply_filter(sample_data, step['config'])
        elif step['type'] == 'normalize':
            result = pipeline.apply_normalization(sample_data, step['config'])
        elif step['type'] == 'segment':
            result = pipeline.apply_segmentation(sample_data, step['config'])
            
        assert result.success
        sample_data = result.data

def test_visualization_sync(app, error_handler, sample_data):
    """Test synchronization between visualization components."""
    # Create components
    comparison_view = ComparisonView()
    history_view = HistoryView()
    
    # Process data
    pipeline = ProcessingPipelineManager()
    
    # Apply filter
    filter_config = {
        'type': 'Bandpass Filter',
        'parameters': {
            'lowcut': 5,
            'highcut': 15,
            'order': 4
        }
    }
    filter_result = pipeline.apply_filter(sample_data, filter_config)
    
    # Update visualizations
    comparison_view.update_original(sample_data)
    comparison_view.update_processed(filter_result.data)
    
    step = history_view.add_step("Filter", filter_config)
    history_view.update_step_metrics(step, filter_result.metrics)
    
    # Verify visualizations are updated
    assert "Signal Metrics:" in comparison_view.original_metrics.text()
    assert "Signal Metrics:" in comparison_view.processed_metrics.text()
    assert history_view.tree.topLevelItemCount() > 0

def test_multi_channel_processing(app, error_handler):
    """Test processing multiple channels."""
    # Create multi-channel data
    t = np.linspace(0, 1, 1000)
    channel1 = np.sin(2 * np.pi * 10 * t)
    channel2 = np.sin(2 * np.pi * 20 * t)
    data = np.vstack([channel1, channel2])
    
    # Create pipeline
    pipeline = ProcessingPipelineManager()
    
    # Process each channel
    results = []
    for channel in data:
        # Apply processing steps
        filter_config = {
            'type': 'Bandpass Filter',
            'parameters': {
                'lowcut': 5,
                'highcut': 25,
                'order': 4
            }
        }
        result = pipeline.apply_filter(channel, filter_config)
        assert result.success
        results.append(result.data)
    
    # Verify results
    processed_data = np.vstack(results)
    assert processed_data.shape == data.shape