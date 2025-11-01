import pytest
import numpy as np
from ui.preprocessing_integration import (
    PreprocessingIntegrator,
    ProcessingStage,
    ProcessingResult
)

@pytest.fixture
def integrator():
    """Create a PreprocessingIntegrator instance."""
    return PreprocessingIntegrator()

@pytest.fixture
def sample_data():
    """Create sample signal data."""
    t = np.linspace(0, 1, 1000)
    clean = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    noise = np.random.normal(0, 0.1, len(t))
    return clean + noise

def test_filter_bandpass(integrator, sample_data):
    """Test bandpass filter integration."""
    config = {
        'type': 'Bandpass Filter',
        'parameters': {
            'lowcut': 5,
            'highcut': 15,
            'order': 4
        }
    }
    
    result = integrator.apply_filter(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0
    assert 'pre_processing' in result.metrics
    assert 'post_processing' in result.metrics
    assert 'improvement' in result.metrics

def test_filter_notch(integrator, sample_data):
    """Test notch filter integration."""
    config = {
        'type': 'Notch Filter',
        'parameters': {
            'center_freq': 50,
            'q_factor': 30
        }
    }
    
    result = integrator.apply_filter(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0

def test_filter_wavelet(integrator, sample_data):
    """Test wavelet denoising integration."""
    config = {
        'type': 'Wavelet Denoising',
        'parameters': {
            'wavelet_type': 'db4',
            'decomp_level': 3
        }
    }
    
    result = integrator.apply_filter(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0

def test_normalization_zscore(integrator, sample_data):
    """Test z-score normalization integration."""
    config = {
        'method': 'Z-score',
        'parameters': {
            'zscore_robust': False
        }
    }
    
    result = integrator.apply_normalization(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0
    assert abs(np.mean(result.data)) < 1e-10  # Should be approximately 0
    assert abs(np.std(result.data) - 1) < 1e-10  # Should be approximately 1

def test_normalization_minmax(integrator, sample_data):
    """Test min-max normalization integration."""
    config = {
        'method': 'Min-Max',
        'parameters': {
            'feature_min': -1,
            'feature_max': 1
        }
    }
    
    result = integrator.apply_normalization(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0
    assert np.min(result.data) >= -1
    assert np.max(result.data) <= 1

def test_normalization_robust(integrator, sample_data):
    """Test robust normalization integration."""
    config = {
        'method': 'Robust',
        'parameters': {}
    }
    
    result = integrator.apply_normalization(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0

def test_segmentation_fixed(integrator, sample_data):
    """Test fixed window segmentation integration."""
    config = {
        'method': 'Fixed Window',
        'parameters': {
            'window_size': 100
        }
    }
    
    result = integrator.apply_segmentation(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0
    assert result.metrics['statistics']['num_segments'] == len(sample_data) // 100

def test_segmentation_overlap(integrator, sample_data):
    """Test overlapping window segmentation integration."""
    config = {
        'method': 'Overlapping Window',
        'parameters': {
            'window_size': 100,
            'overlap': 50
        }
    }
    
    result = integrator.apply_segmentation(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0
    assert result.metrics['statistics']['total_coverage'] > 1  # Due to overlap

def test_segmentation_event(integrator, sample_data):
    """Test event-based segmentation integration."""
    events = np.array([100, 300, 500, 700])  # Sample event markers
    config = {
        'method': 'Event-based',
        'parameters': {
            'pre_event': 50,
            'post_event': 50
        },
        'events': events
    }
    
    result = integrator.apply_segmentation(sample_data, config)
    
    assert result.success
    assert result.error is None
    assert isinstance(result.data, np.ndarray)
    assert len(result.metrics) > 0
    assert result.metrics['statistics']['num_segments'] == len(events)

def test_config_validation(integrator):
    """Test configuration validation."""
    # Test valid bandpass config
    valid_bandpass = {
        'type': 'Bandpass Filter',
        'parameters': {
            'lowcut': 5,
            'highcut': 15,
            'order': 4
        }
    }
    is_valid, error = integrator.validate_config(ProcessingStage.FILTER, valid_bandpass)
    assert is_valid
    assert error is None
    
    # Test invalid bandpass config
    invalid_bandpass = {
        'type': 'Bandpass Filter',
        'parameters': {
            'lowcut': 15,  # Higher than highcut
            'highcut': 5,
            'order': 4
        }
    }
    is_valid, error = integrator.validate_config(ProcessingStage.FILTER, invalid_bandpass)
    assert not is_valid
    assert error is not None
    
    # Test valid normalization config
    valid_norm = {
        'method': 'Min-Max',
        'parameters': {
            'feature_min': -1,
            'feature_max': 1
        }
    }
    is_valid, error = integrator.validate_config(ProcessingStage.NORMALIZE, valid_norm)
    assert is_valid
    assert error is None
    
    # Test invalid normalization config
    invalid_norm = {
        'method': 'Min-Max',
        'parameters': {
            'feature_min': 1,  # Greater than feature_max
            'feature_max': -1
        }
    }
    is_valid, error = integrator.validate_config(ProcessingStage.NORMALIZE, invalid_norm)
    assert not is_valid
    assert error is not None

def test_error_handling(integrator):
    """Test error handling in processing operations."""
    # Test with invalid data
    invalid_data = np.array([])  # Empty array
    
    # Filter should handle error
    result = integrator.apply_filter(invalid_data, {
        'type': 'Bandpass Filter',
        'parameters': {
            'lowcut': 5,
            'highcut': 15,
            'order': 4
        }
    })
    assert not result.success
    assert result.error is not None
    
    # Normalization should handle error
    result = integrator.apply_normalization(invalid_data, {
        'method': 'Z-score',
        'parameters': {}
    })
    assert not result.success
    assert result.error is not None
    
    # Segmentation should handle error
    result = integrator.apply_segmentation(invalid_data, {
        'method': 'Fixed Window',
        'parameters': {
            'window_size': 100
        }
    })
    assert not result.success
    assert result.error is not None

def test_metrics_calculation(integrator, sample_data):
    """Test signal metrics calculation."""
    metrics = integrator._calculate_signal_metrics(sample_data)
    
    # Check required metrics
    assert 'mean' in metrics
    assert 'std' in metrics
    assert 'min' in metrics
    assert 'max' in metrics
    assert 'rms' in metrics
    assert 'snr' in metrics
    assert 'noise_level' in metrics
    assert 'peak_frequency' in metrics
    assert 'spectral_centroid' in metrics
    
    # Check metric values
    assert isinstance(metrics['mean'], float)
    assert isinstance(metrics['std'], float)
    assert metrics['min'] <= metrics['max']
    assert metrics['rms'] >= 0
    assert metrics['noise_level'] >= 0