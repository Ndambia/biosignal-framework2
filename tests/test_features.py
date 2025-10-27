import pytest
import numpy as np
from features import FeatureExtractor, FeatureSchema

@pytest.fixture
def sample_signal():
    # Create synthetic test signal: sine wave + noise
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
    return signal

@pytest.fixture
def feature_extractor():
    return FeatureExtractor(fs=1000.0)

def test_feature_schema():
    schema = FeatureSchema(names=['mean', 'std', 'rms'])
    assert schema.names == ['mean', 'std', 'rms']

def test_time_domain_features(feature_extractor, sample_signal):
    features = feature_extractor.time_domain(sample_signal)
    
    # Check all expected features are present
    expected_features = {
        'mean', 'std', 'rms', 'iemg', 'mav', 'wl', 
        'zc', 'median', 'iqr', 'skew', 'kurtosis'
    }
    assert set(features.keys()) == expected_features
    
    # Test specific feature properties
    assert np.abs(features['mean']) < 0.2  # Should be close to 0 for sine wave
    assert features['std'] > 0  # Should be positive
    assert features['rms'] > 0  # Should be positive
    assert features['iemg'] > 0  # Should be positive
    assert features['mav'] > 0  # Should be positive
    assert features['wl'] > 0  # Should be positive
    assert features['zc'] > 0  # Should have zero crossings
    assert np.abs(features['median']) < 0.2  # Should be close to 0
    assert features['iqr'] > 0  # Should be positive

def test_freq_domain_features(feature_extractor, sample_signal):
    features = feature_extractor.freq_domain(sample_signal)
    
    # Check expected features
    assert 'psd_power' in features
    assert 'psd_med_freq' in features
    
    # Test feature properties
    assert features['psd_power'] > 0  # Total power should be positive
    assert 0 <= features['psd_med_freq'] <= 500  # Nyquist frequency

def test_wavelet_features(feature_extractor, sample_signal):
    features = feature_extractor.wavelet_energy(sample_signal)
    
    # Check number of wavelet coefficients (level 3 decomposition)
    expected_coeffs = 4  # Approximation + 3 detail coefficients
    assert len([k for k in features.keys() if k.startswith('wavelet_e_')]) == expected_coeffs
    
    # Test feature properties
    for v in features.values():
        assert v >= 0  # Energy should be non-negative

def test_extract_window(feature_extractor, sample_signal):
    features = feature_extractor.extract_window(sample_signal)
    
    # Should include all feature types
    assert any(k.startswith('wavelet_e_') for k in features)
    assert 'psd_power' in features
    assert 'mean' in features
    
    # All values should be float type
    assert all(isinstance(v, float) for v in features.values())

def test_sliding_extract(feature_extractor):
    # Create multi-channel test data
    t = np.linspace(0, 1, 1000)
    data = np.vstack([
        np.sin(2 * np.pi * 10 * t),  # 10 Hz sine
        np.sin(2 * np.pi * 20 * t)   # 20 Hz sine
    ])
    
    # Test with default window parameters
    features = feature_extractor.sliding_extract(data)
    
    # Check feature structure
    assert len(features) > 0
    assert all(isinstance(f, dict) for f in features)
    
    # Check channel prefixing
    first_feature = features[0]
    assert any(k.startswith('ch0_') for k in first_feature)
    assert any(k.startswith('ch1_') for k in first_feature)
    
    # Test with custom window parameters
    features = feature_extractor.sliding_extract(data, win_s=0.1, step_s=0.05)
    expected_windows = int((len(t) - int(0.1 * 1000)) / int(0.05 * 1000)) + 1
    assert len(features) == expected_windows

def test_feature_consistency(feature_extractor, sample_signal):
    # Test feature consistency with repeated calls
    features1 = feature_extractor.extract_window(sample_signal)
    features2 = feature_extractor.extract_window(sample_signal)
    
    assert features1.keys() == features2.keys()
    for k in features1:
        assert np.allclose(features1[k], features2[k])

def test_edge_cases(feature_extractor):
    # Test with very short signal
    short_sig = np.array([1.0, -1.0, 1.0])
    features = feature_extractor.extract_window(short_sig)
    assert all(isinstance(v, float) for v in features.values())
    
    # Test with constant signal
    const_sig = np.ones(100)
    features = feature_extractor.extract_window(const_sig)
    assert features['std'] == 0
    assert features['zc'] == 0
    
    # Test with NaN/Inf handling
    bad_sig = np.array([1.0, np.nan, np.inf, -np.inf, 1.0])
    with pytest.raises(Exception):  # Should raise some kind of error
        feature_extractor.extract_window(bad_sig)