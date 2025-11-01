import pytest
import numpy as np
from preprocess import (
    PreprocessingOperator, Notch, Bandpass, Highpass, Lowpass,
    Resample, ArtifactDetector, Pipeline
)

@pytest.fixture
def sample_data():
    # Create synthetic test data: sine wave + noise
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)
    return np.vstack([signal, signal]), t

def test_base_operator():
    op = PreprocessingOperator("test")
    assert op.name == "test"
    assert op.config() == {"name": "test"}
    with pytest.raises(NotImplementedError):
        op.process(np.array([]), np.array([]), 1.0)

def test_notch_filter():
    data, timestamps = sample_data()
    fs = 1000.0
    
    # Test 50Hz notch filter
    notch = Notch(freq=50.0, q=30.0)
    filtered, ts, _ = notch.process(data, timestamps, fs)
    
    # Check shape preservation
    assert filtered.shape == data.shape
    assert ts is timestamps
    
    # Verify attenuation at 50Hz using FFT
    fft_orig = np.fft.fft(data[0])
    fft_filtered = np.fft.fft(filtered[0])
    freqs = np.fft.fftfreq(len(data[0]), 1/fs)
    
    # Find index closest to 50Hz
    idx_50hz = np.argmin(np.abs(freqs - 50))
    assert np.abs(fft_filtered[idx_50hz]) < np.abs(fft_orig[idx_50hz])

def test_bandpass_filter():
    data, timestamps = sample_data()
    fs = 1000.0
    
    bandpass = Bandpass(low=20.0, high=100.0, order=4)
    filtered, ts, _ = bandpass.process(data, timestamps, fs)
    
    # Check shape preservation
    assert filtered.shape == data.shape
    assert ts is timestamps
    
    # Verify frequency response
    fft_filtered = np.fft.fft(filtered[0])
    freqs = np.fft.fftfreq(len(data[0]), 1/fs)
    
    # Check attenuation outside passband
    mask_below = np.abs(freqs) < 20
    mask_above = np.abs(freqs) > 100
    assert np.mean(np.abs(fft_filtered[mask_below])) < np.mean(np.abs(fft_filtered[~mask_below]))
    assert np.mean(np.abs(fft_filtered[mask_above])) < np.mean(np.abs(fft_filtered[~mask_above]))

def test_highpass_filter():
    data, timestamps = sample_data()
    fs = 1000.0
    
    highpass = Highpass(cutoff=1.0, order=2)
    filtered, ts, _ = highpass.process(data, timestamps, fs)
    
    # Check shape preservation
    assert filtered.shape == data.shape
    assert ts is timestamps
    
    # Verify high-pass effect
    fft_filtered = np.fft.fft(filtered[0])
    freqs = np.fft.fftfreq(len(data[0]), 1/fs)
    
    # Check attenuation below cutoff
    mask_below = np.abs(freqs) < 1.0
    assert np.mean(np.abs(fft_filtered[mask_below])) < np.mean(np.abs(fft_filtered[~mask_below]))

def test_lowpass_filter():
    data, timestamps = sample_data()
    fs = 1000.0
    
    lowpass = Lowpass(cutoff=100.0, order=4)
    filtered, ts, _ = lowpass.process(data, timestamps, fs)
    
    # Check shape preservation
    assert filtered.shape == data.shape
    assert ts is timestamps
    
    # Verify low-pass effect
    fft_filtered = np.fft.fft(filtered[0])
    freqs = np.fft.fftfreq(len(data[0]), 1/fs)
    
    # Check attenuation above cutoff
    mask_above = np.abs(freqs) > 100.0
    assert np.mean(np.abs(fft_filtered[mask_above])) < np.mean(np.abs(fft_filtered[~mask_above]))

def test_resample():
    data, timestamps = sample_data()
    fs = 1000.0
    
    # Test downsampling
    resample = Resample(target_fs=250.0)
    resampled, ts, _ = resample.process(data, timestamps, fs)
    
    # Check new length matches target frequency
    expected_length = int(len(timestamps) * (250.0 / fs))
    assert resampled.shape[1] == expected_length
    assert len(ts) == expected_length
    
    # Test no resampling needed
    resample = Resample(target_fs=fs)
    resampled, ts, _ = resample.process(data, timestamps, fs)
    assert resampled is data
    assert ts is timestamps

def test_artifact_detector():
    # Create data with artificial artifacts
    data = np.random.randn(2, 1000)
    artifacts = np.zeros_like(data)
    artifacts[0, 300:350] = 10  # Large artifact in first channel
    artifacts[1, 600:650] = -10  # Large artifact in second channel
    data += artifacts
    timestamps = np.arange(1000) / 1000.0
    
    detector = ArtifactDetector(z_thresh=4.0)
    processed, ts, annotations = detector.process(data, timestamps, fs=1000.0)
    
    # Check data preservation
    assert processed is data
    assert ts is timestamps
    
    # Verify artifact detection
    assert len(annotations['bad_segments']) > 0
    bad_segments = annotations['bad_segments'][0]
    assert set(bad_segments['channel_indices']) == {0, 1}
    assert bad_segments['count'] > 0

def test_pipeline():
    data, timestamps = sample_data()
    fs = 1000.0
    
    # Create pipeline with multiple operators
    pipeline = Pipeline([
        Notch(freq=50.0),
        Bandpass(low=1.0, high=100.0),
        Resample(target_fs=250.0)
    ])
    
    # Process data through pipeline
    processed, ts, annotations = pipeline.run(data, timestamps, fs)
    
    # Check final shape matches resampling
    expected_length = int(len(timestamps) * (250.0 / fs))
    assert processed.shape[1] == expected_length
    assert len(ts) == expected_length
    
    # Check history
    assert len(pipeline.history) == 3
    assert pipeline.history[0]['op'] == 'notch'
    assert pipeline.history[1]['op'] == 'bandpass'
    assert pipeline.history[2]['op'] == 'resample'
    
    # Test pipeline serialization
    pipeline_dict = pipeline.to_dict()
    assert len(pipeline_dict['ops']) == 3
    assert all('name' in op for op in pipeline_dict['ops'])

def test_pipeline_empty():
    pipeline = Pipeline([])
    data, timestamps = sample_data()
    processed, ts, annotations = pipeline.run(data, timestamps, fs=1000.0)
    assert processed is data
    assert ts is timestamps
    assert annotations == {}