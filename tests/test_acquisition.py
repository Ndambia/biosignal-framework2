import pytest
import numpy as np
import os
import tempfile
import h5py
from acquisition import AcquisitionAdapter, FilePlaybackAdapter, SimulatedAdapter

def test_base_adapter():
    adapter = AcquisitionAdapter()
    assert adapter.sample_rate is None
    assert len(adapter.channel_labels) == 0
    assert adapter.device_id is None
    assert not adapter._running
    
    adapter.start()
    assert adapter._running
    
    adapter.stop()
    assert not adapter._running
    
    with pytest.raises(NotImplementedError):
        adapter.read(100)

def create_test_npz():
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        data = np.random.randn(2, 1000)  # 2 channels, 1000 samples
        fs = 250.0
        channel_labels = ['ch1', 'ch2']
        np.savez(f.name, data=data, fs=fs, channel_labels=channel_labels)
        return f.name, data, fs, channel_labels

def create_test_h5():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        data = np.random.randn(2, 1000)
        fs = 250.0
        channel_labels = ['ch1', 'ch2']
        with h5py.File(f.name, 'w') as h5f:
            h5f.create_dataset('data', data=data)
            h5f.attrs['fs'] = fs
            h5f.attrs['channel_labels'] = channel_labels
        return f.name, data, fs, channel_labels

def test_file_playback_npz():
    filename, data, fs, labels = create_test_npz()
    try:
        adapter = FilePlaybackAdapter(filename)
        adapter.start()
        
        assert adapter.sample_rate == fs
        assert adapter.channel_labels == labels
        
        # Test reading chunks
        chunk, ts = adapter.read(500)
        assert chunk.shape == (2, 500)
        np.testing.assert_array_equal(chunk, data[:, :500])
        assert len(ts) == 500
        
        # Test reading remaining data
        chunk, ts = adapter.read(1000)
        assert chunk.shape == (2, 500)
        np.testing.assert_array_equal(chunk, data[:, 500:])
        assert len(ts) == 500
        
        # Test reading when no more data
        chunk, ts = adapter.read(100)
        assert chunk.shape == (2, 0)
        assert len(ts) == 0
        
    finally:
        os.unlink(filename)

def test_file_playback_h5():
    filename, data, fs, labels = create_test_h5()
    try:
        adapter = FilePlaybackAdapter(filename)
        adapter.start()
        
        assert adapter.sample_rate == fs
        assert adapter.channel_labels == labels
        
        # Test reading full data
        chunk, ts = adapter.read(1000)
        assert chunk.shape == (2, 1000)
        np.testing.assert_array_equal(chunk, data)
        assert len(ts) == 1000
        
    finally:
        os.unlink(filename)

def test_file_playback_loop():
    filename, data, fs, labels = create_test_npz()
    try:
        adapter = FilePlaybackAdapter(filename, loop=True)
        adapter.start()
        
        # Read all data
        chunk1, ts1 = adapter.read(1000)
        # Should loop back to start
        chunk2, ts2 = adapter.read(500)
        np.testing.assert_array_equal(chunk2, data[:, :500])
        
    finally:
        os.unlink(filename)

def test_file_playback_errors():
    with pytest.raises(FileNotFoundError):
        adapter = FilePlaybackAdapter('nonexistent.npz')
        adapter.start()
    
    with pytest.raises(ValueError):
        adapter = FilePlaybackAdapter('test.txt')
        adapter.start()
    
    adapter = FilePlaybackAdapter('test.npz')
    with pytest.raises(RuntimeError):
        adapter.read(100)  # Try reading before start()

def test_simulated_adapter():
    channels = ['ch1', 'ch2']
    fs = 1000.0
    duration = 1.0
    adapter = SimulatedAdapter(channels, fs, duration)
    
    assert adapter.sample_rate == fs
    assert adapter.channel_labels == channels
    
    adapter.start()
    
    # Read half the data
    chunk, ts = adapter.read(500)
    assert chunk.shape == (2, 500)
    assert len(ts) == 500
    
    # Read remaining data
    chunk, ts = adapter.read(500)
    assert chunk.shape == (2, 500)
    assert len(ts) == 500
    
    # Should return empty after all data read
    chunk, ts = adapter.read(100)
    assert chunk.shape == (2, 0)
    assert len(ts) == 0

def test_simulated_custom_signals():
    channels = ['ch1']
    fs = 100.0
    duration = 1.0
    
    def custom_signal(t):
        return np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        
    adapter = SimulatedAdapter(channels, fs, duration, signal_fns=[custom_signal])
    adapter.start()
    
    chunk, ts = adapter.read(100)
    assert chunk.shape == (1, 100)
    # Verify it's a sine wave by checking zero crossings
    zero_crossings = np.where(np.diff(np.signbit(chunk[0])))[0]
    assert len(zero_crossings) > 0  # Should have multiple zero crossings