"""Acquisition adapters and unified interface.

Provides:
- AcquisitionAdapter (abstract base)
- FilePlaybackAdapter (.npz/.h5 loader)
- SimulatedAdapter (synthetic signals for tests)
- (Hooks for hardware adapters - OpenBCI/ADS family) via a simple plugin API

All adapters implement:
  start(), stop(), read(n_samples) -> (data, timestamps)

Data convention: ndarray shape (n_channels, n_samples), timestamps shape (n_samples,)
Timestamps are POSIX seconds (float)
"""
from __future__ import annotations
import time
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import os
import h5py

class AcquisitionAdapter:
    """Abstract acquisition adapter interface."""

    def __init__(self):
        self.sample_rate: Optional[float] = None
        self.channel_labels: List[str] = []
        self.device_id: Optional[str] = None
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class FilePlaybackAdapter(AcquisitionAdapter):
    """Plays back pre-recorded data saved in .npz or HDF5.

    Supported file formats:
      - .npz: expects arrays 'data' (n_channels, n_samples), 'fs' scalar, optional 'channel_labels'
      - .h5/.hdf5: expects dataset '/data' and attrs 'fs', 'channel_labels'
    """

    def __init__(self, filename: str, loop: bool = False):
        super().__init__()
        self.filename = filename
        self.loop = loop
        self._pos = 0
        self._loaded = False
        self._data: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None

    def _load_npz(self, path: str):
        d = np.load(path, allow_pickle=True)
        data = d['data']
        fs = float(d['fs'].tolist())
        labels = d.get('channel_labels', None)
        if labels is None:
            labels = [f'ch{i}' for i in range(data.shape[0])]
        return data, fs, list(labels)

    def _load_h5(self, path: str):
        with h5py.File(path, 'r') as f:
            data = f['data'][:]
            fs = float(f.attrs['fs'])
            labels = f.attrs.get('channel_labels', None)
            if labels is not None:
                try:
                    labels = list(labels)
                except Exception:
                    labels = [l.decode('utf8') for l in labels]
            else:
                labels = [f'ch{i}' for i in range(data.shape[0])]
        return data, fs, labels

    def _load(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(self.filename)
        ext = os.path.splitext(self.filename)[1].lower()
        if ext == '.npz':
            data, fs, labels = self._load_npz(self.filename)
        elif ext in ('.h5', '.hdf5'):
            data, fs, labels = self._load_h5(self.filename)
        else:
            raise ValueError('Unsupported file format: ' + ext)
        self._data = np.asarray(data)
        n_samples = self._data.shape[1]
        # create timestamps anchored at current time for playback
        start_ts = time.time()
        self._timestamps = start_ts + np.arange(n_samples) / float(fs)
        self.sample_rate = float(fs)
        self.channel_labels = labels
        self._loaded = True

    def start(self):
        super().start()
        if not self._loaded:
            self._load()
        self._pos = 0

    def stop(self):
        super().stop()
        self._pos = 0

    def read(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._loaded:
            raise RuntimeError('Adapter not loaded. call start()')
        if self._pos >= self._data.shape[1]:
            if self.loop:
                self._pos = 0
            else:
                return np.zeros((self._data.shape[0], 0)), np.zeros((0,))
        end = min(self._pos + n_samples, self._data.shape[1])
        chunk = self._data[:, self._pos:end]
        ts = self._timestamps[self._pos:end]
        self._pos = end
        return chunk, ts

class SimulatedAdapter(AcquisitionAdapter):
    """Generates synthetic signals for testing and CI.

    Parameters:
      - channels: list of channel names
      - fs: sample rate
      - signal_fns: optional list of callables (t->value) per channel
    """

    def __init__(self, channels: List[str], fs: float = 1000.0, duration_s: float = 10.0, signal_fns: Optional[List[Any]] = None):
        super().__init__()
        self.sample_rate = fs
        self.channel_labels = channels[:]
        self.duration_s = duration_s
        self._t = np.arange(0, duration_s, 1.0/fs)
        self._pos = 0
        self._signals = []
        if signal_fns is None:
            # default: simple sinusoids of different frequencies
            for i in range(len(channels)):
                f = 5 + i * 10
                self._signals.append(np.sin(2 * np.pi * f * self._t))
        else:
            for fn in signal_fns:
                self._signals.append(fn(self._t))
        self._data = np.vstack(self._signals)
        self._timestamps = time.time() + self._t

    def start(self):
        super().start()
        self._pos = 0

    def stop(self):
        super().stop()
        self._pos = 0

    def read(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._pos >= self._data.shape[1]:
            return np.zeros((self._data.shape[0], 0)), np.zeros((0,))
        end = min(self._pos + n_samples, self._data.shape[1])
        chunk = self._data[:, self._pos:end]
        ts = self._timestamps[self._pos:end]
        self._pos = end
        return chunk, ts
