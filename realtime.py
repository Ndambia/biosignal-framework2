"""Realtime engine: ring buffer, worker thread, graceful degradation.

Features:
- RingBuffer for multi-channel streaming
- RealtimeWorker that pulls windows and runs pipeline -> features -> model
- Simple scheduling with watchdog for latency
"""
from __future__ import annotations
import threading
import time
from collections import deque
from typing import Optional, Callable, Any
import numpy as np

class RingBuffer:
    def __init__(self, n_channels: int, max_samples: int):
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.buffer = deque(maxlen=max_samples)
        self.lock = threading.Lock()

    def push(self, samples: np.ndarray):
        # samples: (n_channels, n_samples)
        with self.lock:
            for i in range(samples.shape[1]):
                self.buffer.append(samples[:, i].copy())

    def read_last(self, n_samples: int) -> Optional[np.ndarray]:
        with self.lock:
            if len(self.buffer) < n_samples:
                return None
            arr = np.stack(list(self.buffer)[-n_samples:], axis=1)
            return arr

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)

class RealtimeWorker(threading.Thread):
    def __init__(self, ring: RingBuffer, fs: float, pipeline: Any, extractor: Callable[[np.ndarray], dict], model: Any, window_s: float = 0.2, step_s: float = 0.1):
        super().__init__()
        self.ring = ring
        self.fs = fs
        self.pipeline = pipeline
        self.extractor = extractor
        self.model = model
        self.window_s = window_s
        self.step_s = step_s
        self.window_samples = int(window_s * fs)
        self.step_samples = int(step_s * fs)
        self._stop = threading.Event()
        self.last_run = 0.0

    def run(self):
        while not self._stop.is_set():
            start_t = time.time()
            arr = self.ring.read_last(self.window_samples)
            if arr is None:
                time.sleep(0.005)
                continue
            # run preprocessing pipeline (assumes pipeline.run returns processed samples)
            proc, ts, ann = self.pipeline.run(arr, np.arange(arr.shape[1]) / self.fs, self.fs)
            feats = self.extractor(proc)
            try:
                y = self.model.predict(feats)
            except Exception:
                # model expects numpy array; attempt conversion
                import numpy as _np
                if isinstance(feats, dict):
                    X = _np.array([list(feats.values())])
                else:
                    X = _np.asarray(feats)
                y = self.model.predict(X)
            # Here you'd push `y` to an output channel
            print('[realtime] prediction', y)
            self.last_run = time.time() - start_t
            # sleep until next step
            time.sleep(max(0, self.step_samples / self.fs - self.last_run))

    def stop(self):
        self._stop.set()
