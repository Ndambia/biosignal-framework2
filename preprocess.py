"""Preprocessing pipeline and standard operators.

Implements:
- PreprocessingOperator: base class with JSON-serializable config
- Standard operators: Notch, Bandpass, Highpass, Lowpass, Resample, Detrend
- Pipeline: declarative runner that composes operators and records history
- Simple artifact detection & marking
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend, resample
import json

class PreprocessingOperator:
    def __init__(self, name: str):
        self.name = name

    def process(self, data: np.ndarray, timestamps: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def config(self) -> Dict[str, Any]:
        return {'name': self.name}

@dataclass
class Notch(PreprocessingOperator):
    freq: float = 50.0
    q: float = 30.0

    def __post_init__(self):
        super().__init__('notch')

    def process(self, data, timestamps, fs):
        b, a = iirnotch(self.freq / (fs / 2), self.q)
        out = filtfilt(b, a, data, axis=1)
        return out, timestamps, {}

@dataclass
class Bandpass(PreprocessingOperator):
    low: float = 20.0
    high: float = 450.0
    order: int = 4

    def __post_init__(self):
        super().__init__('bandpass')

    def process(self, data, timestamps, fs):
        nyq = fs / 2.0
        low = self.low / nyq
        high = self.high / nyq
        b, a = butter(self.order, [low, high], btype='band')
        out = filtfilt(b, a, data, axis=1)
        return out, timestamps, {}

@dataclass
class Highpass(PreprocessingOperator):
    cutoff: float = 0.5
    order: int = 2

    def __post_init__(self):
        super().__init__('highpass')

    def process(self, data, timestamps, fs):
        b, a = butter(self.order, self.cutoff / (fs / 2), btype='high')
        out = filtfilt(b, a, data, axis=1)
        return out, timestamps, {}

@dataclass
class Lowpass(PreprocessingOperator):
    cutoff: float = 100.0
    order: int = 4

    def __post_init__(self):
        super().__init__('lowpass')

    def process(self, data, timestamps, fs):
        b, a = butter(self.order, self.cutoff / (fs / 2), btype='low')
        out = filtfilt(b, a, data, axis=1)
        return out, timestamps, {}

@dataclass
class Resample(PreprocessingOperator):
    target_fs: float = 250.0

    def __post_init__(self):
        super().__init__('resample')

    def process(self, data, timestamps, fs):
        if fs == self.target_fs:
            return data, timestamps, {}
        num = int(data.shape[1] * (self.target_fs / fs))
        out = resample(data, num, axis=1)
        start = timestamps[0] if timestamps.size else 0.0
        out_ts = start + np.arange(out.shape[1]) / self.target_fs
        return out, out_ts, {}

class ArtifactDetector(PreprocessingOperator):
    """Simple threshold-based artifact detector per-channel.""" 

    def __init__(self, z_thresh: float = 6.0):
        super().__init__('artifact_detector')
        self.z_thresh = z_thresh

    def process(self, data, timestamps, fs):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        z = np.abs((data - mean) / (std + 1e-12))
        mask = (z > self.z_thresh)
        # return annotations as index ranges per channel
        bad_segments = []
        if mask.any():
            inds = np.where(mask)
            bad_segments.append({'channel_indices': list(set(inds[0].tolist())), 'count': int(mask.sum())})
        return data, timestamps, {'bad_segments': bad_segments}

class Pipeline:
    def __init__(self, ops: List[PreprocessingOperator]):
        self.ops = ops
        self.history: List[Dict[str, Any]] = []

    def run(self, data: np.ndarray, timestamps: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        cur = data
        cur_ts = timestamps
        annotations: Dict[str, Any] = {}
        cur_fs = fs
        for op in self.ops:
            cur, cur_ts, ann = op.process(cur, cur_ts, cur_fs)
            annotations[op.name] = ann
            self.history.append({'op': op.name, 'config': getattr(op, '__dict__', {}), 'annotations': ann})
            if isinstance(op, Resample):
                cur_fs = op.target_fs
        return cur, cur_ts, annotations

    def to_dict(self):
        return {'ops': [getattr(op, '__dict__', {'name': getattr(op, 'name', 'unknown')}) for op in self.ops]}

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
