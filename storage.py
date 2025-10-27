"""Storage helpers for raw data, processed data, annotations, and experiments.

- HDF5-backed raw storage with metadata
- Parquet/CSV features (stubs)
- Experiment registry (local JSON)
"""
from __future__ import annotations
import os
import h5py
import json
import numpy as np

def save_raw_h5(path: str, data: np.ndarray, fs: float, channel_labels: list, metadata: dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip')
        f.attrs['fs'] = float(fs)
        f.attrs['channel_labels'] = np.array(channel_labels, dtype='S')
        f.attrs['metadata'] = json.dumps(metadata)

def load_raw_h5(path: str):
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
        fs = float(f.attrs['fs'])
        labels = [l.decode('utf8') for l in f.attrs['channel_labels']]
        metadata = json.loads(f.attrs['metadata'])
    return data, fs, labels, metadata

def register_experiment(root: str, info: dict) -> str:
    os.makedirs(root, exist_ok=True)
    idx = len([n for n in os.listdir(root) if n.startswith('exp_')]) + 1
    path = os.path.join(root, f'exp_{idx}')
    os.makedirs(path)
    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(info, f, indent=2)
    return path
