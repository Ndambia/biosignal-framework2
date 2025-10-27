"""Feature extraction engine with feature registry and common feature sets.

Provides time-domain, frequency-domain and wavelet feature helpers.
Also defines a FeatureSchema to keep track of feature names/types.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from scipy.stats import iqr, skew, kurtosis
from scipy.signal import welch
import pywt

@dataclass
class FeatureSchema:
    names: List[str]

class FeatureExtractor:
    def __init__(self, fs: float):
        self.fs = fs

    def time_domain(self, sig: np.ndarray) -> Dict[str, float]:
        # sig: 1D array
        return {
            'mean': float(np.mean(sig)),
            'std': float(np.std(sig)),
            'rms': float(np.sqrt(np.mean(sig**2))),
            'iemg': float(np.sum(np.abs(sig))),
            'mav': float(np.mean(np.abs(sig))),
            'wl': float(np.sum(np.abs(np.diff(sig)))),
            'zc': float(((sig[:-1] * sig[1:]) < 0).sum()),
            'median': float(np.median(sig)),
            'iqr': float(iqr(sig)),
            'skew': float(skew(sig)),
            'kurtosis': float(kurtosis(sig))
        }

    def freq_domain(self, sig: np.ndarray) -> Dict[str, float]:
        f, pxx = welch(sig, self.fs, nperseg=min(256, len(sig)))
        total = float(np.trapz(pxx, f))
        csum = np.cumsum(pxx)
        medf = float(np.interp(csum[-1] / 2.0, csum, f)) if csum[-1] > 0 else 0.0
        return {'psd_power': total, 'psd_med_freq': medf}

    def wavelet_energy(self, sig: np.ndarray, wavelet: str = 'db4', level: int = 3) -> Dict[str, float]:
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        energies = {f'wavelet_e_{i}': float(np.sum(c**2)) for i, c in enumerate(coeffs)}
        return energies

    def extract_window(self, sig: np.ndarray) -> Dict[str, float]:
        # combine feature sets
        td = self.time_domain(sig)
        fd = self.freq_domain(sig)
        we = self.wavelet_energy(sig)
        return {**td, **fd, **we}

    def sliding_extract(self, data: np.ndarray, win_s: float = 0.2, step_s: float = 0.1) -> List[Dict[str, Any]]:
        n_channels, n_samples = data.shape
        win = int(win_s * self.fs)
        step = int(step_s * self.fs)
        feats = []
        for start in range(0, n_samples - win + 1, step):
            window = data[:, start:start + win]
            # extract per-channel and prefix with channel name index
            window_feats = {}
            for ch in range(n_channels):
                ch_feats = self.extract_window(window[ch])
                # prefix names
                for k, v in ch_feats.items():
                    window_feats[f'ch{ch}_{k}'] = v
            feats.append(window_feats)
        return feats
