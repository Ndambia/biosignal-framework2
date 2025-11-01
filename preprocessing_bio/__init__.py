"""Signal preprocessing module for biosignal processing.

This module provides classes and functions for preprocessing biosignals, including:
- Signal denoising (bandpass, notch, wavelet)
- Normalization (z-score, min-max, robust scaling) 
- Segmentation (fixed-length, overlapping, event-based)
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
import pywt

class SignalDenoising:
    """Class for signal denoising operations."""
    
    @staticmethod
    def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
        """Apply bandpass filter to the signal.
        
        Args:
            data: Input signal array
            lowcut: Lower frequency cutoff
            highcut: Higher frequency cutoff
            fs: Sampling frequency
            order: Filter order
            
        Returns:
            Filtered signal array
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def notch_filter(data: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
        """Apply notch filter to remove power line interference.
        
        Args:
            data: Input signal array
            freq: Frequency to remove (e.g., 50 or 60 Hz)
            fs: Sampling frequency
            q: Quality factor
            
        Returns:
            Filtered signal array
        """
        nyq = 0.5 * fs
        freq_normalized = freq / nyq
        b, a = signal.iirnotch(freq_normalized, q)
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def wavelet_denoise(data: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
        """Apply wavelet denoising.
        
        Args:
            data: Input signal array
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Denoised signal array
        """
        coeffs = pywt.wavedec(data, wavelet, level=level)
        threshold = np.sqrt(2 * np.log(len(data)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)

class SignalNormalization:
    """Class for signal normalization methods."""
    
    @staticmethod
    def zscore_normalize(data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.
        
        Args:
            data: Input signal array
            
        Returns:
            Normalized signal array
        """
        return zscore(data)
    
    @staticmethod
    def minmax_scale(data: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:
        """Apply min-max scaling.
        
        Args:
            data: Input signal array
            feature_range: Desired range of transformed data
            
        Returns:
            Scaled signal array
        """
        min_val, max_val = feature_range
        x_std = (data - data.min()) / (data.max() - data.min())
        return x_std * (max_val - min_val) + min_val
    
    @staticmethod
    def robust_scale(data: np.ndarray) -> np.ndarray:
        """Apply robust scaling using median and IQR.
        
        Args:
            data: Input signal array
            
        Returns:
            Scaled signal array
        """
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        return (data - median) / iqr

class SignalSegmentation:
    """Class for signal segmentation operations."""
    
    @staticmethod
    def fixed_window(data: np.ndarray, window_size: int) -> np.ndarray:
        """Segment signal into fixed-length windows.
        
        Args:
            data: Input signal array
            window_size: Size of each window
            
        Returns:
            Array of segmented windows
        """
        n_windows = len(data) // window_size
        return np.array([data[i*window_size:(i+1)*window_size] for i in range(n_windows)])
    
    @staticmethod
    def overlap_window(data: np.ndarray, window_size: int, overlap: float = 0.5) -> np.ndarray:
        """Segment signal into overlapping windows.
        
        Args:
            data: Input signal array
            window_size: Size of each window
            overlap: Overlap fraction between windows
            
        Returns:
            Array of overlapping windows
        """
        step = int(window_size * (1 - overlap))
        n_windows = (len(data) - window_size) // step + 1
        return np.array([data[i*step:i*step + window_size] for i in range(n_windows)])
    
    @staticmethod
    def event_based_segment(data: np.ndarray, events: np.ndarray, pre_event: int, post_event: int) -> np.ndarray:
        """Segment signal based on events.
        
        Args:
            data: Input signal array
            events: Array of event indices
            pre_event: Number of samples before event
            post_event: Number of samples after event
            
        Returns:
            Array of event-based segments
        """
        segments = []
        for event in events:
            start = max(0, event - pre_event)
            end = min(len(data), event + post_event)
            segments.append(data[start:end])
        return np.array(segments)