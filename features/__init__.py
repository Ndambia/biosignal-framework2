"""
Feature extraction module for biosignals.

This module provides classes for extracting various features from biosignal data:
- Time domain features (statistical and morphological features)
- Frequency domain features (spectral features)
- Nonlinear features (complexity measures)
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft
import warnings

class Feature:
    """Base class for feature extraction."""
    
    def __init__(self, sampling_rate=1000):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate (float): Sampling rate of the signal in Hz
        """
        self.fs = sampling_rate
        
    def validate_input(self, signal_data):
        """Validate input signal data."""
        if not isinstance(signal_data, np.ndarray):
            signal_data = np.array(signal_data)
        if signal_data.ndim != 1:
            raise ValueError("Input signal must be a 1D array")
        return signal_data

class TimeDomainFeatures(Feature):
    """Time domain feature extraction."""
    
    def rms(self, signal_data):
        """Calculate Root Mean Square (RMS) value."""
        signal_data = self.validate_input(signal_data)
        return np.sqrt(np.mean(np.square(signal_data)))
    
    def mav(self, signal_data):
        """Calculate Mean Absolute Value (MAV)."""
        signal_data = self.validate_input(signal_data)
        return np.mean(np.abs(signal_data))
    
    def zero_crossing_rate(self, signal_data):
        """Calculate Zero Crossing Rate."""
        signal_data = self.validate_input(signal_data)
        return np.sum(np.diff(np.signbit(signal_data).astype(int))) / len(signal_data)
    
    def slope_sign_changes(self, signal_data, threshold=0):
        """
        Calculate Slope Sign Changes.
        
        Args:
            signal_data: Input signal
            threshold: Threshold for considering slope changes
        """
        signal_data = self.validate_input(signal_data)
        diff1 = np.diff(signal_data)
        diff2 = np.diff(diff1)
        return np.sum(np.abs(np.sign(diff2)) > threshold) / len(signal_data)
    
    def waveform_length(self, signal_data):
        """Calculate Waveform Length (total amplitude change over the segment)."""
        signal_data = self.validate_input(signal_data)
        return np.sum(np.abs(np.diff(signal_data)))

class FrequencyDomainFeatures(Feature):
    """Frequency domain feature extraction."""
    
    def power_spectral_density(self, signal_data, nperseg=None):
        """
        Calculate Power Spectral Density using Welch's method.
        
        Args:
            signal_data: Input signal
            nperseg: Length of each segment for Welch's method
        """
        signal_data = self.validate_input(signal_data)
        if nperseg is None:
            nperseg = min(256, len(signal_data))
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=nperseg)
        return freqs, psd
    
    def mean_frequency(self, signal_data):
        """Calculate mean frequency."""
        freqs, psd = self.power_spectral_density(signal_data)
        return np.sum(freqs * psd) / np.sum(psd)
    
    def median_frequency(self, signal_data):
        """Calculate median frequency."""
        _, psd = self.power_spectral_density(signal_data)
        cumsum = np.cumsum(psd)
        median_idx = np.where(cumsum >= cumsum[-1]/2)[0][0]
        return median_idx * (self.fs/2) / len(psd)
    
    def frequency_band_power(self, signal_data, freq_bands):
        """
        Calculate power in specific frequency bands.
        
        Args:
            signal_data: Input signal
            freq_bands: List of tuples [(low1, high1), (low2, high2), ...]
        """
        freqs, psd = self.power_spectral_density(signal_data)
        powers = []
        for low, high in freq_bands:
            mask = (freqs >= low) & (freqs <= high)
            powers.append(np.trapz(psd[mask], freqs[mask]))
        return powers
    
    def spectral_entropy(self, signal_data):
        """Calculate spectral entropy."""
        _, psd = self.power_spectral_density(signal_data)
        psd_norm = psd / np.sum(psd)
        return -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps))

class NonlinearFeatures(Feature):
    """Nonlinear feature extraction."""
    
    def sample_entropy(self, signal_data, m=2, r=0.2):
        """
        Calculate Sample Entropy.
        
        Args:
            signal_data: Input signal
            m: Embedding dimension
            r: Tolerance (typically 0.1 to 0.25 times signal std)
        """
        signal_data = self.validate_input(signal_data)
        r = r * np.std(signal_data)
        N = len(signal_data)
        
        def _count_matches(m, r, signal_data):
            templates = np.array([signal_data[i:i+m] for i in range(N-m+1)])
            distances = np.abs(templates[:, np.newaxis] - templates)
            max_distances = np.max(distances, axis=2)
            return np.sum(max_distances < r, axis=1) - 1  # Exclude self-matches
            
        count_m = np.mean(_count_matches(m, r, signal_data))
        count_m1 = np.mean(_count_matches(m+1, r, signal_data))
        
        return -np.log(count_m1 / count_m) if count_m > 0 else np.inf
    
    def approximate_entropy(self, signal_data, m=2, r=0.2):
        """
        Calculate Approximate Entropy.
        
        Args:
            signal_data: Input signal
            m: Embedding dimension
            r: Tolerance
        """
        signal_data = self.validate_input(signal_data)
        r = r * np.std(signal_data)
        N = len(signal_data)
        
        def _phi(m):
            templates = np.array([signal_data[i:i+m] for i in range(N-m+1)])
            distances = np.abs(templates[:, np.newaxis] - templates)
            max_distances = np.max(distances, axis=2)
            count = np.sum(max_distances < r, axis=1)
            return np.mean(np.log(count / (N-m+1)))
            
        return _phi(m) - _phi(m+1)
    
    def fractal_dimension(self, signal_data, k_max=None):
        """
        Calculate Katz Fractal Dimension.
        
        Args:
            signal_data: Input signal
            k_max: Maximum number of points to consider
        """
        signal_data = self.validate_input(signal_data)
        if k_max is None:
            k_max = len(signal_data)
            
        L = np.sum(np.abs(np.diff(signal_data[:k_max])))  # Total length
        d = np.max(np.abs(signal_data[:k_max] - signal_data[0]))  # Extent
        if d == 0:
            return 0
        return np.log10(L/d) / np.log10(k_max)