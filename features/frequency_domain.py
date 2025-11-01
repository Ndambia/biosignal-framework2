import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.stats import entropy

class FrequencyDomainFeatures:
    """
    A collection of frequency-domain feature extraction methods for biosignals.
    """

    def _compute_psd(self, signal: np.ndarray, fs: float, nperseg: int = None):
        """Helper to compute Power Spectral Density (PSD)."""
        if nperseg is None:
            nperseg = min(256, len(signal))
        if nperseg > len(signal):
            nperseg = len(signal)
        
        freqs, psd = welch(signal, fs, nperseg=nperseg)
        return freqs, psd

    def mean_frequency(self, signal: np.ndarray, fs: float, nperseg: int = None) -> float:
        """
        Calculate the Mean Frequency (MNF) of a signal.
        """
        freqs, psd = self._compute_psd(signal, fs, nperseg)
        
        # Avoid division by zero if sum of PSD is zero
        if np.sum(psd) == 0:
            return 0.0
            
        return np.sum(freqs * psd) / np.sum(psd)

    def median_frequency(self, signal: np.ndarray, fs: float, nperseg: int = None) -> float:
        """
        Calculate the Median Frequency (MDF) of a signal.
        """
        freqs, psd = self._compute_psd(signal, fs, nperseg)
        
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        
        if total_power == 0:
            return 0.0

        median_power_idx = np.where(cumulative_power >= total_power / 2)[0][0]
        return freqs[median_power_idx]

    def frequency_band_power(self, signal: np.ndarray, fs: float, 
                             bands: dict, nperseg: int = None) -> dict:
        """
        Calculate the power in specified frequency bands.
        `bands` should be a dictionary like {'alpha': (8, 13), 'beta': (13, 30)}.
        """
        freqs, psd = self._compute_psd(signal, fs, nperseg)
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx_band):
                band_power = simpson(psd[idx_band], freqs[idx_band])
            else:
                band_power = 0.0
            band_powers[band_name] = band_power
            
        return band_powers

    def spectral_entropy(self, signal: np.ndarray, fs: float, nperseg: int = None, normalize: bool = True) -> float:
        """
        Calculate the spectral entropy of a signal.
        """
        freqs, psd = self._compute_psd(signal, fs, nperseg)
        
        # Normalize PSD to get a probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Remove zero values to avoid log(0)
        psd_norm = psd_norm[psd_norm > 0]
        
        spec_entropy = entropy(psd_norm)
        
        if normalize:
            # Max entropy for a discrete distribution is log2(N)
            # where N is the number of bins (frequencies)
            max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 0
            if max_entropy > 0:
                spec_entropy /= max_entropy
            
        return spec_entropy

    def peak_frequency(self, signal: np.ndarray, fs: float, nperseg: int = None) -> float:
        """
        Calculate the peak frequency (frequency with the highest power) of a signal.
        """
        freqs, psd = self._compute_psd(signal, fs, nperseg)
        
        if len(psd) == 0:
            return 0.0
            
        peak_idx = np.argmax(psd)
        return freqs[peak_idx]
