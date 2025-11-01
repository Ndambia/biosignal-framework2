import numpy as np
from scipy import stats

class TimeDomainFeatures:
    """
    A collection of time-domain feature extraction methods for biosignals.
    """

    def rms(self, signal: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Calculate the Root Mean Square (RMS) of a signal.
        If window_size is provided, calculates RMS in sliding windows.
        """
        if window_size:
            # Ensure window_size is an integer
            window_size = int(window_size)
            if window_size <= 0:
                raise ValueError("window_size must be a positive integer.")
            
            num_windows = len(signal) - window_size + 1
            if num_windows <= 0:
                raise ValueError("Signal length is less than window_size.")
            
            rms_values = np.zeros(num_windows)
            for i in range(num_windows):
                window = signal[i : i + window_size]
                rms_values[i] = np.sqrt(np.mean(window**2))
            return rms_values
        else:
            return np.array([np.sqrt(np.mean(signal**2))])

    def mav(self, signal: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Calculate the Mean Absolute Value (MAV) of a signal.
        If window_size is provided, calculates MAV in sliding windows.
        """
        if window_size:
            window_size = int(window_size)
            if window_size <= 0:
                raise ValueError("window_size must be a positive integer.")
            
            num_windows = len(signal) - window_size + 1
            if num_windows <= 0:
                raise ValueError("Signal length is less than window_size.")
            
            mav_values = np.zeros(num_windows)
            for i in range(num_windows):
                window = signal[i : i + window_size]
                mav_values[i] = np.mean(np.abs(window))
            return mav_values
        else:
            return np.array([np.mean(np.abs(signal))])

    def zero_crossing_rate(self, signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Calculate the Zero Crossing Rate (ZCR) of a signal.
        The threshold parameter defines a band around zero within which crossings are ignored.
        """
        # Ensure signal is 1D
        if signal.ndim > 1:
            signal = signal.flatten()

        # Apply threshold: values within [-threshold, threshold] are considered zero
        # This prevents counting noise as zero crossings
        thresholded_signal = np.copy(signal)
        thresholded_signal[np.abs(thresholded_signal) <= threshold] = 0

        # Find where the sign changes
        # np.diff(np.sign(x)) will be non-zero at sign changes
        # A change from positive to negative or negative to positive
        # will result in -2 or 2 respectively.
        zcr = np.where(np.diff(np.sign(thresholded_signal)))[0]
        return np.array([len(zcr) / len(signal)]) if len(signal) > 0 else np.array([0.0])


    def slope_sign_changes(self, signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Calculate the number of Slope Sign Changes (SSC) in a signal.
        A slope sign change occurs when the product of two consecutive differences
        is negative, and the absolute value of the middle point exceeds a threshold.
        """
        if len(signal) < 3:
            return np.array([0.0])

        ssc_count = 0
        for i in range(1, len(signal) - 1):
            diff1 = signal[i] - signal[i-1]
            diff2 = signal[i+1] - signal[i]
            
            # Check for sign change and threshold condition
            if (diff1 * diff2 < 0) and (np.abs(signal[i]) >= threshold):
                ssc_count += 1
        return np.array([ssc_count / len(signal)]) if len(signal) > 0 else np.array([0.0])

    def waveform_length(self, signal: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Calculate the Waveform Length (WL) of a signal.
        If window_size is provided, calculates WL in sliding windows.
        """
        if window_size:
            window_size = int(window_size)
            if window_size <= 0:
                raise ValueError("window_size must be a positive integer.")
            
            num_windows = len(signal) - window_size + 1
            if num_windows <= 0:
                raise ValueError("Signal length is less than window_size.")
            
            wl_values = np.zeros(num_windows)
            for i in range(num_windows):
                window = signal[i : i + window_size]
                wl_values[i] = np.sum(np.abs(np.diff(window)))
            return wl_values
        else:
            return np.array([np.sum(np.abs(np.diff(signal)))])

    def variance(self, signal: np.ndarray) -> float:
        """Calculate the variance of a signal."""
        return np.var(signal)

    def standard_deviation(self, signal: np.ndarray) -> float:
        """Calculate the standard deviation of a signal."""
        return np.std(signal)

    def skewness(self, signal: np.ndarray) -> float:
        """Calculate the skewness of a signal."""
        return stats.skew(signal)

    def kurtosis(self, signal: np.ndarray) -> float:
        """Calculate the kurtosis of a signal."""
        return stats.kurtosis(signal)
