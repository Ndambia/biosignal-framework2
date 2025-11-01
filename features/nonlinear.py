import numpy as np
from typing import Union

class NonlinearFeatures:
    """
    A collection of nonlinear feature extraction methods for biosignals.
    """

    def sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2, normalize: bool = True) -> float:
        """
        Calculate the Sample Entropy (SampEn) of a signal.
        
        Parameters:
        - signal: Input signal (1D numpy array).
        - m: Embedding dimension.
        - r: Tolerance (usually 0.1 to 0.25 times the standard deviation of the signal).
        - normalize: If True, r is interpreted as a factor of the signal's standard deviation.
        """
        N = len(signal)
        if N < m + 2:
            return 0.0 # Not enough data points

        if normalize:
            std_dev = np.std(signal)
            if std_dev == 0:
                return 0.0 # Cannot normalize if std dev is zero
            r *= std_dev

        if r <= 0:
            raise ValueError("Tolerance 'r' must be greater than 0.")

        # Helper function to calculate B_m(r) and A_m(r)
        def _phi(data, M):
            x = np.array([data[i:i+M] for i in range(N - M + 1)])
            B = 0.0
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    if np.max(np.abs(x[i] - x[j])) <= r:
                        B += 1
            return B

        B_m = _phi(signal, m)
        A_m = _phi(signal, m + 1)

        if B_m == 0 or A_m == 0:
            return 0.0 # Avoid log(0)
        
        return -np.log(A_m / B_m)

    def approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2, normalize: bool = True) -> float:
        """
        Calculate the Approximate Entropy (ApEn) of a signal.
        
        Parameters:
        - signal: Input signal (1D numpy array).
        - m: Embedding dimension.
        - r: Tolerance (usually 0.1 to 0.25 times the standard deviation of the signal).
        - normalize: If True, r is interpreted as a factor of the signal's standard deviation.
        """
        N = len(signal)
        if N < m + 1:
            return 0.0 # Not enough data points

        if normalize:
            std_dev = np.std(signal)
            if std_dev == 0:
                return 0.0 # Cannot normalize if std dev is zero
            r *= std_dev

        if r <= 0:
            raise ValueError("Tolerance 'r' must be greater than 0.")

        def _phi(data, M):
            x = np.array([data[i:i+M] for i in range(N - M + 1)])
            C = np.zeros(len(x))
            for i in range(len(x)):
                count = 0
                for j in range(len(x)):
                    if np.max(np.abs(x[i] - x[j])) <= r:
                        count += 1
                C[i] = count / (N - M + 1)
            
            # Avoid log(0)
            C = C[C > 0]
            if len(C) == 0:
                return 0.0
            return np.mean(np.log(C))

        phi_m = _phi(signal, m)
        phi_m_plus_1 = _phi(signal, m + 1)

        return phi_m - phi_m_plus_1

    def fractal_dimension(self, signal: np.ndarray, method: str = 'higuchi', k_max: int = 10) -> float:
        """
        Calculate the Fractal Dimension (FD) of a signal using various methods.
        
        Parameters:
        - signal: Input signal (1D numpy array).
        - method: The method to use ('higuchi' or 'katz').
        - k_max: Maximum 'k' value for Higuchi's method.
        """
        if method == 'higuchi':
            return self._higuchi_fd(signal, k_max)
        elif method == 'katz':
            return self._katz_fd(signal)
        else:
            raise ValueError(f"Unknown fractal dimension method: {method}")

    def _higuchi_fd(self, signal: np.ndarray, k_max: int) -> float:
        """
        Calculates Higuchi Fractal Dimension (HFD).
        """
        N = len(signal)
        L_k = np.zeros(k_max)
        
        for k in range(1, k_max + 1):
            L_m_k = np.zeros(k)
            for m in range(1, k + 1):
                # Construct the m-th sequence
                x_m_k = signal[m-1::k]
                
                if len(x_m_k) <= 1:
                    L_m_k[m-1] = 0
                    continue

                # Calculate length of the m-th sequence
                length = np.sum(np.abs(np.diff(x_m_k))) * (N - 1) / (k * (len(x_m_k) - 1))
                L_m_k[m-1] = length
            
            # Average over m
            L_k[k-1] = np.mean(L_m_k[L_m_k > 0]) # Only average non-zero lengths
            
        # Filter out zero values from L_k and corresponding k values
        valid_k_indices = np.where(L_k > 0)[0]
        if len(valid_k_indices) < 2: # Need at least two points for regression
            return 0.0

        k_values = 1.0 / (np.arange(1, k_max + 1)[valid_k_indices])
        log_L_k = np.log(L_k[valid_k_indices])
        log_k_values = np.log(k_values)

        # Perform linear regression
        # Handle potential issues with all log_k_values being the same
        if np.all(log_k_values == log_k_values[0]):
            return 0.0 # Cannot perform regression if all x-values are the same

        # Using polyfit for robust linear regression
        coeffs = np.polyfit(log_k_values, log_L_k, 1)
        return coeffs[0]

    def _katz_fd(self, signal: np.ndarray) -> float:
        """
        Calculates Katz Fractal Dimension.
        """
        L = np.sum(np.abs(np.diff(signal)))
        d = np.max(np.abs(signal - signal[0]))
        
        if L == 0 or d == 0:
            return 0.0 # Avoid log(0) or division by zero
            
        return np.log10(L) / (np.log10(d) + np.log10(L / d))
