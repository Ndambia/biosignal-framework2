from PyQt6.QtCore import pyqtSignal
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import signal as sig
from .base_worker import BaseWorker, ConfigurationError, OperationError

class ProcessingWorker(BaseWorker):
    """Worker thread for signal processing operations."""
    
    # Additional signals specific to signal processing
    filter_complete = pyqtSignal(np.ndarray, dict)  # filtered_signal, filter_info
    feature_complete = pyqtSignal(dict, dict)  # features, metadata
    segment_complete = pyqtSignal(list, dict)  # segments, segment_info
    
    def __init__(self):
        super().__init__(operation_type="signal_processing")
        self.signal = None
        self.sampling_rate = None
        self.filter_config = {}
        self.feature_config = {}
        self.segment_config = {}
        
    def configure(self, signal: np.ndarray, sampling_rate: float,
                 filter_config: Optional[Dict] = None,
                 feature_config: Optional[Dict] = None,
                 segment_config: Optional[Dict] = None):
        """Configure processing parameters."""
        if signal is None or sampling_rate is None:
            raise ConfigurationError("Signal and sampling rate must be provided")
            
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.filter_config = filter_config or {}
        self.feature_config = feature_config or {}
        self.segment_config = segment_config or {}
        
    def _execute(self) -> Dict[str, Any]:
        """Execute signal processing pipeline."""
        try:
            results = {}
            
            # Apply filters
            if self.filter_config:
                self.report_status("Applying filters...")
                filtered_signal = self._apply_filters(self.signal)
                results['filtered_signal'] = filtered_signal
                self.filter_complete.emit(filtered_signal, self.filter_config)
                self.report_progress(33, "Filtering complete")
            else:
                filtered_signal = self.signal
                
            # Apply segmentation
            if self.segment_config:
                self.report_status("Segmenting signal...")
                segments = self._segment_signal(filtered_signal)
                results['segments'] = segments
                self.segment_complete.emit(segments, self.segment_config)
                self.report_progress(66, "Segmentation complete")
            
            # Extract features
            if self.feature_config:
                self.report_status("Extracting features...")
                features = self._extract_features(filtered_signal)
                results['features'] = features
                self.feature_complete.emit(features, self.feature_config)
                self.report_progress(100, "Feature extraction complete")
                
            return results
            
        except Exception as e:
            raise OperationError(f"Processing failed: {str(e)}")
            
    def _apply_filters(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply configured filters to signal."""
        filtered = signal_data.copy()
        
        for filter_type, params in self.filter_config.items():
            if not params.get('enabled', True):
                continue
                
            self.report_status(f"Applying {filter_type} filter...")
            
            if filter_type == 'bandpass':
                lowcut = params.get('lowcut', 20)
                highcut = params.get('highcut', 450)
                order = params.get('order', 4)
                
                nyquist = 0.5 * self.sampling_rate
                low = lowcut / nyquist
                high = highcut / nyquist
                
                b, a = sig.butter(order, [low, high], btype='band')
                filtered = sig.filtfilt(b, a, filtered)
                
            elif filter_type == 'notch':
                freq = params.get('frequency', 50)
                q = params.get('q_factor', 30)
                
                w0 = freq / (self.sampling_rate/2)
                b, a = sig.iirnotch(w0, q)
                filtered = sig.filtfilt(b, a, filtered)
                
            elif filter_type == 'lowpass':
                cutoff = params.get('cutoff', 100)
                order = params.get('order', 4)
                
                nyquist = 0.5 * self.sampling_rate
                normal_cutoff = cutoff / nyquist
                
                b, a = sig.butter(order, normal_cutoff, btype='low')
                filtered = sig.filtfilt(b, a, filtered)
                
            elif filter_type == 'highpass':
                cutoff = params.get('cutoff', 20)
                order = params.get('order', 4)
                
                nyquist = 0.5 * self.sampling_rate
                normal_cutoff = cutoff / nyquist
                
                b, a = sig.butter(order, normal_cutoff, btype='high')
                filtered = sig.filtfilt(b, a, filtered)
                
        return filtered
        
    def _segment_signal(self, signal_data: np.ndarray) -> List[np.ndarray]:
        """Segment signal according to configuration."""
        segments = []
        
        segment_type = self.segment_config.get('type', 'fixed')
        
        if segment_type == 'fixed':
            window_size = self.segment_config.get('window_size', 1000)
            overlap = self.segment_config.get('overlap', 0)
            
            # Calculate hop size
            hop = window_size - overlap
            
            # Create segments
            for i in range(0, len(signal_data) - window_size + 1, hop):
                segment = signal_data[i:i + window_size]
                segments.append(segment)
                
        elif segment_type == 'event':
            events = self.segment_config.get('events', [])
            pre_event = self.segment_config.get('pre_event', 100)
            post_event = self.segment_config.get('post_event', 100)
            
            for event in events:
                start = max(0, event - pre_event)
                end = min(len(signal_data), event + post_event)
                segment = signal_data[start:end]
                segments.append(segment)
                
        return segments
        
    def _extract_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract features according to configuration."""
        features = {}
        
        for feature_name, params in self.feature_config.items():
            if not params.get('enabled', True):
                continue
                
            self.report_status(f"Extracting {feature_name}...")
            
            if feature_name == 'rms':
                features['rms'] = np.sqrt(np.mean(np.square(signal_data)))
                
            elif feature_name == 'mav':
                features['mav'] = np.mean(np.abs(signal_data))
                
            elif feature_name == 'zero_crossings':
                features['zero_crossings'] = np.sum(np.diff(np.signbit(signal_data)))
                
            elif feature_name == 'mean_frequency':
                freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)
                fft = np.abs(np.fft.fft(signal_data))
                features['mean_frequency'] = np.sum(freqs * fft) / np.sum(fft)
                
            elif feature_name == 'median_frequency':
                freqs = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)
                fft = np.abs(np.fft.fft(signal_data))
                cumsum = np.cumsum(fft)
                features['median_frequency'] = freqs[np.where(cumsum >= cumsum[-1]/2)[0][0]]
                
        return features