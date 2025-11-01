from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from preprocessing_bio import (
    SignalDenoising, SignalNormalization, SignalSegmentation
)

class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    FILTER = "filter"
    NORMALIZE = "normalize"
    SEGMENT = "segment"

@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    data: np.ndarray
    metrics: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class PreprocessingIntegrator:
    """Integrates preprocessing_bio functionality with UI components."""
    
    def __init__(self):
        self.denoiser = SignalDenoising()
        self.normalizer = SignalNormalization()
        self.segmenter = SignalSegmentation()
        
    def apply_filter(self, data: np.ndarray, config: Dict[str, Any], fs: float = 1000.0) -> ProcessingResult:
        """Apply filtering operation."""
        try:
            # Calculate pre-filtering metrics
            pre_metrics = self._calculate_signal_metrics(data)
            
            # Apply appropriate filter
            if config['type'] == "Bandpass Filter":
                filtered = self.denoiser.bandpass_filter(
                    data,
                    config['parameters']['lowcut'],
                    config['parameters']['highcut'],
                    fs,
                    config['parameters']['order']
                )
            elif config['type'] == "Notch Filter":
                filtered = self.denoiser.notch_filter(
                    data,
                    config['parameters']['center_freq'],
                    fs,
                    config['parameters']['q_factor']
                )
            else:  # Wavelet
                filtered = self.denoiser.wavelet_denoise(
                    data,
                    config['parameters']['wavelet_type'],
                    config['parameters']['decomp_level']
                )
                
            # Calculate post-filtering metrics
            post_metrics = self._calculate_signal_metrics(filtered)
            
            # Combine metrics
            metrics = {
                'pre_processing': pre_metrics,
                'post_processing': post_metrics,
                'improvement': {
                    'snr_change': post_metrics['snr'] - pre_metrics['snr'],
                    'noise_reduction': pre_metrics['noise_level'] - post_metrics['noise_level']
                }
            }
            
            return ProcessingResult(filtered, metrics)
            
        except Exception as e:
            return ProcessingResult(
                data,
                {},
                success=False,
                error=f"Filter error: {str(e)}"
            )
            
    def apply_normalization(self, data: np.ndarray, config: Dict[str, Any]) -> ProcessingResult:
        """Apply normalization operation."""
        try:
            # Calculate pre-normalization metrics
            pre_metrics = self._calculate_signal_metrics(data)
            
            # Apply appropriate normalization
            if config['method'] == "Z-score":
                normalized = self.normalizer.zscore_normalize(data)
            elif config['method'] == "Min-Max":
                normalized = self.normalizer.minmax_scale(
                    data,
                    (config['parameters']['feature_min'],
                     config['parameters']['feature_max'])
                )
            else:  # Robust
                normalized = self.normalizer.robust_scale(data)
                
            # Calculate post-normalization metrics
            post_metrics = self._calculate_signal_metrics(normalized)
            
            # Combine metrics
            metrics = {
                'pre_processing': pre_metrics,
                'post_processing': post_metrics,
                'statistics': {
                    'mean_shift': post_metrics['mean'] - pre_metrics['mean'],
                    'std_ratio': post_metrics['std'] / pre_metrics['std']
                }
            }
            
            return ProcessingResult(normalized, metrics)
            
        except Exception as e:
            return ProcessingResult(
                data,
                {},
                success=False,
                error=f"Normalization error: {str(e)}"
            )
            
    def apply_segmentation(self, data: np.ndarray, config: Dict[str, Any]) -> ProcessingResult:
        """Apply segmentation operation."""
        try:
            # Calculate pre-segmentation metrics
            pre_metrics = self._calculate_signal_metrics(data)
            
            # Apply appropriate segmentation
            if config['method'] == "Fixed Window":
                segmented = self.segmenter.fixed_window(
                    data,
                    config['parameters']['window_size']
                )
            elif config['method'] == "Overlapping Window":
                segmented = self.segmenter.overlap_window(
                    data,
                    config['parameters']['window_size'],
                    config['parameters']['overlap'] / 100
                )
            else:  # Event-based
                segmented = self.segmenter.event_based_segment(
                    data,
                    config['events'],
                    config['parameters']['pre_event'],
                    config['parameters']['post_event']
                )
                
            # Calculate metrics for each segment
            segment_metrics = []
            for i, segment in enumerate(segmented):
                metrics = self._calculate_signal_metrics(segment)
                segment_metrics.append({
                    'segment_id': i,
                    'metrics': metrics
                })
                
            # Combine metrics
            metrics = {
                'original_signal': pre_metrics,
                'segments': segment_metrics,
                'statistics': {
                    'num_segments': len(segmented),
                    'avg_segment_length': np.mean([len(s) for s in segmented]),
                    'total_coverage': sum(len(s) for s in segmented) / len(data)
                }
            }
            
            return ProcessingResult(segmented, metrics)
            
        except Exception as e:
            return ProcessingResult(
                data,
                {},
                success=False,
                error=f"Segmentation error: {str(e)}"
            )
            
    def _calculate_signal_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate signal quality metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = float(np.mean(data))
        metrics['std'] = float(np.std(data))
        metrics['min'] = float(np.min(data))
        metrics['max'] = float(np.max(data))
        metrics['rms'] = float(np.sqrt(np.mean(data**2)))
        
        # Signal-to-noise ratio (estimated)
        signal_power = np.mean(data**2)
        noise = data - np.mean(data)
        noise_power = np.var(noise)
        metrics['snr'] = float(10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf'))
        metrics['noise_level'] = float(np.std(noise))
        
        # Frequency domain metrics
        if len(data) > 1:
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))
            magnitude = np.abs(fft)
            
            metrics['peak_frequency'] = float(freqs[np.argmax(magnitude[1:])])
            metrics['spectral_centroid'] = float(np.sum(freqs * magnitude) / np.sum(magnitude))
            
        return metrics
        
    def validate_config(self, stage: ProcessingStage, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate processing configuration."""
        try:
            if stage == ProcessingStage.FILTER:
                if config['type'] == "Bandpass Filter":
                    if not (0 < config['parameters']['lowcut'] < config['parameters']['highcut']):
                        return False, "Invalid cutoff frequencies"
                    if config['parameters']['order'] < 1:
                        return False, "Filter order must be positive"
                        
                elif config['type'] == "Notch Filter":
                    if config['parameters']['center_freq'] <= 0:
                        return False, "Center frequency must be positive"
                    if config['parameters']['q_factor'] <= 0:
                        return False, "Q factor must be positive"
                        
            elif stage == ProcessingStage.NORMALIZE:
                if config['method'] == "Min-Max":
                    if config['parameters']['feature_min'] >= config['parameters']['feature_max']:
                        return False, "Invalid feature range"
                        
            elif stage == ProcessingStage.SEGMENT:
                if config['method'] in ["Fixed Window", "Overlapping Window"]:
                    if config['parameters']['window_size'] < 1:
                        return False, "Window size must be positive"
                        
                if config['method'] == "Overlapping Window":
                    if not (0 <= config['parameters']['overlap'] < 100):
                        return False, "Overlap must be between 0 and 100"
                        
                if config['method'] == "Event-based":
                    if 'events' not in config:
                        return False, "Event markers required"
                    if config['parameters']['pre_event'] < 0 or config['parameters']['post_event'] < 0:
                        return False, "Event windows must be non-negative"
                        
            return True, None
            
        except KeyError as e:
            return False, f"Missing required parameter: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"