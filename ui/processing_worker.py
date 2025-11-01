from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from typing import Dict, Any, Optional
from preprocessing_bio import SignalDenoising
from features import TimeDomainFeatures, FrequencyDomainFeatures, NonlinearFeatures
from models import SVMModel, RandomForestModel, CNNModel, LSTMModel
from .data_manager import DataManager

class ProcessingWorker(QThread):
    """Worker thread for signal processing pipeline."""
    
    # Signals for progress updates and results
    progress = pyqtSignal(int)  # Progress percentage
    status = pyqtSignal(str)    # Status message
    error = pyqtSignal(str)     # Error message
    
    # Processing stage completion signals
    filtering_complete = pyqtSignal(np.ndarray)
    features_complete = pyqtSignal(np.ndarray)
    model_complete = pyqtSignal(Dict[str, Any])
    
    def __init__(self):
        super().__init__()
        self._setup_components()
        self.data_manager = DataManager()
        self.reset()
        
    def _setup_components(self):
        """Initialize processing components."""
        # Signal processing components
        self.denoising = SignalDenoising()
        
        # Feature extractors
        self.feature_extractors = {
            'Time Domain': TimeDomainFeatures(),
            'Frequency Domain': FrequencyDomainFeatures(),
            'Nonlinear': NonlinearFeatures()
        }
        
        # Models
        self.models = {
            'SVM': SVMModel(),
            'Random Forest': RandomForestModel(),
            'CNN': CNNModel(input_shape=(1, 32, 32), num_classes=2),  # Example shape
            'LSTM': LSTMModel(input_size=10, hidden_size=64, num_classes=2)  # Example params
        }
        
    def reset(self):
        """Reset worker state."""
        self.signal_data = None
        self.sampling_rate = None
        self.filter_config = None
        self.features_config = None
        self.model_config = None
        self.processed_data = None
        self.extracted_features = None
        self.results_cache = {}
        
    def configure(self, 
                 signal_data: np.ndarray,
                 sampling_rate: float,
                 filter_config: Dict[str, Any],
                 features_config: Dict[str, Any],
                 model_config: Dict[str, Any]):
        """Configure the processing pipeline.
        
        Args:
            signal_data: Raw signal data
            sampling_rate: Signal sampling rate in Hz
            filter_config: Filter configuration dictionary
            features_config: Feature extraction configuration
            model_config: Model configuration
        """
        self.signal_data = signal_data
        self.sampling_rate = sampling_rate
        self.filter_config = filter_config
        self.features_config = features_config
        self.model_config = model_config
        
    def run(self):
        """Execute the processing pipeline."""
        try:
            # Check if we have all required data
            if any(x is None for x in [self.signal_data, self.sampling_rate,
                                     self.filter_config, self.features_config,
                                     self.model_config]):
                raise ValueError("Processing pipeline not fully configured")
            
            # Check cache first
            config = {
                'filter': self.filter_config,
                'features': self.features_config,
                'model': self.model_config
            }
            cached_result = self.data_manager.get_cached_result(self.signal_data, config)
            
            if cached_result is not None:
                self.status.emit("Loading from cache...")
                self.filtering_complete.emit(cached_result['filtered_data'])
                self.features_complete.emit(cached_result['features'])
                self.model_complete.emit(cached_result['results'])
                self.progress.emit(100)
                self.status.emit("Loaded from cache")
            else:
                # 1. Signal Filtering
                self.status.emit("Filtering signal...")
                self.progress.emit(0)
                
                filtered_data = self._apply_filtering(self.signal_data)
                self.filtering_complete.emit(filtered_data)
                self.progress.emit(33)
                
                # 2. Feature Extraction
                self.status.emit("Extracting features...")
                features = self._extract_features(filtered_data)
                self.features_complete.emit(features)
                self.progress.emit(66)
                
                # 3. Model Processing
                self.status.emit("Applying model...")
                results = self._apply_model(features)
                self.model_complete.emit(results)
                
                # Cache results
                self.data_manager.cache_result(
                    self.signal_data,
                    config,
                    {
                        'filtered_data': filtered_data,
                        'features': features,
                        'results': results
                    }
                )
                
                self.progress.emit(100)
                self.status.emit("Processing complete")
            
        except Exception as e:
            self.error.emit(str(e))
            
    def _apply_filtering(self, data: np.ndarray) -> np.ndarray:
        """Apply configured filters to the signal."""
        filter_type = self.filter_config['type']
        
        if filter_type == 'Bandpass':
            return self.denoising.bandpass_filter(
                data,
                self.filter_config['low_freq'],
                self.filter_config['high_freq'],
                self.sampling_rate,
                self.filter_config['order']
            )
        elif filter_type == 'Notch':
            return self.denoising.notch_filter(
                data,
                self.filter_config['high_freq'],  # Using high_freq as notch frequency
                self.sampling_rate
            )
        elif filter_type == 'Wavelet':
            return self.denoising.wavelet_denoise(
                data,
                wavelet='db4',  # Default wavelet
                level=self.filter_config['order']
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from the filtered signal."""
        feature_type = self.features_config['type']
        extractor = self.feature_extractors.get(feature_type)
        
        if extractor is None:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        # Configure window parameters
        window_size = int(self.features_config['window_size'] * self.sampling_rate)
        overlap = self.features_config['overlap']
        
        # Extract features based on type
        if feature_type == 'Time Domain':
            features = np.array([
                extractor.rms(data),
                extractor.mav(data),
                extractor.zero_crossing_rate(data),
                extractor.waveform_length(data)
            ])
        elif feature_type == 'Frequency Domain':
            freqs, psd = extractor.power_spectral_density(data)
            features = np.array([
                extractor.mean_frequency(data),
                extractor.median_frequency(data),
                extractor.spectral_entropy(data)
            ])
        else:  # Nonlinear
            features = np.array([
                extractor.sample_entropy(data),
                extractor.approximate_entropy(data),
                extractor.fractal_dimension(data)
            ])
            
        return features.reshape(1, -1)  # Reshape for model input
        
    def _apply_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Apply the selected model to the extracted features."""
        model_type = self.model_config['type']
        model = self.models.get(model_type)
        
        if model is None:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # For this example, we'll just return the features and model type
        # In a real application, you would train/load the model and make predictions
        return {
            'model_type': model_type,
            'features': features.tolist(),
            'predictions': None  # Would contain actual predictions in real use
        }
        
    def get_cached_result(self, stage: str) -> Optional[Any]:
        """Retrieve cached result for a processing stage."""
        return self.results_cache.get(stage)