import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal

class SignalBus(QObject):
    """Signal bus for cross-tab communication."""
    
    # Signal generation signals
    signal_generated = pyqtSignal(np.ndarray, np.ndarray, float)  # signal, time, sampling_rate
    signal_modified = pyqtSignal(np.ndarray, np.ndarray)  # signal, time
    parameters_changed = pyqtSignal(str, dict)  # signal_type, parameters
    
    # Preprocessing signals
    filter_applied = pyqtSignal(np.ndarray, dict)  # filtered_signal, filter_info
    normalization_applied = pyqtSignal(np.ndarray, dict)  # normalized_signal, norm_info
    segmentation_applied = pyqtSignal(list, dict)  # segments, segment_info
    
    # Feature extraction signals
    features_extracted = pyqtSignal(dict, dict)  # features, metadata
    features_selected = pyqtSignal(list)  # selected_feature_names
    
    # ML workflow signals
    model_trained = pyqtSignal(object, dict)  # model, training_info
    predictions_made = pyqtSignal(np.ndarray, dict)  # predictions, metadata
    
    # Analysis signals
    analysis_complete = pyqtSignal(dict)  # analysis_results
    visualization_updated = pyqtSignal(str, dict)  # view_type, view_data

    # Data loading signals
    data_loaded_from_file = pyqtSignal(str, np.ndarray, np.ndarray, dict) # file_path, data, labels, metadata
    
    # Batch processing signals
    batch_started = pyqtSignal(str, dict)  # batch_id, config
    batch_progress = pyqtSignal(str, int, dict)  # batch_id, progress, metrics
    batch_completed = pyqtSignal(str, dict)  # batch_id, results
    batch_error = pyqtSignal(str, str)  # batch_id, error_message

class DataManager(QObject):
    """Manages data persistence and cross-tab communication."""
    
    # Status signals
    operation_started = pyqtSignal(str, str)  # operation_id, description
    operation_progress = pyqtSignal(str, int, str)  # operation_id, progress, status
    operation_completed = pyqtSignal(str, str)  # operation_id, status
    operation_failed = pyqtSignal(str, str)  # operation_id, error_message
    
    def __init__(self, cache_dir: str = "cache"):
        super().__init__()
        self.cache_dir = cache_dir
        self.results_cache = {}
        self._ensure_cache_dir()
        
        # Create signal bus
        self.signals = SignalBus()
        
        # Initialize data storage
        self.current_signal = None
        self.current_time = None
        self.current_sampling_rate = None
        self.current_features = None
        self.current_model = None
        self.current_batch = None
        self.batch_results = {}
        
        # Initialize callback registry
        self.callbacks = {
            'signal_update': [],
            'feature_update': [],
            'model_update': [],
            'batch_update': [],
            'error': []
        }
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for specific events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            
    def _notify_callbacks(self, event: str, *args, **kwargs):
        """Notify registered callbacks."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self._notify_callbacks('error', str(e))
                
    def set_signal_data(self, signal: np.ndarray, time: np.ndarray, 
                       sampling_rate: float, metadata: dict = None):
        """Set current signal data."""
        self.current_signal = signal
        self.current_time = time
        self.current_sampling_rate = sampling_rate
        
        # Emit signal
        self.signals.signal_generated.emit(signal, time, sampling_rate)
        
        # Notify callbacks
        self._notify_callbacks('signal_update', signal, time, sampling_rate)
        
    def update_signal(self, signal: np.ndarray, time: np.ndarray):
        """Update current signal data."""
        self.current_signal = signal
        self.current_time = time
        
        # Emit signal
        self.signals.signal_modified.emit(signal, time)
        
        # Notify callbacks
        self._notify_callbacks('signal_update', signal, time)
        
    def set_features(self, features: dict, metadata: dict = None):
        """Set current feature data."""
        self.current_features = features
        
        # Emit signal
        self.signals.features_extracted.emit(features, metadata or {})
        
        # Notify callbacks
        self._notify_callbacks('feature_update', features)
        
    def set_model(self, model: object, info: dict = None):
        """Set current model."""
        self.current_model = model
        
        # Emit signal
        self.signals.model_trained.emit(model, info or {})
        
        # Notify callbacks
        self._notify_callbacks('model_update', model)
        
    def get_cached_result(self,
                         signal_data: np.ndarray,
                         config: Dict[str, Any],
                         batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached results if available.
        
        Args:
            signal_data: Input signal data
            config: Processing configuration
            batch_id: Optional batch identifier for batch-specific caching
            
        Returns:
            Cached results if available, None otherwise
        """
        cache_key = self._generate_cache_key(signal_data, config, batch_id)
        
        # Try memory cache first
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
            
        # Try disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    result = json.load(f)
                # Store in memory cache
                self.results_cache[cache_key] = result
                return result
            except Exception as e:
                self._notify_callbacks('error', f"Cache read error: {str(e)}")
                return None
                
        return None
        
    def cache_result(self,
                    signal_data: np.ndarray,
                    config: Dict[str, Any],
                    result: Dict[str, Any],
                    batch_id: Optional[str] = None,
                    persist: bool = True):
        """Cache processing results.
        
        Args:
            signal_data: Input signal data
            config: Processing configuration
            result: Results to cache
            batch_id: Optional batch identifier for batch-specific caching
            persist: Whether to persist cache to disk
        """
        cache_key = self._generate_cache_key(signal_data, config, batch_id)
        self.results_cache[cache_key] = result
        
        if persist:
            # Save to disk for persistence
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self._make_serializable(result), f)
            except Exception as e:
                self._notify_callbacks('error', f"Cache write error: {str(e)}")
            
    def _generate_cache_key(self,
                          signal_data: np.ndarray,
                          config: Dict[str, Any],
                          batch_id: Optional[str] = None) -> str:
        """Generate a unique key for caching.
        
        Args:
            signal_data: Input signal data
            config: Processing configuration
            batch_id: Optional batch identifier
            
        Returns:
            Unique cache key string
        """
        data_hash = hash(signal_data.tobytes())
        config_str = json.dumps(config, sort_keys=True)
        key_parts = [str(data_hash), str(hash(config_str))]
        
        if batch_id:
            key_parts.append(batch_id)
            
        return "_".join(key_parts)
        
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj
        
    def export_results(self, 
                      signal_data: np.ndarray,
                      filtered_data: np.ndarray,
                      features: Dict[str, float],
                      results: Dict[str, Any],
                      export_dir: str = "exports"):
        """Export processing results to various formats."""
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export signal data as CSV
        signals_df = pd.DataFrame({
            'raw_signal': signal_data,
            'filtered_signal': filtered_data
        })
        signals_df.to_csv(
            os.path.join(export_dir, f"signals_{timestamp}.csv"),
            index=False
        )
        
        # Export features as CSV
        features_df = pd.DataFrame([features])
        features_df.to_csv(
            os.path.join(export_dir, f"features_{timestamp}.csv"),
            index=False
        )
        
        # Export full results as JSON
        full_results = {
            'features': features,
            'model_results': results
        }
        with open(os.path.join(export_dir, f"results_{timestamp}.json"), 'w') as f:
            json.dump(self._make_serializable(full_results), f, indent=2)
            
    def clear_cache(self, batch_id: Optional[str] = None):
        """Clear cached results.
        
        Args:
            batch_id: Optional batch identifier to clear only batch-specific cache
        """
        if batch_id:
            # Clear only batch-specific cache
            batch_keys = [k for k in self.results_cache if batch_id in k]
            for key in batch_keys:
                self.results_cache.pop(key, None)
                
            # Remove batch-specific cache files
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json') and batch_id in file:
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        self._notify_callbacks('error', f"Cache delete error: {str(e)}")
        else:
            # Clear all cache
            self.results_cache.clear()
            self.batch_results.clear()
            
            # Remove all cache files
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        self._notify_callbacks('error', f"Cache delete error: {str(e)}")

    def get_batch_cache_size(self, batch_id: str) -> int:
        """Get size of cached data for a specific batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Size of cached data in bytes
        """
        total_size = 0
        
        # Memory cache size
        batch_keys = [k for k in self.results_cache if batch_id in k]
        for key in batch_keys:
            total_size += sys.getsizeof(self.results_cache[key])
            
        # Disk cache size
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json') and batch_id in file:
                try:
                    total_size += os.path.getsize(os.path.join(self.cache_dir, file))
                except Exception:
                    pass
                    
        return total_size

    def prune_batch_cache(self, max_size_mb: int = 1000) -> None:
        """Prune batch cache to stay under size limit.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        
        # Get all batch IDs
        batch_ids = set()
        for key in self.results_cache:
            parts = key.split('_')
            if len(parts) >= 3:  # Has batch ID
                batch_ids.add(parts[2])
                
        # Calculate sizes and timestamps
        batch_info = []
        for batch_id in batch_ids:
            size = self.get_batch_cache_size(batch_id)
            # Get latest modification time of batch files
            latest_mod_time = 0
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json') and batch_id in file:
                    mod_time = os.path.getmtime(os.path.join(self.cache_dir, file))
                    latest_mod_time = max(latest_mod_time, mod_time)
            batch_info.append((batch_id, size, latest_mod_time))
            
        # Sort by modification time (oldest first)
        batch_info.sort(key=lambda x: x[2])
        
        # Calculate total size
        total_size = sum(info[1] for info in batch_info)
        
        # Remove oldest batches until under limit
        while total_size > max_size and batch_info:
            batch_id, size, _ = batch_info.pop(0)
            self.clear_cache(batch_id)
            total_size -= size
                
    def start_batch_processing(self, batch_id: str, config: Dict[str, Any]) -> None:
        """Start a new batch processing operation.
        
        Args:
            batch_id: Unique identifier for this batch
            config: Batch processing configuration
        """
        self.current_batch = {
            'id': batch_id,
            'config': config,
            'status': 'running',
            'progress': 0,
            'results': {}
        }
        self.signals.batch_started.emit(batch_id, config)
        self._notify_callbacks('batch_update', self.current_batch)
        
    def update_batch_progress(self, batch_id: str, progress: int, metrics: Dict[str, Any]) -> None:
        """Update batch processing progress.
        
        Args:
            batch_id: Batch identifier
            progress: Current progress (0-100)
            metrics: Current performance metrics
        """
        if self.current_batch and self.current_batch['id'] == batch_id:
            self.current_batch['progress'] = progress
            self.current_batch['metrics'] = metrics
            self.signals.batch_progress.emit(batch_id, progress, metrics)
            self._notify_callbacks('batch_update', self.current_batch)
            
    def complete_batch_processing(self, batch_id: str, results: Dict[str, Any]) -> None:
        """Complete a batch processing operation.
        
        Args:
            batch_id: Batch identifier
            results: Final batch results
        """
        if self.current_batch and self.current_batch['id'] == batch_id:
            self.current_batch['status'] = 'completed'
            self.current_batch['results'] = results
            self.batch_results[batch_id] = results
            self.signals.batch_completed.emit(batch_id, results)
            self._notify_callbacks('batch_update', self.current_batch)
            
    def get_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch results if available, None otherwise
        """
        return self.batch_results.get(batch_id)
        
    def get_current_batch_status(self) -> Optional[Dict[str, Any]]:
        """Get current batch processing status.
        
        Returns:
            Current batch status information if a batch is running,
            None otherwise
        """
        return self.current_batch