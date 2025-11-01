from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable
import traceback

class FeatureWorker(QObject):
    """Worker for computing features in a background thread."""
    
    # Signals
    progress = pyqtSignal(int)  # Progress percentage
    feature_computed = pyqtSignal(str, np.ndarray)  # feature_name, result
    error = pyqtSignal(str, str)  # feature_name, error_message
    finished = pyqtSignal(pd.DataFrame)  # All computed features
    
    def __init__(self, signal_data: np.ndarray):
        super().__init__()
        self.signal_data = signal_data
        self.features_to_compute = {}
        self.results = {}
        
    def add_feature(self, name: str, function: Callable, parameters: Dict[str, Any] = None):
        """Add a feature to be computed."""
        self.features_to_compute[name] = {
            'function': function,
            'parameters': parameters or {}
        }
        
    def compute(self):
        """Compute all registered features."""
        total_features = len(self.features_to_compute)
        if total_features == 0:
            return
            
        for i, (name, config) in enumerate(self.features_to_compute.items()):
            try:
                result = config['function'](self.signal_data, **config['parameters'])
                self.results[name] = result
                self.feature_computed.emit(name, result)
            except Exception as e:
                error_msg = f"Error computing {name}: {str(e)}\n{traceback.format_exc()}"
                self.error.emit(name, error_msg)
                continue
                
            # Update progress
            progress = int(((i + 1) / total_features) * 100)
            self.progress.emit(progress)
            
        # Emit final results
        if self.results:
            df = pd.DataFrame(self.results)
            self.finished.emit(df)
            
    def clear(self):
        """Clear all registered features and results."""
        self.features_to_compute.clear()
        self.results.clear()