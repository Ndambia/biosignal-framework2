from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QThread, pyqtSlot
import numpy as np
import pandas as pd
from typing import Dict, Any

from .feature_panel import BaseFeaturePanel, NumericParameter, EnumParameter
from features.nonlinear import NonlinearFeatures
from ui.workers.feature_worker import FeatureWorker

class NonlinearFeaturePanel(BaseFeaturePanel):
    """Panel for nonlinear feature extraction."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_features()
        self._init_worker()
        
    def _init_worker(self):
        """Initialize the feature computation worker thread."""
        self.worker_thread = QThread()
        self.worker = None
        
        # Connect base progress bar
        self.computation_progress.connect(self.progress_bar.setValue)
        self.computation_finished.connect(self._on_computation_finished)
        
    def _init_features(self):
        """Initialize available nonlinear features."""
        features = NonlinearFeatures()
        
        # Sample Entropy
        self.add_feature("Sample Entropy", features.sample_entropy, {
            "m": {
                "type": "numeric",
                "min": 1,
                "max": 5,
                "step": 1,
                "decimals": 0,
                "default": 2
            },
            "r": {
                "type": "numeric",
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "decimals": 2,
                "default": 0.2
            },
            "normalize": {
                "type": "bool",
                "label": "Normalize r by Std Dev",
                "default": True
            }
        })
        
        # Approximate Entropy
        self.add_feature("Approximate Entropy", features.approximate_entropy, {
            "m": {
                "type": "numeric",
                "min": 1,
                "max": 5,
                "step": 1,
                "decimals": 0,
                "default": 2
            },
            "r": {
                "type": "numeric",
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "decimals": 2,
                "default": 0.2
            },
            "normalize": {
                "type": "bool",
                "label": "Normalize r by Std Dev",
                "default": True
            }
        })
        
        # Fractal Dimension
        self.add_feature("Fractal Dimension", features.fractal_dimension, {
            "method": {
                "type": "enum",
                "options": ["higuchi", "katz"],
                "default": "higuchi"
            },
            "k_max": {
                "type": "numeric",
                "min": 5,
                "max": 20,
                "step": 1,
                "decimals": 0,
                "default": 10
            }
        })
        
    def compute_selected_features(self):
        """Compute all selected features using a worker thread."""
        if not hasattr(self, 'signal_data') or self.signal_data is None:
            print("No signal data available for feature computation.")
            return
            
        if not self.selected_features:
            print("No features selected for computation.")
            return
            
        # Clean up previous worker if it exists
        if self.worker:
            self.worker.deleteLater()
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.worker = FeatureWorker(self.signal_data)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect worker signals to panel slots
        self.worker.progress.connect(self.computation_progress)
        self.worker.feature_computed.connect(self._on_feature_computed)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self.computation_finished)
        
        # Add selected features to the worker
        for feature_name in self.selected_features:
            feature_config = self.features[feature_name]
            params = self.get_feature_parameters(feature_name)
            self.worker.add_feature(feature_name, feature_config['function'], params)
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.compute_btn.setEnabled(False)
        
        # Start computation in the worker thread
        self.worker_thread.started.connect(self.worker.compute)
        self.worker_thread.start()
        
    @pyqtSlot(str, np.ndarray)
    def _on_feature_computed(self, feature_name: str, result: np.ndarray):
        """Slot to handle individual feature computation results."""
        self.cached_results[feature_name] = result
        self.feature_computed.emit(feature_name, result)
        
    @pyqtSlot(pd.DataFrame)
    def _on_computation_finished(self, df: pd.DataFrame):
        """Slot to handle all feature computation finished."""
        self.compute_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.computation_finished.emit(df) # Emit the final DataFrame
        
    @pyqtSlot(str, str)
    def _on_worker_error(self, feature_name: str, error_message: str):
        """Slot to handle errors from the worker."""
        print(f"Error in worker for {feature_name}: {error_message}")
        # TODO: Display error to user, e.g., via a log dock or message box
        
    def set_signal_data(self, data: np.ndarray):
        """Set the signal data to compute features from."""
        self.signal_data = data
        self.clear_cache()  # Clear cache when new data is set
        
    def closeEvent(self, event):
        """Ensure worker thread is terminated when panel is closed."""
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        super().closeEvent(event)