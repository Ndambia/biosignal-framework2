from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QDialogButtonBox
from PyQt6.QtCore import QThread, pyqtSlot
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .feature_panel import BaseFeaturePanel, NumericParameter
from features.frequency_domain import FrequencyDomainFeatures
from ui.workers.feature_worker import FeatureWorker

class BandConfigDialog(QDialog):
    """Dialog for configuring frequency bands."""
    def __init__(self, parent=None, initial_bands: Dict[str, tuple] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure Frequency Bands")
        self.bands = initial_bands if initial_bands is not None else {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 100)
        }
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        self.band_layouts = QVBoxLayout()
        main_layout.addLayout(self.band_layouts)

        self.add_band_button = QPushButton("Add Band")
        self.add_band_button.clicked.connect(self._add_band_row)
        main_layout.addWidget(self.add_band_button)

        self._populate_bands()

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _populate_bands(self):
        for name, (low, high) in self.bands.items():
            self._add_band_row(name, low, high)

    def _add_band_row(self, name: str = "", low: float = 0.0, high: float = 0.0):
        h_layout = QHBoxLayout()
        name_edit = QLineEdit(name)
        low_edit = QLineEdit(str(low))
        high_edit = QLineEdit(str(high))
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self._remove_band_row(h_layout))

        h_layout.addWidget(QLabel("Name:"))
        h_layout.addWidget(name_edit)
        h_layout.addWidget(QLabel("Low (Hz):"))
        h_layout.addWidget(low_edit)
        h_layout.addWidget(QLabel("High (Hz):"))
        h_layout.addWidget(high_edit)
        h_layout.addWidget(remove_button)
        self.band_layouts.addLayout(h_layout)

    def _remove_band_row(self, layout_to_remove: QHBoxLayout):
        for i in reversed(range(layout_to_remove.count())):
            widget = layout_to_remove.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.band_layouts.removeItem(layout_to_remove)
        layout_to_remove.deleteLater()

    def get_bands(self) -> Dict[str, tuple]:
        new_bands = {}
        for i in range(self.band_layouts.count()):
            h_layout = self.band_layouts.itemAt(i).layout()
            if h_layout:
                name = h_layout.itemAt(1).widget().text()
                low = float(h_layout.itemAt(3).widget().text())
                high = float(h_layout.itemAt(5).widget().text())
                new_bands[name] = (low, high)
        return new_bands


class FrequencyDomainFeaturePanel(BaseFeaturePanel):
    """Panel for frequency domain feature extraction."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fs = 1000  # Default sampling frequency, should be set externally
        self.frequency_bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 100)
        }
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
        """Initialize available frequency domain features."""
        features = FrequencyDomainFeatures()
        
        # Mean Frequency
        self.add_feature("Mean Frequency", lambda signal, nperseg: features.mean_frequency(signal, self.fs, nperseg), {
            "nperseg": {
                "type": "numeric",
                "min": 64,
                "max": 2048,
                "step": 64,
                "decimals": 0,
                "default": 256
            }
        })
        
        # Median Frequency
        self.add_feature("Median Frequency", lambda signal, nperseg: features.median_frequency(signal, self.fs, nperseg), {
            "nperseg": {
                "type": "numeric",
                "min": 64,
                "max": 2048,
                "step": 64,
                "decimals": 0,
                "default": 256
            }
        })
        
        # Frequency Band Power
        self.add_feature("Band Power", lambda signal, nperseg: features.frequency_band_power(signal, self.fs, self.frequency_bands, nperseg), {
            "nperseg": {
                "type": "numeric",
                "min": 64,
                "max": 2048,
                "step": 64,
                "decimals": 0,
                "default": 256
            }
        })
        
        # Spectral Entropy
        self.add_feature("Spectral Entropy", lambda signal, nperseg, normalize: features.spectral_entropy(signal, self.fs, nperseg, normalize), {
            "nperseg": {
                "type": "numeric",
                "min": 64,
                "max": 2048,
                "step": 64,
                "decimals": 0,
                "default": 256
            },
            "normalize": {
                "type": "bool",
                "label": "Normalize Entropy",
                "default": True
            }
        })

        # Peak Frequency
        self.add_feature("Peak Frequency", lambda signal, nperseg: features.peak_frequency(signal, self.fs, nperseg), {
            "nperseg": {
                "type": "numeric",
                "min": 64,
                "max": 2048,
                "step": 64,
                "decimals": 0,
                "default": 256
            }
        })

        # Add a button to configure frequency bands
        self.config_bands_btn = QPushButton("Configure Bands")
        self.config_bands_btn.clicked.connect(self._configure_bands)
        self.feature_group.layout().addWidget(self.config_bands_btn)

    def _configure_bands(self):
        dialog = BandConfigDialog(self, initial_bands=self.frequency_bands)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.frequency_bands = dialog.get_bands()
            print(f"Updated frequency bands: {self.frequency_bands}")
            # Optionally, re-emit parameters_changed if this affects feature computation
            # self.parameters_changed.emit(self.get_parameters())

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
        
    def set_signal_data(self, data: np.ndarray, fs: float):
        """Set the signal data and sampling frequency to compute features from."""
        self.signal_data = data
        self.fs = fs
        self.clear_cache()  # Clear cache when new data is set
        
    def closeEvent(self, event):
        """Ensure worker thread is terminated when panel is closed."""
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        super().closeEvent(event)