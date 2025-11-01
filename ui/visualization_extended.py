from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QFormLayout, QLabel, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCharts import QLineSeries
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor
import numpy as np
from .visualization import VisualizationWidget

class ProcessingVisualization(VisualizationWidget):
    """Extended visualization widget for processing pipeline results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.series_dict = {}
        self._setup_extended_ui()
        
    def _setup_extended_ui(self):
        """Set up additional UI components for processing visualization."""
        # Create features table
        self.features_group = QGroupBox("Extracted Features")
        features_layout = QVBoxLayout()
        
        self.features_table = QTableWidget(0, 2)  # 0 rows, 2 columns initially
        self.features_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.features_table.horizontalHeader().setStretchLastSection(True)
        features_layout.addWidget(self.features_table)
        
        self.features_group.setLayout(features_layout)
        self.layout().addWidget(self.features_group)
        
        # Create results group
        self.results_group = QGroupBox("Processing Results")
        results_layout = QFormLayout()
        
        self.model_label = QLabel("Model: None")
        self.prediction_label = QLabel("Prediction: None")
        self.confidence_label = QLabel("Confidence: N/A")
        
        results_layout.addRow(self.model_label)
        results_layout.addRow(self.prediction_label)
        results_layout.addRow(self.confidence_label)
        
        self.results_group.setLayout(results_layout)
        self.layout().addWidget(self.results_group)
        
    def update_plot(self, data: np.ndarray, time_values: np.ndarray, series_name: str = "Raw"):
        """Update or add a new data series to the plot.
        
        Args:
            data: Signal data array
            time_values: Time points array
            series_name: Name of the data series
        """
        # Create new series if it doesn't exist
        if series_name not in self.series_dict:
            new_series = QLineSeries()
            new_series.setName(series_name)
            
            # Set different colors for different series
            if series_name == "Raw":
                pen = QPen(QColor("#1f77b4"))  # Blue
            elif series_name == "Filtered":
                pen = QPen(QColor("#2ca02c"))  # Green
            else:
                pen = QPen(QColor("#ff7f0e"))  # Orange
                
            pen.setWidth(2)
            new_series.setPen(pen)
            
            self.chart.addSeries(new_series)
            new_series.attachAxis(self.axis_x)
            new_series.attachAxis(self.axis_y)
            
            self.series_dict[series_name] = new_series
            
        # Update the series data
        series = self.series_dict[series_name]
        series.clear()
        
        for t, y in zip(time_values, data):
            series.append(float(t), float(y))
            
        # Auto-scale Y axis if needed
        if data.size > 0:
            all_data = np.concatenate([
                np.array([p.y() for p in s.points()])
                for s in self.series_dict.values()
            ])
            margin = 0.1 * (np.max(all_data) - np.min(all_data))
            self.axis_y.setRange(
                float(np.min(all_data) - margin),
                float(np.max(all_data) + margin)
            )
            
    def update_features(self, features: dict):
        """Update the features table with extracted features.
        
        Args:
            features: Dictionary of feature names and values
        """
        self.features_table.setRowCount(len(features))
        
        for i, (name, value) in enumerate(features.items()):
            name_item = QTableWidgetItem(str(name))
            value_item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, float) else str(value))
            
            self.features_table.setItem(i, 0, name_item)
            self.features_table.setItem(i, 1, value_item)
            
    def update_results(self, results: dict):
        """Update the results display.
        
        Args:
            results: Dictionary containing model results
        """
        self.model_label.setText(f"Model: {results.get('model_type', 'None')}")
        self.prediction_label.setText(f"Prediction: {results.get('prediction', 'None')}")
        
        confidence = results.get('confidence')
        if confidence is not None:
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")
        else:
            self.confidence_label.setText("Confidence: N/A")
            
    def clear_all(self):
        """Clear all visualizations."""
        for series in self.series_dict.values():
            series.clear()
        self.features_table.setRowCount(0)
        self.model_label.setText("Model: None")
        self.prediction_label.setText("Prediction: None")
        self.confidence_label.setText("Confidence: N/A")