from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QComboBox,
    QPushButton, QHeaderView, QSplitter, QLabel,
    QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from typing import Dict, Any, List, Optional
import numpy as np

from .base_panel import BaseControlPanel
from ..error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
from ..visualization.base_view import BaseVisualizationView

class MetricsTable(QTableWidget):
    """Table widget for displaying comparison metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize table UI."""
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Metric", "Best", "Average", "Std Dev"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
    def update_metrics(self, metrics: Dict[str, List[float]]):
        """Update table with new metrics."""
        self.setRowCount(len(metrics))
        
        for i, (metric, values) in enumerate(metrics.items()):
            # Metric name
            self.setItem(i, 0, QTableWidgetItem(metric))
            
            # Best value
            best = max(values) if metric in ['accuracy', 'f1_score'] else min(values)
            best_item = QTableWidgetItem(f"{best:.4f}")
            best_item.setBackground(QColor(200, 255, 200))  # Light green
            self.setItem(i, 1, best_item)
            
            # Average
            avg = np.mean(values)
            self.setItem(i, 2, QTableWidgetItem(f"{avg:.4f}"))
            
            # Standard deviation
            std = np.std(values)
            self.setItem(i, 3, QTableWidgetItem(f"{std:.4f}"))

class ResultPlotView(BaseVisualizationView):
    """Custom plot view for result visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        
    def plot_comparison(self, data: Dict[str, Any], plot_type: str):
        """Plot comparison data based on selected type."""
        self.clear()
        
        if plot_type == "bar":
            self._plot_bar_comparison(data)
        elif plot_type == "box":
            self._plot_box_comparison(data)
        elif plot_type == "scatter":
            self._plot_scatter_comparison(data)
        elif plot_type == "line":
            self._plot_line_comparison(data)
            
    def _plot_bar_comparison(self, data: Dict[str, Any]):
        """Create bar plot comparison."""
        # Implementation using PyQtGraph
        pass
        
    def _plot_box_comparison(self, data: Dict[str, Any]):
        """Create box plot comparison."""
        # Implementation using PyQtGraph
        pass
        
    def _plot_scatter_comparison(self, data: Dict[str, Any]):
        """Create scatter plot comparison."""
        # Implementation using PyQtGraph
        pass
        
    def _plot_line_comparison(self, data: Dict[str, Any]):
        """Create line plot comparison."""
        # Implementation using PyQtGraph
        pass

class ResultComparisonPanel(BaseControlPanel):
    """Panel for comparing and analyzing batch processing results."""
    
    export_requested = pyqtSignal(str, dict)  # format, data
    
    def __init__(self, error_handler: ErrorHandler, parent=None):
        self.error_handler = error_handler
        self.results_data: Dict[str, Any] = {}
        super().__init__(parent)
        
    def _init_ui(self):
        """Initialize the comparison interface."""
        super()._init_ui()
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Plot controls
        control_layout = QHBoxLayout()
        
        # Metric selector
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.currentTextChanged.connect(self._update_visualization)
        metric_layout.addWidget(self.metric_combo)
        control_layout.addLayout(metric_layout)
        
        # Plot type selector
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["bar", "box", "scatter", "line"])
        self.plot_type_combo.currentTextChanged.connect(self._update_visualization)
        plot_layout.addWidget(self.plot_type_combo)
        control_layout.addLayout(plot_layout)
        
        # Add controls to viz layout
        viz_layout.addLayout(control_layout)
        
        # Plot view
        self.plot_view = ResultPlotView()
        viz_layout.addWidget(self.plot_view)
        
        # Bottom section - Metrics
        metrics_group = QGroupBox("Comparison Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Metrics table
        self.metrics_table = MetricsTable()
        metrics_layout.addWidget(self.metrics_table)
        
        # Export controls
        export_layout = QHBoxLayout()
        
        # Format selection
        self.export_combo = QComboBox()
        self.export_combo.addItems(["CSV", "JSON", "Excel"])
        export_layout.addWidget(self.export_combo)
        
        # Export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self._export_results)
        export_layout.addWidget(export_btn)
        
        metrics_layout.addLayout(export_layout)
        
        # Add groups to splitter
        splitter.addWidget(viz_group)
        splitter.addWidget(metrics_group)
        
        # Add splitter to main layout
        self.layout.addWidget(splitter)
        
    def update_results(self, results: Dict[str, Any]):
        """Update panel with new results data."""
        self.results_data = results
        
        # Update metric selector
        metrics = list(results.get("metrics", {}).keys())
        self.metric_combo.clear()
        self.metric_combo.addItems(metrics)
        
        # Update metrics table
        self.metrics_table.update_metrics(results.get("metrics", {}))
        
        # Update visualization
        self._update_visualization()
        
    def _update_visualization(self):
        """Update plot based on current selections."""
        if not self.results_data:
            return
            
        metric = self.metric_combo.currentText()
        plot_type = self.plot_type_combo.currentText()
        
        if metric and metric in self.results_data.get("metrics", {}):
            plot_data = {
                "values": self.results_data["metrics"][metric],
                "labels": self.results_data.get("labels", []),
                "title": f"{metric} Comparison",
                "ylabel": metric,
                "xlabel": "Models/Configurations"
            }
            
            self.plot_view.plot_comparison(plot_data, plot_type)
            
    def _export_results(self):
        """Export results in selected format."""
        export_format = self.export_combo.currentText().lower()
        self.export_requested.emit(export_format, self.results_data)
        
    def clear(self):
        """Clear all results and reset the panel."""
        self.results_data = {}
        self.metric_combo.clear()
        self.metrics_table.setRowCount(0)
        self.plot_view.clear()