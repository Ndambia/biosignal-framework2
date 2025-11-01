from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QSizePolicy
)
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPen, QColor, QPainter
import numpy as np

class VisualizationWidget(QWidget):
    """Widget for real-time signal visualization using PyQtChart"""
    
    window_changed = pyqtSignal(float, float)  # emit time window changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._setup_chart()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create chart view
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Statistics group
        stats_group = QGroupBox("Signal Statistics")
        stats_layout = QFormLayout()
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Set size policy
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
    def _setup_chart(self):
        # Create chart
        self.chart = QChart()
        self.chart.setTitle("Signal Visualization")
        
        # Create series for the signal
        self.signal_series = QLineSeries()
        self.signal_series.setName("Signal")
        pen = QPen(QColor("#1f77b4"))  # Use a nice blue color
        pen.setWidth(2)
        self.signal_series.setPen(pen)
        
        # Add series to chart
        self.chart.addSeries(self.signal_series)
        
        # Create axes
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Time (s)")
        self.axis_x.setRange(0, 10)  # Default 10 second window
        
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Amplitude")
        self.axis_y.setRange(-1, 1)  # Default range
        
        # Add axes to chart
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        
        # Attach axes to series
        self.signal_series.attachAxis(self.axis_x)
        self.signal_series.attachAxis(self.axis_y)
        
        # Set chart to view
        self.chart_view.setChart(self.chart)
        
    def update_plot(self, data: np.ndarray, time_values: np.ndarray = None):
        """Update the plot with new data"""
        # Clear existing points
        self.signal_series.clear()
        
        # Create time values if not provided
        if time_values is None:
            time_values = np.linspace(0, len(data) / 1000, len(data))
        
        # Add new points
        for t, y in zip(time_values, data):
            self.signal_series.append(float(t), float(y))
        
        # Auto-scale Y axis if needed
        if data.size > 0:
            margin = 0.1 * (np.max(data) - np.min(data))
            self.axis_y.setRange(
                float(np.min(data) - margin),
                float(np.max(data) + margin)
            )
    
    def set_time_window(self, start: float, duration: float):
        """Set the visible time window"""
        self.axis_x.setRange(start, start + duration)
        self.window_changed.emit(start, start + duration)
    
    def clear(self):
        """Clear all data from the plot"""
        self.signal_series.clear()