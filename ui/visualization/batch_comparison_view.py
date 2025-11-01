from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem,
    QSpinBox, QWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import pyqtgraph as pg
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Deque
from collections import deque
import time

from .base_view import BaseVisualizationView

class BatchComparisonView(BaseVisualizationView):
    """Specialized view for batch data comparison visualization."""
    
    # Additional signals for real-time updates
    update_rate_changed = pyqtSignal(int)  # Update rate in Hz
    buffer_size_changed = pyqtSignal(int)  # Buffer size in samples
    
    def __init__(self, parent=None, update_rate: int = 10, buffer_size: int = 1000):
        super().__init__(parent)
        
        # Real-time visualization settings
        self.update_rate = update_rate  # Hz
        self.buffer_size = buffer_size  # samples
        self.batch_buffers = {}  # Dict[str, Deque[np.ndarray]]
        self.last_update_time = {}  # Dict[str, float]
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._on_update_timer)
        self.update_timer.setInterval(1000 // self.update_rate)  # Convert Hz to ms
        
        self.setup_comparison_view()
        
    def setup_comparison_view(self):
        """Set up comparison-specific UI elements."""
        # Add statistics panel
        stats_group = QGroupBox("Batch Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_table = QTableWidget()
        self.stats_table.setRowCount(0)
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(['Batch', 'Mean', 'Std', 'Min', 'Max'])
        stats_layout.addWidget(self.stats_table)
        
        # Add to main layout after plot widget
        self.layout().insertWidget(self.layout().count() - 1, stats_group)
        
        # Add metric selector
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Metric:"))
        self.metric_selector = QComboBox()
        self.metric_selector.addItems(['amplitude', 'frequency', 'phase'])
        self.metric_selector.currentTextChanged.connect(self._on_metric_changed)
        metric_layout.addWidget(self.metric_selector)
        
        # Add visualization type selector
        metric_layout.addWidget(QLabel("View:"))
        self.view_selector = QComboBox()
        self.view_selector.addItems(['time-series', 'histogram', 'box-plot'])
        self.view_selector.currentTextChanged.connect(self._on_view_changed)
        metric_layout.addWidget(self.view_selector)
        
        # Add real-time controls
        realtime_group = QGroupBox("Real-time Settings")
        realtime_layout = QHBoxLayout(realtime_group)
        
        # Update rate control
        realtime_layout.addWidget(QLabel("Update Rate (Hz):"))
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(1, 60)
        self.update_rate_spin.setValue(self.update_rate)
        self.update_rate_spin.valueChanged.connect(self._on_update_rate_changed)
        realtime_layout.addWidget(self.update_rate_spin)
        
        # Buffer size control
        realtime_layout.addWidget(QLabel("Buffer Size:"))
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(100, 10000)
        self.buffer_size_spin.setSingleStep(100)
        self.buffer_size_spin.setValue(self.buffer_size)
        self.buffer_size_spin.valueChanged.connect(self._on_buffer_size_changed)
        realtime_layout.addWidget(self.buffer_size_spin)
        
        metric_layout.addWidget(realtime_group)
        
        # Add to toolbar
        metric_widget = QWidget()
        metric_widget.setLayout(metric_layout)
        self.toolbar.addWidget(metric_widget)
        
    def set_batch_data(self, batch_data: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]):
        """Set multiple batch datasets at once."""
        self.batch_data.clear()
        self.batch_selector.clear()
        self.batch_buffers.clear()
        self.last_update_time.clear()
        
        for batch_id, (data, time) in batch_data.items():
            self.set_data(data, time, batch_id)
            # Initialize real-time buffer
            self.batch_buffers[batch_id] = deque(maxlen=self.buffer_size)
            self.last_update_time[batch_id] = time.time()
            
        self._update_statistics()
        self._update_view()
        
    def update_batch_data(self, batch_id: str, new_data: np.ndarray, timestamp: Optional[float] = None):
        """Update batch data in real-time.
        
        Args:
            batch_id: Batch identifier
            new_data: New data points to add
            timestamp: Optional timestamp for the data
        """
        if batch_id not in self.batch_buffers:
            self.batch_buffers[batch_id] = deque(maxlen=self.buffer_size)
            self.last_update_time[batch_id] = time.time()
            
        # Add new data to buffer
        self.batch_buffers[batch_id].extend(new_data)
        
        # Update timestamp
        self.last_update_time[batch_id] = timestamp or time.time()
        
        # Start update timer if not running
        if not self.update_timer.isActive():
            self.update_timer.start()
        
    def _update_statistics(self):
        """Update batch statistics table."""
        self.stats_table.setRowCount(len(self.batch_data))
        
        for row, (batch_id, (data, _)) in enumerate(self.batch_data.items()):
            # Calculate statistics
            stats = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            
            # Update table
            self.stats_table.setItem(row, 0, QTableWidgetItem(batch_id))
            self.stats_table.setItem(row, 1, QTableWidgetItem(f"{stats['mean']:.3f}"))
            self.stats_table.setItem(row, 2, QTableWidgetItem(f"{stats['std']:.3f}"))
            self.stats_table.setItem(row, 3, QTableWidgetItem(f"{stats['min']:.3f}"))
            self.stats_table.setItem(row, 4, QTableWidgetItem(f"{stats['max']:.3f}"))
            
    def _on_metric_changed(self, metric: str):
        """Handle metric selection change."""
        self._update_view()
            
    def _on_view_changed(self, view_type: str):
        """Handle visualization type change."""
        self._update_view()
        
    def _update_view(self):
        """Update the visualization based on current settings."""
        self.plot.clear()
        
        if not self.batch_data:
            return
            
        view_type = self.view_selector.currentText()
        
        if view_type == 'time-series':
            self._plot_time_series()
        elif view_type == 'histogram':
            self._plot_histograms()
        elif view_type == 'box-plot':
            self._plot_box_plots()
            
    def _plot_time_series(self):
        """Plot time series comparison."""
        if self.comparison_mode == 'overlay':
            # Plot all batches overlaid
            for batch_id, (data, time) in self.batch_data.items():
                color = self._get_batch_color(batch_id)
                self.plot.plot(time if time is not None else np.arange(len(data)),
                             data,
                             pen=color,
                             name=batch_id)
        else:
            # Create side-by-side plots
            self.plot.clear()
            
            # Calculate grid layout
            n_plots = len(self.batch_data)
            cols = min(3, n_plots)  # Max 3 columns
            rows = (n_plots + cols - 1) // cols
            
            # Create subplot for each batch
            for i, (batch_id, (data, time)) in enumerate(self.batch_data.items()):
                row = i // cols
                col = i % cols
                
                subplot = self.plot.addPlot(row=row, col=col, title=batch_id)
                subplot.plot(time if time is not None else np.arange(len(data)),
                           data,
                           pen=self._get_batch_color(batch_id))
                           
                # Link x/y axes for synchronized zooming
                if i > 0:
                    subplot.setXLink(self.plot.getItem(0, 0))
                    subplot.setYLink(self.plot.getItem(0, 0))
                    
    def _plot_histograms(self):
        """Plot histogram comparison."""
        if self.comparison_mode == 'overlay':
            # Plot overlaid histograms
            for batch_id, (data, _) in self.batch_data.items():
                y, x = np.histogram(data, bins=50)
                color = self._get_batch_color(batch_id)
                self.plot.plot(x[:-1], y, stepMode=True, fillLevel=0,
                             pen=color, brush=color + (50,), name=batch_id)
        else:
            # Create side-by-side histograms
            n_plots = len(self.batch_data)
            cols = min(3, n_plots)
            rows = (n_plots + cols - 1) // cols
            
            for i, (batch_id, (data, _)) in enumerate(self.batch_data.items()):
                row = i // cols
                col = i % cols
                
                subplot = self.plot.addPlot(row=row, col=col, title=batch_id)
                y, x = np.histogram(data, bins=50)
                color = self._get_batch_color(batch_id)
                subplot.plot(x[:-1], y, stepMode=True, fillLevel=0,
                           pen=color, brush=color + (50,))
                           
    def _plot_box_plots(self):
        """Plot box plot comparison."""
        data_list = []
        labels = []
        positions = []
        
        for i, (batch_id, (data, _)) in enumerate(self.batch_data.items()):
            data_list.append(data)
            labels.append(batch_id)
            positions.append(i)
            
        # Create box plot
        box_plot = pg.PlotItem()
        self.plot.addItem(box_plot)
        
        # Calculate box plot statistics
        for i, data in enumerate(data_list):
            # Calculate quartiles
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            whisker_min = max(np.min(data), q1 - 1.5 * iqr)
            whisker_max = min(np.max(data), q3 + 1.5 * iqr)
            
            # Draw box
            box_plot.addItem(pg.QtGui.QGraphicsRectItem(i-0.25, q1, 0.5, q3-q1))
            
            # Draw median line
            box_plot.plot([i-0.25, i+0.25], [median, median], pen='r')
            
            # Draw whiskers
            box_plot.plot([i, i], [whisker_min, q1], pen='b')
            box_plot.plot([i, i], [q3, whisker_max], pen='b')
            
            # Draw outliers
            outliers = data[(data < whisker_min) | (data > whisker_max)]
            if len(outliers) > 0:
                box_plot.plot([i] * len(outliers), outliers, pen=None,
                            symbol='o', symbolSize=3, symbolBrush='r')
                            
        # Set axis labels and ticks
        box_plot.getAxis('bottom').setTicks([list(enumerate(labels))])
        
    def _on_update_timer(self):
        """Handle timer-based view updates."""
        current_time = time.time()
        
        # Check if any batch has new data
        has_updates = False
        for batch_id in self.batch_buffers:
            if (current_time - self.last_update_time.get(batch_id, 0)) < (2.0 / self.update_rate):
                has_updates = True
                break
                
        if has_updates:
            self._update_view()
        else:
            # Stop timer if no recent updates
            self.update_timer.stop()
            
    def _on_update_rate_changed(self, rate: int):
        """Handle update rate change."""
        self.update_rate = rate
        self.update_timer.setInterval(1000 // rate)
        self.update_rate_changed.emit(rate)
        
    def _on_buffer_size_changed(self, size: int):
        """Handle buffer size change."""
        self.buffer_size = size
        # Update buffer sizes
        for batch_id in self.batch_buffers:
            current_data = list(self.batch_buffers[batch_id])
            self.batch_buffers[batch_id] = deque(current_data[-size:], maxlen=size)
        self.buffer_size_changed.emit(size)
        
    def start_real_time_updates(self):
        """Start real-time updates."""
        if not self.update_timer.isActive():
            self.update_timer.start()
            
    def stop_real_time_updates(self):
        """Stop real-time updates."""
        if self.update_timer.isActive():
            self.update_timer.stop()
            
    def cleanup(self):
        """Clean up resources.
        
        Performs cleanup tasks:
        - Stops real-time updates
        - Releases memory buffers
        - Cleans up plot resources
        - Disconnects signals
        - Releases system resources
        """
        # Stop updates
        self.stop_real_time_updates()
        
        # Clear data buffers
        self.batch_buffers.clear()
        self.last_update_time.clear()
        self.batch_data.clear()
        
        # Clean up plot resources
        self.plot.clear()
        self.stats_table.clearContents()
        self.stats_table.setRowCount(0)
        
        # Disconnect signals
        self.metric_selector.currentTextChanged.disconnect(self._on_metric_changed)
        self.view_selector.currentTextChanged.disconnect(self._on_view_changed)
        self.update_rate_spin.valueChanged.disconnect(self._on_update_rate_changed)
        self.buffer_size_spin.valueChanged.disconnect(self._on_buffer_size_changed)
        self.auto_update_check.toggled.disconnect(self._on_auto_update_toggled)
        
        # Stop and disconnect timer
        if self.update_timer.isActive():
            self.update_timer.stop()
        self.update_timer.timeout.disconnect(self._on_update_timer)
        
        # Reset state
        self.current_batch = None
        self.overlay_enabled = False
        self.comparison_mode = 'overlay'
        
        # Call base class cleanup
        super().cleanup()