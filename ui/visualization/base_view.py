from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
import pyqtgraph as pg
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

class BaseVisualizationView(QWidget):
    """Base class for visualization views."""
    
    # Signals
    view_changed = pyqtSignal(str)  # View type changed
    data_updated = pyqtSignal()  # Data was updated
    batch_selection_changed = pyqtSignal(str)  # Selected batch changed
    overlay_toggled = pyqtSignal(bool)  # Overlay mode toggled
    comparison_mode_changed = pyqtSignal(str)  # Comparison mode changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_data = {}  # Store batch visualization data
        self.current_batch = None
        self.overlay_enabled = False
        self.comparison_mode = 'overlay'  # 'overlay' or 'side-by-side'
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbar
        self.toolbar = QToolBar()
        layout.addWidget(self.toolbar)
        
        # Add common toolbar actions
        self.zoom_in_action = QAction("Zoom In", self)
        self.zoom_in_action.triggered.connect(self._zoom_in)
        self.toolbar.addAction(self.zoom_in_action)
        
        self.zoom_out_action = QAction("Zoom Out", self)
        self.zoom_out_action.triggered.connect(self._zoom_out)
        self.toolbar.addAction(self.zoom_out_action)
        
        self.reset_view_action = QAction("Reset View", self)
        self.reset_view_action.triggered.connect(self._reset_view)
        self.toolbar.addAction(self.reset_view_action)
        
        self.toolbar.addSeparator()
        
        # Add batch visualization controls
        batch_group = QGroupBox("Batch Visualization")
        batch_layout = QVBoxLayout(batch_group)
        
        # Batch selector
        self.batch_selector = QComboBox()
        self.batch_selector.currentTextChanged.connect(self._on_batch_selected)
        batch_layout.addWidget(self.batch_selector)
        
        # Visualization controls
        controls_layout = QHBoxLayout()
        
        # Overlay toggle
        self.overlay_checkbox = QCheckBox("Enable Overlay")
        self.overlay_checkbox.toggled.connect(self._on_overlay_toggled)
        controls_layout.addWidget(self.overlay_checkbox)
        
        # Comparison mode selector
        self.comparison_selector = QComboBox()
        self.comparison_selector.addItems(['overlay', 'side-by-side'])
        self.comparison_selector.currentTextChanged.connect(self._on_comparison_mode_changed)
        controls_layout.addWidget(self.comparison_selector)
        
        batch_layout.addLayout(controls_layout)
        layout.addWidget(batch_group)
        
        self.toolbar.addSeparator()
        
        # Add export actions
        self.export_action = QAction("Export", self)
        self.export_action.triggered.connect(self._export_view)
        self.toolbar.addAction(self.export_action)
        
        self.copy_action = QAction("Copy", self)
        self.copy_action.triggered.connect(self._copy_view)
        self.toolbar.addAction(self.copy_action)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background
        layout.addWidget(self.plot_widget)
        
        # Set up plot
        self.plot = self.plot_widget.getPlotItem()
        self.plot.showGrid(x=True, y=True)
        
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(False)
        
        # Create status bar
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(4, 0, 4, 0)
        
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)
        
        self.cursor_label = QLabel()
        status_layout.addWidget(self.cursor_label)
        
        layout.addWidget(status_bar)
        
    def _zoom_in(self):
        """Zoom in on plot."""
        self.plot_widget.getViewBox().scaleBy((0.5, 0.5))
        
    def _zoom_out(self):
        """Zoom out on plot."""
        self.plot_widget.getViewBox().scaleBy((2.0, 2.0))
        
    def _reset_view(self):
        """Reset plot view."""
        self.plot_widget.getViewBox().autoRange()
        
    def _export_view(self):
        """Export view as image."""
        # To be implemented by subclasses
        pass
        
    def _copy_view(self):
        """Copy view to clipboard."""
        # To be implemented by subclasses
        pass
        
    def set_data(self, data: np.ndarray, time: np.ndarray = None, batch_id: Optional[str] = None):
        """Set data to display.
        
        Args:
            data: Data array to display
            time: Optional time array
            batch_id: Optional batch identifier for batch visualization
        """
        if batch_id:
            self.batch_data[batch_id] = (data, time)
            if batch_id not in [self.batch_selector.itemText(i) for i in range(self.batch_selector.count())]:
                self.batch_selector.addItem(batch_id)
            if not self.current_batch:
                self.current_batch = batch_id
                self.batch_selector.setCurrentText(batch_id)
        else:
            super().set_data(data, time)
        
    def clear(self, batch_id: Optional[str] = None):
        """Clear data.
        
        Args:
            batch_id: Optional batch identifier to clear specific batch data
        """
        if batch_id:
            if batch_id in self.batch_data:
                del self.batch_data[batch_id]
                idx = self.batch_selector.findText(batch_id)
                if idx >= 0:
                    self.batch_selector.removeItem(idx)
            if batch_id == self.current_batch:
                self.current_batch = None
                self._update_view()
        else:
            self.plot.clear()
            self.status_label.clear()
            self.cursor_label.clear()
            self.batch_data.clear()
            self.batch_selector.clear()
            self.current_batch = None
        
    def update_status(self, message: str):
        """Update status message."""
        self.status_label.setText(message)
        
    def update_cursor_info(self, x: float, y: float):
        """Update cursor position info."""
        self.cursor_label.setText(f"X: {x:.3f}, Y: {y:.3f}")
        
    def enable_interaction(self, enabled: bool):
        """Enable/disable user interaction."""
        self.plot_widget.setMouseEnabled(x=enabled, y=enabled)
        self.zoom_in_action.setEnabled(enabled)
        self.zoom_out_action.setEnabled(enabled)
        self.reset_view_action.setEnabled(enabled)
        self.export_action.setEnabled(enabled)
        self.copy_action.setEnabled(enabled)
        self.batch_selector.setEnabled(enabled)
        self.overlay_checkbox.setEnabled(enabled)
        self.comparison_selector.setEnabled(enabled)
        
    def _on_batch_selected(self, batch_id: str):
        """Handle batch selection change."""
        if batch_id != self.current_batch:
            self.current_batch = batch_id
            self.batch_selection_changed.emit(batch_id)
            self._update_view()
            
    def _on_overlay_toggled(self, enabled: bool):
        """Handle overlay mode toggle."""
        self.overlay_enabled = enabled
        self.overlay_toggled.emit(enabled)
        self._update_view()
        
    def _on_comparison_mode_changed(self, mode: str):
        """Handle comparison mode change."""
        self.comparison_mode = mode
        self.comparison_mode_changed.emit(mode)
        self._update_view()
        
    def _update_view(self):
        """Update the view based on current settings."""
        self.plot.clear()
        
        if not self.batch_data:
            return
            
        if self.overlay_enabled:
            # Plot all batches overlaid
            for batch_id, (data, time) in self.batch_data.items():
                color = self._get_batch_color(batch_id)
                self.plot.plot(time if time is not None else np.arange(len(data)),
                             data,
                             pen=color,
                             name=batch_id)
        else:
            # Plot only current batch
            if self.current_batch and self.current_batch in self.batch_data:
                data, time = self.batch_data[self.current_batch]
                self.plot.plot(time if time is not None else np.arange(len(data)),
                             data,
                             pen='b',
                             name=self.current_batch)
                             
    def _get_batch_color(self, batch_id: str) -> Tuple[int, int, int]:
        """Get a unique color for a batch."""
        # Generate a unique color based on batch_id hash
        hash_val = hash(batch_id)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        return (r, g, b)

class SignalPlotView(BaseVisualizationView):
    """View for displaying time-domain signals."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_signal_plot()
        
    def _setup_signal_plot(self):
        """Set up signal plotting."""
        # Add signal-specific toolbar actions
        self.toolbar.addSeparator()
        
        self.show_grid_action = QAction("Show Grid", self)
        self.show_grid_action.setCheckable(True)
        self.show_grid_action.setChecked(True)
        self.show_grid_action.triggered.connect(self._toggle_grid)
        self.toolbar.addAction(self.show_grid_action)
        
        self.show_markers_action = QAction("Show Markers", self)
        self.show_markers_action.setCheckable(True)
        self.show_markers_action.triggered.connect(self._toggle_markers)
        self.toolbar.addAction(self.show_markers_action)
        
        # Set up plot
        self.plot.setLabel('bottom', 'Time', 's')
        self.plot.setLabel('left', 'Amplitude', 'V')
        
        # Create plot curves
        self.signal_curve = self.plot.plot(pen='b')
        self.marker_scatter = pg.ScatterPlotItem(pen='r', brush='r')
        self.plot.addItem(self.marker_scatter)
        self.marker_scatter.hide()
        
    def set_data(self, data: np.ndarray, time: np.ndarray = None):
        """Set signal data to display."""
        if time is None:
            time = np.arange(len(data)) / 1000.0  # Assume 1kHz sampling
            
        self.signal_curve.setData(time, data)
        self._update_markers(data, time)
        self.plot_widget.getViewBox().autoRange()
        
        # Update status
        stats = {
            'Mean': np.mean(data),
            'Std': np.std(data),
            'Min': np.min(data),
            'Max': np.max(data)
        }
        self.update_status(
            f"Mean: {stats['Mean']:.3f}, Std: {stats['Std']:.3f}, "
            f"Range: [{stats['Min']:.3f}, {stats['Max']:.3f}]"
        )
        
    def _update_markers(self, data: np.ndarray, time: np.ndarray):
        """Update signal markers (peaks, etc.)."""
        if self.show_markers_action.isChecked():
            # Find peaks
            peaks = self._find_peaks(data)
            self.marker_scatter.setData(
                time[peaks],
                data[peaks]
            )
            
    def _find_peaks(self, data: np.ndarray) -> np.ndarray:
        """Find signal peaks."""
        # Simple peak finding - can be made more sophisticated
        peaks = []
        for i in range(1, len(data)-1):
            if data[i-1] < data[i] > data[i+1]:
                peaks.append(i)
        return np.array(peaks)
        
    def _toggle_grid(self, show: bool):
        """Toggle grid visibility."""
        self.plot.showGrid(x=show, y=show)
        
    def _toggle_markers(self, show: bool):
        """Toggle marker visibility."""
        self.marker_scatter.setVisible(show)
        if show and self.signal_curve.getData()[0] is not None:
            x, y = self.signal_curve.getData()
            self._update_markers(y, x)