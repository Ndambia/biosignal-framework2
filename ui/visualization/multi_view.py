from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
    QLabel, QComboBox, QPushButton, QMenu,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
import pyqtgraph as pg
import numpy as np
from .base_view import BaseVisualizationView

class SignalItem:
    """Container for signal data and display properties."""
    
    def __init__(self, name: str, data: np.ndarray, time: np.ndarray,
                 color: QColor = None):
        self.name = name
        self.data = data
        self.time = time
        self.color = color or QColor(0, 0, 255)  # Default blue
        self.visible = True
        self.curve = None  # Plot curve reference

class MultiSignalView(BaseVisualizationView):
    """View for comparing multiple signals."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = {}  # name -> SignalItem
        self._setup_multi_view()
        
    def _setup_multi_view(self):
        """Set up multi-signal visualization."""
        # Add view-specific toolbar controls
        self.toolbar.addSeparator()
        
        # View mode selection
        self.toolbar.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Overlay', 'Stack', 'Grid'])
        self.mode_combo.currentTextChanged.connect(self._update_view_mode)
        self.toolbar.addWidget(self.mode_combo)
        
        # Signal list
        self.signal_list = QListWidget()
        self.signal_list.setMaximumWidth(200)
        self.signal_list.itemChanged.connect(self._on_signal_visibility_changed)
        
        # Create splitter layout
        splitter_layout = QHBoxLayout()
        splitter_layout.addWidget(self.signal_list)
        splitter_layout.addWidget(self.plot_widget)
        
        # Replace plot widget in layout
        self.layout().removeWidget(self.plot_widget)
        self.layout().addLayout(splitter_layout)
        
        # Set up synchronized views for stacked mode
        self.stacked_plots = []
        
        # Set up grid layout for grid mode
        self.grid_layout = pg.GraphicsLayoutWidget()
        self.grid_plots = []
        
    def add_signal(self, name: str, data: np.ndarray, time: np.ndarray = None,
                  color: QColor = None):
        """Add a signal to compare."""
        if time is None:
            time = np.arange(len(data)) / 1000.0
            
        # Create signal item
        signal = SignalItem(name, data, time, color)
        self.signals[name] = signal
        
        # Add to list widget
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.signal_list.addItem(item)
        
        # Update view
        self._update_view()
        
    def remove_signal(self, name: str):
        """Remove a signal."""
        if name in self.signals:
            # Remove from signals dict
            signal = self.signals.pop(name)
            
            # Remove curve if it exists
            if signal.curve is not None:
                self.plot.removeItem(signal.curve)
                
            # Remove from list widget
            items = self.signal_list.findItems(name, Qt.MatchFlag.MatchExactly)
            for item in items:
                self.signal_list.takeItem(self.signal_list.row(item))
                
            # Update view
            self._update_view()
            
    def clear(self):
        """Clear all signals."""
        super().clear()
        self.signals.clear()
        self.signal_list.clear()
        
        # Clear stacked plots
        for plot in self.stacked_plots:
            plot.clear()
        self.stacked_plots.clear()
        
        # Clear grid plots
        self.grid_layout.clear()
        self.grid_plots.clear()
        
    def _update_view_mode(self, mode: str):
        """Update view layout mode."""
        # Clear existing plots
        self.plot.clear()
        for plot in self.stacked_plots:
            plot.clear()
        self.grid_layout.clear()
        
        # Update layout
        if mode == 'Overlay':
            self.plot_widget.show()
            self.grid_layout.hide()
            self._setup_overlay_view()
        elif mode == 'Stack':
            self.plot_widget.show()
            self.grid_layout.hide()
            self._setup_stacked_view()
        else:  # Grid
            self.plot_widget.hide()
            self.grid_layout.show()
            self._setup_grid_view()
            
        # Update plots
        self._update_view()
        
    def _setup_overlay_view(self):
        """Set up overlay view."""
        self.plot.clear()
        self.plot.setLabel('bottom', 'Time', 's')
        self.plot.setLabel('left', 'Amplitude')
        
    def _setup_stacked_view(self):
        """Set up stacked view."""
        # Create plot for each signal
        self.stacked_plots = []
        for name, signal in self.signals.items():
            if signal.visible:
                plot = pg.PlotItem()
                plot.setLabel('left', name)
                self.stacked_plots.append(plot)
                
        # Link x axes
        for plot in self.stacked_plots[1:]:
            plot.setXLink(self.stacked_plots[0])
            
    def _setup_grid_view(self):
        """Set up grid view."""
        # Calculate grid dimensions
        n_signals = len([s for s in self.signals.values() if s.visible])
        cols = int(np.ceil(np.sqrt(n_signals)))
        rows = int(np.ceil(n_signals / cols))
        
        # Create plots
        self.grid_plots = []
        i = 0
        for name, signal in self.signals.items():
            if signal.visible:
                plot = self.grid_layout.addPlot(row=i//cols, col=i%cols)
                plot.setTitle(name)
                self.grid_plots.append(plot)
                i += 1
                
    def _update_view(self):
        """Update plot with current signals."""
        mode = self.mode_combo.currentText()
        
        if mode == 'Overlay':
            self._update_overlay_view()
        elif mode == 'Stack':
            self._update_stacked_view()
        else:  # Grid
            self._update_grid_view()
            
    def _update_overlay_view(self):
        """Update overlay view."""
        self.plot.clear()
        
        for name, signal in self.signals.items():
            if signal.visible:
                signal.curve = self.plot.plot(
                    signal.time,
                    signal.data,
                    name=name,
                    pen=pg.mkPen(signal.color, width=1)
                )
                
        self.plot.addLegend()
        
    def _update_stacked_view(self):
        """Update stacked view."""
        for plot, (name, signal) in zip(self.stacked_plots,
                                      [(n,s) for n,s in self.signals.items()
                                       if s.visible]):
            plot.clear()
            signal.curve = plot.plot(
                signal.time,
                signal.data,
                pen=pg.mkPen(signal.color, width=1)
            )
            
    def _update_grid_view(self):
        """Update grid view."""
        for plot, (name, signal) in zip(self.grid_plots,
                                      [(n,s) for n,s in self.signals.items()
                                       if s.visible]):
            plot.clear()
            signal.curve = plot.plot(
                signal.time,
                signal.data,
                pen=pg.mkPen(signal.color, width=1)
            )
            
    def _on_signal_visibility_changed(self, item: QListWidgetItem):
        """Handle signal visibility toggle."""
        name = item.text()
        if name in self.signals:
            self.signals[name].visible = (
                item.checkState() == Qt.CheckState.Checked
            )
            self._update_view()
            
    def set_signal_color(self, name: str, color: QColor):
        """Set signal color."""
        if name in self.signals:
            self.signals[name].color = color
            self._update_view()