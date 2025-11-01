from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog
)
from PyQt6.QtCore import pyqtSignal
import pyqtgraph as pg
import numpy as np

from .base_panel import BaseControlPanel, NumericParameter, EnumParameter, BoolParameter
from preprocessing_bio import SignalSegmentation

class SegmentationPanel(BaseControlPanel):
    """Panel for signal segmentation with visual preview."""
    
    segmentation_changed = pyqtSignal(dict)  # Emitted when segmentation parameters change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.segmenter = SignalSegmentation()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        super()._init_ui()
        
        # Create main layout sections
        self.seg_controls = self.add_parameter_group("Segmentation Settings")
        self.preview = self.add_parameter_group("Segment Preview")
        
        # Add segmentation method selector
        self.seg_method = EnumParameter("seg_method", [
            "Fixed Window",
            "Overlapping Window",
            "Event-based"
        ])
        self.add_parameter(self.seg_controls, "Method:", self.seg_method)
        
        # Initialize parameter groups for each method
        self._init_fixed_params()
        self._init_overlap_params()
        self._init_event_params()
        
        # Add preview visualization
        self._init_preview()
        
        # Add navigation controls
        self._init_navigation()
        
        # Connect signals
        self.seg_method.value_changed.connect(self._on_method_changed)
        self.parameters_changed.connect(self._on_params_changed)
        
        # Show initial method
        self._on_method_changed("seg_method", "Fixed Window")
        
    def _init_fixed_params(self):
        """Initialize fixed window parameters."""
        self.fixed_group = QGroupBox()
        layout = QVBoxLayout(self.fixed_group)
        
        # Add parameters
        self.window_size = NumericParameter("window_size", 100, 10000, 100, 0)
        self.add_parameter(self.fixed_group, "Window Size (samples):", self.window_size)
        
        self.seg_controls.layout().addWidget(self.fixed_group)
        
    def _init_overlap_params(self):
        """Initialize overlapping window parameters."""
        self.overlap_group = QGroupBox()
        layout = QVBoxLayout(self.overlap_group)
        
        # Add parameters
        self.overlap_size = NumericParameter("window_size", 100, 10000, 100, 0)
        self.overlap_percent = NumericParameter("overlap", 0, 90, 1, 0)
        
        self.add_parameter(self.overlap_group, "Window Size (samples):", self.overlap_size)
        self.add_parameter(self.overlap_group, "Overlap (%):", self.overlap_percent)
        
        self.seg_controls.layout().addWidget(self.overlap_group)
        self.overlap_group.hide()
        
    def _init_event_params(self):
        """Initialize event-based parameters."""
        self.event_group = QGroupBox()
        layout = QVBoxLayout(self.event_group)
        
        # Add parameters
        self.pre_event = NumericParameter("pre_event", 0, 5000, 100, 0)
        self.post_event = NumericParameter("post_event", 0, 5000, 100, 0)
        
        self.add_parameter(self.event_group, "Pre-event (samples):", self.pre_event)
        self.add_parameter(self.event_group, "Post-event (samples):", self.post_event)
        
        # Add event file selection
        self.event_file_btn = QPushButton("Load Event File")
        self.event_file_btn.clicked.connect(self._load_event_file)
        layout.addWidget(self.event_file_btn)
        
        self.event_file_label = QLabel("No event file loaded")
        layout.addWidget(self.event_file_label)
        
        self.seg_controls.layout().addWidget(self.event_group)
        self.event_group.hide()
        
    def _init_preview(self):
        """Initialize the segment preview visualization."""
        layout = QVBoxLayout()
        
        # Create signal plot
        self.signal_plot = pg.PlotWidget(title="Signal Preview")
        self.signal_plot.setLabel('left', 'Amplitude')
        self.signal_plot.setLabel('bottom', 'Sample')
        self.signal_plot.showGrid(x=True, y=True)
        
        # Add segment overlay
        self.segment_region = pg.LinearRegionItem()
        self.segment_region.setZValue(10)
        self.signal_plot.addItem(self.segment_region)
        
        layout.addWidget(self.signal_plot)
        self.preview.setLayout(layout)
        
    def _init_navigation(self):
        """Initialize segment navigation controls."""
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Previous")
        self.next_btn = QPushButton("Next →")
        self.segment_label = QLabel("Segment: 0/0")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.segment_label)
        nav_layout.addWidget(self.next_btn)
        
        self.prev_btn.clicked.connect(self._prev_segment)
        self.next_btn.clicked.connect(self._next_segment)
        
        self.layout.addLayout(nav_layout)
        
        # Initialize navigation state
        self.current_segment = 0
        self.total_segments = 0
        self._update_navigation()
        
    def _on_method_changed(self, name: str, value: str):
        """Handle segmentation method changes."""
        # Hide all parameter groups
        self.fixed_group.hide()
        self.overlap_group.hide()
        self.event_group.hide()
        
        # Show selected group
        if value == "Fixed Window":
            self.fixed_group.show()
        elif value == "Overlapping Window":
            self.overlap_group.show()
        elif value == "Event-based":
            self.event_group.show()
            
        self._update_preview()
        
    def _on_params_changed(self, params: dict):
        """Handle parameter changes."""
        self._update_preview()
        self.segmentation_changed.emit(params)
        
    def _load_event_file(self):
        """Load event markers from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Event File",
            "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if filename:
            try:
                # Load events (implementation depends on file format)
                self.events = np.loadtxt(filename)
                self.event_file_label.setText(f"Events loaded: {len(self.events)}")
                self._update_preview()
            except Exception as e:
                self.event_file_label.setText(f"Error loading file: {str(e)}")
        
    def _update_preview(self):
        """Update the segmentation preview."""
        # This would be connected to actual signal data in practice
        # For now, generate sample data for visualization
        t = np.linspace(0, 10, 1000)
        data = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
        
        # Update signal plot
        self.signal_plot.clear()
        self.signal_plot.plot(data)
        
        # Calculate segment boundaries
        method = self.seg_method.get_value()
        if method == "Fixed Window":
            window = self.window_size.get_value()
            self.total_segments = len(data) // window
            start = self.current_segment * window
            end = start + window
            
        elif method == "Overlapping Window":
            window = self.overlap_size.get_value()
            overlap = self.overlap_percent.get_value() / 100
            step = int(window * (1 - overlap))
            self.total_segments = (len(data) - window) // step + 1
            start = self.current_segment * step
            end = start + window
            
        else:  # Event-based
            if hasattr(self, 'events') and len(self.events) > 0:
                self.total_segments = len(self.events)
                event = self.events[self.current_segment]
                start = max(0, event - self.pre_event.get_value())
                end = min(len(data), event + self.post_event.get_value())
            else:
                start = 0
                end = 100
                self.total_segments = 0
        
        # Update segment overlay
        self.segment_region.setRegion((start, end))
        
        # Update navigation
        self._update_navigation()
        
    def _prev_segment(self):
        """Navigate to previous segment."""
        if self.current_segment > 0:
            self.current_segment -= 1
            self._update_preview()
        
    def _next_segment(self):
        """Navigate to next segment."""
        if self.current_segment < self.total_segments - 1:
            self.current_segment += 1
            self._update_preview()
        
    def _update_navigation(self):
        """Update navigation controls state."""
        self.segment_label.setText(f"Segment: {self.current_segment + 1}/{self.total_segments}")
        self.prev_btn.setEnabled(self.current_segment > 0)
        self.next_btn.setEnabled(self.current_segment < self.total_segments - 1)
        
    def get_segmentation_config(self) -> dict:
        """Get current segmentation configuration."""
        params = self.get_parameters()
        return {
            'method': params['seg_method'],
            'parameters': params
        }
        
    def reset_parameters(self):
        """Reset parameters to defaults."""
        if self.seg_method.get_value() == "Fixed Window":
            self.window_size.set_value(1000)
        elif self.seg_method.get_value() == "Overlapping Window":
            self.overlap_size.set_value(1000)
            self.overlap_percent.set_value(50)
        else:  # Event-based
            self.pre_event.set_value(500)
            self.post_event.set_value(500)
            
    def update_signal(self, data: np.ndarray):
        """Update preview with actual signal data."""
        if data is None or len(data) == 0:
            return
            
        self.signal_plot.clear()
        self.signal_plot.plot(data)
        self._update_preview()