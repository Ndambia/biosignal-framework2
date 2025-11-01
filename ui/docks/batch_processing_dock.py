from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_dock import BaseDock
from ..panels.batch_configuration_panel import BatchConfigurationPanel
from ..panels.batch_monitor_panel import BatchMonitorPanel
from ..panels.batch_processing_panel import BatchProcessingPanel
from ..error_handling import ErrorHandler
from ..data_manager import DataManager

class BatchProcessingDock(BaseDock):
    """Dockable panel for batch processing operations."""
    
    state_changed = pyqtSignal(str)  # Emits current state (config/monitor/process)
    
    def __init__(self, title="Batch Processing", parent=None):
        super().__init__(title, parent)
        self.error_handler = ErrorHandler()
        self.data_manager = DataManager()
        self._current_state = "config"  # Default state
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.add_widget(scroll)
        
        # Create main container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(container)
        
        # Create stacked widget for different batch processing states
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        
        # Create panels for different batch processing states
        self.config_panel = BatchConfigurationPanel(error_handler=self.error_handler)
        self.monitor_panel = BatchMonitorPanel(error_handler=self.error_handler)
        self.process_panel = BatchProcessingPanel(data_manager=self.data_manager, error_handler=self.error_handler)
        
        # Add panels to stack
        self.stack.addWidget(self.config_panel)
        self.stack.addWidget(self.monitor_panel)
        self.stack.addWidget(self.process_panel)
        
    def _connect_signals(self):
        """Connect all panel signals."""
        try:
            # Connect configuration signals
            if hasattr(self.config_panel, 'configuration_complete'):
                self.config_panel.configuration_complete.connect(self.show_monitor)
            if hasattr(self.config_panel, 'config_changed'):
                self.config_panel.config_changed.connect(self._on_config_changed)
            
            # Connect monitoring signals
            if hasattr(self.monitor_panel, 'monitoring_started'):
                self.monitor_panel.monitoring_started.connect(self.show_processing)
            if hasattr(self.monitor_panel, 'task_completed'):
                self.monitor_panel.task_completed.connect(self._on_task_completed)
            if hasattr(self.monitor_panel, 'error_occurred'):
                self.monitor_panel.error_occurred.connect(self._on_task_error)
            
            # Connect processing signals
            if hasattr(self.process_panel, 'processing_complete'):
                self.process_panel.processing_complete.connect(self.show_config)
            if hasattr(self.process_panel, 'batch_started'):
                self.process_panel.batch_started.connect(self._on_batch_started)
            if hasattr(self.process_panel, 'batch_completed'):
                self.process_panel.batch_completed.connect(self._on_batch_completed)
            if hasattr(self.process_panel, 'batch_error'):
                self.process_panel.batch_error.connect(self._on_batch_error)
        except Exception as e:
            print(f"Error connecting signals: {str(e)}")
        
    def show_config(self):
        """Show batch configuration panel."""
        self.stack.setCurrentWidget(self.config_panel)
        self._current_state = "config"
        self.state_changed.emit(self._current_state)
        
    def show_monitor(self):
        """Show batch monitoring panel."""
        self.stack.setCurrentWidget(self.monitor_panel)
        self._current_state = "monitor"
        self.state_changed.emit(self._current_state)
        
    def show_processing(self):
        """Show batch processing panel."""
        self.stack.setCurrentWidget(self.process_panel)
        self._current_state = "process"
        self.state_changed.emit(self._current_state)
        
    def save_state(self):
        """Save current dock state."""
        state = {
            "current_panel": self.stack.currentIndex(),
            "current_state": self._current_state,
            "config_state": self.config_panel.save_state(),
            "monitor_state": self.monitor_panel.save_state(),
            "process_state": self.process_panel.save_state(),
            "visible": self.isVisible()
        }
        return state
        
    def _on_config_changed(self, config):
        """Handle configuration changes."""
        if self._current_state == "config":
            # Only handle config changes in config state
            self.state_changed.emit("config_updated")
            
    def _on_task_completed(self, task_id, metrics):
        """Handle task completion."""
        if self._current_state == "monitor":
            # Update monitoring display
            self.state_changed.emit("task_completed")
            
    def _on_task_error(self, task_id, error_info):
        """Handle task errors."""
        self.error_handler.handle_error(error_info)
        self.state_changed.emit("error")
        
    def _on_batch_started(self):
        """Handle batch processing start."""
        self.show_processing()
        self.state_changed.emit("processing_started")
        
    def _on_batch_completed(self):
        """Handle batch processing completion."""
        self.show_config()
        self.state_changed.emit("processing_completed")
        
    def _on_batch_error(self, error_info):
        """Handle batch processing errors."""
        self.error_handler.handle_error(error_info)
        self.show_config()
        self.state_changed.emit("error")
        
    def restore_state(self, state):
        """Restore dock state from saved state."""
        if not state:
            return
            
        self.stack.setCurrentIndex(state.get("current_panel", 0))
        self._current_state = state.get("current_state", "config")
        self.config_panel.restore_state(state.get("config_state"))
        self.monitor_panel.restore_state(state.get("monitor_state"))
        self.process_panel.restore_state(state.get("process_state"))
        self.setVisible(state.get("visible", True))