from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QProgressBar, QPushButton,
    QHBoxLayout, QLabel, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from datetime import datetime, timedelta
from .base_dock import BaseDock

class ProgressItem(QWidget):
    """Widget representing a single operation's progress."""
    
    cancelled = pyqtSignal(str)  # operation_id
    
    def __init__(self, operation_id: str, description: str, parent=None):
        super().__init__(parent)
        self.operation_id = operation_id
        self._init_ui(description)
        self.start_time = datetime.now()
        
    def _init_ui(self, description: str):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Add description and time
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        self.description_label = QLabel(description)
        self.time_label = QLabel("00:00")
        
        info_layout.addWidget(self.description_label)
        info_layout.addStretch()
        info_layout.addWidget(self.time_label)
        
        layout.addWidget(info_widget)
        
        # Add progress bar and cancel button
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMaximumWidth(60)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.cancel_btn)
        
        layout.addWidget(progress_widget)
        
        # Add status label
        self.status_label = QLabel("Starting...")
        layout.addWidget(self.status_label)
        
        # Connect signals
        self.cancel_btn.clicked.connect(
            lambda: self.cancelled.emit(self.operation_id)
        )
        
        # Start timer for elapsed time
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.timer.start(1000)  # Update every second
        
    def _update_time(self):
        """Update elapsed time display."""
        elapsed = datetime.now() - self.start_time
        self.time_label.setText(str(elapsed).split('.')[0])  # Remove microseconds
        
    def update_progress(self, value: int, status: str = None):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
            
    def complete(self, status: str = "Completed"):
        """Mark operation as complete."""
        self.progress_bar.setValue(100)
        self.status_label.setText(status)
        self.cancel_btn.setEnabled(False)
        self.timer.stop()
        
    def fail(self, error_message: str):
        """Mark operation as failed."""
        self.status_label.setText(f"Failed: {error_message}")
        self.cancel_btn.setEnabled(False)
        self.timer.stop()

class ProgressDock(BaseDock):
    """Dockable panel for tracking operation progress."""
    
    operation_cancelled = pyqtSignal(str)  # operation_id
    
    def __init__(self, title="Progress Tracker", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        self.operations = {}  # operation_id -> ProgressItem
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.add_widget(scroll)
        
        # Create container for progress items
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(8)
        self.layout.addStretch()
        
        scroll.setWidget(self.container)
        
        # Create control bar
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(4, 4, 4, 4)
        
        self.clear_completed_btn = QPushButton("Clear Completed")
        self.cancel_all_btn = QPushButton("Cancel All")
        
        control_layout.addWidget(self.clear_completed_btn)
        control_layout.addWidget(self.cancel_all_btn)
        
        self.add_widget(control_widget)
        
        # Connect signals
        self.clear_completed_btn.clicked.connect(self._clear_completed)
        self.cancel_all_btn.clicked.connect(self._cancel_all)
        
    def add_operation(self, operation_id: str, description: str) -> None:
        """Add a new operation to track."""
        # Create progress item
        item = ProgressItem(operation_id, description)
        
        # Insert before the stretch at the end
        self.layout.insertWidget(self.layout.count() - 1, item)
        
        # Connect cancel signal
        item.cancelled.connect(self.operation_cancelled)
        
        # Store reference
        self.operations[operation_id] = item
        
    def update_operation(self, operation_id: str, progress: int, status: str = None) -> None:
        """Update an operation's progress."""
        if operation_id in self.operations:
            self.operations[operation_id].update_progress(progress, status)
            
    def complete_operation(self, operation_id: str, status: str = "Completed") -> None:
        """Mark an operation as complete."""
        if operation_id in self.operations:
            self.operations[operation_id].complete(status)
            
    def fail_operation(self, operation_id: str, error_message: str) -> None:
        """Mark an operation as failed."""
        if operation_id in self.operations:
            self.operations[operation_id].fail(error_message)
            
    def remove_operation(self, operation_id: str) -> None:
        """Remove an operation from tracking."""
        if operation_id in self.operations:
            item = self.operations.pop(operation_id)
            self.layout.removeWidget(item)
            item.deleteLater()
            
    def _clear_completed(self):
        """Remove all completed operations."""
        completed = []
        for operation_id, item in self.operations.items():
            if item.progress_bar.value() == 100:
                completed.append(operation_id)
                
        for operation_id in completed:
            self.remove_operation(operation_id)
            
    def _cancel_all(self):
        """Cancel all active operations."""
        for operation_id in list(self.operations.keys()):
            self.operation_cancelled.emit(operation_id)