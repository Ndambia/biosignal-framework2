from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QProgressBar, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from typing import Dict, Any, List, Optional
import time

from .base_panel import BaseControlPanel
from ..error_handling import ErrorHandler, ErrorCategory, ErrorSeverity, ErrorInfo

class TaskProgressWidget(QWidget):
    """Widget for displaying individual task progress."""
    
    def __init__(self, task_id: str, task_name: str, parent=None):
        super().__init__(parent)
        self.task_id = task_id
        self.task_name = task_name
        self.start_time = time.time()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Task info
        info_layout = QHBoxLayout()
        self.name_label = QLabel(self.task_name)
        self.status_label = QLabel("Pending")
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.status_label)
        layout.addLayout(info_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Metrics
        metrics_layout = QHBoxLayout()
        self.time_label = QLabel("Time: 0:00")
        self.memory_label = QLabel("Memory: 0 MB")
        metrics_layout.addWidget(self.time_label)
        metrics_layout.addWidget(self.memory_label)
        layout.addLayout(metrics_layout)
        
    def update_progress(self, percentage: int, status: str, metrics: Dict[str, Any]):
        """Update task progress and metrics."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(status)
        
        # Update time
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.time_label.setText(f"Time: {minutes}:{seconds:02d}")
        
        # Update memory if provided
        if "memory_usage" in metrics:
            memory_mb = metrics["memory_usage"] / (1024 * 1024)  # Convert to MB
            self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
            
    def set_error(self, error_info: ErrorInfo):
        """Display error state."""
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: red")
        self.setToolTip(error_info.message)

class BatchMonitorPanel(BaseControlPanel):
    """Panel for monitoring batch processing progress and metrics."""
    
    task_completed = pyqtSignal(str, dict)  # task_id, metrics
    error_occurred = pyqtSignal(str, ErrorInfo)  # task_id, error_info
    
    def __init__(self, error_handler: ErrorHandler, parent=None):
        self.error_handler = error_handler
        self.task_widgets: Dict[str, TaskProgressWidget] = {}
        super().__init__(parent)
        
    def _init_ui(self):
        """Initialize the monitor interface."""
        super()._init_ui()
        
        # Tasks progress section
        tasks_group = QGroupBox("Active Tasks")
        tasks_layout = QVBoxLayout(tasks_group)
        
        # Scrollable task area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        scroll.setWidget(self.tasks_container)
        tasks_layout.addWidget(scroll)
        
        # Metrics table
        metrics_group = QGroupBox("Batch Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget(0, 3)  # Rows will be added dynamically
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Current", "Average"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        metrics_layout.addWidget(self.metrics_table)
        
        # Resource monitor
        resource_group = QGroupBox("Resource Usage")
        resource_layout = QHBoxLayout(resource_group)
        
        # CPU usage
        cpu_layout = QVBoxLayout()
        self.cpu_label = QLabel("CPU Usage")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addWidget(self.cpu_progress)
        resource_layout.addLayout(cpu_layout)
        
        # Memory usage
        memory_layout = QVBoxLayout()
        self.memory_label = QLabel("Memory Usage")
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        memory_layout.addWidget(self.memory_label)
        memory_layout.addWidget(self.memory_progress)
        resource_layout.addLayout(memory_layout)
        
        # Add all groups to main layout
        self.layout.addWidget(tasks_group)
        self.layout.addWidget(metrics_group)
        self.layout.addWidget(resource_group)
        
    def add_task(self, task_id: str, task_name: str):
        """Add a new task to monitor."""
        if task_id not in self.task_widgets:
            task_widget = TaskProgressWidget(task_id, task_name)
            self.task_widgets[task_id] = task_widget
            self.tasks_layout.addWidget(task_widget)
            
    def update_task_progress(self, task_id: str, percentage: int, 
                           status: str, metrics: Dict[str, Any]):
        """Update progress for a specific task."""
        if task_id in self.task_widgets:
            self.task_widgets[task_id].update_progress(percentage, status, metrics)
            self._update_batch_metrics(metrics)
            
    def set_task_error(self, task_id: str, error_info: ErrorInfo):
        """Set error state for a task."""
        if task_id in self.task_widgets:
            self.task_widgets[task_id].set_error(error_info)
            self.error_occurred.emit(task_id, error_info)
            
    def remove_task(self, task_id: str):
        """Remove a completed task."""
        if task_id in self.task_widgets:
            self.task_widgets[task_id].deleteLater()
            del self.task_widgets[task_id]
            
    def _update_batch_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics table with new values."""
        current_rows = self.metrics_table.rowCount()
        
        for i, (metric, value) in enumerate(metrics.items()):
            if i >= current_rows:
                self.metrics_table.insertRow(i)
                self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
                
            # Update current value
            current_item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, float) else str(value))
            self.metrics_table.setItem(i, 1, current_item)
            
            # Calculate and update average (if numeric)
            try:
                avg_item = self.metrics_table.item(i, 2)
                if avg_item is None:
                    avg_item = QTableWidgetItem("0")
                    self.metrics_table.setItem(i, 2, avg_item)
                    
                current_avg = float(avg_item.text())
                new_avg = (current_avg + float(value)) / 2
                avg_item.setText(f"{new_avg:.4f}")
            except (ValueError, TypeError):
                pass
                
    def update_resource_usage(self, cpu_percent: float, memory_percent: float):
        """Update resource usage indicators."""
        self.cpu_progress.setValue(int(cpu_percent))
        self.memory_progress.setValue(int(memory_percent))
        
        self.cpu_label.setText(f"CPU Usage: {cpu_percent:.1f}%")
        self.memory_label.setText(f"Memory Usage: {memory_percent:.1f}%")
        
        # Set color based on usage
        self._set_progress_color(self.cpu_progress, cpu_percent)
        self._set_progress_color(self.memory_progress, memory_percent)
        
    def _set_progress_color(self, progress_bar: QProgressBar, value: float):
        """Set progress bar color based on value."""
        if value < 60:
            color = QColor(0, 255, 0)  # Green
        elif value < 80:
            color = QColor(255, 165, 0)  # Orange
        else:
            color = QColor(255, 0, 0)  # Red
            
        style = f"""
            QProgressBar {{
                border: 1px solid grey;
                border-radius: 2px;
                text-align: center;
            }}
            
            QProgressBar::chunk {{
                background-color: {color.name()};
            }}
        """
        progress_bar.setStyleSheet(style)
        
    def clear_all(self):
        """Clear all tasks and reset metrics."""
        for task_id in list(self.task_widgets.keys()):
            self.remove_task(task_id)
            
        self.metrics_table.setRowCount(0)
        self.update_resource_usage(0, 0)