from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QTreeWidget, QTreeWidgetItem,
    QMenu, QDialog, QTextEdit
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction
import json
from datetime import datetime
from typing import List, Dict, Any

class ProcessingStep:
    """Represents a single processing operation."""
    
    def __init__(self, operation: str, parameters: Dict[str, Any]):
        self.operation = operation
        self.parameters = parameters
        self.timestamp = datetime.now()
        self.metrics = {}  # Signal metrics before/after
        
    def to_dict(self) -> dict:
        """Convert step to dictionary."""
        return {
            'operation': self.operation,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingStep':
        """Create step from dictionary."""
        step = cls(data['operation'], data['parameters'])
        step.timestamp = datetime.fromisoformat(data['timestamp'])
        step.metrics = data['metrics']
        return step

class ProcessingHistory:
    """Manages the history of processing operations."""
    
    def __init__(self):
        self.steps: List[ProcessingStep] = []
        
    def add_step(self, operation: str, parameters: Dict[str, Any]):
        """Add a processing step."""
        step = ProcessingStep(operation, parameters)
        self.steps.append(step)
        return step
        
    def clear(self):
        """Clear processing history."""
        self.steps.clear()
        
    def to_dict(self) -> dict:
        """Convert history to dictionary."""
        return {
            'steps': [step.to_dict() for step in self.steps]
        }
        
    def save(self, filename: str):
        """Save history to file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filename: str) -> 'ProcessingHistory':
        """Load history from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            history = cls()
            for step_data in data['steps']:
                step = ProcessingStep.from_dict(step_data)
                history.steps.append(step)
            return history

class StepDetailsDialog(QDialog):
    """Dialog for displaying detailed step information."""
    
    def __init__(self, step: ProcessingStep, parent=None):
        super().__init__(parent)
        self.step = step
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle(f"Step Details: {self.step.operation}")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Add timestamp
        layout.addWidget(QLabel(
            f"Timestamp: {self.step.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ))
        
        # Add parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        params_text = QTextEdit()
        params_text.setPlainText(
            json.dumps(self.step.parameters, indent=2)
        )
        params_text.setReadOnly(True)
        params_layout.addWidget(params_text)
        layout.addWidget(params_group)
        
        # Add metrics
        if self.step.metrics:
            metrics_group = QGroupBox("Signal Metrics")
            metrics_layout = QVBoxLayout(metrics_group)
            metrics_text = QTextEdit()
            metrics_text.setPlainText(
                json.dumps(self.step.metrics, indent=2)
            )
            metrics_text.setReadOnly(True)
            metrics_layout.addWidget(metrics_text)
            layout.addWidget(metrics_group)
            
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class HistoryView(QWidget):
    """Widget for displaying processing history."""
    
    step_selected = pyqtSignal(ProcessingStep)  # Emitted when step is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = ProcessingHistory()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Create history tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Operation", "Time", "Status"])
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._show_step_details)
        layout.addWidget(self.tree)
        
        # Create action buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save History")
        self.load_btn = QPushButton("Load History")
        self.clear_btn = QPushButton("Clear History")
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.save_btn.clicked.connect(self._save_history)
        self.load_btn.clicked.connect(self._load_history)
        self.clear_btn.clicked.connect(self._clear_history)
        
    def add_step(self, operation: str, parameters: Dict[str, Any]):
        """Add a processing step to history."""
        step = self.history.add_step(operation, parameters)
        self._add_step_item(step)
        return step
        
    def _add_step_item(self, step: ProcessingStep):
        """Add step to tree widget."""
        item = QTreeWidgetItem(self.tree)
        item.setText(0, step.operation)
        item.setText(1, step.timestamp.strftime("%H:%M:%S"))
        item.setText(2, "Completed")
        item.setData(0, Qt.ItemDataRole.UserRole, step)
        self.tree.scrollToItem(item)
        
    def _show_context_menu(self, position):
        """Show context menu for history item."""
        item = self.tree.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        details_action = QAction("Show Details", self)
        details_action.triggered.connect(lambda: self._show_step_details(item))
        menu.addAction(details_action)
        
        menu.exec(self.tree.viewport().mapToGlobal(position))
        
    def _show_step_details(self, item: QTreeWidgetItem):
        """Show details dialog for step."""
        step = item.data(0, Qt.ItemDataRole.UserRole)
        if step:
            dialog = StepDetailsDialog(step, self)
            dialog.exec()
            
    def _save_history(self):
        """Save processing history to file."""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing History",
            "",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.history.save(filename)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to save history: {str(e)}"
                )
                
    def _load_history(self):
        """Load processing history from file."""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Processing History",
            "",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.history = ProcessingHistory.load(filename)
                self._update_tree()
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to load history: {str(e)}"
                )
                
    def _clear_history(self):
        """Clear processing history."""
        self.history.clear()
        self.tree.clear()
        
    def _update_tree(self):
        """Update tree widget with current history."""
        self.tree.clear()
        for step in self.history.steps:
            self._add_step_item(step)
            
    def update_step_metrics(self, step: ProcessingStep, metrics: Dict[str, Any]):
        """Update signal metrics for a processing step."""
        step.metrics = metrics
        # Find and update tree item
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == step:
                item.setText(2, "Completed")
                break