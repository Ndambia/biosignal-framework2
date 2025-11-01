from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QProgressBar, QButtonGroup, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QFont
from .base_dock import BaseDock

class ActionButton(QPushButton):
    """Custom button with icon and text."""
    
    def __init__(self, text: str, icon_name: str = None, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(32)
        if icon_name:
            self.setIcon(QIcon(f":/icons/{icon_name}"))
        self.setCheckable(True)

class ActionGroup(QWidget):
    """Group of related actions."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._init_ui(title)
        
    def _init_ui(self, title: str):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)
        
        # Add title
        title_label = QLabel(title)
        font = QFont()
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # Create button container
        self.button_container = QWidget()
        self.button_layout = QVBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(4)
        layout.addWidget(self.button_container)
        
    def add_action(self, button: ActionButton):
        """Add an action button to the group."""
        self.button_layout.addWidget(button)

class QuickActionsDock(BaseDock):
    """Dockable panel for quick access to common actions."""
    
    # Action signals
    run_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    
    def __init__(self, title="Quick Actions", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create playback control group
        playback_group = ActionGroup("Playback Control")
        
        # Create playback buttons
        self.run_btn = ActionButton("Run", "run")
        self.pause_btn = ActionButton("Pause", "pause")
        self.stop_btn = ActionButton("Stop", "stop")
        
        # Add buttons to group
        playback_group.add_action(self.run_btn)
        playback_group.add_action(self.pause_btn)
        playback_group.add_action(self.stop_btn)
        
        # Create button group for mutual exclusion
        self.playback_group = QButtonGroup()
        self.playback_group.addButton(self.run_btn)
        self.playback_group.addButton(self.pause_btn)
        self.playback_group.addButton(self.stop_btn)
        
        # Add playback group to dock
        self.add_widget(playback_group)
        
        # Create data actions group
        data_group = ActionGroup("Data Actions")
        
        # Create data action buttons
        self.save_btn = ActionButton("Save", "save")
        self.export_btn = ActionButton("Export", "export")
        self.clear_btn = ActionButton("Clear", "clear")
        
        # Add buttons to group
        data_group.add_action(self.save_btn)
        data_group.add_action(self.export_btn)
        data_group.add_action(self.clear_btn)
        
        # Add data group to dock
        self.add_widget(data_group)
        
        # Create progress bar
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 8, 0, 0)
        
        progress_label = QLabel("Progress")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # Add progress widget to dock
        self.add_widget(progress_widget)
        
        # Connect signals
        self.run_btn.clicked.connect(self._on_run)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn.clicked.connect(self._on_stop)
        self.save_btn.clicked.connect(self.save_clicked)
        self.export_btn.clicked.connect(self.export_clicked)
        self.clear_btn.clicked.connect(self.clear_clicked)
        
        # Initialize state
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
    def _on_run(self):
        """Handle run button click."""
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.run_clicked.emit()
        
    def _on_pause(self):
        """Handle pause button click."""
        self.pause_clicked.emit()
        
    def _on_stop(self):
        """Handle stop button click."""
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.run_btn.setChecked(False)
        self.stop_clicked.emit()
        
    def set_progress(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)
        
    def reset_progress(self):
        """Reset progress bar."""
        self.progress_bar.setValue(0)
        
    def enable_actions(self, enabled: bool):
        """Enable or disable all actions."""
        self.run_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)