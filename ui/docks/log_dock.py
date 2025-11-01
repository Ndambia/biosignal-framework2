from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QPushButton,
    QHBoxLayout, QComboBox, QLabel, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat, QFont
from datetime import datetime
from enum import Enum, auto
from .base_dock import BaseDock

class LogLevel(Enum):
    """Log message severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    DEBUG = auto()

class LogColors:
    """Color definitions for different log levels."""
    INFO = QColor("#000000")  # Black
    WARNING = QColor("#FFA500")  # Orange
    ERROR = QColor("#FF0000")  # Red
    DEBUG = QColor("#808080")  # Gray
    TIMESTAMP = QColor("#0000FF")  # Blue

class LogWidget(QTextEdit):
    """Widget for displaying and managing log messages."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Set font
        font = QFont("Consolas", 9)
        self.setFont(font)
        
    def _show_context_menu(self, position):
        """Show context menu."""
        menu = QMenu()
        
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.copy)
        
        clear_action = menu.addAction("Clear")
        clear_action.triggered.connect(self.clear)
        
        select_all_action = menu.addAction("Select All")
        select_all_action.triggered.connect(self.selectAll)
        
        menu.addSeparator()
        
        save_action = menu.addAction("Save Logs")
        save_action.triggered.connect(self._save_logs)
        
        menu.exec(self.viewport().mapToGlobal(position))
        
    def _save_logs(self):
        """Save logs to file."""
        # To be implemented
        pass
        
    def append_message(self, message: str, level: LogLevel = LogLevel.INFO):
        """Append a new log message with timestamp."""
        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Create formats
        timestamp_format = QTextCharFormat()
        timestamp_format.setForeground(LogColors.TIMESTAMP)
        
        message_format = QTextCharFormat()
        message_format.setForeground(getattr(LogColors, level.name))
        
        # Get cursor and start block
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Insert timestamp
        cursor.insertText("[")
        cursor.insertText(timestamp, timestamp_format)
        cursor.insertText("] ")
        
        # Insert level indicator and message
        cursor.insertText(f"{level.name}: ", message_format)
        cursor.insertText(f"{message}\n", message_format)
        
        # Scroll to bottom
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

class LogDock(BaseDock):
    """Dockable panel for displaying log messages."""
    
    def __init__(self, title="Processing Log", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create log widget
        self.log_widget = LogWidget()
        self.add_widget(self.log_widget)
        
        # Create control bar
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add level filter
        level_label = QLabel("Level:")
        self.level_combo = QComboBox()
        self.level_combo.addItems([level.name for level in LogLevel])
        self.level_combo.setCurrentText("INFO")
        
        # Add buttons
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save")
        
        control_layout.addWidget(level_label)
        control_layout.addWidget(self.level_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.save_btn)
        
        self.add_widget(control_widget)
        
        # Connect signals
        self.clear_btn.clicked.connect(self.clear_logs)
        self.save_btn.clicked.connect(self._save_logs)
        self.level_combo.currentTextChanged.connect(self._filter_changed)
        
        # Initialize with welcome message
        self.log_info("Log system initialized")
        
    def log_info(self, message: str):
        """Log an info message."""
        self._log_message(message, LogLevel.INFO)
        
    def log_warning(self, message: str):
        """Log a warning message."""
        self._log_message(message, LogLevel.WARNING)
        
    def log_error(self, message: str):
        """Log an error message."""
        self._log_message(message, LogLevel.ERROR)
        
    def log_debug(self, message: str):
        """Log a debug message."""
        self._log_message(message, LogLevel.DEBUG)
        
    def _log_message(self, message: str, level: LogLevel):
        """Log a message if its level is sufficient."""
        current_level = LogLevel[self.level_combo.currentText()]
        if level.value >= current_level.value:
            self.log_widget.append_message(message, level)
            
    def clear_logs(self):
        """Clear all log messages."""
        self.log_widget.clear()
        self.log_info("Log cleared")
        
    def _save_logs(self):
        """Save logs to file."""
        # To be implemented
        pass
        
    def _filter_changed(self, level: str):
        """Handle log level filter changes."""
        self.log_info(f"Log level filter changed to {level}")