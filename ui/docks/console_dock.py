from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPlainTextEdit, QPushButton,
    QHBoxLayout, QLabel, QMenu, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat, QFont, QSyntaxHighlighter
import traceback
from datetime import datetime
from .base_dock import BaseDock

class SyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for console output."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._create_formats()
        
    def _create_formats(self):
        """Create text formats for different types of output."""
        # Error format (red)
        self.error_format = QTextCharFormat()
        self.error_format.setForeground(QColor("#FF0000"))
        
        # Warning format (orange)
        self.warning_format = QTextCharFormat()
        self.warning_format.setForeground(QColor("#FFA500"))
        
        # Exception format (dark red)
        self.exception_format = QTextCharFormat()
        self.exception_format.setForeground(QColor("#8B0000"))
        
        # Stack trace format (gray)
        self.stack_format = QTextCharFormat()
        self.stack_format.setForeground(QColor("#808080"))
        
    def highlightBlock(self, text: str):
        """Apply syntax highlighting to text block."""
        # Highlight errors
        if "Error:" in text:
            self.setFormat(0, len(text), self.error_format)
            
        # Highlight warnings
        elif "Warning:" in text:
            self.setFormat(0, len(text), self.warning_format)
            
        # Highlight exceptions
        elif "Exception:" in text:
            self.setFormat(0, len(text), self.exception_format)
            
        # Highlight stack traces
        elif "  File" in text or "    at" in text:
            self.setFormat(0, len(text), self.stack_format)

class ConsoleWidget(QPlainTextEdit):
    """Widget for displaying console output with syntax highlighting."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setMaximumBlockCount(1000)  # Limit number of lines
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Set font
        font = QFont("Consolas", 9)
        self.setFont(font)
        
        # Add syntax highlighter
        self.highlighter = SyntaxHighlighter(self.document())
        
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
        
        save_action = menu.addAction("Save Console Output")
        save_action.triggered.connect(self._save_output)
        
        menu.exec(self.viewport().mapToGlobal(position))
        
    def _save_output(self):
        """Save console output to file."""
        # To be implemented
        pass
        
    def append_message(self, message: str):
        """Append a message to the console."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"{message}\n")
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

class ConsoleDock(BaseDock):
    """Dockable panel for displaying error console."""
    
    def __init__(self, title="Error Console", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create console widget
        self.console = ConsoleWidget()
        self.add_widget(self.console)
        
        # Create control bar
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add auto-scroll checkbox
        self.auto_scroll = QCheckBox("Auto-scroll")
        self.auto_scroll.setChecked(True)
        
        # Add buttons
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save")
        
        control_layout.addWidget(self.auto_scroll)
        control_layout.addStretch()
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.save_btn)
        
        self.add_widget(control_widget)
        
        # Connect signals
        self.clear_btn.clicked.connect(self.clear_console)
        self.save_btn.clicked.connect(self._save_console)
        
        # Initialize with welcome message
        self.write_message("Console initialized")
        
    def write_message(self, message: str):
        """Write a regular message to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.append_message(f"[{timestamp}] {message}")
        
    def write_error(self, error: Exception):
        """Write an error with stack trace to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Write error message
        self.console.append_message(f"[{timestamp}] Error: {str(error)}")
        
        # Write stack trace
        stack_trace = "".join(traceback.format_tb(error.__traceback__))
        self.console.append_message(stack_trace)
        
        if self.auto_scroll.isChecked():
            self.console.verticalScrollBar().setValue(
                self.console.verticalScrollBar().maximum()
            )
            
    def write_warning(self, message: str):
        """Write a warning message to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.append_message(f"[{timestamp}] Warning: {message}")
        
    def clear_console(self):
        """Clear the console."""
        self.console.clear()
        self.write_message("Console cleared")
        
    def _save_console(self):
        """Save console output to file."""
        # To be implemented
        pass