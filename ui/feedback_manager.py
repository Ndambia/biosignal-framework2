from PyQt6.QtWidgets import QStatusBar, QMessageBox
from PyQt6.QtCore import QTimer, pyqtSignal, QObject

from .error_handling import ErrorInfo, ErrorSeverity

class FeedbackManager(QObject):
    """Manages user feedback, including status messages and error dialogs."""

    status_message_changed = pyqtSignal(str, int) # message, timeout_ms

    def __init__(self, status_bar: QStatusBar, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self.status_bar.showMessage("Ready")

    def show_status_message(self, message: str, timeout_ms: int = 3000):
        """Display a temporary status message in the status bar."""
        self.status_bar.showMessage(message, timeout_ms)
        self.status_message_changed.emit(message, timeout_ms)

    def show_error_dialog(self, error_info: ErrorInfo):
        """Show a detailed error dialog to the user."""
        dialog = QMessageBox()
        
        # Set icon based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            dialog.setIcon(QMessageBox.Icon.Critical)
        elif error_info.severity == ErrorSeverity.ERROR:
            dialog.setIcon(QMessageBox.Icon.Critical)
        elif error_info.severity == ErrorSeverity.WARNING:
            dialog.setIcon(QMessageBox.Icon.Warning)
        else:
            dialog.setIcon(QMessageBox.Icon.Information)
            
        # Set title and text
        dialog.setWindowTitle(f"{error_info.category.value.title()} Error")
        dialog.setText(error_info.message)
        
        # Add details if available
        detailed_text = []
        if error_info.details:
            detailed_text.append(f"Details: {error_info.details}")
            
        if error_info.suggestions:
            detailed_text.append("\nSuggestions:")
            for suggestion in error_info.suggestions:
                detailed_text.append(f"• {suggestion}")
                
        if detailed_text:
            dialog.setDetailedText("\n".join(detailed_text))
            
        dialog.exec()
