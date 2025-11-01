from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any, Optional
import traceback
import uuid
from datetime import datetime

class BaseWorker(QThread):
    """Base class for all worker threads."""
    
    # Status signals
    started = pyqtSignal(str)  # operation_id
    progress = pyqtSignal(str, int, str)  # operation_id, progress_value, status_message
    status = pyqtSignal(str, str)  # operation_id, status_message
    error = pyqtSignal(str, str)  # operation_id, error_message
    completed = pyqtSignal(str, dict)  # operation_id, results
    cancelled = pyqtSignal(str)  # operation_id
    
    def __init__(self, operation_type: str = "generic"):
        super().__init__()
        self.operation_type = operation_type
        self.operation_id = self._generate_operation_id()
        self.is_cancelled = False
        self._results = {}
        self._error = None
        
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{self.operation_type}_{timestamp}_{unique_id}"
        
    def configure(self, **kwargs):
        """Configure worker parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def run(self):
        """Main execution method."""
        try:
            self.started.emit(self.operation_id)
            self._results = self._execute()
            if not self.is_cancelled:
                self.completed.emit(self.operation_id, self._results)
        except Exception as e:
            self._error = str(e)
            self.error.emit(self.operation_id, f"{str(e)}\n{traceback.format_exc()}")
            
    def _execute(self) -> Dict[str, Any]:
        """Execute worker operation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute method")
        
    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True
        self.cancelled.emit(self.operation_id)
        
    def report_progress(self, value: int, message: str = None):
        """Report operation progress."""
        self.progress.emit(self.operation_id, value, message or "")
        
    def report_status(self, message: str):
        """Report operation status."""
        self.status.emit(self.operation_id, message)
        
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get operation results."""
        return self._results if not self._error else None
        
    def get_error(self) -> Optional[str]:
        """Get operation error if any."""
        return self._error
        
    def is_successful(self) -> bool:
        """Check if operation completed successfully."""
        return not self._error and not self.is_cancelled

class OperationError(Exception):
    """Custom exception for operation errors."""
    pass

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass