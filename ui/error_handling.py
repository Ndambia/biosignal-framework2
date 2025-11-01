from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import logging
import traceback
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal

class BiosignalException(Exception):
    """Base exception for the Biosignal Framework."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class StateError(BiosignalException):
    """Exception raised for state management failures."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Categories of errors."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    FILE_IO = "file_io"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    UI = "ui"
    ML_TRAINING = "ml_training"
    ML_EVALUATION = "ml_evaluation"
    DATA_LOADING = "data_loading"
    FEATURE_EXTRACTION = "feature_extraction"

@dataclass
class ValidationRule:
    """Defines a validation rule."""
    condition: Callable
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.VALIDATION

@dataclass
class ErrorInfo:
    """Detailed error information."""
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    details: Optional[str] = None
    traceback: Optional[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

class BiosignalException(Exception):
    """Base exception for the Biosignal Framework."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class ValidationError(BiosignalException):
    """Exception raised for validation failures."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class ProcessingError(BiosignalException):
    """Exception raised for processing failures."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class ConfigurationError(BiosignalException):
    """Exception raised for configuration errors."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class MLTrainingError(BiosignalException):
    """Exception raised for ML training failures."""
    def __init__(self, message: str, details: Optional[str] = None, model_state: Optional[Dict] = None):
        self.message = message
        self.details = details
        self.model_state = model_state
        super().__init__(self.message)

class MLEvaluationError(BiosignalException):
    """Exception raised for ML evaluation failures."""
    def __init__(self, message: str, details: Optional[str] = None, metrics: Optional[Dict] = None):
        self.message = message
        self.details = details
        self.metrics = metrics
        super().__init__(self.message)

class DataLoadingError(BiosignalException):
    """Exception raised for data loading/preprocessing failures."""
    def __init__(self, message: str, details: Optional[str] = None, file_path: Optional[str] = None):
        self.message = message
        self.details = details
        self.file_path = file_path
        super().__init__(self.message)

class FeatureExtractionError(BiosignalException):
    """Exception raised for feature extraction failures."""
    def __init__(self, message: str, details: Optional[str] = None, feature_name: Optional[str] = None):
        self.message = message
        self.details = details
        self.feature_name = feature_name
        super().__init__(self.message)

class ErrorHandler(QObject):
    """Centralized error handling system."""
    
    error_occurred = pyqtSignal(ErrorInfo)  # Emitted when an error occurs
    progress_updated = pyqtSignal(str, int)  # Emitted for progress updates (operation, percentage)
    status_changed = pyqtSignal(str)  # Emitted when status message changes
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.current_operation: Optional[str] = None
        self.error_states: Dict[str, List[ErrorInfo]] = {}
        
    def _setup_logging(self):
        """Configure enhanced logging with ML-specific context."""
        # Create formatters for different log types
        standard_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(operation)s] [%(category)s] - %(message)s\n'
            'Details: %(details)s\n'
            'Context: %(context)s'
        )
        
        # File handlers for different log types
        main_file_handler = logging.FileHandler('app.log')
        main_file_handler.setFormatter(standard_formatter)
        
        ml_file_handler = logging.FileHandler('ml_operations.log')
        ml_file_handler.setFormatter(detailed_formatter)
        
        error_file_handler = logging.FileHandler('errors.log')
        error_file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(standard_formatter)
        
        # Add all handlers
        self.logger.addHandler(main_file_handler)
        self.logger.addHandler(ml_file_handler)
        self.logger.addHandler(error_file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.setLevel(logging.INFO)
        
    def _log_with_context(self, level: int, msg: str, category: ErrorCategory,
                         operation: Optional[str] = None, details: Optional[str] = None,
                         context: Optional[Dict] = None):
        """Log message with additional context."""
        extra = {
            'operation': operation or self.current_operation or 'unknown',
            'category': category.value,
            'details': details or 'No additional details',
            'context': str(context) if context else 'No context provided'
        }
        
        self.logger.log(level, msg, extra=extra)
        
    def add_validation_rule(self, key: str, rule: ValidationRule):
        """Add a validation rule."""
        if key not in self.validation_rules:
            self.validation_rules[key] = []
        self.validation_rules[key].append(rule)
        
    def validate(self, key: str, data: Any) -> Tuple[bool, Optional[str]]:
        """Validate data against registered rules."""
        if key not in self.validation_rules:
            return True, None
            
        for rule in self.validation_rules[key]:
            try:
                if not rule.condition(data):
                    return False, rule.message
            except Exception as e:
                return False, f"Validation error: {str(e)}"
                
        return True, None
        
    def handle_error(self, error: Exception, severity: ErrorSeverity,
                    category: ErrorCategory, suggestions: List[str] = None) -> ErrorInfo:
        """Handle an error and return error information."""
        error_info = ErrorInfo(
            message=str(error),
            severity=severity,
            category=category,
            details=getattr(error, 'details', None),
            traceback=traceback.format_exc(),
            suggestions=suggestions or []
        )
        
        # Extract context based on error type
        context = {}
        if isinstance(error, MLTrainingError) and error.model_state:
            context['model_state'] = error.model_state
        elif isinstance(error, MLEvaluationError) and error.metrics:
            context['metrics'] = error.metrics
        elif isinstance(error, DataLoadingError):
            context['file_path'] = error.file_path
        elif isinstance(error, FeatureExtractionError):
            context['feature_name'] = error.feature_name
            
        # Log the error with context
        log_message = self.format_ml_error(error)
        
        if severity == ErrorSeverity.CRITICAL:
            self._log_with_context(logging.CRITICAL, log_message, category,
                                details=error_info.details, context=context)
        elif severity == ErrorSeverity.ERROR:
            self._log_with_context(logging.ERROR, log_message, category,
                                details=error_info.details, context=context)
        elif severity == ErrorSeverity.WARNING:
            self._log_with_context(logging.WARNING, log_message, category,
                                details=error_info.details, context=context)
        else:
            self._log_with_context(logging.INFO, log_message, category,
                                details=error_info.details, context=context)
            
        # Emit error signal
        self.error_occurred.emit(error_info)
        
        # Store error state
        if error_info.category.value not in self.error_states:
            self.error_states[error_info.category.value] = []
        self.error_states[error_info.category.value].append(error_info)
        
        return error_info

    def get_error_state(self, category: Optional[ErrorCategory] = None) -> List[ErrorInfo]:
        """Get current error state for a category or all categories."""
        if category:
            return self.error_states.get(category.value, [])
        return [error for errors in self.error_states.values() for error in errors]

    def clear_error_state(self, category: Optional[ErrorCategory] = None):
        """Clear error state for a category or all categories."""
        if category:
            self.error_states.pop(category.value, None)
        else:
            self.error_states.clear()

    def update_progress(self, operation: str, percentage: int):
        """Update progress for long-running operations."""
        self.current_operation = operation
        self.progress_updated.emit(operation, percentage)
        self.logger.info(f"Progress update - {operation}: {percentage}%")

    def set_status(self, message: str):
        """Update status message."""
        self.status_changed.emit(message)
        self.logger.info(f"Status update: {message}")

    def show_error_dialog(self, error_info: ErrorInfo):
        """Show error dialog to user."""
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

    def format_ml_error(self, error: Exception) -> str:
        """Format ML-specific error messages with additional context."""
        if isinstance(error, MLTrainingError):
            msg = f"Training Error: {error.message}"
            if error.model_state:
                msg += "\nModel State:"
                for key, value in error.model_state.items():
                    msg += f"\n- {key}: {value}"
            return msg
            
        elif isinstance(error, MLEvaluationError):
            msg = f"Evaluation Error: {error.message}"
            if error.metrics:
                msg += "\nMetrics at failure:"
                for metric, value in error.metrics.items():
                    msg += f"\n- {metric}: {value}"
            return msg
            
        elif isinstance(error, DataLoadingError):
            msg = f"Data Loading Error: {error.message}"
            if error.file_path:
                msg += f"\nFile: {error.file_path}"
            return msg
            
        elif isinstance(error, FeatureExtractionError):
            msg = f"Feature Extraction Error: {error.message}"
            if error.feature_name:
                msg += f"\nFeature: {error.feature_name}"
            return msg
            
        return str(error)

    def get_suggestions_for_error(self, error: Exception) -> List[str]:
        """Get context-specific suggestions for different error types."""
        if isinstance(error, MLTrainingError):
            return [
                "Check if the training data is properly preprocessed",
                "Verify model hyperparameters",
                "Ensure sufficient training data is available",
                "Check for data imbalance issues"
            ]
        elif isinstance(error, DataLoadingError):
            return [
                "Verify file format and encoding",
                "Check file permissions",
                "Ensure data follows expected schema",
                "Validate data integrity"
            ]
        elif isinstance(error, FeatureExtractionError):
            return [
                "Check input signal quality",
                "Verify feature parameters",
                "Ensure sufficient data length",
                "Check for missing or invalid values"
            ]
        return []

class Validator:
    """Utility class for common validation rules."""
    
    @staticmethod
    def create_range_rule(min_val: float, max_val: float,
                         message: Optional[str] = None) -> ValidationRule:
        """Create a numeric range validation rule."""
        def check_range(value):
            try:
                num_value = float(value)
                return min_val <= num_value <= max_val
            except (TypeError, ValueError):
                return False
                
        return ValidationRule(
            condition=check_range,
            message=message or f"Value must be between {min_val} and {max_val}"
        )
        
    @staticmethod
    def create_type_rule(expected_type: type,
                        message: Optional[str] = None) -> ValidationRule:
        """Create a type validation rule."""
        return ValidationRule(
            condition=lambda x: isinstance(x, expected_type),
            message=message or f"Value must be of type {expected_type.__name__}"
        )
        
    @staticmethod
    def create_required_rule(message: Optional[str] = None) -> ValidationRule:
        """Create a required field validation rule."""
        return ValidationRule(
            condition=lambda x: x is not None and str(x).strip() != "",
            message=message or "This field is required"
        )
        
    @staticmethod
    def create_regex_rule(pattern: str,
                         message: Optional[str] = None) -> ValidationRule:
        """Create a regex validation rule."""
        import re
        return ValidationRule(
            condition=lambda x: bool(re.match(pattern, str(x))),
            message=message or f"Value must match pattern: {pattern}"
        )
        
    @staticmethod
    def create_custom_rule(condition: Callable,
                          message: str,
                          severity: ErrorSeverity = ErrorSeverity.ERROR) -> ValidationRule:
        """Create a custom validation rule."""
        return ValidationRule(
            condition=condition,
            message=message,
            severity=severity
        )

# Example usage:
"""
# Create error handler
error_handler = ErrorHandler()

# Add validation rules
error_handler.add_validation_rule(
    'frequency',
    Validator.create_range_rule(0, 1000, "Frequency must be between 0 and 1000 Hz")
)

error_handler.add_validation_rule(
    'window_size',
    Validator.create_range_rule(10, 10000, "Window size must be between 10 and 10000 samples")
)

# Validate data
is_valid, message = error_handler.validate('frequency', 1500)
if not is_valid:
    error_info = error_handler.handle_error(
        ValidationError(message),
        ErrorSeverity.ERROR,
        ErrorCategory.VALIDATION,
        suggestions=["Try using a value between 0 and 1000 Hz"]
    )
    error_handler.show_error_dialog(error_info)

# Handle processing error
try:
    # Some processing code
    raise ProcessingError("Failed to process signal", "Invalid input format")
except ProcessingError as e:
    error_info = error_handler.handle_error(
        e,
        ErrorSeverity.ERROR,
        ErrorCategory.PROCESSING,
        suggestions=[
            "Check input signal format",
            "Verify signal parameters"
        ]
    )
    error_handler.show_error_dialog(error_info)
"""