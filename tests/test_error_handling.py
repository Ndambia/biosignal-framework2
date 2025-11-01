import unittest
import os
import logging
from unittest.mock import Mock, patch
from PyQt6.QtCore import QObject

from ui.error_handling import (
    ErrorHandler, ErrorSeverity, ErrorCategory, ErrorInfo,
    MLTrainingError, MLEvaluationError, DataLoadingError, FeatureExtractionError
)

class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.error_handler = ErrorHandler()
        # Ensure clean log files for each test
        self._clean_log_files()
        
    def tearDown(self):
        # Clean up log files after tests
        self._clean_log_files()
        
    def _clean_log_files(self):
        log_files = ['app.log', 'ml_operations.log', 'errors.log']
        for file in log_files:
            if os.path.exists(file):
                os.remove(file)

    def test_ml_training_error_handling(self):
        """Test ML training error handling and logging"""
        model_state = {'epoch': 5, 'loss': 2.5}
        error = MLTrainingError(
            message="Training failed",
            details="Gradient explosion detected",
            model_state=model_state
        )
        
        with self.assertLogs(level='ERROR') as log:
            error_info = self.error_handler.handle_error(
                error,
                ErrorSeverity.ERROR,
                ErrorCategory.ML_TRAINING
            )
            
        # Verify error info
        self.assertEqual(error_info.message, "Training failed")
        self.assertEqual(error_info.details, "Gradient explosion detected")
        self.assertEqual(error_info.category, ErrorCategory.ML_TRAINING)
        
        # Verify logging
        self.assertTrue(any("Training Error: Training failed" in msg for msg in log.output))
        self.assertTrue(any("model_state" in msg for msg in log.output))

    def test_error_state_management(self):
        """Test error state tracking and retrieval"""
        # Create test errors
        error1 = MLTrainingError("Error 1")
        error2 = DataLoadingError("Error 2")
        
        # Handle errors
        self.error_handler.handle_error(error1, ErrorSeverity.ERROR, ErrorCategory.ML_TRAINING)
        self.error_handler.handle_error(error2, ErrorSeverity.ERROR, ErrorCategory.DATA_LOADING)
        
        # Test getting errors by category
        ml_errors = self.error_handler.get_error_state(ErrorCategory.ML_TRAINING)
        self.assertEqual(len(ml_errors), 1)
        self.assertEqual(ml_errors[0].message, "Error 1")
        
        # Test getting all errors
        all_errors = self.error_handler.get_error_state()
        self.assertEqual(len(all_errors), 2)
        
        # Test clearing errors
        self.error_handler.clear_error_state(ErrorCategory.ML_TRAINING)
        ml_errors = self.error_handler.get_error_state(ErrorCategory.ML_TRAINING)
        self.assertEqual(len(ml_errors), 0)

    def test_progress_tracking(self):
        """Test progress update functionality"""
        progress_callback = Mock()
        self.error_handler.progress_updated.connect(progress_callback)
        
        self.error_handler.update_progress("Training model", 50)
        
        progress_callback.assert_called_once_with("Training model", 50)

    def test_status_messages(self):
        """Test status message functionality"""
        status_callback = Mock()
        self.error_handler.status_changed.connect(status_callback)
        
        self.error_handler.set_status("Processing features...")
        
        status_callback.assert_called_once_with("Processing features...")

    def test_ml_error_formatting(self):
        """Test ML-specific error message formatting"""
        # Test MLTrainingError formatting
        training_error = MLTrainingError(
            "Training failed",
            model_state={'epoch': 5, 'loss': 2.5}
        )
        formatted_msg = self.error_handler.format_ml_error(training_error)
        self.assertIn("Training Error:", formatted_msg)
        self.assertIn("epoch", formatted_msg)
        
        # Test MLEvaluationError formatting
        eval_error = MLEvaluationError(
            "Evaluation failed",
            metrics={'accuracy': 0.85}
        )
        formatted_msg = self.error_handler.format_ml_error(eval_error)
        self.assertIn("Evaluation Error:", formatted_msg)
        self.assertIn("accuracy", formatted_msg)

    def test_error_suggestions(self):
        """Test error-specific suggestions"""
        # Test ML training suggestions
        training_error = MLTrainingError("Training failed")
        suggestions = self.error_handler.get_suggestions_for_error(training_error)
        self.assertTrue(len(suggestions) > 0)
        self.assertTrue(any("training data" in s for s in suggestions))
        
        # Test data loading suggestions
        data_error = DataLoadingError("Failed to load data")
        suggestions = self.error_handler.get_suggestions_for_error(data_error)
        self.assertTrue(len(suggestions) > 0)
        self.assertTrue(any("file format" in s for s in suggestions))

    @patch('PyQt6.QtWidgets.QMessageBox')
    def test_error_dialog(self, mock_dialog):
        """Test error dialog display"""
        error_info = ErrorInfo(
            message="Test error",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.ML_TRAINING,
            details="Test details",
            suggestions=["Try this", "Try that"]
        )
        
        self.error_handler.show_error_dialog(error_info)
        
        # Verify dialog setup
        mock_dialog.return_value.setIcon.assert_called()
        mock_dialog.return_value.setWindowTitle.assert_called()
        mock_dialog.return_value.setText.assert_called_with("Test error")
        mock_dialog.return_value.setDetailedText.assert_called()

if __name__ == '__main__':
    unittest.main()