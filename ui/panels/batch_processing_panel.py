"""
Batch Processing Panel for ML Workflow Management

This module provides a comprehensive interface for managing batch processing of ML workflows,
including configuration, monitoring, and results analysis. It integrates configuration,
monitoring, and result comparison panels into a unified interface.

Key Features:
- Batch data loading and preprocessing controls
- Training configuration for multiple models
- Batch evaluation interface
- Progress tracking integration
- Real-time monitoring of batch tasks
- Result comparison and visualization

Classes:
    BatchProcessingPanel: Main panel for batch processing ML workflows
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QProgressBar, QLabel, QSplitter,
    QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

from .base_panel import BaseControlPanel
from .batch_configuration_panel import BatchConfigurationPanel
from .batch_monitor_panel import BatchMonitorPanel
from .result_comparison_panel import ResultComparisonPanel
from ..error_handling import ErrorHandler, ErrorCategory, ErrorSeverity, ErrorInfo
from ..workers.base_worker import BaseWorker
from ..visualization.batch_comparison_view import BatchComparisonView

class BatchProcessingPanel(BaseControlPanel):
    """Main panel for batch processing ML workflows.
    
    This panel serves as the primary interface for managing batch processing operations,
    integrating configuration, monitoring, and result analysis capabilities.
    
    Attributes:
        batch_started (pyqtSignal): Emitted when batch processing begins
        batch_completed (pyqtSignal): Emitted when batch processing finishes
        batch_error (pyqtSignal[ErrorInfo]): Emitted when an error occurs
        progress_updated (pyqtSignal[str, int]): Emitted with task progress updates
        
    Properties:
        is_running (bool): Whether batch processing is currently active
        current_batch (Optional[Dict[str, Any]]): Current batch configuration
        workers (List[BaseWorker]): Active worker threads
    """
    
    # Signals for batch operations
    batch_started = pyqtSignal()
    batch_completed = pyqtSignal()
    batch_error = pyqtSignal(ErrorInfo)
    progress_updated = pyqtSignal(str, int)  # task_name, percentage

    def __init__(self, error_handler: ErrorHandler, data_manager, parent=None):
        super().__init__(parent)
        self.error_handler = error_handler
        self.data_manager = data_manager
        self.current_batch: Optional[Dict[str, Any]] = None
        self.workers: List[BaseWorker] = []
        
        # Visualization update settings
        self.auto_update_enabled = True
        self.update_threshold = 5  # Minimum percentage change to trigger update
        self.last_update_progress = 0
        self.update_buffer = []  # Buffer for data points between updates
        self.buffer_size = 100  # Maximum buffer size before forcing update
        
        # Connect to data manager signals
        self.data_manager.signals.batch_started.connect(self._on_batch_started)
        self.data_manager.signals.batch_progress.connect(self._on_batch_progress)
        self.data_manager.signals.batch_completed.connect(self._on_batch_completed)
        self.data_manager.signals.batch_error.connect(self._on_batch_error)
        
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the batch processing interface."""
        main_layout = QVBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Configuration and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Initialize configuration panel
        self.config_panel = BatchConfigurationPanel(self.error_handler)
        self.config_panel.config_changed.connect(self._on_config_changed)
        left_layout.addWidget(self.config_panel)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Batch")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        
        self.start_button.clicked.connect(self._start_batch)
        self.pause_button.clicked.connect(self._pause_batch)
        self.stop_button.clicked.connect(self._stop_batch)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)
        
        # Overall progress
        progress_group = QGroupBox("Overall Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        left_layout.addWidget(progress_group)
        
        # Right side - Monitoring and results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Monitor section
        # Initialize monitor panel
        self.monitor_panel = BatchMonitorPanel(self.error_handler)
        self.monitor_panel.task_completed.connect(self._on_task_completed)
        self.monitor_panel.error_occurred.connect(self._on_task_error)
        right_layout.addWidget(self.monitor_panel)
        
        # Initialize visualization panels
        visualization_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Results panel
        self.results_panel = ResultComparisonPanel(self.error_handler)
        self.results_panel.export_requested.connect(self._on_export_requested)
        visualization_splitter.addWidget(self.results_panel)
        
        # Batch comparison view
        self.comparison_view = BatchComparisonView(update_rate=5)  # 5 Hz update rate
        self.comparison_view.batch_selection_changed.connect(self._on_batch_selected)
        self.comparison_view.update_rate_changed.connect(self._on_update_rate_changed)
        visualization_splitter.addWidget(self.comparison_view)
        
        # Add visualization controls
        viz_controls = QGroupBox("Visualization Controls")
        viz_layout = QHBoxLayout(viz_controls)
        
        # Auto-update toggle
        self.auto_update_check = QCheckBox("Auto Update")
        self.auto_update_check.setChecked(self.auto_update_enabled)
        self.auto_update_check.toggled.connect(self._on_auto_update_toggled)
        viz_layout.addWidget(self.auto_update_check)
        
        # Update threshold control
        viz_layout.addWidget(QLabel("Update Threshold (%):"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 20)
        self.threshold_spin.setValue(self.update_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        viz_layout.addWidget(self.threshold_spin)
        
        # Buffer size control
        viz_layout.addWidget(QLabel("Buffer Size:"))
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(10, 1000)
        self.buffer_spin.setValue(self.buffer_size)
        self.buffer_spin.valueChanged.connect(self._on_buffer_size_changed)
        viz_layout.addWidget(self.buffer_spin)
        
        # Force update button
        self.update_button = QPushButton("Force Update")
        self.update_button.clicked.connect(self._force_visualization_update)
        viz_layout.addWidget(self.update_button)
        
        right_layout.addWidget(viz_controls)
        right_layout.addWidget(visualization_splitter)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set initial states
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _start_batch(self) -> None:
        """Start batch processing.
        
        Validates configuration, initializes workers, and begins processing.
        Handles any startup errors through the error handling system.
        
        Raises:
            ConfigurationError: If batch configuration is invalid
            ProcessingError: If worker initialization fails
        """
        try:
            # Validate configuration before starting
            if not self._validate_batch_config():
                return
            
            config = self.config_panel.get_configuration()
            batch_id = f"batch_{int(time.time())}"
            
            # Start batch through data manager
            self.data_manager.start_batch_processing(batch_id, config)
            
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
            # Initialize batch processing
            self._initialize_batch()
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.PROCESSING,
                ["Check batch configuration", "Verify data paths"]
            )

    def _pause_batch(self) -> None:
        """Pause current batch processing.
        
        Suspends all active workers while preserving state.
        Workers can be resumed by calling _start_batch again.
        """
        for worker in self.workers:
            worker.pause()
        self.status_label.setText("Paused")

    def _stop_batch(self) -> None:
        """Stop current batch processing.
        
        Terminates all active workers and resets the UI state.
        Any incomplete tasks will be abandoned.
        """
        for worker in self.workers:
            worker.stop()
        self._reset_ui()
        self.status_label.setText("Stopped")

    def _validate_batch_config(self) -> Tuple[bool, Optional[str]]:
        """Validate batch configuration before starting.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
            
        Validates:
            - Dataset paths exist and are readable
            - Model configuration is complete
            - Processing parameters are within valid ranges
            - Resource requirements can be met
        """
        try:
            config = self.config_panel.get_configuration()
            
            # Validate dataset configuration
            if not config["dataset_paths"]:
                return False, "No dataset files selected"
                
            # Validate model configuration
            if not self._validate_model_config(config):
                return False, "Invalid model configuration"
                
            # Validate processing parameters
            if not self._validate_processing_params(config):
                return False, "Invalid processing parameters"
                
            return True, None
            
        except Exception as e:
            return False, str(e)

    def _initialize_batch(self) -> None:
        """Initialize batch processing workers and start execution.
        
        Sets up worker threads for:
        - Data loading and preprocessing
        - Model training
        - Evaluation and metrics collection
        
        Each worker is configured based on the current batch configuration
        and connected to appropriate progress and error handling signals.
        """
        try:
            config = self.config_panel.get_configuration()
            
            # Initialize workers based on configuration
            self._setup_data_worker(config)
            self._setup_training_worker(config)
            self._setup_evaluation_worker(config)
            
            # Start processing
            for worker in self.workers:
                worker.start()
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.PROCESSING,
                ["Check worker initialization", "Verify resource availability"]
            )

    def _reset_ui(self) -> None:
        """Reset UI to initial state.
        
        Resets all UI elements to their default state:
        - Enables/disables appropriate buttons
        - Clears progress indicators
        - Resets status messages
        - Maintains any completed results
        """
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")

    def _update_progress(self, task_name: str, percentage: int, metrics: Dict[str, Any] = None, data: Optional[np.ndarray] = None) -> None:
        """Update progress bar and status label.
        
        Args:
            task_name: Name or description of the current task
            percentage: Progress percentage (0-100)
            metrics: Optional performance metrics
            
        Updates both the overall progress bar and individual task
        progress in the monitor panel.
        """
        self.progress_bar.setValue(percentage)
        self.status_label.setText(f"Processing: {task_name}")
        self.progress_updated.emit(task_name, percentage)
        
        if self.current_batch:
            self.data_manager.update_batch_progress(
                self.current_batch['id'],
                percentage,
                metrics or {}
            )
            
            # Update visualization if data is provided
            if data is not None:
                self.comparison_view.update_batch_data(
                    self.current_batch['id'],
                    data,
                    time.time()
                )

    def _on_config_changed(self, config: Dict[str, Any]) -> None:
        """Handle configuration changes.
        
        Args:
            config: Updated configuration dictionary
            
        Updates internal state and validates new configuration.
        """
        self.current_batch = config
        is_valid, _ = self._validate_batch_config()
        self.start_button.setEnabled(is_valid)

    def _on_task_completed(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """Handle task completion.
        
        Args:
            task_id: Identifier of the completed task
            metrics: Performance metrics and results
            
        Updates results panel, visualization, and progress tracking.
        """
        self.results_panel.update_results(metrics)
        self.monitor_panel.remove_task(task_id)
        
        if self.current_batch:
            batch_id = self.current_batch['id']
            
            # Update visualization with final results
            if 'data' in metrics:
                self.comparison_view.set_data(
                    metrics['data'],
                    metrics.get('time'),
                    batch_id
                )
            
            self.data_manager.complete_batch_processing(batch_id, metrics)

    def _on_task_error(self, task_id: str, error_info: ErrorInfo) -> None:
        """Handle task errors.
        
        Args:
            task_id: Identifier of the failed task
            error_info: Detailed error information
            
        Updates UI and notifies error handling system.
        """
        self.batch_error.emit(error_info)
        self.monitor_panel.set_task_error(task_id, error_info)
        
        if self.current_batch:
            self.data_manager.signals.batch_error.emit(
                self.current_batch['id'],
                str(error_info.message)
            )

    def _on_export_requested(self, format: str, data: Dict[str, Any]) -> None:
        """Handle result export requests.
        
        Args:
            format: Desired export format (e.g., 'csv', 'json')
            data: Results data to export
            
        Initiates export process in requested format.
        """
        try:
            self._export_results(format, data)
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.FILE_IO,
                ["Check file permissions", "Verify export format"]
            )

    def cleanup(self) -> None:
        """Clean up resources before panel is destroyed.
        
        Performs cleanup tasks:
        - Stops all active workers
        - Releases system resources
        - Saves any necessary state
        - Disconnects signals
        - Cleans up visualization resources
        - Releases memory buffers
        """
        # Stop processing
        self._stop_batch()
        
        # Clean up workers
        for worker in self.workers:
            worker.cleanup()
        self.workers.clear()
        
        # Disconnect data manager signals
        self.data_manager.signals.batch_started.disconnect(self._on_batch_started)
        self.data_manager.signals.batch_progress.disconnect(self._on_batch_progress)
        self.data_manager.signals.batch_completed.disconnect(self._on_batch_completed)
        self.data_manager.signals.batch_error.disconnect(self._on_batch_error)
        
        # Clean up panels
        self.config_panel.cleanup()
        self.monitor_panel.cleanup()
        self.results_panel.cleanup()
        self.comparison_view.cleanup()
        
        # Clear buffers and state
        self.update_buffer.clear()
        self.current_batch = None
        
        # Call base class cleanup
        super().cleanup()
            
    def _on_batch_started(self, batch_id: str, config: Dict[str, Any]) -> None:
        """Handle batch start notification from data manager."""
        self.batch_started.emit()
        
    def _on_batch_progress(self, batch_id: str, progress: int, metrics: Dict[str, Any]) -> None:
        """Handle batch progress update from data manager."""
        if metrics.get('task_name'):
            self._update_progress(metrics['task_name'], progress)
            
    def _on_batch_completed(self, batch_id: str, results: Dict[str, Any]) -> None:
        """Handle batch completion notification from data manager."""
        self.batch_completed.emit()
        self.results_panel.update_results(results)
        self._reset_ui()
        
    def _on_batch_selected(self, batch_id: str):
        """Handle batch selection in comparison view."""
        if batch_id in self.data_manager.batch_results:
            results = self.data_manager.get_batch_results(batch_id)
            self.results_panel.update_results(results)
            
    def _on_update_rate_changed(self, rate: int):
        """Handle visualization update rate change."""
        if self.auto_update_enabled:
            self._adjust_update_parameters(rate)
            
    def _handle_visualization_update(self, data: np.ndarray) -> None:
        """Handle new data for visualization updates.
        
        Args:
            data: New data points to visualize
        """
        if not self.auto_update_enabled:
            # Buffer data for manual update
            self.update_buffer.extend(data)
            if len(self.update_buffer) >= self.buffer_size:
                self._force_visualization_update()
            return
            
        progress_delta = abs(self.progress_bar.value() - self.last_update_progress)
        
        # Check if update is needed
        should_update = (
            progress_delta >= self.update_threshold or
            len(self.update_buffer) >= self.buffer_size
        )
        
        # Buffer the data
        self.update_buffer.extend(data)
        
        if should_update:
            self._force_visualization_update()
            
    def _force_visualization_update(self) -> None:
        """Force visualization update with current buffer."""
        if not self.update_buffer:
            return
            
        if self.current_batch:
            # Update visualization with buffered data
            self.comparison_view.update_batch_data(
                self.current_batch['id'],
                np.array(self.update_buffer),
                time.time()
            )
            
            # Clear buffer and update progress tracking
            self.update_buffer = []
            self.last_update_progress = self.progress_bar.value()
            
    def _adjust_update_parameters(self, update_rate: int) -> None:
        """Adjust update parameters based on update rate.
        
        Args:
            update_rate: New update rate in Hz
        """
        # Adjust threshold based on update rate
        self.update_threshold = max(1, int(100 / (update_rate * 2)))
        self.threshold_spin.setValue(self.update_threshold)
        
        # Adjust buffer size based on update rate
        self.buffer_size = max(10, int(update_rate * 10))  # 10 seconds worth of data
        self.buffer_spin.setValue(self.buffer_size)
        
    def _on_auto_update_toggled(self, enabled: bool) -> None:
        """Handle auto-update toggle."""
        self.auto_update_enabled = enabled
        self.threshold_spin.setEnabled(enabled)
        if enabled and self.update_buffer:
            self._force_visualization_update()
            
    def _on_threshold_changed(self, value: int) -> None:
        """Handle update threshold change."""
        self.update_threshold = value
        
    def _on_buffer_size_changed(self, size: int) -> None:
        """Handle buffer size change."""
        self.buffer_size = size
        if len(self.update_buffer) >= size:
            self._force_visualization_update()
        
    def _on_batch_error(self, batch_id: str, error_message: str) -> None:
        """Handle batch error notification from data manager."""
        error_info = ErrorInfo(
            message=error_message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.PROCESSING
        )
        self.batch_error.emit(error_info)