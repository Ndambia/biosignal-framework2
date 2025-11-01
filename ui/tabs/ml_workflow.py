from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget
from ui.panels.data_loader_panel import DataLoaderPanel
from ui.panels.model_selection_panel import ModelSelectionPanel
from ui.panels.training_panel import TrainingPanel
from ui.panels.evaluation_panel import EvaluationPanel
from models.model_manager import ModelManager
from ui.workers.training_worker import TrainingWorker
from ui.workers.evaluation_worker import EvaluationWorker
from PyQt6.QtCore import QThreadPool, pyqtSignal
from typing import Any
from ui.error_handling import ErrorHandler, ErrorSeverity, ErrorCategory
from ui.feedback_manager import FeedbackManager
import numpy as np
class MLWorkflowTab(QWidget):
    """
    Main tab for the Machine Learning Workflow, integrating data loading,
    model selection, training, and evaluation panels.
    """
    # Signals for communicating with other parts of the application
    # For example, to update a global data manager or log errors
    
    def __init__(self, data_manager: Any, error_handler: ErrorHandler, feedback_manager: FeedbackManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.error_handler = error_handler
        self.feedback_manager = feedback_manager
        self.model_manager = ModelManager()
        self.thread_pool = QThreadPool()
        self.setup_ui()
        self._connect_signals()
        self.current_data = None # Actual loaded data
        self.current_labels = None # Actual loaded labels
        self.current_model_type = None
        self.current_model_params = {}

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Navigation buttons for the different ML workflow stages
        nav_layout = QHBoxLayout()
        self.data_loader_btn = QPushButton("1. Data Loading")
        self.model_selection_btn = QPushButton("2. Model Selection")
        self.training_btn = QPushButton("3. Training")
        self.evaluation_btn = QPushButton("4. Evaluation")

        nav_layout.addWidget(self.data_loader_btn)
        nav_layout.addWidget(self.model_selection_btn)
        nav_layout.addWidget(self.training_btn)
        nav_layout.addWidget(self.evaluation_btn)
        main_layout.addLayout(nav_layout)

        # Stacked widget to hold different panels
        self.stacked_widget = QStackedWidget(self)
        main_layout.addWidget(self.stacked_widget)

        # 1. Data Loader Panel
        self.data_loader_panel = DataLoaderPanel(self)
        self.stacked_widget.addWidget(self.data_loader_panel)

        # 2. Model Selection Panel
        self.model_selection_panel = ModelSelectionPanel(self)
        self.stacked_widget.addWidget(self.model_selection_panel)

        # 3. Training Panel
        self.training_panel = TrainingPanel(self)
        self.stacked_widget.addWidget(self.training_panel)

        # 4. Evaluation Panel
        self.evaluation_panel = EvaluationPanel(self)
        self.stacked_widget.addWidget(self.evaluation_panel)

        self.setLayout(main_layout)

    def _connect_signals(self):
        # Navigation
        self.data_loader_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.data_loader_panel))
        self.model_selection_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.model_selection_panel))
        self.training_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.training_panel))
        self.evaluation_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.evaluation_panel))

        # DataLoaderPanel signals
        self.data_loader_panel.data_loaded.connect(self._handle_data_loaded)
        self.data_loader_panel.data_split_requested.connect(self._handle_data_split_requested)
        self.data_loader_panel.data_augmentation_requested.connect(self._handle_data_augmentation_requested)
        self.data_loader_panel.labels_updated.connect(self._handle_labels_updated)
        self.data_manager.signals.data_loaded_from_file.connect(self._on_data_manager_data_loaded)

        # ModelSelectionPanel signals
        self.model_selection_panel.model_selected.connect(self._handle_model_selected)
        self.model_selection_panel.parameters_changed.connect(self._handle_model_parameters_changed)
        self.model_selection_panel.load_pretrained_requested.connect(self._handle_load_pretrained_model)

        # TrainingPanel signals
        self.training_panel.training_started.connect(self._handle_training_started)
        self.training_panel.training_stopped.connect(self._handle_training_stopped)

        # EvaluationPanel signals
        self.evaluation_panel.evaluation_requested.connect(self._handle_evaluation_requested)

    def _handle_data_loaded(self, file_path: str):
        self.feedback_manager.show_status_message(f"Data load requested for {file_path}. Triggering DataManager.")
        # In a real application, DataManager would handle the actual loading
        # For now, we'll simulate DataManager emitting data_loaded_from_file
        # self.data_manager.load_data_from_file(file_path) # This would be the actual call
        
        # Simulate DataManager emitting the signal after loading
        simulated_data = np.random.rand(1000, 10) # 1000 samples, 10 features
        simulated_labels = np.random.randint(0, 2, 1000) # 2 classes
        self.data_manager.signals.data_loaded_from_file.emit(file_path, simulated_data, simulated_labels, {"num_samples": 1000})

    def _on_data_manager_data_loaded(self, file_path: str, data: Any, labels: Any, metadata: dict):
        self.current_data = data
        self.current_labels = labels
        self.data_loader_panel.on_data_loaded_success(file_path, metadata.get("num_samples", len(data)))
        unique_labels = list(np.unique(labels)) if labels is not None else []
        self.data_loader_panel.on_labels_loaded([str(label) for label in unique_labels])
        self.feedback_manager.show_status_message(f"DataManager reported data loaded. Samples: {len(data)}, Labels: {unique_labels}")

    def _handle_data_split_requested(self, train_ratio: float, test_ratio: float, val_ratio: float):
        self.feedback_manager.show_status_message(f"Data split requested: Train={train_ratio}, Test={test_ratio}, Val={val_ratio}")
        # DataManager would handle the actual splitting
        if self.current_data is not None and self.current_labels is not None:
            total_samples = len(self.current_data)
            train_count = int(total_samples * train_ratio)
            test_count = int(total_samples * test_ratio)
            val_count = int(total_samples * val_ratio)
            # In a real scenario, DataManager would return the split data
            self.data_loader_panel.on_data_split_success(train_count, test_count, val_count)
            self.feedback_manager.show_status_message("Data split simulated successfully.")
        else:
            self.error_handler.handle_error(
                Exception("No data available in DataManager to split."),
                ErrorSeverity.ERROR,
                ErrorCategory.DATA_LOADING
            )

    def _handle_data_augmentation_requested(self, aug_params: dict):
        self.feedback_manager.show_status_message(f"Data augmentation requested with params: {aug_params}")
        # DataManager would handle the actual augmentation
        if self.current_data is not None:
            # Simulate augmented data size
            num_augmented_samples = len(self.current_data) * 1.2 # Example
            self.data_loader_panel.on_data_augmented_success(int(num_augmented_samples))
            self.feedback_manager.show_status_message("Data augmentation simulated successfully.")
        else:
            self.error_handler.handle_error(
                Exception("No data available in DataManager to augment."),
                ErrorSeverity.ERROR,
                ErrorCategory.DATA_LOADING
            )

    def _handle_labels_updated(self, labels: list):
        print(f"MLWorkflowTab: Labels updated: {labels}")
        self.current_labels = labels

    def _handle_model_selected(self, model_type: str):
        print(f"MLWorkflowTab: Model type selected: {model_type}")
        self.current_model_type = model_type

    def _handle_model_parameters_changed(self, model_type: str, params: dict):
        print(f"MLWorkflowTab: Model '{model_type}' parameters changed: {params}")
        self.current_model_params = params

    def _handle_load_pretrained_model(self, file_path: str):
        print(f"MLWorkflowTab: Load pre-trained model requested: {file_path}")
        # In a real app, use model_manager to load
        try:
            # Simulate loading a model and getting its ID
            # For now, just pass the file_path as a dummy model_id
            dummy_model_id = f"pretrained_{file_path.split('/')[-1].split('.')[0]}"
            self.model_selection_panel.on_model_loaded_success(dummy_model_id)
            # Optionally, load the model into model_manager if it's a valid format
            # self.model_manager.load_model(dummy_model_id)
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            # self.model_selection_panel.on_model_load_error(str(e))

    def _handle_training_started(self, config: dict):
        print(f"MLWorkflowTab: Training started with config: {config}")
        try:
            if (self.current_data is None or self.current_labels is None or
                not isinstance(self.current_data, np.ndarray) or not isinstance(self.current_labels, (np.ndarray, list)) or
                self.current_data.size == 0 or len(self.current_labels) == 0):
                self.training_panel.on_training_error("No data or labels loaded for training.")
                return
        except Exception as e:
            self.training_panel.on_training_error(f"Error validating training data: {str(e)}")
            return

        # Create and start training worker
        worker = TrainingWorker(
            model_type=self.current_model_type,
            model_params=self.current_model_params,
            training_config=config,
            data=self.current_data, # Pass actual data here
            labels=self.current_labels, # Pass actual labels here
            model_manager=self.model_manager
        )
        worker.signals.progress.connect(self.training_panel.on_training_progress)
        worker.signals.finished.connect(self._handle_training_finished)
        worker.signals.error.connect(self.training_panel.on_training_error)
        worker.signals.stopped.connect(self.training_panel.on_training_finished) # Treat stopped as finished for UI update
        self.thread_pool.start(worker)
        self.current_training_worker = worker # Keep a reference to stop it if needed

    def _handle_training_stopped(self):
        print("MLWorkflowTab: Training stop requested.")
        if hasattr(self, 'current_training_worker') and self.current_training_worker:
            self.current_training_worker.stop()
            self.current_training_worker = None

    def _handle_training_finished(self, final_metrics: dict, model_id: str):
        print(f"MLWorkflowTab: Training finished. Model ID: {model_id}, Metrics: {final_metrics}")
        self.training_panel.on_training_finished(final_metrics)
        self.current_training_worker = None
        # Update evaluation panel with new model
        self.evaluation_panel.update_available_models(list(self.model_manager.list_all_models().keys()))

    def _handle_evaluation_requested(self, model_id: str):
        print(f"MLWorkflowTab: Evaluation requested for model ID: {model_id}")
        try:
            if (self.current_data is None or self.current_labels is None or
                not isinstance(self.current_data, np.ndarray) or not isinstance(self.current_labels, (np.ndarray, list)) or
                self.current_data.size == 0 or len(self.current_labels) == 0):
                self.evaluation_panel.on_evaluation_error("No data or labels loaded for evaluation.")
                return
        except Exception as e:
            self.evaluation_panel.on_evaluation_error(f"Error validating evaluation data: {str(e)}")
            return

        # Create and start evaluation worker
        worker = EvaluationWorker(
            model_id=model_id,
            data=self.current_data, # Pass actual data here
            labels=self.current_labels, # Pass actual labels here
            model_manager=self.model_manager
        )
        worker.signals.finished.connect(self.evaluation_panel.on_evaluation_results)
        worker.signals.error.connect(self.evaluation_panel.on_evaluation_error)
        self.thread_pool.start(worker)