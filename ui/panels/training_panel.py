from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QPushButton, QWidget, QComboBox,
    QGroupBox, QFormLayout, QProgressBar, QTextEdit, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
from ui.panels.base_panel import BaseControlPanel, NumericParameter, SliderParameter, BoolParameter, EnumParameter

class TrainingPanel(BaseControlPanel):
    training_started = pyqtSignal(dict) # Emits training configuration
    training_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Training Configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QFormLayout(config_group)

        self.epochs_param = NumericParameter("epochs", 1, 1000, 1, 0)
        self.epochs_param.set_value(10)
        config_layout.addRow("Epochs:", self.epochs_param)

        self.batch_size_param = NumericParameter("batch_size", 1, 256, 1, 0)
        self.batch_size_param.set_value(32)
        config_layout.addRow("Batch Size:", self.batch_size_param)

        self.learning_rate_param = SliderParameter("learning_rate", 0.0001, 0.1, 0.0001, 4)
        self.learning_rate_param.set_value(0.001)
        config_layout.addRow("Learning Rate:", self.learning_rate_param)

        self.optimizer_param = EnumParameter("optimizer", ["Adam", "SGD", "RMSprop"])
        config_layout.addRow("Optimizer:", self.optimizer_param)

        self.early_stopping_checkbox = BoolParameter("early_stopping", "Enable Early Stopping")
        config_layout.addRow("Early Stopping:", self.early_stopping_checkbox)

        self.l2_regularization_param = SliderParameter("l2_regularization", 0.0, 0.1, 0.001, 3)
        self.l2_regularization_param.set_value(0.001)
        config_layout.addRow("L2 Regularization:", self.l2_regularization_param)
        
        main_layout.addWidget(config_group)

        # Training Controls
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training", self)
        self.start_button.clicked.connect(self._start_training)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Training", self)
        self.stop_button.clicked.connect(self._stop_training)
        self.stop_button.setEnabled(False) # Initially disabled
        control_layout.addWidget(self.stop_button)
        main_layout.addLayout(control_layout)

        # 2. Real-time Training Progress Visualization
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.loss_label = QLabel("Loss: N/A")
        progress_layout.addWidget(self.loss_label)
        self.accuracy_label = QLabel("Accuracy: N/A")
        progress_layout.addWidget(self.accuracy_label)
        
        # Placeholder for plot (e.g., PyQtGraph widget would go here)
        self.plot_placeholder = QLabel("Loss/Accuracy Curves Plot (Placeholder)")
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.plot_placeholder)

        main_layout.addWidget(progress_group)

        # 3. Training Logs and Metrics Display
        logs_group = QGroupBox("Training Logs")
        logs_layout = QVBoxLayout(logs_group)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        logs_layout.addWidget(self.log_output)
        main_layout.addWidget(logs_group)
        
        self.setLayout(main_layout)

    def _start_training(self):
        config = {
            "epochs": self.epochs_param.get_value(),
            "batch_size": self.batch_size_param.get_value(),
            "learning_rate": self.learning_rate_param.get_value(),
            "optimizer": self.optimizer_param.get_value(),
            "early_stopping": self.early_stopping_checkbox.get_value(),
            "l2_regularization": self.l2_regularization_param.get_value(),
        }
        print(f"Starting training with config: {config}")
        self.training_started.emit(config)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.log_output.append("Training started...")

    def _stop_training(self):
        print("Stopping training.")
        self.training_stopped.emit()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.append("Training stopped by user.")

    def on_training_progress(self, epoch: int, total_epochs: int, loss: float, accuracy: float):
        progress_percent = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress_percent)
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.accuracy_label.setText(f"Accuracy: {accuracy:.4f}")
        self.log_output.append(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        # Update plot here

    def on_training_finished(self, final_metrics: dict):
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.append("Training finished!")
        self.log_output.append(f"Final Metrics: {final_metrics}")
        print(f"Training finished with metrics: {final_metrics}")

    def on_training_error(self, error_message: str):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.append(f"Training Error: {error_message}")
        print(f"Training Error: {error_message}")