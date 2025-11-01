from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QPushButton, QWidget, QComboBox,
    QStackedWidget, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit
)
from PyQt6.QtCore import pyqtSignal, Qt
from ui.panels.base_panel import BaseControlPanel, NumericParameter, SliderParameter, BoolParameter, EnumParameter

class ModelSelectionPanel(BaseControlPanel):
    model_selected = pyqtSignal(str) # Emits selected model type
    parameters_changed = pyqtSignal(str, dict) # Emits model type and its parameters
    load_pretrained_requested = pyqtSignal(str) # Emits path to pre-trained model

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Model Type Selection
        model_selection_group = QGroupBox("Select Model Type")
        model_selection_layout = QVBoxLayout(model_selection_group)

        self.model_type_combo = QComboBox(self)
        self.model_type_combo.addItems(["SVM", "RandomForest", "CNN", "LSTM", "Ensemble"])
        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        model_selection_layout.addWidget(self.model_type_combo)
        main_layout.addWidget(model_selection_group)

        # 2. Dynamic Parameter Controls
        self.parameter_stack = QStackedWidget(self)
        model_selection_layout.addWidget(self.parameter_stack)

        self._create_svm_params()
        self._create_random_forest_params()
        self._create_cnn_params()
        self._create_lstm_params()
        self._create_ensemble_params()

        # 3. Pre-trained Model Loading
        pretrained_group = QGroupBox("Pre-trained Model")
        pretrained_layout = QHBoxLayout(pretrained_group)

        self.pretrained_path_input = QLineEdit(self)
        self.pretrained_path_input.setPlaceholderText("No pre-trained model selected")
        self.pretrained_path_input.setReadOnly(True)
        pretrained_layout.addWidget(self.pretrained_path_input)

        self.browse_pretrained_button = QPushButton("Browse", self)
        self.browse_pretrained_button.clicked.connect(self._browse_pretrained_model)
        pretrained_layout.addWidget(self.browse_pretrained_button)

        self.load_pretrained_button = QPushButton("Load Pre-trained", self)
        self.load_pretrained_button.clicked.connect(self._load_pretrained_model)
        pretrained_layout.addWidget(self.load_pretrained_button)
        main_layout.addWidget(pretrained_group)

        # 4. Model Architecture Visualization (Placeholder)
        arch_viz_group = QGroupBox("Model Architecture Visualization")
        arch_viz_layout = QVBoxLayout(arch_viz_group)
        self.arch_viz_label = QLabel("Architecture visualization will appear here for deep learning models.")
        arch_viz_layout.addWidget(self.arch_viz_label)
        main_layout.addWidget(arch_viz_group)

        self.setLayout(main_layout)
        self._on_model_type_changed(0) # Initialize with the first model

    def _create_svm_params(self):
        panel = QWidget()
        layout = QFormLayout(panel)
        self.svm_kernel = EnumParameter("kernel", ["linear", "poly", "rbf", "sigmoid"])
        self.svm_c = NumericParameter("C", 0.1, 10.0, 0.1, 1)
        layout.addRow("Kernel:", self.svm_kernel)
        layout.addRow("C:", self.svm_c)
        self.parameter_stack.addWidget(panel)
        self.svm_kernel.value_changed.connect(self._emit_parameters)
        self.svm_c.value_changed.connect(self._emit_parameters)

    def _create_random_forest_params(self):
        panel = QWidget()
        layout = QFormLayout(panel)
        self.rf_estimators = NumericParameter("n_estimators", 10, 500, 10, 0)
        self.rf_max_depth = NumericParameter("max_depth", 1, 50, 1, 0)
        layout.addRow("N Estimators:", self.rf_estimators)
        layout.addRow("Max Depth:", self.rf_max_depth)
        self.parameter_stack.addWidget(panel)
        self.rf_estimators.value_changed.connect(self._emit_parameters)
        self.rf_max_depth.value_changed.connect(self._emit_parameters)

    def _create_cnn_params(self):
        panel = QWidget()
        layout = QFormLayout(panel)
        self.cnn_layers = NumericParameter("num_layers", 1, 10, 1, 0)
        self.cnn_filters = NumericParameter("num_filters", 16, 128, 16, 0)
        layout.addRow("Num Layers:", self.cnn_layers)
        layout.addRow("Num Filters:", self.cnn_filters)
        self.parameter_stack.addWidget(panel)
        self.cnn_layers.value_changed.connect(self._emit_parameters)
        self.cnn_filters.value_changed.connect(self._emit_parameters)

    def _create_lstm_params(self):
        panel = QWidget()
        layout = QFormLayout(panel)
        self.lstm_units = NumericParameter("lstm_units", 32, 256, 32, 0)
        self.lstm_dropout = SliderParameter("dropout", 0.0, 0.5, 0.05, 2)
        layout.addRow("LSTM Units:", self.lstm_units)
        layout.addRow("Dropout:", self.lstm_dropout)
        self.parameter_stack.addWidget(panel)
        self.lstm_units.value_changed.connect(self._emit_parameters)
        self.lstm_dropout.value_changed.connect(self._emit_parameters)

    def _create_ensemble_params(self):
        panel = QWidget()
        layout = QFormLayout(panel)
        self.ensemble_method = EnumParameter("method", ["Voting", "Stacking"])
        self.ensemble_n_models = NumericParameter("n_models", 2, 10, 1, 0)
        layout.addRow("Method:", self.ensemble_method)
        layout.addRow("Num Models:", self.ensemble_n_models)
        self.parameter_stack.addWidget(panel)
        self.ensemble_method.value_changed.connect(self._emit_parameters)
        self.ensemble_n_models.value_changed.connect(self._emit_parameters)

    def _on_model_type_changed(self, index: int):
        model_type = self.model_type_combo.currentText()
        self.parameter_stack.setCurrentIndex(index)
        self.model_selected.emit(model_type)
        self._emit_parameters() # Emit parameters for the newly selected model

    def _emit_parameters(self):
        model_type = self.model_type_combo.currentText()
        current_widget = self.parameter_stack.currentWidget()
        params = {}
        if model_type == "SVM":
            params = {"kernel": self.svm_kernel.get_value(), "C": self.svm_c.get_value()}
        elif model_type == "RandomForest":
            params = {"n_estimators": self.rf_estimators.get_value(), "max_depth": self.rf_max_depth.get_value()}
        elif model_type == "CNN":
            params = {"num_layers": self.cnn_layers.get_value(), "num_filters": self.cnn_filters.get_value()}
        elif model_type == "LSTM":
            params = {"lstm_units": self.lstm_units.get_value(), "dropout": self.lstm_dropout.get_value()}
        elif model_type == "Ensemble":
            params = {"method": self.ensemble_method.get_value(), "n_models": self.ensemble_n_models.get_value()}
        
        self.parameters_changed.emit(model_type, params)
        print(f"Model: {model_type}, Parameters: {params}")

    def _browse_pretrained_model(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Select Pre-trained Model", "", "Model Files (*.pkl *.h5 *.pth);;All Files (*)")
        if file_path:
            self.pretrained_path_input.setText(file_path)

    def _load_pretrained_model(self):
        file_path = self.pretrained_path_input.text()
        if file_path:
            print(f"Request to load pre-trained model from: {file_path}")
            self.load_pretrained_requested.emit(file_path)
        else:
            print("No pre-trained model file selected.")

    def on_model_loaded_success(self, model_name: str):
        print(f"Pre-trained model '{model_name}' loaded successfully.")
        # Update UI to reflect loaded model