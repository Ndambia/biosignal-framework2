from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QPushButton, QWidget, QFileDialog,
    QHBoxLayout, QLineEdit, QGroupBox, QFormLayout, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import pyqtSignal, Qt
from ui.panels.base_panel import BaseControlPanel, NumericParameter, SliderParameter, BoolParameter, EnumParameter

class DataLoaderPanel(BaseControlPanel):
    data_loaded = pyqtSignal(str) # Emits path of loaded data
    data_split_requested = pyqtSignal(float, float, float) # train, test, val ratios
    data_augmentation_requested = pyqtSignal(dict) # augmentation parameters
    labels_updated = pyqtSignal(list) # list of labels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.loaded_file_path = ""

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # 1. Data Loading Interface
        data_loading_group = QGroupBox("Data Loading")
        data_loading_layout = QVBoxLayout(data_loading_group)

        file_selection_layout = QHBoxLayout()
        self.file_path_input = QLineEdit(self)
        self.file_path_input.setPlaceholderText("No file selected")
        self.file_path_input.setReadOnly(True)
        file_selection_layout.addWidget(self.file_path_input)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self._browse_file)
        file_selection_layout.addWidget(self.browse_button)
        data_loading_layout.addLayout(file_selection_layout)

        self.load_button = QPushButton("Load Data", self)
        self.load_button.clicked.connect(self._load_data)
        data_loading_layout.addWidget(self.load_button)
        main_layout.addWidget(data_loading_group)

        # 2. Data Splitting Controls
        data_splitting_group = QGroupBox("Data Splitting")
        data_splitting_layout = QFormLayout(data_splitting_group)

        self.train_ratio_param = SliderParameter("train_ratio", 0.0, 1.0, 0.01, 2)
        self.train_ratio_param.set_value(0.7)
        data_splitting_layout.addRow("Train Ratio:", self.train_ratio_param)

        self.test_ratio_param = SliderParameter("test_ratio", 0.0, 1.0, 0.01, 2)
        self.test_ratio_param.set_value(0.2)
        data_splitting_layout.addRow("Test Ratio:", self.test_ratio_param)

        self.val_ratio_param = SliderParameter("val_ratio", 0.0, 1.0, 0.01, 2)
        self.val_ratio_param.set_value(0.1)
        data_splitting_layout.addRow("Validation Ratio:", self.val_ratio_param)

        self.split_button = QPushButton("Apply Split", self)
        self.split_button.clicked.connect(self._apply_split)
        data_splitting_layout.addWidget(self.split_button)
        main_layout.addWidget(data_splitting_group)

        # 3. Data Augmentation Controls
        data_augmentation_group = QGroupBox("Data Augmentation")
        data_augmentation_layout = QVBoxLayout(data_augmentation_group)

        self.noise_aug_checkbox = BoolParameter("add_noise", "Add Noise Augmentation")
        data_augmentation_layout.addWidget(self.noise_aug_checkbox)

        self.shift_aug_checkbox = BoolParameter("time_shift", "Add Time Shift Augmentation")
        data_augmentation_layout.addWidget(self.shift_aug_checkbox)

        self.augment_button = QPushButton("Apply Augmentation", self)
        self.augment_button.clicked.connect(self._apply_augmentation)
        data_augmentation_layout.addWidget(self.augment_button)
        main_layout.addWidget(data_augmentation_group)

        # 4. Label Management and Visualization
        label_management_group = QGroupBox("Label Management")
        label_management_layout = QVBoxLayout(label_management_group)

        self.label_list_widget = QListWidget(self)
        label_management_layout.addWidget(self.label_list_widget)

        self.add_label_button = QPushButton("Add Label", self)
        self.add_label_button.clicked.connect(self._add_label)
        label_management_layout.addWidget(self.add_label_button)
        main_layout.addWidget(label_management_group)
        
        self.setLayout(main_layout)

    def _browse_file(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Select Data File", "", "All Files (*);;HDF5 Files (*.h5);;CSV Files (*.csv)")
        if file_path:
            self.file_path_input.setText(file_path)
            self.loaded_file_path = file_path

    def _load_data(self):
        if self.loaded_file_path:
            print(f"Attempting to load data from: {self.loaded_file_path}")
            self.data_loaded.emit(self.loaded_file_path)
        else:
            print("No file selected to load.")

    def _apply_split(self):
        train_ratio = self.train_ratio_param.get_value()
        test_ratio = self.test_ratio_param.get_value()
        val_ratio = self.val_ratio_param.get_value()
        total_ratio = train_ratio + test_ratio + val_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: Split ratios do not sum to 1.0 (current sum: {total_ratio:.2f}). Adjusting...")
            # Simple normalization if they don't sum to 1
            if total_ratio > 0:
                train_ratio /= total_ratio
                test_ratio /= total_ratio
                val_ratio /= total_ratio
                self.train_ratio_param.set_value(train_ratio)
                self.test_ratio_param.set_value(test_ratio)
                self.val_ratio_param.set_value(val_ratio)
                print(f"Adjusted ratios: Train={train_ratio:.2f}, Test={test_ratio:.2f}, Val={val_ratio:.2f}")
            else:
                print("Error: All split ratios are zero.")
                return

        print(f"Applying data split: Train={train_ratio:.2f}, Test={test_ratio:.2f}, Val={val_ratio:.2f}")
        self.data_split_requested.emit(train_ratio, test_ratio, val_ratio)

    def _apply_augmentation(self):
        augmentation_params = {
            "add_noise": self.noise_aug_checkbox.get_value(),
            "time_shift": self.shift_aug_checkbox.get_value(),
            # Add more augmentation parameters here
        }
        print(f"Applying data augmentation with params: {augmentation_params}")
        self.data_augmentation_requested.emit(augmentation_params)

    def _add_label(self):
        # In a real application, this would open a dialog to add/edit labels
        new_label_text = f"Label {self.label_list_widget.count() + 1}"
        item = QListWidgetItem(new_label_text)
        self.label_list_widget.addItem(item)
        current_labels = [self.label_list_widget.item(i).text() for i in range(self.label_list_widget.count())]
        self.labels_updated.emit(current_labels)
        print(f"Added label: {new_label_text}")

    # These methods would be called by external components (e.g., worker threads)
    def on_data_loaded_success(self, file_path: str, num_samples: int):
        print(f"Successfully loaded data from {file_path}. Samples: {num_samples}")
        # Update UI to reflect loaded data, e.g., enable other controls
        self.file_path_input.setText(file_path)

    def on_data_split_success(self, train_count: int, test_count: int, val_count: int):
        print(f"Data split successful: Train={train_count}, Test={test_count}, Val={val_count}")

    def on_data_augmented_success(self, num_augmented_samples: int):
        print(f"Data augmentation successful. Total augmented samples: {num_augmented_samples}")

    def on_labels_loaded(self, labels: list):
        self.label_list_widget.clear()
        for label in labels:
            self.label_list_widget.addItem(QListWidgetItem(label))
        print(f"Labels loaded: {labels}")