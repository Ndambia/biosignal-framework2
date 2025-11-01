from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QGroupBox,
    QLabel, QFormLayout
)
from PyQt6.QtCore import pyqtSignal

class SignalTypeSelector(QWidget):
    signal_type_changed = pyqtSignal(str)
    
    SIGNAL_TYPES = ["EMG", "ECG", "EOG"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create and configure the combo box
        self.type_selector = QComboBox()
        self.type_selector.addItems(self.SIGNAL_TYPES)
        self.type_selector.currentTextChanged.connect(self.signal_type_changed.emit)
        
        # Add to layout with label
        group = QGroupBox("Signal Type")
        group_layout = QFormLayout()
        group_layout.addRow("Type:", self.type_selector)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
    def current_signal_type(self) -> str:
        return self.type_selector.currentText()

class SignalControlDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Signal type selector
        self.type_selector = SignalTypeSelector()
        layout.addWidget(self.type_selector)
        
        # Common controls group
        common_group = QGroupBox("Common Parameters")
        common_layout = QFormLayout()
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)
        
        # Processing controls group
        processing_group = QGroupBox("Processing Controls")
        processing_layout = QFormLayout()
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        # Add stretch to push widgets to top
        layout.addStretch()
        
    def get_signal_type(self) -> str:
        return self.type_selector.current_signal_type()