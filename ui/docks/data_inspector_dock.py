from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QHBoxLayout, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import numpy as np
from .base_dock import BaseDock

class DataTreeWidget(QTreeWidget):
    """Tree widget for displaying hierarchical data information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI."""
        self.setHeaderLabels(["Property", "Value"])
        self.setColumnWidth(0, 150)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
    def _show_context_menu(self, position):
        """Show context menu for copying values."""
        menu = QMenu()
        item = self.itemAt(position)
        
        if item:
            copy_value = menu.addAction("Copy Value")
            copy_value.triggered.connect(lambda: self._copy_value(item))
            
            if item.childCount() > 0:
                copy_all = menu.addAction("Copy All")
                copy_all.triggered.connect(lambda: self._copy_subtree(item))
            
            menu.exec(self.viewport().mapToGlobal(position))
            
    def _copy_value(self, item):
        """Copy item value to clipboard."""
        if item.text(1):  # Only copy if there's a value
            QApplication.clipboard().setText(item.text(1))
            
    def _copy_subtree(self, item):
        """Copy entire subtree to clipboard."""
        text = self._get_subtree_text(item)
        QApplication.clipboard().setText(text)
        
    def _get_subtree_text(self, item, level=0):
        """Get formatted text representation of subtree."""
        text = "  " * level + f"{item.text(0)}: {item.text(1)}\n"
        for i in range(item.childCount()):
            text += self._get_subtree_text(item.child(i), level + 1)
        return text
        
    def update_signal_info(self, signal: np.ndarray, sampling_rate: float):
        """Update tree with signal information."""
        self.clear()
        
        # Basic signal properties
        signal_root = QTreeWidgetItem(["Signal Properties"])
        self.addTopLevelItem(signal_root)
        
        properties = [
            ("Shape", str(signal.shape)),
            ("Length", str(len(signal))),
            ("Duration", f"{len(signal)/sampling_rate:.2f} s"),
            ("Sampling Rate", f"{sampling_rate} Hz"),
            ("Data Type", str(signal.dtype))
        ]
        
        for name, value in properties:
            item = QTreeWidgetItem([name, value])
            signal_root.addChild(item)
            
        # Statistical properties
        stats_root = QTreeWidgetItem(["Statistics"])
        self.addTopLevelItem(stats_root)
        
        statistics = [
            ("Mean", f"{np.mean(signal):.4f}"),
            ("Std Dev", f"{np.std(signal):.4f}"),
            ("Min", f"{np.min(signal):.4f}"),
            ("Max", f"{np.max(signal):.4f}"),
            ("RMS", f"{np.sqrt(np.mean(np.square(signal))):.4f}")
        ]
        
        for name, value in statistics:
            item = QTreeWidgetItem([name, value])
            stats_root.addChild(item)
            
        # Expand all items
        self.expandAll()
        
    def update_processing_info(self, info: dict):
        """Update tree with processing information."""
        self.clear()
        
        for category, data in info.items():
            root = QTreeWidgetItem([category])
            self.addTopLevelItem(root)
            
            if isinstance(data, dict):
                for key, value in data.items():
                    item = QTreeWidgetItem([key, str(value)])
                    root.addChild(item)
            else:
                root.setText(1, str(data))
                
        self.expandAll()

class DataInspectorDock(BaseDock):
    """Dockable panel for inspecting data properties."""
    
    def __init__(self, title="Data Inspector", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create data tree
        self.data_tree = DataTreeWidget()
        self.add_widget(self.data_tree)
        
        # Create button bar
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.refresh_btn = QPushButton("Refresh")
        self.export_btn = QPushButton("Export")
        
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.export_btn)
        
        self.add_widget(button_widget)
        
        # Add status label
        self.status_label = QLabel("No data selected")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setItalic(True)
        self.status_label.setFont(font)
        self.add_widget(self.status_label)
        
        # Connect signals
        self.refresh_btn.clicked.connect(self._refresh_data)
        self.export_btn.clicked.connect(self._export_data)
        
    def update_signal_data(self, signal: np.ndarray, sampling_rate: float):
        """Update inspector with signal data."""
        if signal is not None:
            self.data_tree.update_signal_info(signal, sampling_rate)
            self.status_label.setText(f"Inspecting signal data")
        else:
            self.status_label.setText("No signal data available")
            
    def update_processing_info(self, info: dict):
        """Update inspector with processing information."""
        if info:
            self.data_tree.update_processing_info(info)
            self.status_label.setText(f"Inspecting processing results")
        else:
            self.status_label.setText("No processing information available")
            
    def clear(self):
        """Clear all displayed data."""
        self.data_tree.clear()
        self.status_label.setText("No data selected")
        
    def _refresh_data(self):
        """Refresh current data display."""
        # To be implemented based on current context
        pass
        
    def _export_data(self):
        """Export displayed data to file."""
        # To be implemented
        pass