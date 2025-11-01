from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QDialog, QFormLayout, QLineEdit, QTextEdit,
    QComboBox, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Optional, Dict, Any

from .preset_manager import PresetManager, PresetConfig
from ..error_handling import ErrorHandler, ErrorSeverity, ErrorCategory

class PresetDialog(QDialog):
    """Dialog for creating/editing presets."""
    
    def __init__(self, preset: Optional[PresetConfig] = None, parent=None):
        super().__init__(parent)
        self.preset = preset
        self._init_ui()
        if preset:
            self._load_preset(preset)
            
    def _init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Preset Configuration")
        layout = QFormLayout(self)
        
        # Basic info
        self.name_edit = QLineEdit()
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        
        # Categories
        self.signal_type_combo = QComboBox()
        self.signal_type_combo.addItems(["EMG", "ECG", "EOG"])
        
        self.category_edit = QLineEdit()
        self.subcategory_edit = QLineEdit()
        
        # Add fields
        layout.addRow("Name:", self.name_edit)
        layout.addRow("Description:", self.description_edit)
        layout.addRow("Signal Type:", self.signal_type_combo)
        layout.addRow("Category:", self.category_edit)
        layout.addRow("Subcategory:", self.subcategory_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addRow(button_layout)
        
        # Connect signals
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
    def _load_preset(self, preset: PresetConfig):
        """Load preset data into form."""
        self.name_edit.setText(preset.name)
        self.description_edit.setText(preset.description)
        self.signal_type_combo.setCurrentText(preset.signal_type)
        self.category_edit.setText(preset.category)
        self.subcategory_edit.setText(preset.subcategory)
        
    def get_preset_data(self) -> Dict[str, Any]:
        """Get preset data from form."""
        return {
            'name': self.name_edit.text(),
            'description': self.description_edit.toPlainText(),
            'signal_type': self.signal_type_combo.currentText(),
            'category': self.category_edit.text(),
            'subcategory': self.subcategory_edit.text(),
            'parameters': self.preset.parameters if self.preset else {},
            'metadata': self.preset.metadata if self.preset else {
                'author': 'User',
                'version': '1.0'
            }
        }

class PresetWidget(QWidget):
    """Widget for managing presets."""
    
    preset_selected = pyqtSignal(PresetConfig)  # Emitted when preset is selected
    
    def __init__(self, preset_manager: PresetManager, error_handler: ErrorHandler, parent=None):
        super().__init__(parent)
        self.preset_manager = preset_manager
        self.error_handler = error_handler
        self._init_ui()
        
        # Connect preset manager signals
        self.preset_manager.preset_added.connect(self._on_preset_added)
        self.preset_manager.preset_removed.connect(self._on_preset_removed)
        self.preset_manager.preset_modified.connect(self._on_preset_modified)
        
    def _init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Create tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Category", "Description"])
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.new_btn = QPushButton("New Preset")
        self.import_btn = QPushButton("Import")
        self.export_btn = QPushButton("Export")
        
        button_layout.addWidget(self.new_btn)
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        
        # Add widgets to layout
        layout.addWidget(self.tree)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.new_btn.clicked.connect(self._create_preset)
        self.import_btn.clicked.connect(self._import_preset)
        self.export_btn.clicked.connect(self._export_preset)
        
        # Load initial presets
        self._load_presets()
        
    def _load_presets(self):
        """Load presets into tree widget."""
        self.tree.clear()
        
        # Create top-level items for signal types
        signal_items = {}
        for signal_type in ["EMG", "ECG", "EOG"]:
            item = QTreeWidgetItem([signal_type, "", ""])
            self.tree.addTopLevelItem(item)
            signal_items[signal_type] = item
            
        # Add presets
        for preset in self.preset_manager.list_presets():
            signal_item = signal_items[preset.signal_type]
            
            # Find or create category item
            category_item = None
            for i in range(signal_item.childCount()):
                if signal_item.child(i).text(0) == preset.category:
                    category_item = signal_item.child(i)
                    break
                    
            if not category_item:
                category_item = QTreeWidgetItem([preset.category, "", ""])
                signal_item.addChild(category_item)
                
            # Find or create subcategory item
            if preset.subcategory:
                subcategory_item = None
                for i in range(category_item.childCount()):
                    if category_item.child(i).text(0) == preset.subcategory:
                        subcategory_item = category_item.child(i)
                        break
                        
                if not subcategory_item:
                    subcategory_item = QTreeWidgetItem([preset.subcategory, "", ""])
                    category_item.addChild(subcategory_item)
                    
                parent_item = subcategory_item
            else:
                parent_item = category_item
                
            # Add preset item
            preset_item = QTreeWidgetItem([
                preset.name,
                preset.category + ("/" + preset.subcategory if preset.subcategory else ""),
                preset.description
            ])
            preset_item.setData(0, Qt.ItemDataRole.UserRole, preset)
            parent_item.addChild(preset_item)
            
        self.tree.expandAll()
        
    def _show_context_menu(self, position):
        """Show context menu for tree item."""
        item = self.tree.itemAt(position)
        if not item:
            return
            
        preset = item.data(0, Qt.ItemDataRole.UserRole)
        if not preset:
            return
            
        menu = QMenu()
        
        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(lambda: self._edit_preset(preset))
        
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_preset(preset))
        
        menu.exec(self.tree.viewport().mapToGlobal(position))
        
    def _create_preset(self):
        """Create a new preset."""
        dialog = PresetDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                preset = PresetConfig(**dialog.get_preset_data())
                self.preset_manager.add_preset(preset)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                )
                
    def _edit_preset(self, preset: PresetConfig):
        """Edit an existing preset."""
        dialog = PresetDialog(preset, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                modified_preset = PresetConfig(**dialog.get_preset_data())
                self.preset_manager.modify_preset(modified_preset)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                )
                
    def _delete_preset(self, preset: PresetConfig):
        """Delete a preset."""
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Are you sure you want to delete preset '{preset.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.preset_manager.remove_preset(
                preset.name,
                preset.signal_type,
                preset.category,
                preset.subcategory
            )
            
    def _import_preset(self):
        """Import preset from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Preset",
            "",
            "JSON Files (*.json)"
        )
        
        if filename:
            preset = self.preset_manager.import_preset(filename)
            if preset:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Imported preset '{preset.name}'"
                )
                
    def _export_preset(self):
        """Export preset to file."""
        item = self.tree.currentItem()
        if not item:
            return
            
        preset = item.data(0, Qt.ItemDataRole.UserRole)
        if not preset:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Preset",
            f"{preset.name}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            if self.preset_manager.export_preset(preset, filename):
                QMessageBox.information(
                    self,
                    "Success",
                    f"Exported preset '{preset.name}'"
                )
                
    def _on_item_double_clicked(self, item: QTreeWidgetItem):
        """Handle item double click."""
        preset = item.data(0, Qt.ItemDataRole.UserRole)
        if preset:
            self.preset_selected.emit(preset)
            
    def _on_preset_added(self, preset: PresetConfig):
        """Handle preset added."""
        self._load_presets()
        
    def _on_preset_removed(self, name: str, category: str, subcategory: str):
        """Handle preset removed."""
        self._load_presets()
        
    def _on_preset_modified(self, preset: PresetConfig):
        """Handle preset modified."""
        self._load_presets()