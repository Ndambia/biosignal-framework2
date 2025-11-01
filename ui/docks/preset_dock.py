from PyQt6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QPushButton, 
    QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QMenu, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_dock import BaseDock

class PresetTreeWidget(QTreeWidget):
    """Hierarchical tree widget for preset organization."""
    
    preset_selected = pyqtSignal(dict)  # Emits preset data when selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Presets")
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
    def _show_context_menu(self, position):
        menu = QMenu()
        item = self.itemAt(position)
        
        if item is None:
            # Clicked on empty area
            add_category = menu.addAction("Add Category")
            add_category.triggered.connect(self._add_category)
        else:
            # Clicked on an item
            if item.parent() is None:
                # Category item
                add_preset = menu.addAction("Add Preset")
                add_preset.triggered.connect(lambda: self._add_preset(item))
                rename = menu.addAction("Rename Category")
                rename.triggered.connect(lambda: self._rename_item(item))
            else:
                # Preset item
                edit = menu.addAction("Edit Preset")
                edit.triggered.connect(lambda: self._edit_preset(item))
                rename = menu.addAction("Rename Preset")
                rename.triggered.connect(lambda: self._rename_item(item))
            
            menu.addSeparator()
            delete = menu.addAction("Delete")
            delete.triggered.connect(lambda: self._delete_item(item))
        
        menu.exec(self.viewport().mapToGlobal(position))
        
    def _add_category(self):
        """Add a new top-level category."""
        name, ok = QInputDialog.getText(self, "New Category", "Category name:")
        if ok and name:
            item = QTreeWidgetItem([name])
            self.addTopLevelItem(item)
            
    def _add_preset(self, category_item):
        """Add a new preset to a category."""
        name, ok = QInputDialog.getText(self, "New Preset", "Preset name:")
        if ok and name:
            preset = QTreeWidgetItem([name])
            category_item.addChild(preset)
            
    def _rename_item(self, item):
        """Rename a category or preset."""
        name, ok = QInputDialog.getText(self, "Rename", "New name:", 
                                      text=item.text(0))
        if ok and name:
            item.setText(0, name)
            
    def _edit_preset(self, item):
        """Edit preset parameters."""
        # This will be implemented when we have the parameter editing dialog
        pass
        
    def _delete_item(self, item):
        """Delete a category or preset."""
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            index = self.indexOfTopLevelItem(item)
            self.takeTopLevelItem(index)

class PresetDock(BaseDock):
    """Dockable panel for preset management."""
    
    preset_selected = pyqtSignal(dict)  # Forward preset selection
    
    def __init__(self, title="Preset Library", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        self._populate_default_presets()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create preset tree
        self.preset_tree = PresetTreeWidget()
        self.preset_tree.preset_selected.connect(self.preset_selected.emit)
        self.add_widget(self.preset_tree)
        
        # Create button bar
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.import_btn = QPushButton("Import")
        self.export_btn = QPushButton("Export")
        self.refresh_btn = QPushButton("Refresh")
        
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.refresh_btn)
        
        self.add_widget(button_widget)
        
        # Connect signals
        self.import_btn.clicked.connect(self._import_presets)
        self.export_btn.clicked.connect(self._export_presets)
        self.refresh_btn.clicked.connect(self._refresh_presets)
        
    def _populate_default_presets(self):
        """Add default preset categories and items."""
        # EMG presets
        emg = QTreeWidgetItem(["EMG"])
        self.preset_tree.addTopLevelItem(emg)
        
        emg_categories = {
            "Isometric": ["Light", "Moderate", "Heavy"],
            "Dynamic": ["Ramp Up", "Ramp Down", "Cyclic"],
            "Complex": ["Multi-Movement", "Fatigue Test"]
        }
        
        for category, presets in emg_categories.items():
            cat_item = QTreeWidgetItem([category])
            emg.addChild(cat_item)
            for preset in presets:
                cat_item.addChild(QTreeWidgetItem([preset]))
                
        # ECG presets
        ecg = QTreeWidgetItem(["ECG"])
        self.preset_tree.addTopLevelItem(ecg)
        
        ecg_categories = {
            "Normal": ["Resting", "Exercise", "Recovery"],
            "Arrhythmia": ["PVC", "AF", "VT"],
            "Ischemia": ["STEMI", "NSTEMI", "Angina"]
        }
        
        for category, presets in ecg_categories.items():
            cat_item = QTreeWidgetItem([category])
            ecg.addChild(cat_item)
            for preset in presets:
                cat_item.addChild(QTreeWidgetItem([preset]))
                
        # EOG presets
        eog = QTreeWidgetItem(["EOG"])
        self.preset_tree.addTopLevelItem(eog)
        
        eog_categories = {
            "Saccades": ["Horizontal", "Vertical", "Random"],
            "Pursuit": ["Sinusoidal", "Circular", "Complex"],
            "Combined": ["Reading", "Visual Search"]
        }
        
        for category, presets in eog_categories.items():
            cat_item = QTreeWidgetItem([category])
            eog.addChild(cat_item)
            for preset in presets:
                cat_item.addChild(QTreeWidgetItem([preset]))
                
    def _import_presets(self):
        """Import presets from file."""
        # To be implemented
        pass
        
    def _export_presets(self):
        """Export presets to file."""
        # To be implemented
        pass
        
    def _refresh_presets(self):
        """Refresh preset list from storage."""
        # To be implemented
        pass