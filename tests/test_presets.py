import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import json
import os

from ui.presets.preset_manager import (
    PresetManager, PresetConfig, PresetCategory
)
from ui.presets.preset_widget import PresetWidget, PresetDialog
from ui.error_handling import ErrorHandler

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication([])

@pytest.fixture
def error_handler():
    """Create an ErrorHandler instance."""
    return ErrorHandler()

@pytest.fixture
def preset_manager(error_handler):
    """Create a PresetManager instance."""
    return PresetManager(error_handler)

@pytest.fixture
def preset_widget(preset_manager, error_handler):
    """Create a PresetWidget instance."""
    return PresetWidget(preset_manager, error_handler)

@pytest.fixture
def sample_preset():
    """Create a sample preset configuration."""
    return PresetConfig(
        name="Test Preset",
        description="Test preset description",
        signal_type="EMG",
        category="Test Category",
        subcategory="Test Subcategory",
        parameters={
            'filter': {
                'type': 'Bandpass Filter',
                'parameters': {
                    'lowcut': 20,
                    'highcut': 450,
                    'order': 4
                }
            }
        },
        metadata={
            'author': 'Test',
            'version': '1.0'
        }
    )

def test_preset_config_serialization(sample_preset):
    """Test PresetConfig serialization."""
    # Convert to dict
    data = sample_preset.to_dict()
    
    # Create new preset from dict
    new_preset = PresetConfig.from_dict(data)
    
    # Verify attributes
    assert new_preset.name == sample_preset.name
    assert new_preset.description == sample_preset.description
    assert new_preset.signal_type == sample_preset.signal_type
    assert new_preset.category == sample_preset.category
    assert new_preset.subcategory == sample_preset.subcategory
    assert new_preset.parameters == sample_preset.parameters
    assert new_preset.metadata == sample_preset.metadata

def test_preset_category_operations():
    """Test PresetCategory operations."""
    category = PresetCategory("Test")
    
    # Add preset
    preset = PresetConfig(
        name="Test",
        description="Test",
        signal_type="EMG",
        category="Test",
        subcategory="Sub",
        parameters={},
        metadata={}
    )
    category.add_preset(preset)
    
    # Get preset
    retrieved = category.get_preset("Test", "Sub")
    assert retrieved == preset
    
    # Convert to dict
    data = category.to_dict()
    
    # Create from dict
    new_category = PresetCategory.from_dict(data)
    assert new_category.name == category.name
    assert len(new_category.subcategories) == len(category.subcategories)

def test_preset_manager_operations(preset_manager, sample_preset):
    """Test PresetManager operations."""
    # Add preset
    assert preset_manager.add_preset(sample_preset)
    
    # Get preset
    retrieved = preset_manager.get_preset(
        sample_preset.name,
        sample_preset.signal_type,
        sample_preset.category,
        sample_preset.subcategory
    )
    assert retrieved == sample_preset
    
    # List presets
    presets = preset_manager.list_presets("EMG")
    assert sample_preset in presets
    
    # Modify preset
    modified = PresetConfig(
        **{**sample_preset.to_dict(), 'description': 'Modified description'}
    )
    assert preset_manager.modify_preset(modified)
    
    # Remove preset
    assert preset_manager.remove_preset(
        sample_preset.name,
        sample_preset.signal_type,
        sample_preset.category,
        sample_preset.subcategory
    )

def test_preset_manager_file_operations(preset_manager, sample_preset, tmp_path):
    """Test PresetManager file operations."""
    # Add preset
    preset_manager.add_preset(sample_preset)
    
    # Save presets
    filepath = tmp_path / "presets.json"
    preset_manager.save_presets(str(filepath))
    
    # Create new manager and load presets
    new_manager = PresetManager(preset_manager.error_handler)
    new_manager.load_presets(str(filepath))
    
    # Verify loaded presets
    loaded = new_manager.get_preset(
        sample_preset.name,
        sample_preset.signal_type,
        sample_preset.category,
        sample_preset.subcategory
    )
    assert loaded is not None
    assert loaded.to_dict() == sample_preset.to_dict()

def test_preset_manager_import_export(preset_manager, sample_preset, tmp_path):
    """Test preset import/export operations."""
    # Export preset
    export_path = tmp_path / "preset.json"
    assert preset_manager.export_preset(sample_preset, str(export_path))
    
    # Import preset
    imported = preset_manager.import_preset(str(export_path))
    assert imported is not None
    assert imported.to_dict() == sample_preset.to_dict()

def test_preset_widget_tree(preset_widget, sample_preset, qtbot):
    """Test PresetWidget tree structure."""
    # Add preset
    preset_widget.preset_manager.add_preset(sample_preset)
    
    # Find preset item
    def find_preset_item(parent, name):
        for i in range(parent.childCount()):
            item = parent.child(i)
            if item.text(0) == name:
                return item
            result = find_preset_item(item, name)
            if result:
                return result
        return None
        
    root = preset_widget.tree.topLevelItem(0)  # EMG
    preset_item = find_preset_item(root, sample_preset.name)
    assert preset_item is not None
    
    # Verify item data
    assert preset_item.text(0) == sample_preset.name
    assert preset_item.text(1) == f"{sample_preset.category}/{sample_preset.subcategory}"
    assert preset_item.text(2) == sample_preset.description

def test_preset_dialog(app, sample_preset, qtbot):
    """Test PresetDialog functionality."""
    # Create dialog with existing preset
    dialog = PresetDialog(sample_preset)
    
    # Verify form fields
    assert dialog.name_edit.text() == sample_preset.name
    assert dialog.description_edit.toPlainText() == sample_preset.description
    assert dialog.signal_type_combo.currentText() == sample_preset.signal_type
    assert dialog.category_edit.text() == sample_preset.category
    assert dialog.subcategory_edit.text() == sample_preset.subcategory
    
    # Modify fields
    dialog.name_edit.setText("Modified Name")
    dialog.description_edit.setText("Modified description")
    
    # Get data
    data = dialog.get_preset_data()
    assert data['name'] == "Modified Name"
    assert data['description'] == "Modified description"
    assert data['parameters'] == sample_preset.parameters

def test_preset_widget_signals(preset_widget, sample_preset, qtbot):
    """Test PresetWidget signals."""
    # Add preset
    preset_widget.preset_manager.add_preset(sample_preset)
    
    # Setup signal spy
    with qtbot.waitSignal(preset_widget.preset_selected) as blocker:
        # Find and double-click preset item
        def find_and_click_preset():
            for i in range(preset_widget.tree.topLevelItemCount()):
                top_item = preset_widget.tree.topLevelItem(i)
                for j in range(top_item.childCount()):
                    category_item = top_item.child(j)
                    for k in range(category_item.childCount()):
                        subcategory_item = category_item.child(k)
                        for l in range(subcategory_item.childCount()):
                            preset_item = subcategory_item.child(l)
                            if preset_item.text(0) == sample_preset.name:
                                qtbot.mouseClick(
                                    preset_widget.tree.viewport(),
                                    Qt.MouseButton.LeftButton,
                                    pos=preset_widget.tree.visualItemRect(preset_item).center()
                                )
                                qtbot.mouseDClick(
                                    preset_widget.tree.viewport(),
                                    Qt.MouseButton.LeftButton,
                                    pos=preset_widget.tree.visualItemRect(preset_item).center()
                                )
                                return
                                
        find_and_click_preset()
    
    # Verify signal
    assert blocker.args[0].name == sample_preset.name

def test_preset_widget_context_menu(preset_widget, sample_preset, qtbot, monkeypatch):
    """Test PresetWidget context menu."""
    # Add preset
    preset_widget.preset_manager.add_preset(sample_preset)
    
    # Mock QMenu.exec
    exec_called = False
    def mock_exec(self, pos):
        nonlocal exec_called
        exec_called = True
        
    monkeypatch.setattr('PyQt6.QtWidgets.QMenu.exec', mock_exec)
    
    # Find preset item and show context menu
    def find_and_show_menu():
        for i in range(preset_widget.tree.topLevelItemCount()):
            top_item = preset_widget.tree.topLevelItem(i)
            for j in range(top_item.childCount()):
                category_item = top_item.child(j)
                for k in range(category_item.childCount()):
                    subcategory_item = category_item.child(k)
                    for l in range(subcategory_item.childCount()):
                        preset_item = subcategory_item.child(l)
                        if preset_item.text(0) == sample_preset.name:
                            rect = preset_widget.tree.visualItemRect(preset_item)
                            preset_widget._show_context_menu(rect.center())
                            return
                            
    find_and_show_menu()
    
    # Verify menu was shown
    assert exec_called