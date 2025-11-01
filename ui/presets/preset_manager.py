from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal

from ..error_handling import ErrorHandler, ErrorSeverity, ErrorCategory, ValidationError

@dataclass
class PresetConfig:
    """Configuration data for a preset."""
    name: str
    description: str
    signal_type: str  # EMG, ECG, EOG
    category: str  # e.g., "Muscle Contraction", "Arrhythmias"
    subcategory: str  # e.g., "Isometric", "PVC"
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert preset to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PresetConfig':
        """Create preset from dictionary."""
        return cls(**data)

class PresetCategory:
    """Represents a category in the preset hierarchy."""
    
    def __init__(self, name: str):
        self.name = name
        self.subcategories: Dict[str, 'PresetCategory'] = {}
        self.presets: Dict[str, PresetConfig] = {}
        
    def add_preset(self, preset: PresetConfig):
        """Add a preset to this category."""
        if preset.subcategory:
            if preset.subcategory not in self.subcategories:
                self.subcategories[preset.subcategory] = PresetCategory(preset.subcategory)
            self.subcategories[preset.subcategory].presets[preset.name] = preset
        else:
            self.presets[preset.name] = preset
            
    def get_preset(self, name: str, subcategory: Optional[str] = None) -> Optional[PresetConfig]:
        """Get a preset by name and optional subcategory."""
        if subcategory:
            if subcategory in self.subcategories:
                return self.subcategories[subcategory].presets.get(name)
        return self.presets.get(name)
        
    def to_dict(self) -> dict:
        """Convert category to dictionary."""
        return {
            'name': self.name,
            'subcategories': {
                name: cat.to_dict() for name, cat in self.subcategories.items()
            },
            'presets': {
                name: preset.to_dict() for name, preset in self.presets.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'PresetCategory':
        """Create category from dictionary."""
        category = cls(data['name'])
        
        for name, subcat_data in data['subcategories'].items():
            category.subcategories[name] = cls.from_dict(subcat_data)
            
        for name, preset_data in data['presets'].items():
            category.presets[name] = PresetConfig.from_dict(preset_data)
            
        return category

class PresetManager(QObject):
    """Manages processing presets."""
    
    preset_added = pyqtSignal(PresetConfig)  # Emitted when preset is added
    preset_removed = pyqtSignal(str, str, str)  # name, category, subcategory
    preset_modified = pyqtSignal(PresetConfig)  # Emitted when preset is modified
    
    def __init__(self, error_handler: ErrorHandler):
        super().__init__()
        self.error_handler = error_handler
        self.categories: Dict[str, PresetCategory] = {
            'EMG': PresetCategory('EMG'),
            'ECG': PresetCategory('ECG'),
            'EOG': PresetCategory('EOG')
        }
        
        # Load default presets
        self._load_default_presets()
        
    def _load_default_presets(self):
        """Load default presets from configuration."""
        default_presets = {
            'EMG': {
                'Muscle Contraction': {
                    'Isometric': {
                        'name': 'Isometric - Basic',
                        'description': 'Basic isometric contraction processing',
                        'signal_type': 'EMG',
                        'category': 'Muscle Contraction',
                        'subcategory': 'Isometric',
                        'parameters': {
                            'filter': {
                                'type': 'Bandpass Filter',
                                'parameters': {
                                    'lowcut': 20,
                                    'highcut': 450,
                                    'order': 4
                                }
                            },
                            'normalize': {
                                'method': 'Z-score',
                                'parameters': {}
                            }
                        },
                        'metadata': {
                            'author': 'System',
                            'version': '1.0'
                        }
                    }
                }
            },
            'ECG': {
                'Normal Rhythms': {
                    'Sinus': {
                        'name': 'Normal Sinus Rhythm',
                        'description': 'Standard ECG processing for NSR',
                        'signal_type': 'ECG',
                        'category': 'Normal Rhythms',
                        'subcategory': 'Sinus',
                        'parameters': {
                            'filter': {
                                'type': 'Bandpass Filter',
                                'parameters': {
                                    'lowcut': 0.5,
                                    'highcut': 40,
                                    'order': 4
                                }
                            },
                            'normalize': {
                                'method': 'Z-score',
                                'parameters': {}
                            }
                        },
                        'metadata': {
                            'author': 'System',
                            'version': '1.0'
                        }
                    }
                }
            }
        }
        
        for signal_type, categories in default_presets.items():
            for category, subcategories in categories.items():
                for subcategory, preset_data in subcategories.items():
                    preset = PresetConfig.from_dict(preset_data)
                    self.add_preset(preset)
                    
    def add_preset(self, preset: PresetConfig) -> bool:
        """Add a new preset."""
        try:
            if preset.signal_type not in self.categories:
                raise ValidationError(
                    f"Invalid signal type: {preset.signal_type}",
                    "Signal type must be EMG, ECG, or EOG"
                )
                
            category = self.categories[preset.signal_type]
            category.add_preset(preset)
            self.preset_added.emit(preset)
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.CONFIGURATION,
                ["Check preset configuration format"]
            )
            return False
            
    def remove_preset(self, name: str, signal_type: str,
                     category: str, subcategory: Optional[str] = None) -> bool:
        """Remove a preset."""
        try:
            if signal_type not in self.categories:
                raise ValidationError(f"Invalid signal type: {signal_type}")
                
            cat = self.categories[signal_type]
            if subcategory:
                if subcategory not in cat.subcategories:
                    raise ValidationError(f"Invalid subcategory: {subcategory}")
                del cat.subcategories[subcategory].presets[name]
            else:
                del cat.presets[name]
                
            self.preset_removed.emit(name, category, subcategory or "")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.CONFIGURATION
            )
            return False
            
    def get_preset(self, name: str, signal_type: str,
                   category: str, subcategory: Optional[str] = None) -> Optional[PresetConfig]:
        """Get a preset by name and category."""
        if signal_type in self.categories:
            return self.categories[signal_type].get_preset(name, subcategory)
        return None
        
    def list_presets(self, signal_type: Optional[str] = None) -> List[PresetConfig]:
        """List all presets, optionally filtered by signal type."""
        presets = []
        
        categories = [self.categories[signal_type]] if signal_type else self.categories.values()
        
        for category in categories:
            # Add top-level presets
            presets.extend(category.presets.values())
            
            # Add subcategory presets
            for subcategory in category.subcategories.values():
                presets.extend(subcategory.presets.values())
                
        return presets
        
    def save_presets(self, filepath: str):
        """Save all presets to file."""
        try:
            data = {
                name: category.to_dict()
                for name, category in self.categories.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.FILE_IO,
                ["Check file permissions", "Verify file path"]
            )
            
    def load_presets(self, filepath: str):
        """Load presets from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.categories = {
                name: PresetCategory.from_dict(cat_data)
                for name, cat_data in data.items()
            }
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.FILE_IO,
                ["Check file format", "Verify file exists"]
            )
            
    def modify_preset(self, preset: PresetConfig) -> bool:
        """Modify an existing preset."""
        try:
            if preset.signal_type not in self.categories:
                raise ValidationError(f"Invalid signal type: {preset.signal_type}")
                
            # Remove old preset
            self.remove_preset(
                preset.name,
                preset.signal_type,
                preset.category,
                preset.subcategory
            )
            
            # Add modified preset
            self.add_preset(preset)
            self.preset_modified.emit(preset)
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.CONFIGURATION
            )
            return False
            
    def import_preset(self, filepath: str) -> Optional[PresetConfig]:
        """Import a preset from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            preset = PresetConfig.from_dict(data)
            if self.add_preset(preset):
                return preset
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.FILE_IO,
                ["Check file format", "Verify file exists"]
            )
            
        return None
        
    def export_preset(self, preset: PresetConfig, filepath: str) -> bool:
        """Export a preset to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(preset.to_dict(), f, indent=2)
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorSeverity.ERROR,
                ErrorCategory.FILE_IO,
                ["Check file permissions", "Verify file path"]
            )
            return False