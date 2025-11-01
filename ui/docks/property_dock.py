from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QStackedWidget, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from .base_dock import BaseDock

class PropertyWidget(QWidget):
    """Base class for property widgets."""
    
    property_changed = pyqtSignal(str, object)  # (property_name, new_value)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
    def add_property(self, label: str, widget: QWidget):
        """Add a property control with label."""
        self.layout.addRow(label, widget)
        
    def clear_properties(self):
        """Remove all property controls."""
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

class SignalPropertyWidget(PropertyWidget):
    """Properties for signal generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_signal_properties()
        
    def _setup_signal_properties(self):
        # To be populated with signal-specific controls
        pass
        
    def update_for_signal_type(self, signal_type: str):
        """Update properties for specific signal type."""
        self.clear_properties()
        # Will be implemented with signal-specific controls

class FilterPropertyWidget(PropertyWidget):
    """Properties for signal filtering."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_filter_properties()
        
    def _setup_filter_properties(self):
        # To be populated with filter controls
        pass
        
    def update_for_filter_type(self, filter_type: str):
        """Update properties for specific filter type."""
        self.clear_properties()
        # Will be implemented with filter-specific controls

class FeaturePropertyWidget(PropertyWidget):
    """Properties for feature extraction."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_feature_properties()
        
    def _setup_feature_properties(self):
        # To be populated with feature extraction controls
        pass
        
    def update_for_feature_type(self, feature_type: str):
        """Update properties for specific feature type."""
        self.clear_properties()
        # Will be implemented with feature-specific controls

class PropertyDock(BaseDock):
    """Dockable panel for context-sensitive properties."""
    
    property_changed = pyqtSignal(str, str, object)  # (context, property_name, new_value)
    
    def __init__(self, title="Properties", parent=None):
        super().__init__(title, parent)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.add_widget(scroll)
        
        # Create main container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(container)
        
        # Create stacked widget for different property contexts
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        
        # Create property widgets for different contexts
        self.signal_properties = SignalPropertyWidget()
        self.filter_properties = FilterPropertyWidget()
        self.feature_properties = FeaturePropertyWidget()
        
        # Add widgets to stack
        self.stack.addWidget(self.signal_properties)
        self.stack.addWidget(self.filter_properties)
        self.stack.addWidget(self.feature_properties)
        
        # Connect property change signals
        self.signal_properties.property_changed.connect(
            lambda name, value: self.property_changed.emit("signal", name, value)
        )
        self.filter_properties.property_changed.connect(
            lambda name, value: self.property_changed.emit("filter", name, value)
        )
        self.feature_properties.property_changed.connect(
            lambda name, value: self.property_changed.emit("feature", name, value)
        )
        
        # Add placeholder text when no properties are available
        self.placeholder = QLabel("No properties available")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.placeholder)
        
    def show_signal_properties(self, signal_type: str = None):
        """Show signal generation properties."""
        if signal_type:
            self.signal_properties.update_for_signal_type(signal_type)
        self.stack.setCurrentWidget(self.signal_properties)
        
    def show_filter_properties(self, filter_type: str = None):
        """Show filter properties."""
        if filter_type:
            self.filter_properties.update_for_filter_type(filter_type)
        self.stack.setCurrentWidget(self.filter_properties)
        
    def show_feature_properties(self, feature_type: str = None):
        """Show feature extraction properties."""
        if feature_type:
            self.feature_properties.update_for_feature_type(feature_type)
        self.stack.setCurrentWidget(self.feature_properties)
        
    def show_placeholder(self):
        """Show placeholder when no properties are available."""
        self.stack.setCurrentWidget(self.placeholder)
        
    def clear_properties(self):
        """Clear all property widgets."""
        self.signal_properties.clear_properties()
        self.filter_properties.clear_properties()
        self.feature_properties.clear_properties()