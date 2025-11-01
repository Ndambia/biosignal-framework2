from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

class BaseDock(QDockWidget):
    """Base class for all dockable panels."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        
        # Set dock features
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)
        
        self.setWidget(self.main_widget)
        
    def add_widget(self, widget: QWidget):
        """Add a widget to the dock's layout."""
        self.main_layout.addWidget(widget)
        
    def clear_layout(self):
        """Remove all widgets from the layout."""
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()