import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow
from ui.theme import ThemeManager
from ui.data_manager import DataManager
from ui.error_handling import ErrorHandler
from ui.presets import PresetManager

class BiosignalFramework:
    """Main application class."""
    
    def __init__(self):
        # Create application
        self.app = QApplication(sys.argv)
        
        # Set up theme
        ThemeManager.set_theme("light")
        
        # Create managers
        self.error_handler = ErrorHandler()
        self.data_manager = DataManager()
        self.preset_manager = PresetManager(self.error_handler)
        
        # Create main window
        self.main_window = MainWindow()
        
        # Set up ML workflow tab
        self._setup_ml_workflow()
        
        # Set up batch processing tab
        self._setup_batch_processing()
        
        # Connect signals
        self._connect_signals()
        
    def _setup_ml_workflow(self):
        """Set up ML workflow tab."""
        pass  # To be implemented
        
    def _setup_data_management(self):
        """Set up data management tab."""
        pass  # To be implemented
        
    def _setup_analysis(self):
        """Set up analysis tab."""
        pass  # To be implemented
        
    def _setup_batch_processing(self):
        """Set up batch processing tab."""
        pass  # To be implemented
        
    def _connect_signals(self):
        """Connect signals between components."""
        pass  # To be implemented
        
    def run(self):
        """Run the application."""
        self.main_window.show()
        return self.app.exec()

def main():
    """Application entry point."""
    app = BiosignalFramework()
    sys.exit(app.run())

if __name__ == "__main__":
    main()