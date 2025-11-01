from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

class ThemeManager:
    """Manages application-wide theme settings"""
    
    @staticmethod
    def set_theme(theme: str = "light"):
        """Set application theme to either 'light' or 'dark'"""
        if theme.lower() == "dark":
            ThemeManager._set_dark_theme()
        else:
            ThemeManager._set_light_theme()
    
    @staticmethod
    def _set_dark_theme():
        """Apply dark theme palette"""
        app = QApplication.instance()
        palette = QPalette()
        
        # Set colors
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        app.setPalette(palette)
        
        # Set stylesheet for fine-tuning
        app.setStyleSheet("""
            QToolTip { 
                color: #ffffff; 
                background-color: #2a82da;
                border: 1px solid white;
            }
            QDockWidget {
                border: 1px solid #3d3d3d;
            }
            QComboBox {
                background-color: #353535;
                color: white;
                border: 1px solid #5c5c5c;
                padding: 4px;
            }
            QComboBox:drop-down {
                border: 0px;
            }
            QComboBox:down-arrow {
                image: url(none);
                border-width: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #353535;
                color: white;
                selection-background-color: #2a82da;
            }
        """)
    
    @staticmethod
    def _set_light_theme():
        """Apply light theme palette"""
        app = QApplication.instance()
        palette = QPalette()
        
        # Set colors
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        
        app.setPalette(palette)
        
        # Set stylesheet for fine-tuning
        app.setStyleSheet("""
            QToolTip { 
                color: #000000; 
                background-color: #ffffff;
                border: 1px solid #76797C;
            }
            QDockWidget {
                border: 1px solid #cccccc;
            }
            QComboBox {
                background-color: white;
                color: black;
                border: 1px solid #cccccc;
                padding: 4px;
            }
            QComboBox:drop-down {
                border: 0px;
            }
            QComboBox:down-arrow {
                image: url(none);
                border-width: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #76797C;
            }
        """)