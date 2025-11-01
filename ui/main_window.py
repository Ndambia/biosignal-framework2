from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QDockWidget, QMenuBar, QStatusBar,
    QWidget, QVBoxLayout, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QKeySequence, QAction

from .error_handling import ErrorHandler, ErrorSeverity, BiosignalException
from .feedback_manager import FeedbackManager
from .state_manager import StateManager
from .data_manager import DataManager

# Import tab widgets
from .tabs.ml_workflow import MLWorkflowTab

# Import dock widgets
from .docks.preset_dock import PresetDock
from .docks.property_dock import PropertyDock
from .docks.data_inspector_dock import DataInspectorDock
from .docks.quick_actions_dock import QuickActionsDock
from .docks.log_dock import LogDock
from .docks.console_dock import ConsoleDock
from .docks.progress_dock import ProgressDock
from .docks.batch_processing_dock import BatchProcessingDock

class MainWindow(QMainWindow):
    """Main application window with tab-based interface and dockable panels."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Biosignal Framework")
        self.resize(1200, 800)
        
        # Initialize managers
        self.error_handler = ErrorHandler()
        self.error_handler.error_occurred.connect(self._on_error)
        
        self.state_manager = StateManager()
        self.state_manager.state_changed.connect(self._on_state_changed)
        self.state_manager.state_restored.connect(self._on_state_restored)
        
        self.data_manager = DataManager()
        
        # Initialize UI
        self._create_menu_bar()
        self._create_status_bar()
        
        # Initialize feedback manager with status bar
        self.feedback_manager = FeedbackManager(self.statusBar())
        
        self._create_central_widget()
        self._create_docks()
        
        # Set up window state
        self.setDockNestingEnabled(True)
        self.settings = QSettings('BiosignalFramework', 'App')
        self._load_window_state()
        
        # Load saved application state
        self.state_manager.load_state()

    def _create_central_widget(self):
        """Create the central tab widget and initialize all tabs."""
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Create and add tabs
        self.tabs = {
            # 'generate': SignalGenerationTab(), # Removed as it no longer exists
            'ml': MLWorkflowTab(data_manager=self.data_manager,
                                error_handler=self.error_handler,
                                feedback_manager=self.feedback_manager)
        }
        
        # Add tabs in specific order
        # self.tab_widget.addTab(self.tabs['generate'], "Generate") # Removed as it no longer exists
        self.tab_widget.addTab(self.tabs['ml'], "ML")
        
        # Connect tab signals
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._redo)
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        # Add dock widget visibility toggles
        self.dock_visibility_actions = {}
        dock_names = {
            'presets': 'Presets',
            'properties': 'Properties',
            'inspector': 'Data Inspector',
            'actions': 'Quick Actions',
            'logs': 'Logs',
            'console': 'Console',
            'progress': 'Progress',
            'batch_processing': 'Batch Processing'
        }
        for dock_id, display_name in dock_names.items():
            action = QAction(display_name, self)
            action.setCheckable(True)
            action.setChecked(True)
            view_menu.addAction(action)
            self.dock_visibility_actions[dock_id] = action
        
        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        
        # Analysis menu
        analysis_menu = menu_bar.addMenu("&Analysis")
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")

    def _create_status_bar(self):
        """Create the status bar with progress indicator."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.feedback_manager = FeedbackManager(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.status_bar.showMessage("Ready")

    def _create_docks(self):
        """Create and set up all dockable panels."""
        # Create dock widgets
        self.docks = {
            'presets': PresetDock("Preset Library"),
            'properties': PropertyDock("Properties"),
            'inspector': DataInspectorDock("Data Inspector"),
            'actions': QuickActionsDock("Quick Actions"),
            'logs': LogDock("Processing Log"),
            'console': ConsoleDock("Error Console"),
            'progress': ProgressDock("Progress Tracker"),
            'batch_processing': BatchProcessingDock("Batch Processing")
        }
        
        # Set up dock locations and properties
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.docks['presets'])
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.docks['properties'])
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.docks['inspector'])
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.docks['actions'])
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docks['logs'])
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docks['console'])
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.docks['progress'])
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.docks['batch_processing'])
        
        # Set dock features
        for dock in self.docks.values():
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable |
                QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            
        # Connect dock visibility toggles
        for dock_id, action in self.dock_visibility_actions.items():
            dock = self.docks[dock_id]
            action.triggered.connect(lambda checked, d=dock: d.setVisible(checked))
            dock.visibilityChanged.connect(lambda visible, a=action: a.setChecked(visible))

    def _on_tab_changed(self, index: int):
        """Handle tab change events."""
        tab_name = self.tab_widget.tabText(index)
        self.status_bar.showMessage(f"Current tab: {tab_name}")
        
        # Update state
        self.state_manager.update_state('ui', {'current_tab': tab_name})
        
        # Update dock widgets based on current tab
        self._update_docks_for_tab(tab_name)

    def _update_docks_for_tab(self, tab_name: str):
        """Update dock widget contents based on current tab."""
        # Update property panel
        # Only ML tab exists now, so no specific dock updates needed based on tab name
        # If other tabs are added later, this logic will need to be re-evaluated.

    def _on_error(self, error_info):
        """Handle error signals from error handler."""
        self.feedback_manager.show_status_message(f"{error_info.severity.value}: {error_info.message}", 5000)
        self.feedback_manager.show_error_dialog(error_info)

    def _on_state_changed(self, state: dict):
        """Handle state changes."""
        try:
            self._apply_state(state)
        except Exception as e:
            self._handle_error(e)

    def _on_state_restored(self, state: dict):
        """Handle state restoration."""
        try:
            self._apply_state(state)
            self.feedback_manager.show_status_message("State restored successfully", 3000)
        except Exception as e:
            self._handle_error(e)

    def _apply_state(self, state: dict):
        """Apply a state to the UI."""
        try:
            # Apply UI state
            if 'ui' in state:
                ui_state = state['ui']
                if 'current_tab' in ui_state:
                    tab_index = self.tab_widget.findText(ui_state['current_tab'])
                    if tab_index >= 0:
                        self.tab_widget.setCurrentIndex(tab_index)
            
            # Apply dock visibility state
            if 'docks' in state:
                dock_state = state['docks']
                for dock_name, visible in dock_state.items():
                    if dock_name in self.docks:
                        self.docks[dock_name].setVisible(visible)
            
        except Exception as e:
            self._handle_error(e)

    def _save_state(self):
        """Save application state."""
        try:
            # Save window geometry and state
            self.settings.setValue("windowGeometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
            
            # Save dock visibility states
            dock_state = {name: dock.isVisible() for name, dock in self.docks.items()}
            self.state_manager.update_state('docks', dock_state)
            
        except Exception as e:
            self._handle_error(e)

    def _load_window_state(self):
        """Load saved window state."""
        try:
            if self.settings.value("windowGeometry"):
                self.restoreGeometry(self.settings.value("windowGeometry"))
            if self.settings.value("windowState"):
                self.restoreState(self.settings.value("windowState"))
        except Exception as e:
            self._handle_error(e)

    def _undo(self):
        """Handle undo action."""
        try:
            if state := self.state_manager.undo():
                self._apply_state(state)
                self.feedback_manager.show_status_message("Undo successful", 2000)
        except Exception as e:
            self._handle_error(e)

    def _redo(self):
        """Handle redo action."""
        try:
            if state := self.state_manager.redo():
                self._apply_state(state)
                self.feedback_manager.show_status_message("Redo successful", 2000)
        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error):
        """Handle errors."""
        if isinstance(error, str):
            error = BiosignalException(error)
        self.error_handler.handle_error(error)

    def closeEvent(self, event):
        """Handle application close."""
        try:
            self._save_state()
            event.accept()
        except Exception as e:
            self._handle_error(e)
            event.ignore()