import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from .error_handling import StateError, ErrorHandler

class StateManager(QObject):
    """Manages application state persistence and recovery"""
    
    state_changed = pyqtSignal(dict)
    state_restored = pyqtSignal(dict)
    
    def __init__(self, auto_save_interval: int = 60):
        """
        Initialize StateManager
        
        Args:
            auto_save_interval: Time between auto-saves in seconds
        """
        super().__init__()
        self.error_handler = ErrorHandler()
        self.state_file = "app_state.json"
        self.backup_dir = "state_backups"
        self.current_state: Dict[str, Any] = {}
        self.undo_stack: list = []
        self.redo_stack: list = []
        
        # Set up auto-save
        self.auto_save_timer = QTimer()
        self.auto_save_timer.setInterval(auto_save_interval * 1000)  # Convert to milliseconds
        self.auto_save_timer.timeout.connect(self.save_state)
        self.auto_save_timer.start()
        
        # Create backup directory if it doesn't exist
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            
    def update_state(self, category: str, data: Dict[str, Any]) -> None:
        """
        Update a category of application state
        
        Args:
            category: State category (e.g., 'parameters', 'visualization')
            data: New state data
        """
        try:
            # Save current state for undo
            self.undo_stack.append(self.current_state.copy())
            if len(self.undo_stack) > 20:  # Limit stack size
                self.undo_stack.pop(0)
            
            # Clear redo stack on new change
            self.redo_stack.clear()
            
            # Update state
            if category not in self.current_state:
                self.current_state[category] = {}
            self.current_state[category].update(data)
            
            # Emit change notification
            self.state_changed.emit(self.current_state)
            
        except Exception as e:
            self.error_handler.handle_error(StateError(f"Failed to update state: {str(e)}"))
            
    def save_state(self) -> None:
        """Save current state to file"""
        try:
            # Create backup of current state file if it exists
            if os.path.exists(self.state_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(
                    self.backup_dir, 
                    f"app_state_backup_{timestamp}.json"
                )
                os.rename(self.state_file, backup_file)
            
            # Save current state
            with open(self.state_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'state': self.current_state
                }, f, indent=2)
                
        except Exception as e:
            self.error_handler.handle_error(StateError(f"Failed to save state: {str(e)}"))
            
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load state from file"""
        try:
            if not os.path.exists(self.state_file):
                return None
                
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.current_state = data['state']
                self.state_restored.emit(self.current_state)
                return self.current_state
                
        except Exception as e:
            self.error_handler.handle_error(StateError(f"Failed to load state: {str(e)}"))
            return None
            
    def get_state(self, category: str) -> Optional[Dict[str, Any]]:
        """Get current state for a category"""
        return self.current_state.get(category)
        
    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last state change"""
        try:
            if not self.undo_stack:
                return None
                
            # Save current state for redo
            self.redo_stack.append(self.current_state.copy())
            
            # Restore previous state
            self.current_state = self.undo_stack.pop()
            self.state_changed.emit(self.current_state)
            return self.current_state
            
        except Exception as e:
            self.error_handler.handle_error(StateError(f"Failed to undo: {str(e)}"))
            return None
            
    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undone change"""
        try:
            if not self.redo_stack:
                return None
                
            # Save current state for undo
            self.undo_stack.append(self.current_state.copy())
            
            # Restore next state
            self.current_state = self.redo_stack.pop()
            self.state_changed.emit(self.current_state)
            return self.current_state
            
        except Exception as e:
            self.error_handler.handle_error(StateError(f"Failed to redo: {str(e)}"))
            return None
            
    def clear_state(self) -> None:
        """Clear all state data"""
        try:
            self.current_state = {}
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                
        except Exception as e:
            self.error_handler.handle_error(StateError(f"Failed to clear state: {str(e)}"))