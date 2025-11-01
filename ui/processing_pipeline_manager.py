from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QProgressBar, QComboBox,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, QThread
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import time

from preprocessing_bio import SignalDenoising, SignalNormalization, SignalSegmentation
from .panels.filter_designer_panel import FilterDesignerPanel
from .panels.normalization_panel import NormalizationPanel
from .panels.segmentation_panel import SegmentationPanel

@dataclass
class PipelineStep:
    """Represents a step in the processing pipeline."""
    type: str
    config: Dict[str, Any]
    enabled: bool = True

@dataclass
class PipelineTemplate:
    """Processing pipeline template configuration."""
    name: str
    description: str
    steps: List[PipelineStep]
    
    def to_dict(self) -> dict:
        """Convert template to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineTemplate':
        """Create template from dictionary."""
        steps = [PipelineStep(**step) for step in data['steps']]
        return cls(
            name=data['name'],
            description=data['description'],
            steps=steps
        )

class ProcessingWorker(QThread):
    """Worker thread for pipeline processing."""
    
    progress = pyqtSignal(int)  # Progress percentage
    step_completed = pyqtSignal(str, object)  # Step name and result
    error = pyqtSignal(str)  # Error message
    finished = pyqtSignal()
    
    def __init__(self, pipeline: List[PipelineStep], data: np.ndarray):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.stop_flag = False
        
    def run(self):
        """Execute the processing pipeline."""
        try:
            result = self.data.copy()
            total_steps = len([step for step in self.pipeline if step.enabled])
            completed_steps = 0
            
            for step in self.pipeline:
                if self.stop_flag:
                    break
                    
                if not step.enabled:
                    continue
                    
                # Process step
                if step.type == "filter":
                    result = self._apply_filter(result, step.config)
                elif step.type == "normalize":
                    result = self._apply_normalization(result, step.config)
                elif step.type == "segment":
                    result = self._apply_segmentation(result, step.config)
                
                completed_steps += 1
                progress = int((completed_steps / total_steps) * 100)
                self.progress.emit(progress)
                self.step_completed.emit(step.type, result)
                
            if not self.stop_flag:
                self.finished.emit()
                
        except Exception as e:
            self.error.emit(str(e))
            
    def stop(self):
        """Stop processing."""
        self.stop_flag = True
        
    def _apply_filter(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Apply filtering step."""
        denoiser = SignalDenoising()
        
        if config['type'] == "Bandpass Filter":
            return denoiser.bandpass_filter(
                data,
                config['parameters']['lowcut'],
                config['parameters']['highcut'],
                1000,  # sampling rate (should be passed from signal)
                config['parameters']['order']
            )
        elif config['type'] == "Notch Filter":
            return denoiser.notch_filter(
                data,
                config['parameters']['center_freq'],
                1000,  # sampling rate
                config['parameters']['q_factor']
            )
        else:  # Wavelet
            return denoiser.wavelet_denoise(
                data,
                config['parameters']['wavelet_type'],
                config['parameters']['decomp_level']
            )
            
    def _apply_normalization(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Apply normalization step."""
        normalizer = SignalNormalization()
        
        if config['method'] == "Z-score":
            return normalizer.zscore_normalize(data)
        elif config['method'] == "Min-Max":
            return normalizer.minmax_scale(
                data,
                (config['parameters']['feature_min'],
                 config['parameters']['feature_max'])
            )
        else:  # Robust
            return normalizer.robust_scale(data)
            
    def _apply_segmentation(self, data: np.ndarray, config: dict) -> np.ndarray:
        """Apply segmentation step."""
        segmenter = SignalSegmentation()
        
        if config['method'] == "Fixed Window":
            return segmenter.fixed_window(
                data,
                config['parameters']['window_size']
            )
        elif config['method'] == "Overlapping Window":
            return segmenter.overlap_window(
                data,
                config['parameters']['window_size'],
                config['parameters']['overlap'] / 100
            )
        else:  # Event-based
            return segmenter.event_based_segment(
                data,
                config['events'],  # Should be loaded from file
                config['parameters']['pre_event'],
                config['parameters']['post_event']
            )

class ProcessingPipelineManager(QWidget):
    """Manager for the signal processing pipeline."""
    
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = []
        self.worker = None
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Create pipeline controls
        controls = QHBoxLayout()
        
        # Template selection
        self.template_combo = QComboBox()
        self.template_combo.addItem("Custom Pipeline")
        self.load_templates()  # Load saved templates
        controls.addWidget(QLabel("Template:"))
        controls.addWidget(self.template_combo)
        
        # Template management buttons
        self.save_btn = QPushButton("Save Template")
        self.load_btn = QPushButton("Load Template")
        controls.addWidget(self.save_btn)
        controls.addWidget(self.load_btn)
        
        layout.addLayout(controls)
        
        # Create pipeline view
        self.pipeline_group = QGroupBox("Processing Pipeline")
        pipeline_layout = QVBoxLayout(self.pipeline_group)
        
        # Add step buttons
        step_buttons = QHBoxLayout()
        self.add_filter_btn = QPushButton("Add Filter")
        self.add_norm_btn = QPushButton("Add Normalization")
        self.add_seg_btn = QPushButton("Add Segmentation")
        step_buttons.addWidget(self.add_filter_btn)
        step_buttons.addWidget(self.add_norm_btn)
        step_buttons.addWidget(self.add_seg_btn)
        pipeline_layout.addLayout(step_buttons)
        
        layout.addWidget(self.pipeline_group)
        
        # Create progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.time_label = QLabel("Time remaining: --:--")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.time_label)
        
        layout.addWidget(progress_group)
        
        # Create action buttons
        actions = QHBoxLayout()
        self.run_btn = QPushButton("Run Pipeline")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.stop_btn)
        layout.addLayout(actions)
        
        # Connect signals
        self.save_btn.clicked.connect(self._save_template)
        self.load_btn.clicked.connect(self._load_template)
        self.add_filter_btn.clicked.connect(lambda: self._add_step("filter"))
        self.add_norm_btn.clicked.connect(lambda: self._add_step("normalize"))
        self.add_seg_btn.clicked.connect(lambda: self._add_step("segment"))
        self.run_btn.clicked.connect(self.run_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        
    def load_templates(self):
        """Load saved pipeline templates."""
        try:
            with open('pipeline_templates.json', 'r') as f:
                templates = json.load(f)
                for template in templates:
                    self.template_combo.addItem(template['name'])
        except FileNotFoundError:
            pass
            
    def _save_template(self):
        """Save current pipeline as template."""
        name, ok = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline Template",
            "",
            "JSON Files (*.json)"
        )
        
        if ok and name:
            template = PipelineTemplate(
                name=name,
                description="Custom pipeline template",
                steps=self.pipeline
            )
            
            try:
                with open(name, 'w') as f:
                    json.dump(template.to_dict(), f, indent=2)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save template: {str(e)}")
                
    def _load_template(self):
        """Load pipeline template from file."""
        name, ok = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline Template",
            "",
            "JSON Files (*.json)"
        )
        
        if ok and name:
            try:
                with open(name, 'r') as f:
                    data = json.load(f)
                    template = PipelineTemplate.from_dict(data)
                    self.pipeline = template.steps
                    self._update_pipeline_view()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load template: {str(e)}")
                
    def _add_step(self, step_type: str):
        """Add a processing step to the pipeline."""
        if step_type == "filter":
            panel = FilterDesignerPanel()
            config = panel.get_filter_config()
        elif step_type == "normalize":
            panel = NormalizationPanel()
            config = panel.get_normalization_config()
        else:  # segment
            panel = SegmentationPanel()
            config = panel.get_segmentation_config()
            
        step = PipelineStep(type=step_type, config=config)
        self.pipeline.append(step)
        self._update_pipeline_view()
        
    def _update_pipeline_view(self):
        """Update the pipeline view in the UI."""
        # Clear existing view
        while self.pipeline_group.layout().count() > 1:  # Keep step buttons
            item = self.pipeline_group.layout().takeAt(1)
            if item.widget():
                item.widget().deleteLater()
                
        # Add step widgets
        for i, step in enumerate(self.pipeline):
            step_widget = QWidget()
            step_layout = QHBoxLayout(step_widget)
            
            # Step info
            step_layout.addWidget(QLabel(f"{i+1}. {step.type.title()}"))
            
            # Enable/disable checkbox
            enabled_btn = QPushButton("✓" if step.enabled else "×")
            enabled_btn.clicked.connect(lambda checked, s=step: self._toggle_step(s))
            step_layout.addWidget(enabled_btn)
            
            # Move buttons
            if i > 0:
                up_btn = QPushButton("↑")
                up_btn.clicked.connect(lambda _, idx=i: self._move_step(idx, idx-1))
                step_layout.addWidget(up_btn)
                
            if i < len(self.pipeline) - 1:
                down_btn = QPushButton("↓")
                down_btn.clicked.connect(lambda _, idx=i: self._move_step(idx, idx+1))
                step_layout.addWidget(down_btn)
                
            # Remove button
            remove_btn = QPushButton("🗑")
            remove_btn.clicked.connect(lambda _, idx=i: self._remove_step(idx))
            step_layout.addWidget(remove_btn)
            
            self.pipeline_group.layout().addWidget(step_widget)
            
    def _toggle_step(self, step: PipelineStep):
        """Toggle step enabled state."""
        step.enabled = not step.enabled
        self._update_pipeline_view()
        
    def _move_step(self, from_idx: int, to_idx: int):
        """Move step in pipeline."""
        if 0 <= to_idx < len(self.pipeline):
            self.pipeline[from_idx], self.pipeline[to_idx] = \
                self.pipeline[to_idx], self.pipeline[from_idx]
            self._update_pipeline_view()
            
    def _remove_step(self, idx: int):
        """Remove step from pipeline."""
        del self.pipeline[idx]
        self._update_pipeline_view()
        
    def run_pipeline(self, data: np.ndarray):
        """Run the processing pipeline on input data."""
        if not self.pipeline:
            QMessageBox.warning(self, "Error", "Pipeline is empty")
            return
            
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Error", "Pipeline is already running")
            return
            
        # Create and start worker
        self.worker = ProcessingWorker(self.pipeline, data)
        self.worker.progress.connect(self._update_progress)
        self.worker.step_completed.connect(self._step_completed)
        self.worker.error.connect(self._processing_error)
        self.worker.finished.connect(self._processing_finished)
        
        self.processing_started.emit()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Processing...")
        self.progress_bar.setValue(0)
        
        self.start_time = time.time()
        self.worker.start()
        
    def stop_pipeline(self):
        """Stop the processing pipeline."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.status_label.setText("Processing stopped")
            self.stop_btn.setEnabled(False)
            self.run_btn.setEnabled(True)
            
    def _update_progress(self, value: int):
        """Update progress bar and estimated time."""
        self.progress_bar.setValue(value)
        
        if value > 0:
            elapsed = time.time() - self.start_time
            total = elapsed * 100 / value
            remaining = total - elapsed
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            self.time_label.setText(f"Time remaining: {mins:02d}:{secs:02d}")
            
    def _step_completed(self, step_type: str, result: np.ndarray):
        """Handle step completion."""
        self.status_label.setText(f"Completed {step_type}")
        
    def _processing_error(self, error: str):
        """Handle processing error."""
        QMessageBox.critical(self, "Error", f"Processing failed: {error}")
        self.status_label.setText("Error")
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.processing_error.emit(error)
        
    def _processing_finished(self):
        """Handle pipeline completion."""
        self.status_label.setText("Processing complete")
        self.time_label.setText("Time remaining: --:--")
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.processing_finished.emit()