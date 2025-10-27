"""Utility functions for I/O operations, visualization, and helper functions.

This module provides a comprehensive set of utilities for:
1. I/O Operations - File handling, data persistence, config management
2. Visualization Tools - Time series, frequency spectra, and model performance plots
3. Helper Functions - Data validation, type checking, and progress tracking
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# I/O Operations
def detect_file_format(filepath: str) -> str:
    """Detect file format based on extension and content."""
    ext = Path(filepath).suffix.lower()
    format_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.npy': 'numpy',
        '.mat': 'matlab',
        '.h5': 'hdf5',
        '.hdf5': 'hdf5'
    }
    return format_map.get(ext, 'unknown')

def load_data(filepath: str, **kwargs) -> Any:
    """Load data from various file formats."""
    format_type = detect_file_format(filepath)
    try:
        if format_type == 'csv':
            return pd.read_csv(filepath, **kwargs)
        elif format_type == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format_type == 'yaml':
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        elif format_type == 'numpy':
            return np.load(filepath)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {str(e)}")
        raise

def save_data(data: Any, filepath: str, **kwargs) -> None:
    """Save data to various file formats."""
    format_type = detect_file_format(filepath)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if format_type == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, **kwargs)
            else:
                pd.DataFrame(data).to_csv(filepath, **kwargs)
        elif format_type == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type == 'yaml':
            with open(filepath, 'w') as f:
                yaml.safe_dump(data, f)
        elif format_type == 'numpy':
            np.save(filepath, data)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {str(e)}")
        raise

class Config:
    """Configuration management utility."""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        if config_path:
            self.load_config()

    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            format_type = detect_file_format(self.config_path)
            if format_type in ['yaml', 'json']:
                self.config = load_data(self.config_path)
            else:
                raise ValueError("Config file must be YAML or JSON")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def save_config(self) -> None:
        """Save configuration to file."""
        if self.config_path:
            save_data(self.config, self.config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

# Visualization Tools
def plot_time_series(
    data: Union[np.ndarray, pd.Series],
    sampling_rate: float,
    title: str = "Time Series Plot",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot time series data."""
    fig, ax = plt.subplots(figsize=figsize)
    time = np.arange(len(data)) / sampling_rate
    ax.plot(time, data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    return fig

def plot_spectrum(
    data: np.ndarray,
    sampling_rate: float,
    title: str = "Frequency Spectrum",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot frequency spectrum."""
    fig, ax = plt.subplots(figsize=figsize)
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/sampling_rate)
    ax.plot(freqs, np.abs(spectrum))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.grid(True)
    return fig

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    cm = pd.crosstab(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    return fig

# Helper Functions
def validate_data(data: np.ndarray, requirements: Dict[str, Any]) -> bool:
    """Validate data against requirements."""
    try:
        if requirements.get('shape'):
            assert data.shape == requirements['shape'], f"Expected shape {requirements['shape']}, got {data.shape}"
        if requirements.get('dtype'):
            assert data.dtype == requirements['dtype'], f"Expected dtype {requirements['dtype']}, got {data.dtype}"
        if requirements.get('range'):
            min_val, max_val = requirements['range']
            assert np.all((data >= min_val) & (data <= max_val)), "Data outside expected range"
        return True
    except AssertionError as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

def check_type(obj: Any, expected_type: Union[type, Tuple[type, ...]]) -> bool:
    """Check if object is of expected type."""
    return isinstance(obj, expected_type)

class ProgressTracker:
    """Track progress of long-running operations."""
    def __init__(self, total: int, desc: str = "Progress"):
        self.pbar = tqdm(total=total, desc=desc)
        self.start_time = datetime.now()

    def update(self, n: int = 1) -> None:
        """Update progress."""
        self.pbar.update(n)

    def close(self) -> None:
        """Close progress tracker."""
        self.pbar.close()
        duration = datetime.now() - self.start_time
        logger.info(f"Operation completed in {duration}")

def safe_operation(func: callable) -> callable:
    """Decorator for safe operation execution with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper