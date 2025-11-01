import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

from ui.panels.normalization_panel import NormalizationPanel

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication(sys.argv)

@pytest.fixture
def norm_panel(app):
    """Create a NormalizationPanel instance."""
    return NormalizationPanel()

def test_method_selection(norm_panel):
    """Test normalization method selection changes."""
    # Check default state
    assert norm_panel.norm_method.get_value() == "Z-score"
    assert norm_panel.zscore_group.isVisible()
    assert not norm_panel.minmax_group.isVisible()
    assert not norm_panel.robust_group.isVisible()
    
    # Test min-max selection
    norm_panel.norm_method.set_value("Min-Max")
    assert not norm_panel.zscore_group.isVisible()
    assert norm_panel.minmax_group.isVisible()
    assert not norm_panel.robust_group.isVisible()
    
    # Test robust selection
    norm_panel.norm_method.set_value("Robust")
    assert not norm_panel.zscore_group.isVisible()
    assert not norm_panel.minmax_group.isVisible()
    assert norm_panel.robust_group.isVisible()

def test_zscore_parameters(norm_panel):
    """Test z-score normalization parameters."""
    norm_panel.norm_method.set_value("Z-score")
    
    # Test robust option
    norm_panel.zscore_robust.set_value(True)
    assert norm_panel.zscore_robust.get_value() is True
    
    norm_panel.zscore_robust.set_value(False)
    assert norm_panel.zscore_robust.get_value() is False

def test_minmax_parameters(norm_panel):
    """Test min-max scaling parameters."""
    norm_panel.norm_method.set_value("Min-Max")
    
    # Test parameter ranges
    norm_panel.feature_min.set_value(-5)
    assert norm_panel.feature_min.get_value() == -5
    
    norm_panel.feature_max.set_value(5)
    assert norm_panel.feature_max.get_value() == 5
    
    # Test parameter validation
    norm_panel.feature_min.set_value(-15)  # Should clamp to min
    assert norm_panel.feature_min.get_value() == -10
    
    norm_panel.feature_max.set_value(15)  # Should clamp to max
    assert norm_panel.feature_max.get_value() == 10

def test_robust_parameters(norm_panel):
    """Test robust scaling parameters."""
    norm_panel.norm_method.set_value("Robust")
    
    # Test quantile range
    norm_panel.quantile_range.set_value(75)
    assert norm_panel.quantile_range.get_value() == 75
    
    # Test parameter validation
    norm_panel.quantile_range.set_value(-10)  # Should clamp to min
    assert norm_panel.quantile_range.get_value() == 0
    
    norm_panel.quantile_range.set_value(150)  # Should clamp to max
    assert norm_panel.quantile_range.get_value() == 100

def test_reset_parameters(norm_panel):
    """Test parameter reset functionality."""
    # Test z-score reset
    norm_panel.norm_method.set_value("Z-score")
    norm_panel.zscore_robust.set_value(True)
    
    norm_panel.reset_parameters()
    assert norm_panel.zscore_robust.get_value() is False
    
    # Test min-max reset
    norm_panel.norm_method.set_value("Min-Max")
    norm_panel.feature_min.set_value(-5)
    norm_panel.feature_max.set_value(5)
    
    norm_panel.reset_parameters()
    assert norm_panel.feature_min.get_value() == 0
    assert norm_panel.feature_max.get_value() == 1
    
    # Test robust reset
    norm_panel.norm_method.set_value("Robust")
    norm_panel.quantile_range.set_value(75)
    
    norm_panel.reset_parameters()
    assert norm_panel.quantile_range.get_value() == 50

def test_get_normalization_config(norm_panel):
    """Test normalization configuration retrieval."""
    # Test z-score config
    norm_panel.norm_method.set_value("Z-score")
    norm_panel.zscore_robust.set_value(True)
    
    config = norm_panel.get_normalization_config()
    assert config['method'] == "Z-score"
    assert config['parameters']['zscore_robust'] is True
    
    # Test min-max config
    norm_panel.norm_method.set_value("Min-Max")
    norm_panel.feature_min.set_value(-1)
    norm_panel.feature_max.set_value(1)
    
    config = norm_panel.get_normalization_config()
    assert config['method'] == "Min-Max"
    assert config['parameters']['feature_min'] == -1
    assert config['parameters']['feature_max'] == 1
    
    # Test robust config
    norm_panel.norm_method.set_value("Robust")
    norm_panel.quantile_range.set_value(75)
    
    config = norm_panel.get_normalization_config()
    assert config['method'] == "Robust"
    assert config['parameters']['quantile_range'] == 75

def test_statistics_update(norm_panel):
    """Test statistics display update."""
    # Create test data
    data = np.array([1, 2, 3, 4, 5])
    
    # Update statistics
    norm_panel.update_statistics(data)
    
    # Check statistics labels
    assert "Mean: 3.000" in norm_panel.mean_label.text()
    assert "Std Dev: 1.414" in norm_panel.std_label.text()
    assert "Min: 1.000" in norm_panel.min_label.text()
    assert "Max: 5.000" in norm_panel.max_label.text()
    assert "Median: 3.000" in norm_panel.median_label.text()
    assert "IQR: 2.000" in norm_panel.iqr_label.text()
    
    # Test with empty data
    norm_panel.update_statistics(np.array([]))
    # Should not raise any errors