import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

from ui.panels.filter_designer_panel import FilterDesignerPanel

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication(sys.argv)

@pytest.fixture
def filter_panel(app):
    """Create a FilterDesignerPanel instance."""
    return FilterDesignerPanel()

def test_filter_type_selection(filter_panel):
    """Test filter type selection changes."""
    # Check default state
    assert filter_panel.filter_type.get_value() == "Bandpass Filter"
    assert filter_panel.bandpass_group.isVisible()
    assert not filter_panel.notch_group.isVisible()
    assert not filter_panel.wavelet_group.isVisible()
    
    # Test notch filter selection
    filter_panel.filter_type.set_value("Notch Filter")
    assert not filter_panel.bandpass_group.isVisible()
    assert filter_panel.notch_group.isVisible()
    assert not filter_panel.wavelet_group.isVisible()
    
    # Test wavelet filter selection
    filter_panel.filter_type.set_value("Wavelet Denoising")
    assert not filter_panel.bandpass_group.isVisible()
    assert not filter_panel.notch_group.isVisible()
    assert filter_panel.wavelet_group.isVisible()

def test_bandpass_parameters(filter_panel):
    """Test bandpass filter parameter changes."""
    filter_panel.filter_type.set_value("Bandpass Filter")
    
    # Test parameter ranges
    filter_panel.lowcut.set_value(10)
    assert filter_panel.lowcut.get_value() == 10
    
    filter_panel.highcut.set_value(200)
    assert filter_panel.highcut.get_value() == 200
    
    filter_panel.order.set_value(6)
    assert filter_panel.order.get_value() == 6
    
    # Test parameter validation
    filter_panel.lowcut.set_value(-1)  # Should clamp to min
    assert filter_panel.lowcut.get_value() == 0.1
    
    filter_panel.highcut.set_value(1000)  # Should clamp to max
    assert filter_panel.highcut.get_value() == 500

def test_notch_parameters(filter_panel):
    """Test notch filter parameter changes."""
    filter_panel.filter_type.set_value("Notch Filter")
    
    # Test parameter ranges
    filter_panel.center_freq.set_value(50)
    assert filter_panel.center_freq.get_value() == 50
    
    filter_panel.q_factor.set_value(45)
    assert filter_panel.q_factor.get_value() == 45

def test_wavelet_parameters(filter_panel):
    """Test wavelet denoising parameter changes."""
    filter_panel.filter_type.set_value("Wavelet Denoising")
    
    # Test wavelet type selection
    filter_panel.wavelet_type.set_value("db6")
    assert filter_panel.wavelet_type.get_value() == "db6"
    
    # Test decomposition level
    filter_panel.decomp_level.set_value(5)
    assert filter_panel.decomp_level.get_value() == 5

def test_reset_parameters(filter_panel):
    """Test parameter reset functionality."""
    # Test bandpass reset
    filter_panel.filter_type.set_value("Bandpass Filter")
    filter_panel.lowcut.set_value(100)
    filter_panel.highcut.set_value(300)
    filter_panel.order.set_value(8)
    
    filter_panel.reset_parameters()
    assert filter_panel.lowcut.get_value() == 20
    assert filter_panel.highcut.get_value() == 450
    assert filter_panel.order.get_value() == 4
    
    # Test notch reset
    filter_panel.filter_type.set_value("Notch Filter")
    filter_panel.center_freq.set_value(60)
    filter_panel.q_factor.set_value(50)
    
    filter_panel.reset_parameters()
    assert filter_panel.center_freq.get_value() == 50
    assert filter_panel.q_factor.get_value() == 30
    
    # Test wavelet reset
    filter_panel.filter_type.set_value("Wavelet Denoising")
    filter_panel.wavelet_type.set_value("sym4")
    filter_panel.decomp_level.set_value(7)
    
    filter_panel.reset_parameters()
    assert filter_panel.wavelet_type.get_value() == "db4"
    assert filter_panel.decomp_level.get_value() == 3

def test_get_filter_config(filter_panel):
    """Test filter configuration retrieval."""
    # Test bandpass config
    filter_panel.filter_type.set_value("Bandpass Filter")
    filter_panel.lowcut.set_value(20)
    filter_panel.highcut.set_value(450)
    filter_panel.order.set_value(4)
    
    config = filter_panel.get_filter_config()
    assert config['type'] == "Bandpass Filter"
    assert config['parameters']['lowcut'] == 20
    assert config['parameters']['highcut'] == 450
    assert config['parameters']['order'] == 4
    
    # Test notch config
    filter_panel.filter_type.set_value("Notch Filter")
    filter_panel.center_freq.set_value(50)
    filter_panel.q_factor.set_value(30)
    
    config = filter_panel.get_filter_config()
    assert config['type'] == "Notch Filter"
    assert config['parameters']['center_freq'] == 50
    assert config['parameters']['q_factor'] == 30
    
    # Test wavelet config
    filter_panel.filter_type.set_value("Wavelet Denoising")
    filter_panel.wavelet_type.set_value("db4")
    filter_panel.decomp_level.set_value(3)
    
    config = filter_panel.get_filter_config()
    assert config['type'] == "Wavelet Denoising"
    assert config['parameters']['wavelet_type'] == "db4"
    assert config['parameters']['decomp_level'] == 3