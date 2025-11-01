import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import numpy as np

from ui.panels.segmentation_panel import SegmentationPanel

@pytest.fixture
def app():
    """Create a Qt application instance."""
    return QApplication(sys.argv)

@pytest.fixture
def seg_panel(app):
    """Create a SegmentationPanel instance."""
    return SegmentationPanel()

def test_method_selection(seg_panel):
    """Test segmentation method selection changes."""
    # Check default state
    assert seg_panel.seg_method.get_value() == "Fixed Window"
    assert seg_panel.fixed_group.isVisible()
    assert not seg_panel.overlap_group.isVisible()
    assert not seg_panel.event_group.isVisible()
    
    # Test overlapping window selection
    seg_panel.seg_method.set_value("Overlapping Window")
    assert not seg_panel.fixed_group.isVisible()
    assert seg_panel.overlap_group.isVisible()
    assert not seg_panel.event_group.isVisible()
    
    # Test event-based selection
    seg_panel.seg_method.set_value("Event-based")
    assert not seg_panel.fixed_group.isVisible()
    assert not seg_panel.overlap_group.isVisible()
    assert seg_panel.event_group.isVisible()

def test_fixed_window_parameters(seg_panel):
    """Test fixed window segmentation parameters."""
    seg_panel.seg_method.set_value("Fixed Window")
    
    # Test window size
    seg_panel.window_size.set_value(500)
    assert seg_panel.window_size.get_value() == 500
    
    # Test parameter validation
    seg_panel.window_size.set_value(50)  # Below min
    assert seg_panel.window_size.get_value() == 100
    
    seg_panel.window_size.set_value(15000)  # Above max
    assert seg_panel.window_size.get_value() == 10000

def test_overlap_parameters(seg_panel):
    """Test overlapping window parameters."""
    seg_panel.seg_method.set_value("Overlapping Window")
    
    # Test window size
    seg_panel.overlap_size.set_value(500)
    assert seg_panel.overlap_size.get_value() == 500
    
    # Test overlap percentage
    seg_panel.overlap_percent.set_value(25)
    assert seg_panel.overlap_percent.get_value() == 25
    
    # Test parameter validation
    seg_panel.overlap_percent.set_value(-10)  # Below min
    assert seg_panel.overlap_percent.get_value() == 0
    
    seg_panel.overlap_percent.set_value(100)  # Above max
    assert seg_panel.overlap_percent.get_value() == 90

def test_event_parameters(seg_panel):
    """Test event-based segmentation parameters."""
    seg_panel.seg_method.set_value("Event-based")
    
    # Test pre/post event windows
    seg_panel.pre_event.set_value(200)
    assert seg_panel.pre_event.get_value() == 200
    
    seg_panel.post_event.set_value(300)
    assert seg_panel.post_event.get_value() == 300
    
    # Test parameter validation
    seg_panel.pre_event.set_value(-100)  # Below min
    assert seg_panel.pre_event.get_value() == 0
    
    seg_panel.post_event.set_value(6000)  # Above max
    assert seg_panel.post_event.get_value() == 5000

def test_navigation_controls(seg_panel):
    """Test segment navigation controls."""
    # Initial state
    assert seg_panel.current_segment == 0
    assert seg_panel.total_segments == 0
    assert not seg_panel.prev_btn.isEnabled()
    assert not seg_panel.next_btn.isEnabled()
    
    # Simulate having multiple segments
    seg_panel.seg_method.set_value("Fixed Window")
    seg_panel.window_size.set_value(100)
    
    # Update with sample data (1000 points)
    data = np.zeros(1000)
    seg_panel.update_signal(data)
    
    # Should have 10 segments of size 100
    assert seg_panel.total_segments == 10
    assert not seg_panel.prev_btn.isEnabled()  # At first segment
    assert seg_panel.next_btn.isEnabled()
    
    # Navigate forward
    seg_panel.next_btn.click()
    assert seg_panel.current_segment == 1
    assert seg_panel.prev_btn.isEnabled()
    assert seg_panel.next_btn.isEnabled()
    
    # Navigate to last segment
    for _ in range(8):
        seg_panel.next_btn.click()
    assert seg_panel.current_segment == 9
    assert seg_panel.prev_btn.isEnabled()
    assert not seg_panel.next_btn.isEnabled()
    
    # Navigate back
    seg_panel.prev_btn.click()
    assert seg_panel.current_segment == 8
    assert seg_panel.prev_btn.isEnabled()
    assert seg_panel.next_btn.isEnabled()

def test_reset_parameters(seg_panel):
    """Test parameter reset functionality."""
    # Test fixed window reset
    seg_panel.seg_method.set_value("Fixed Window")
    seg_panel.window_size.set_value(500)
    
    seg_panel.reset_parameters()
    assert seg_panel.window_size.get_value() == 1000
    
    # Test overlapping window reset
    seg_panel.seg_method.set_value("Overlapping Window")
    seg_panel.overlap_size.set_value(500)
    seg_panel.overlap_percent.set_value(25)
    
    seg_panel.reset_parameters()
    assert seg_panel.overlap_size.get_value() == 1000
    assert seg_panel.overlap_percent.get_value() == 50
    
    # Test event-based reset
    seg_panel.seg_method.set_value("Event-based")
    seg_panel.pre_event.set_value(200)
    seg_panel.post_event.set_value(300)
    
    seg_panel.reset_parameters()
    assert seg_panel.pre_event.get_value() == 500
    assert seg_panel.post_event.get_value() == 500

def test_get_segmentation_config(seg_panel):
    """Test segmentation configuration retrieval."""
    # Test fixed window config
    seg_panel.seg_method.set_value("Fixed Window")
    seg_panel.window_size.set_value(1000)
    
    config = seg_panel.get_segmentation_config()
    assert config['method'] == "Fixed Window"
    assert config['parameters']['window_size'] == 1000
    
    # Test overlapping window config
    seg_panel.seg_method.set_value("Overlapping Window")
    seg_panel.overlap_size.set_value(1000)
    seg_panel.overlap_percent.set_value(50)
    
    config = seg_panel.get_segmentation_config()
    assert config['method'] == "Overlapping Window"
    assert config['parameters']['window_size'] == 1000
    assert config['parameters']['overlap'] == 50
    
    # Test event-based config
    seg_panel.seg_method.set_value("Event-based")
    seg_panel.pre_event.set_value(500)
    seg_panel.post_event.set_value(500)
    
    config = seg_panel.get_segmentation_config()
    assert config['method'] == "Event-based"
    assert config['parameters']['pre_event'] == 500
    assert config['parameters']['post_event'] == 500

def test_signal_update(seg_panel):
    """Test signal data update."""
    # Test with valid data
    data = np.sin(np.linspace(0, 10, 1000))
    seg_panel.update_signal(data)
    
    # Test with empty data
    seg_panel.update_signal(np.array([]))
    # Should not raise any errors
    
    # Test with None
    seg_panel.update_signal(None)
    # Should not raise any errors