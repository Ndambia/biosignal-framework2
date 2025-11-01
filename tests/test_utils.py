import pytest
import os
import tempfile
from utils import validate_metadata, ensure_dir, DEFAULT_METADATA_SCHEMA

def test_metadata_validation_success():
    valid_metadata = {
        'sample_rate': 1000.0,
        'channel_labels': ['ch1', 'ch2'],
        'channel_units': ['uV', 'uV'],
        'device_id': 'dev123',
        'device_model': 'test_device',
        'firmware_version': '1.0.0',
        'subject_id': 'sub001',
        'session_id': 'sess001',
        'start_time_utc': '2025-01-01T00:00:00Z',
        'consent_version': '1.0'
    }
    
    assert validate_metadata(valid_metadata) is True

def test_metadata_validation_missing_fields():
    invalid_metadata = {
        'sample_rate': 1000.0,
        'channel_labels': ['ch1', 'ch2']
        # Missing required fields
    }
    
    with pytest.raises(ValueError) as exc_info:
        validate_metadata(invalid_metadata)
    
    error_msg = str(exc_info.value)
    assert 'Missing metadata keys' in error_msg
    assert 'channel_units' in error_msg
    assert 'device_id' in error_msg

def test_metadata_validation_empty():
    with pytest.raises(ValueError) as exc_info:
        validate_metadata({})
    
    error_msg = str(exc_info.value)
    assert 'Missing metadata keys' in error_msg
    assert all(key in error_msg for key in DEFAULT_METADATA_SCHEMA.keys())

def test_metadata_schema_types():
    # Verify schema types are correctly defined
    assert DEFAULT_METADATA_SCHEMA['sample_rate'] == 'float'
    assert DEFAULT_METADATA_SCHEMA['channel_labels'] == 'list[str]'
    assert DEFAULT_METADATA_SCHEMA['channel_units'] == 'list[str]'
    assert DEFAULT_METADATA_SCHEMA['device_id'] == 'str'
    assert DEFAULT_METADATA_SCHEMA['device_model'] == 'str'
    assert DEFAULT_METADATA_SCHEMA['firmware_version'] == 'str'
    assert DEFAULT_METADATA_SCHEMA['subject_id'] == 'str'
    assert DEFAULT_METADATA_SCHEMA['session_id'] == 'str'
    assert DEFAULT_METADATA_SCHEMA['start_time_utc'] == 'str'
    assert DEFAULT_METADATA_SCHEMA['consent_version'] == 'str'

def test_ensure_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test creating a single directory
        test_dir = os.path.join(tmpdir, 'test_dir')
        ensure_dir(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
        
        # Test creating nested directories
        nested_dir = os.path.join(tmpdir, 'parent/child/grandchild')
        ensure_dir(nested_dir)
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)
        
        # Test with existing directory (should not raise error)
        ensure_dir(test_dir)
        assert os.path.exists(test_dir)

def test_ensure_dir_with_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file
        file_path = os.path.join(tmpdir, 'test_file')
        with open(file_path, 'w') as f:
            f.write('test')
        
        # Attempt to create directory with same name as file
        with pytest.raises(Exception):
            ensure_dir(file_path)

def test_metadata_validation_extra_fields():
    metadata = {
        'sample_rate': 1000.0,
        'channel_labels': ['ch1', 'ch2'],
        'channel_units': ['uV', 'uV'],
        'device_id': 'dev123',
        'device_model': 'test_device',
        'firmware_version': '1.0.0',
        'subject_id': 'sub001',
        'session_id': 'sess001',
        'start_time_utc': '2025-01-01T00:00:00Z',
        'consent_version': '1.0',
        'extra_field': 'extra_value'  # Additional field not in schema
    }
    
    # Should pass validation since all required fields are present
    assert validate_metadata(metadata) is True