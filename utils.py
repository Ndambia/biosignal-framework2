"""Utility helpers: metadata schema, validators, small CLI helpers.
"""
from __future__ import annotations
import json
from typing import Dict, Any
import os

DEFAULT_METADATA_SCHEMA = {
    'sample_rate': 'float',
    'channel_labels': 'list[str]',
    'channel_units': 'list[str]',
    'device_id': 'str',
    'device_model': 'str',
    'firmware_version': 'str',
    'subject_id': 'str',
    'session_id': 'str',
    'start_time_utc': 'str',
    'consent_version': 'str'
}

def validate_metadata(meta: Dict[str, Any]) -> bool:
    missing = [k for k in DEFAULT_METADATA_SCHEMA.keys() if k not in meta]
    if missing:
        raise ValueError(f"Missing metadata keys: {missing}")
    return True

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
