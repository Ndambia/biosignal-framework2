from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
from .error_handling import ValidationError

class ValidationType(Enum):
    REQUIRED = "required"
    RANGE = "range"
    REGEX = "regex"
    CUSTOM = "custom"
    OPTIONS = "options"

@dataclass
class ValidationRule:
    type: ValidationType
    message: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    options: Optional[List[Any]] = None
    custom_validator: Optional[callable] = None

class ParameterValidator:
    """Validates input parameters against defined rules"""
    
    def __init__(self):
        self.validation_rules: Dict[str, Dict[str, ValidationRule]] = {
            'signal': {
                'sampling_rate': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Sampling rate must be between 1 and 100000 Hz",
                    min_value=1,
                    max_value=100000
                ),
                'duration': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Duration must be between 0.1 and 3600 seconds",
                    min_value=0.1,
                    max_value=3600
                )
            },
            'emg': {
                'activation_level': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Activation level must be between 0 and 1",
                    min_value=0,
                    max_value=1
                ),
                'contraction_type': ValidationRule(
                    type=ValidationType.OPTIONS,
                    message="Contraction type must be either 'isometric' or 'dynamic'",
                    options=['isometric', 'dynamic']
                )
            },
            'ecg': {
                'heart_rate': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Heart rate must be between 20 and 250 BPM",
                    min_value=20,
                    max_value=250
                )
            },
            'eog': {
                'amplitude': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Amplitude must be between 0 and 5000 μV",
                    min_value=0,
                    max_value=5000
                ),
                'frequency': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Frequency must be between 0.1 and 100 Hz",
                    min_value=0.1,
                    max_value=100
                )
            },
            'noise': {
                'amplitude': ValidationRule(
                    type=ValidationType.RANGE,
                    message="Noise amplitude must be between 0 and 1",
                    min_value=0,
                    max_value=1
                ),
                'type': ValidationRule(
                    type=ValidationType.OPTIONS,
                    message="Noise type must be one of: gaussian, powerline, motion",
                    options=['gaussian', 'powerline', 'motion']
                )
            }
        }

    def validate_parameter(self, category: str, param_name: str, value: Any) -> None:
        """Validate a single parameter against its rules"""
        if category not in self.validation_rules:
            return
            
        if param_name not in self.validation_rules[category]:
            return
            
        rule = self.validation_rules[category][param_name]
        
        # Required check
        if rule.type == ValidationType.REQUIRED and value is None:
            raise ValidationError(rule.message)
            
        # Skip validation if value is None and not required
        if value is None:
            return
            
        # Range check
        if rule.type == ValidationType.RANGE:
            try:
                value_float = float(value)
                if rule.min_value is not None and value_float < rule.min_value:
                    raise ValidationError(rule.message)
                if rule.max_value is not None and value_float > rule.max_value:
                    raise ValidationError(rule.message)
            except (TypeError, ValueError):
                raise ValidationError(f"Invalid numeric value for {param_name}")
                
        # Options check
        if rule.type == ValidationType.OPTIONS:
            if rule.options and value not in rule.options:
                raise ValidationError(rule.message)
                
        # Custom validator
        if rule.type == ValidationType.CUSTOM and rule.custom_validator:
            if not rule.custom_validator(value):
                raise ValidationError(rule.message)

    def validate_parameters(self, category: str, parameters: Dict[str, Any]) -> None:
        """Validate multiple parameters for a category"""
        for param_name, value in parameters.items():
            self.validate_parameter(category, param_name, value)

    def add_validation_rule(self, category: str, param_name: str, rule: ValidationRule) -> None:
        """Add or update a validation rule"""
        if category not in self.validation_rules:
            self.validation_rules[category] = {}
        self.validation_rules[category][param_name] = rule

    def get_parameter_limits(self, category: str, param_name: str) -> Optional[Dict[str, float]]:
        """Get the min/max limits for a parameter if they exist"""
        if (category in self.validation_rules and 
            param_name in self.validation_rules[category] and
            self.validation_rules[category][param_name].type == ValidationType.RANGE):
            rule = self.validation_rules[category][param_name]
            return {
                'min': rule.min_value,
                'max': rule.max_value
            }
        return None

    def get_parameter_options(self, category: str, param_name: str) -> Optional[List[Any]]:
        """Get the valid options for a parameter if they exist"""
        if (category in self.validation_rules and 
            param_name in self.validation_rules[category] and
            self.validation_rules[category][param_name].type == ValidationType.OPTIONS):
            return self.validation_rules[category][param_name].options
        return None