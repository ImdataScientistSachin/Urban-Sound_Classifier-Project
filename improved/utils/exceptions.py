from typing import Optional, Any, Dict, List

class UrbanSoundClassifierError(Exception):
    """
    Base exception class for all Urban Sound Classifier errors.
    
    Attributes:
        message (str): Error message
        details (Optional[Dict[str, Any]]): Additional error details
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if not self.details:
            return self.message
        
        details_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.message} [{details_str}]"

# Audio Processing Errors
class AudioProcessingError(UrbanSoundClassifierError):
    """Base class for audio processing errors"""
    pass

class AudioLoadError(AudioProcessingError):
    """Raised when an audio file cannot be loaded"""
    pass

class AudioFormatError(AudioProcessingError):
    """Raised when an audio file has an unsupported format"""
    pass

class AudioQualityError(AudioProcessingError):
    """Raised when an audio file has quality issues (e.g., too short, corrupted)"""
    pass

# Feature Extraction Errors
class FeatureExtractionError(UrbanSoundClassifierError):
    """Base class for feature extraction errors"""
    pass

class FeatureNormalizationError(FeatureExtractionError):
    """Raised when feature normalization fails"""
    pass

class InvalidFeatureError(FeatureExtractionError):
    """Raised when extracted features are invalid (e.g., NaN, Inf)"""
    pass

class FeatureShapeError(FeatureExtractionError):
    """Raised when features have unexpected shapes"""
    pass

# Model Errors
class ModelError(UrbanSoundClassifierError):
    """Base class for model-related errors"""
    pass

class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded"""
    pass

class ModelSaveError(ModelError):
    """Raised when a model cannot be saved"""
    pass

class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found"""
    pass

class ModelArchitectureError(ModelError):
    """Raised when there's an issue with the model architecture"""
    pass

# Training Errors
class TrainingError(UrbanSoundClassifierError):
    """Base class for training-related errors"""
    pass

class DatasetError(TrainingError):
    """Raised when there's an issue with the dataset"""
    pass

class ValidationError(TrainingError):
    """Raised when validation fails during training"""
    pass

class EarlyStoppingError(TrainingError):
    """Raised when training is stopped early due to issues"""
    pass

class MemoryError(TrainingError):
    """Raised when there's insufficient memory for training"""
    pass

# Configuration Errors
class ConfigError(UrbanSoundClassifierError):
    """Base class for configuration-related errors"""
    pass

class ConfigFileError(ConfigError):
    """Raised when a configuration file cannot be loaded or parsed"""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails"""
    pass

class MissingConfigError(ConfigError):
    """Raised when a required configuration parameter is missing"""
    pass

# Utility Functions
def format_exception_with_context(exception: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Format an exception with additional context information.
    
    Args:
        exception (Exception): The exception to format
        context (Optional[Dict[str, Any]]): Additional context information
        
    Returns:
        str: Formatted exception message with context
    """
    if isinstance(exception, UrbanSoundClassifierError) and exception.details:
        # Custom exception with details
        details = exception.details.copy()
        if context:
            details.update(context)
        return f"{exception.__class__.__name__}: {exception.message} [{', '.join(f'{k}={v}' for k, v in details.items())}]"
    elif context:
        # Standard exception with context
        return f"{exception.__class__.__name__}: {str(exception)} [{', '.join(f'{k}={v}' for k, v in context.items())}]"
    else:
        # Standard exception without context
        return f"{exception.__class__.__name__}: {str(exception)}"

def create_error_response(error: Exception, status_code: int = 500, request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.
    
    Args:
        error (Exception): The exception that occurred
        status_code (int): HTTP status code
        request_id (Optional[str]): Request ID for tracking
        
    Returns:
        Dict[str, Any]: Standardized error response dictionary
    """
    error_type = error.__class__.__name__
    
    # Extract details if it's our custom exception
    details = getattr(error, 'details', {}) if isinstance(error, UrbanSoundClassifierError) else {}
    
    response = {
        "error": {
            "type": error_type,
            "message": str(error),
            "status_code": status_code
        }
    }
    
    # Add details if available
    if details:
        response["error"]["details"] = details
        
    # Add request ID if available
    if request_id:
        response["error"]["request_id"] = request_id
        
    return response