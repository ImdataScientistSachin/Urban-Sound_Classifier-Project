import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union

class ConfigManager:
    """
    Configuration Manager for Urban Sound Classifier.
    
    This class handles loading, validating, and providing access to configuration
    parameters from various sources (default values, configuration files, command-line arguments).
    
    Attributes:
        config (Dict[str, Any]): The complete configuration dictionary
        config_path (Optional[str]): Path to the configuration file if loaded from file
    """
    
    def __init__(self, config_path: Optional[str] = None, cli_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConfigManager with optional configuration file and CLI arguments.
        
        Args:
            config_path (Optional[str]): Path to a configuration file (JSON or YAML)
            cli_args (Optional[Dict[str, Any]]): Command-line arguments that override file settings
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_default_config()
        
        # Load from file if provided
        if config_path:
            self._load_from_file(config_path)
            
        # Override with CLI arguments if provided
        if cli_args:
            self._update_from_cli_args(cli_args)
            
        # Validate the final configuration
        self._validate_config()
        
        self.logger.info(f"Configuration loaded successfully")
        
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration values.
        
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        # Base directory for relative paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        return {
            "SYSTEM": {
                "seed": 42,
                "num_classes": 10,
                "verbose": True,
                "debug": False,
                "log_level": "INFO",
                "memory_optimization": True,
                "use_mixed_precision": True,
                "num_threads": 4
            },
            "PATHS": {
                "base_dir": base_dir,
                "dataset_root": os.path.join(base_dir, "data", "UrbanSound8K"),
                "metadata_file": os.path.join(base_dir, "data", "UrbanSound8K", "UrbanSound8K.csv"),
                "output_dir": os.path.join(base_dir, "output"),
                "models_dir": os.path.join(base_dir, "models"),
                "logs_dir": os.path.join(base_dir, "logs"),
                "cache_dir": os.path.join(base_dir, "cache")
            },
            "AUDIO": {
                "sample_rate": 22050,
                "duration": 4.0,
                "mono": True,
                "normalize_audio": True,
                "remove_dc_offset": True,
                "high_pass_filter": True,
                "high_pass_cutoff": 20.0
            },
            "FEATURES": {
                "mel_spectrogram": {
                    "enabled": True,
                    "n_mels": 128,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "multi_resolution": True
                },
                "mfcc": {
                    "enabled": True,
                    "n_mfcc": 40,
                    "include_deltas": True
                },
                "harmonic_percussive": {
                    "enabled": True,
                    "margin": 1.0
                },
                "tonal": {
                    "enabled": True,
                    "chroma": True,
                    "tonnetz": True,
                    "spectral_contrast": True
                },
                "temporal": {
                    "enabled": True,
                    "zero_crossing_rate": True
                },
                "statistical": {
                    "enabled": True,
                    "time_domain": True,
                    "energy": True,
                    "zero_crossing": True
                },
                "normalization": {
                    "method": "robust",  # Options: "standard", "robust", "minmax"
                    "feature_wise": True,
                    "clip_outliers": True,
                    "clip_percentile": 99.5
                }
            },
            "TRAINING": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "early_stopping": True,
                "patience": 15,
                "reduce_lr_on_plateau": True,
                "lr_patience": 5,
                "lr_factor": 0.5,
                "min_lr": 1e-6,
                "n_folds": 5,
                "validation_split": 0.2,
                "class_weight": "balanced",
                "label_smoothing": 0.1,
                "mixup": {
                    "enabled": True,
                    "alpha": 0.2
                },
                "augmentation": {
                    "enabled": True,
                    "time_stretch": True,
                    "pitch_shift": True,
                    "add_noise": True,
                    "time_mask": True,
                    "freq_mask": True
                }
            },
            "MODELS": {
                "enhanced_unet": {
                    "enabled": True,
                    "filters_base": 32,
                    "dropout_rate": 0.3,
                    "l2_reg": 1e-5,
                    "attention": True
                },
                "efficientnet_hybrid": {
                    "enabled": True,
                    "freeze_backbone": False,
                    "custom_head": True,
                    "dropout_rate": 0.5
                },
                "transformer_cnn": {
                    "enabled": True,
                    "num_heads": 4,
                    "d_model": 128,
                    "num_transformer_blocks": 2,
                    "dropout_rate": 0.3
                }
            },
            "ENSEMBLE": {
                "models": ["enhanced_unet", "efficientnet_hybrid", "transformer_cnn"],
                "weight_optimization": True,
                "stacking": True
            },
            "HYPEROPT": {
                "enabled": False,
                "n_trials": 20,
                "timeout": 3600,  # 1 hour
                "best_params": None
            },
            "MONITORING": {
                "save_best_models": True,
                "save_checkpoints": True,
                "tensorboard": True,
                "plot_training_curves": True,
                "plot_confusion_matrix": True,
                "plot_class_distribution": True
            },
            "CLASS_NAMES": [
                "air_conditioner",
                "car_horn",
                "children_playing",
                "dog_bark",
                "drilling",
                "engine_idling",
                "gun_shot",
                "jackhammer",
                "siren",
                "street_music"
            ]
        }
    
    def _load_from_file(self, path: str) -> None:
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            path (str): Path to the configuration file
            
        Raises:
            ValueError: If the file format is not supported or file cannot be read
            FileNotFoundError: If the configuration file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        try:
            with open(path, 'r') as f:
                if path.endswith('.json'):
                    user_config = json.load(f)
                elif path.endswith('.yaml') or path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {path}. Use .json, .yaml, or .yml")
                
            # Update config with user values
            self._update_config(user_config)
            self.logger.info(f"Configuration loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration from {path}: {str(e)}")
            raise
    
    def _update_config(self, user_config: Dict[str, Any], prefix: str = "") -> None:
        """
        Recursively update configuration with user values.
        
        Args:
            user_config (Dict[str, Any]): User configuration dictionary
            prefix (str): Current key prefix for nested dictionaries
        """
        for key, value in user_config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # If this is a nested dictionary and the key exists in the current config
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                self._update_config(value, key)
            else:
                # Validate the value before updating
                try:
                    self._validate_config_value(full_key, value)
                    self.config[key] = value
                except ValueError as e:
                    self.logger.warning(f"Invalid configuration value for {full_key}: {str(e)}. Using default.")
    
    def _update_from_cli_args(self, cli_args: Dict[str, Any]) -> None:
        """
        Update configuration with command-line arguments.
        
        Args:
            cli_args (Dict[str, Any]): Command-line arguments dictionary
        """
        for key, value in cli_args.items():
            # Convert CLI arg format (e.g., "audio.sample_rate") to nested dict updates
            if '.' in key:
                parts = key.split('.')
                current = self.config
                
                # Navigate to the correct nested dictionary
                for part in parts[:-1]:
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
                
                # Set the value in the deepest level
                try:
                    self._validate_config_value(key, value)
                    current[parts[-1]] = value
                except ValueError as e:
                    self.logger.warning(f"Invalid CLI value for {key}: {str(e)}. Using default.")
            else:
                # Top-level key
                try:
                    self._validate_config_value(key, value)
                    self.config[key] = value
                except ValueError as e:
                    self.logger.warning(f"Invalid CLI value for {key}: {str(e)}. Using default.")
    
    def _validate_config_value(self, key: str, value: Any) -> None:
        """
        Validate a specific configuration value.
        
        Args:
            key (str): Configuration key
            value (Any): Configuration value
            
        Raises:
            ValueError: If the value is invalid for the given key
        """
        # Example validations - extend as needed
        if key == "SYSTEM.seed" and not isinstance(value, int):
            raise ValueError(f"Seed must be an integer, got {type(value).__name__}")
            
        if key == "AUDIO.sample_rate" and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"Sample rate must be a positive integer, got {value}")
            
        if key == "AUDIO.duration" and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"Duration must be a positive number, got {value}")
            
        if key == "TRAINING.batch_size" and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"Batch size must be a positive integer, got {value}")
            
        if key == "TRAINING.learning_rate" and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"Learning rate must be a positive number, got {value}")
    
    def _validate_config(self) -> None:
        """
        Validate the entire configuration for consistency and required values.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check for required top-level sections
        required_sections = ["SYSTEM", "PATHS", "AUDIO", "FEATURES", "TRAINING", "MODELS"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate paths exist or can be created
        for key, path in self.config["PATHS"].items():
            if key in ["dataset_root", "metadata_file"] and not os.path.exists(path):
                self.logger.warning(f"Dataset path does not exist: {path}")
            
            # Ensure output directories can be created
            if key in ["output_dir", "models_dir", "logs_dir", "cache_dir"]:
                os.makedirs(path, exist_ok=True)
        
        # Ensure at least one model is enabled
        if not any(model_config.get("enabled", False) for model_config in self.config["MODELS"].values()):
            raise ValueError("At least one model must be enabled")
        
        # Ensure ensemble models exist if ensemble is used
        if self.config["ENSEMBLE"]["models"]:
            for model_name in self.config["ENSEMBLE"]["models"]:
                if model_name not in self.config["MODELS"] or not self.config["MODELS"][model_name].get("enabled", False):
                    self.logger.warning(f"Ensemble references disabled or non-existent model: {model_name}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with optional default.
        
        Args:
            key (str): Configuration key (can use dot notation for nested keys)
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value or default
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested keys
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key (str): Configuration key (can use dot notation for nested keys)
            value (Any): Value to set
            
        Raises:
            ValueError: If the key format is invalid or value is invalid
        """
        try:
            self._validate_config_value(key, value)
        except ValueError as e:
            self.logger.warning(f"Invalid value for {key}: {str(e)}. Not updating.")
            return
        
        if '.' not in key:
            self.config[key] = value
            return
        
        # Handle nested keys
        parts = key.split('.')
        current = self.config
        
        # Navigate to the correct nested dictionary
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        # Set the value in the deepest level
        current[parts[-1]] = value
    
    def save_to_file(self, path: str, format: str = 'json') -> None:
        """
        Save the current configuration to a file.
        
        Args:
            path (str): Path to save the configuration file
            format (str): Format to save as ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is not supported
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(self.config, f, indent=4)
                elif format.lower() in ['yaml', 'yml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
                    
            self.logger.info(f"Configuration saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {path}: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """
        Return a string representation of the configuration.
        
        Returns:
            str: String representation of the configuration
        """
        return json.dumps(self.config, indent=4)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the ConfigManager instance.
        
        Returns:
            str: String representation of the ConfigManager
        """
        return f"ConfigManager(config_path={self.config_path})"