import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List

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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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
                    "tonnetz": True
                }
            },
            "MODEL": {
                "architecture": "double_unet",
                "input_shape": [128, 173, 1],
                "filters_base": 32,
                "dropout_rate": 0.3,
                "l2_reg": 0.0001,
                "use_attention": True,
                "use_residual": True,
                "use_batch_norm": True
            },
            "TRAINING": {
                "k_folds": 5,
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 10,
                "reduce_lr_patience": 5,
                "min_lr": 1e-7,
                "use_class_weights": True,
                "use_mixup": True,
                "mixup_alpha": 0.2,
                "use_label_smoothing": True,
                "label_smoothing": 0.1
            },
            "AUGMENTATION": {
                "enabled": True,
                "noise": {
                    "enabled": True,
                    "factor": 0.01
                },
                "time_mask": {
                    "enabled": True,
                    "probability": 0.5,
                    "size": 20
                },
                "freq_mask": {
                    "enabled": True,
                    "probability": 0.5,
                    "size": 15
                },
                "pitch_shift": {
                    "enabled": True,
                    "steps": [-2, -1, 1, 2]
                },
                "time_stretch": {
                    "enabled": True,
                    "rates": [0.9, 1.1]
                }
            }
        }
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a file (JSON or YAML).
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If the configuration file does not exist
            ValueError: If the file format is not supported or the file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            with open(config_path, 'r') as f:
                if file_ext == '.json':
                    file_config = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            # Update the configuration with values from the file
            self._update_config_recursive(self.config, file_config)
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid configuration file: {e}")
    
    def _update_from_cli_args(self, cli_args: Dict[str, Any]) -> None:
        """
        Update configuration with command-line arguments.
        
        Args:
            cli_args: Dictionary of command-line arguments
        """
        # Flatten CLI args and update config
        flat_args = self._flatten_dict(cli_args)
        
        for key, value in flat_args.items():
            # Skip None values (not specified in CLI)
            if value is None:
                continue
                
            # Convert key to config path (e.g., "model.architecture" -> ["MODEL", "architecture"])
            path = key.split('.')
            
            # Convert to uppercase for top-level keys
            if len(path) > 1:
                path[0] = path[0].upper()
            
            # Update the config value
            self._set_nested_value(self.config, path, value)
            
        self.logger.info(f"Updated configuration with CLI arguments")
    
    def _update_config_recursive(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update target dictionary with values from source dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            # Convert key to uppercase for top-level keys
            config_key = key.upper() if key.lower() in [k.lower() for k in target.keys()] else key
            
            # Find the actual key in target (case-insensitive)
            actual_key = next((k for k in target.keys() if k.lower() == config_key.lower()), config_key)
            
            if isinstance(value, dict) and actual_key in target and isinstance(target[actual_key], dict):
                # Recursively update nested dictionaries
                self._update_config_recursive(target[actual_key], value)
            else:
                # Update value
                target[actual_key] = value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested dictionaries
            sep: Separator for keys
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)
    
    def _set_nested_value(self, d: Dict[str, Any], path: List[str], value: Any) -> None:
        """
        Set a value in a nested dictionary using a path.
        
        Args:
            d: Dictionary to update
            path: Path to the value
            value: Value to set
        """
        current = d
        
        # Navigate to the nested dictionary
        for i, key in enumerate(path[:-1]):
            # Find the actual key (case-insensitive)
            actual_key = next((k for k in current.keys() if k.lower() == key.lower()), key)
            
            # Create nested dictionaries if they don't exist
            if actual_key not in current:
                current[actual_key] = {}
            elif not isinstance(current[actual_key], dict):
                current[actual_key] = {}
                
            current = current[actual_key]
        
        # Set the value
        last_key = path[-1]
        actual_last_key = next((k for k in current.keys() if k.lower() == last_key.lower()), last_key)
        current[actual_last_key] = value
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Ensure required sections exist
        required_sections = ['SYSTEM', 'PATHS', 'AUDIO', 'FEATURES', 'MODEL', 'TRAINING']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific values
        if self.config['AUDIO']['sample_rate'] <= 0:
            raise ValueError(f"Invalid sample rate: {self.config['AUDIO']['sample_rate']}")
            
        if self.config['AUDIO']['duration'] <= 0:
            raise ValueError(f"Invalid audio duration: {self.config['AUDIO']['duration']}")
            
        if self.config['TRAINING']['batch_size'] <= 0:
            raise ValueError(f"Invalid batch size: {self.config['TRAINING']['batch_size']}")
            
        if self.config['TRAINING']['learning_rate'] <= 0:
            raise ValueError(f"Invalid learning rate: {self.config['TRAINING']['learning_rate']}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (e.g., 'AUDIO.sample_rate')
            default: Default value if the key does not exist
            
        Returns:
            Any: Configuration value or default
        """
        keys = key.split('.')
        
        # Convert first key to uppercase for top-level keys
        if keys[0].lower() in [k.lower() for k in self.config.keys()]:
            keys[0] = keys[0].upper()
        
        # Navigate through the nested dictionaries
        current = self.config
        for k in keys:
            # Find the actual key (case-insensitive)
            actual_key = next((ck for ck in current.keys() if ck.lower() == k.lower()), None)
            
            if actual_key is None:
                return default
                
            current = current[actual_key]
            
        return current
    
    def update(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        # Flatten the updates dictionary
        flat_updates = self._flatten_dict(config_updates)
        
        # Update each value
        for key, value in flat_updates.items():
            # Skip None values
            if value is None:
                continue
                
            # Convert key to config path
            path = key.split('.')
            
            # Convert to uppercase for top-level keys
            if len(path) > 1:
                path[0] = path[0].upper()
            
            # Update the config value
            self._set_nested_value(self.config, path, value)
            
        self.logger.info(f"Updated configuration with new values")
    
    def save(self, output_path: str, format: str = 'yaml') -> None:
        """
        Save the configuration to a file.
        
        Args:
            output_path: Path to save the configuration
            format: File format ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is not supported
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == 'json':
                    json.dump(self.config, f, indent=2)
                elif format.lower() in ['yaml', 'yml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                    
            self.logger.info(f"Saved configuration to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise