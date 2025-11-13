import os
import time
import numpy as np
import tensorflow as tf
import logging
import threading
from typing import List, Dict, Any, Optional, Union, Tuple

from ..config.config_manager import ConfigManager
from ..feature_extraction import FeatureExtractor, MelSpectrogramExtractor, MFCCExtractor
from .loaders.model_loader import ModelLoader
from ..utils.audio import AudioUtils

class Predictor:
    """
    Thread-safe class for making predictions using trained models.
    
    This class handles the prediction pipeline, including audio preprocessing,
    feature extraction, model inference, and post-processing of results.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        model_loader (ModelLoader): Model loader instance
        feature_extractor (FeatureExtractor): Feature extractor instance
        class_labels (List[str]): List of class labels
        models (List[tf.keras.Model]): List of loaded models
        _lock (threading.RLock): Reentrant lock for thread safety
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the Predictor.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model loader
        self.model_loader = ModelLoader(config)
        
        # Initialize feature extractor based on config
        feature_type = config.get('FEATURES.type', 'mel_spectrogram')
        if feature_type == 'mel_spectrogram':
            self.feature_extractor = MelSpectrogramExtractor(config)
        elif feature_type == 'mfcc':
            self.feature_extractor = MFCCExtractor(config)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Load class labels
        self.class_labels = self._load_class_labels()
        
        # Initialize models list
        self.models = []
        
        # Initialize lock for thread safety
        self._lock = threading.RLock()
    
    def _load_class_labels(self) -> List[str]:
        """
        Load class labels from the configuration or a file.
        
        Returns:
            List[str]: List of class labels
        """
        # Try to get class labels from config
        class_labels = self.config.get('MODEL.class_labels', None)
        
        if class_labels is not None:
            return class_labels
        
        # If not in config, try to load from a file
        labels_file = self.config.get('PATHS.class_labels_file', None)
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                class_labels = [line.strip() for line in f.readlines()]
            return class_labels
        
        # Default urban sound class labels if nothing else is available
        default_labels = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
        ]
        
        self.logger.warning("No class labels found in config or file. Using default urban sound labels.")
        return default_labels
    
    def load_models(self, model_paths: List[str] = None) -> None:
        """
        Load models for prediction in a thread-safe manner.
        
        Args:
            model_paths (List[str], optional): List of paths to model files.
                If None, will attempt to load all models from the models directory.
        """
        with self._lock:
            self.models = self.model_loader.load_ensemble(model_paths)
            self.logger.info(f"Loaded {len(self.models)} models for prediction")
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file for prediction.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Preprocessed audio data
            
        Raises:
            FileNotFoundError: If the audio file does not exist
            ValueError: If the audio file is empty or cannot be processed
            TypeError: If the input is not a valid audio file
        """
        self.logger.debug(f"Starting audio preprocessing for: {audio_path}")
        
        # Validate input
        if not isinstance(audio_path, str):
            self.logger.error(f"Expected string path, got {type(audio_path)}")
            raise TypeError(f"Audio path must be a string, got {type(audio_path)}")
            
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        if os.path.getsize(audio_path) == 0:
            self.logger.error(f"Audio file is empty: {audio_path}")
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        # Convert audio to WAV format if needed
        sample_rate = self.config.get('AUDIO.sample_rate', 22050)
        temp_path = audio_path
        temp_created = False
        
        try:
            if not audio_path.lower().endswith('.wav'):
                self.logger.debug(f"Converting audio to WAV format: {audio_path}")
                temp_dir = self.config.get('PATHS.temp_dir', '/tmp')
                os.makedirs(temp_dir, exist_ok=True)
                
                temp_path = os.path.join(
                    temp_dir,
                    f"temp_{int(time.time())}_{os.path.basename(audio_path)}.wav"
                )
                
                try:
                    temp_path = AudioUtils.convert_audio_to_wav(audio_path, output_path=temp_path, sample_rate=sample_rate)
                    temp_created = True
                    self.logger.debug(f"Successfully converted to WAV: {temp_path}")
                except Exception as e:
                    self.logger.error(f"Error converting audio to WAV: {e}")
                    raise ValueError(f"Failed to convert audio to WAV format: {e}")
            
            # Extract features
            self.logger.debug("Extracting features from preprocessed audio")
            try:
                features = self.feature_extractor.extract_features_from_file(temp_path)
                if isinstance(features, dict):
                    self.logger.debug(f"Features extracted successfully with keys: {list(features.keys())}")
                else:
                    self.logger.debug(f"Features extracted successfully with shape: {features.shape}")
            except Exception as e:
                self.logger.error(f"Error extracting features: {e}")
                raise ValueError(f"Failed to extract features from audio: {e}")
            
            # Validate features
            if features is None:
                self.logger.error("Extracted features are None")
                raise ValueError("Extracted features are None")
            
            # Handle dictionary features
            if isinstance(features, dict):
                if not features:  # Empty dictionary
                    self.logger.error("Extracted features dictionary is empty")
                    raise ValueError("Extracted features dictionary is empty")
                # We'll return the dictionary and handle it in the predict method
                return features
            
            # For numpy array features
            if features.size == 0:
                self.logger.error("Extracted features array is empty")
                raise ValueError("Extracted features array is empty")
            
            # Handle scalar values (including numpy scalar types)
            if np.isscalar(features) or (hasattr(features, 'ndim') and features.ndim == 0):
                self.logger.debug(f"Converting scalar feature value {features} (type: {type(features)}) to numpy array")
                # For numpy scalar types, we need to explicitly create a new array
                if isinstance(features, np.number):
                    # Create a new array with the scalar value
                    features = np.array([float(features)], dtype=np.float32)
                else:
                    # For other scalar types
                    features = np.atleast_1d(features).astype(np.float32)
                
            if np.isnan(features).any() or np.isinf(features).any():
                self.logger.warning("Features contain NaN or Inf values, replacing with zeros")
                features = np.nan_to_num(features)
            
            # Reshape features for model input if needed
            if len(self.models) > 0:
                # Get input shape from the first model
                model = self.models[0]
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                    # If input_shape is a list/tuple of shapes (multiple inputs), take the first one
                    if isinstance(input_shape, list) or (isinstance(input_shape, tuple) and 
                                                      isinstance(input_shape[0], tuple)):
                        input_shape = input_shape[0]
                    
                    # Remove batch dimension (None) if present
                    if input_shape and input_shape[0] is None:
                        input_shape = input_shape[1:]
                    
                    # Reshape features to match model input shape
                    if input_shape and features.shape != input_shape:
                        self.logger.debug(f"Reshaping features from {features.shape} to match model input shape {input_shape}")
                        # Add batch dimension if needed
                        if len(features.shape) == len(input_shape) - 1:
                            features = np.expand_dims(features, axis=0)
                        # Reshape to match expected dimensions
                        elif len(features.shape) == len(input_shape):
                            features = np.reshape(features, (-1,) + input_shape[1:])
            
            return features
            
        finally:
            # Clean up temporary file if created
            if temp_created and temp_path != audio_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.logger.debug(f"Removed temporary file: {temp_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

    
    def _make_predictions_with_models(self, features: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Make predictions using the loaded models with the provided features.
        
        Args:
            features (np.ndarray): Preprocessed feature array ready for model input
            top_k (int): Number of top predictions to return
            
        Returns:
            List[Dict[str, Any]]: List of top predictions with class labels and probabilities
        """
        # Check if features need reshaping before prediction
        if hasattr(features, 'shape') and (features.shape == (32, 1) or 
                                          (isinstance(features, tf.Tensor) and features.shape == (32, 1))):
            try:
                # Get the expected shape for CNN models
                target_shape = (1, 128, 173, 1)
                self.logger.info(f"Detected problematic (32, 1) shape, reshaping to {target_shape}")
                
                # Convert to numpy array for easier manipulation if it's a TensorFlow tensor
                if isinstance(features, tf.Tensor):
                    features_np = features.numpy()
                else:
                    features_np = features
                
                # Create a new array with the expected shape
                new_features = np.zeros(target_shape)
                
                # Distribute the 32 values across the spectrogram in a more meaningful way
                # Create a pattern that repeats the frequency values across time
                for i in range(min(32, features_np.shape[0])):
                    freq_value = features_np[i, 0] if features_np.shape[1] > 0 else 0
                    # Repeat this value across all time steps for this frequency bin
                    new_features[0, i, :, 0] = freq_value
                
                # Fill the remaining frequency bins with zeros or patterns if needed
                if features_np.shape[0] < target_shape[1]:
                    # Optionally, you could repeat the pattern or interpolate
                    pass
                
                # Convert back to the appropriate type
                if isinstance(features, tf.Tensor):
                    features = tf.convert_to_tensor(new_features, dtype=tf.float32)
                else:
                    features = new_features
                    
                self.logger.info(f"Successfully reshaped features to {features.shape}")
            except Exception as e:
                self.logger.error(f"Error reshaping features: {e}", exc_info=True)
                # Fall back to default error prediction
                return [{'label': 'Error', 'probability': 0.0}]
        
        # Make predictions with each model
        all_predictions = []
        
        # Get a thread-safe copy of the models
        with self._lock:
            models_copy = self.models.copy()
        
        for i, model in enumerate(models_copy):
            try:
                self.logger.info(f"Making prediction with model {i+1}/{len(models_copy)}")
                # Make prediction
                if hasattr(model, 'predict'):
                    self.logger.info(f"Using model.predict() method")
                    preds = model.predict(features)
                else:
                    # For custom model wrappers like TFLite models
                    self.logger.info(f"Using model as callable")
                    preds = model(features)
                
                self.logger.info(f"Prediction shape from model {i+1}: {preds.shape}")
                all_predictions.append(preds)
            except Exception as e:
                self.logger.error(f"Error making prediction with model {i+1}: {str(e)}", exc_info=True)
        
        # Process predictions
        self.logger.info(f"Processing ensemble predictions from {len(all_predictions)} models")
        if not all_predictions:
            self.logger.error("No predictions available from any model")
            return [{'label': 'Error', 'probability': 0.0}]
        
        # Average predictions from all models
        if len(all_predictions) > 1:
            # Ensure all predictions have the same shape
            shapes = [p.shape for p in all_predictions]
            if not all(s == shapes[0] for s in shapes):
                self.logger.warning(f"Inconsistent prediction shapes: {shapes}")
                # Try to reshape predictions to match
                for i in range(1, len(all_predictions)):
                    if all_predictions[i].shape != shapes[0]:
                        try:
                            all_predictions[i] = all_predictions[i].reshape(shapes[0])
                        except Exception as e:
                            self.logger.error(f"Failed to reshape prediction {i}: {e}")
                            # Remove this prediction
                            all_predictions[i] = None
                
                # Filter out None predictions
                all_predictions = [p for p in all_predictions if p is not None]
                if not all_predictions:
                    self.logger.error("No valid predictions after shape correction")
                    return [{'label': 'Error', 'probability': 0.0}]
            
            # Average predictions
            ensemble_preds = np.mean(all_predictions, axis=0)
        else:
            ensemble_preds = all_predictions[0]
        
        # Get top-k predictions
        if ensemble_preds.ndim > 1 and ensemble_preds.shape[0] == 1:
            # Remove batch dimension if present
            ensemble_preds = ensemble_preds[0]
        
        # Get indices of top-k predictions
        try:
            top_indices = np.argsort(ensemble_preds)[-top_k:][::-1]
        except Exception as e:
            self.logger.error(f"Error getting top indices: {e}, prediction shape: {ensemble_preds.shape}")
            return [{'label': 'Error', 'probability': 0.0}]
        
        # Format predictions
        predictions = []
        for idx in top_indices:
            if idx < len(self.class_names):
                label = self.class_names[idx]
                prob = float(ensemble_preds[idx])
                predictions.append({
                    'label': label,
                    'probability': prob
                })
            else:
                self.logger.warning(f"Index {idx} out of range for class_names (length {len(self.class_names)})")
        
        return predictions
    
    def predict(self, input_data: Union[str, np.ndarray], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Make a prediction for an audio file or feature array in a thread-safe manner.
        
        Args:
            input_data (Union[str, np.ndarray]): Either a path to an audio file or a preprocessed feature array
            top_k (int): Number of top predictions to return
            
        Returns:
            List[Dict[str, Any]]: List of top predictions with class labels and probabilities
            
        Raises:
            ValueError: If no models are loaded or if input data is invalid
        """
        with self._lock:
            if not self.models:
                self.logger.error("No models loaded. Call load_models() first.")
                raise ValueError("No models loaded. Call load_models() first.")
            self.logger.info(f"Using {len(self.models)} models for prediction")
        
        # Handle different input types
        if isinstance(input_data, str):
            # Input is a file path
            self.logger.info(f"Starting prediction for audio file: {input_data}")
            
            # Check if file exists
            if not os.path.exists(input_data):
                self.logger.error(f"Audio file not found: {input_data}")
                raise FileNotFoundError(f"Audio file not found: {input_data}")
                
            # Check if file is empty
            if os.path.getsize(input_data) == 0:
                self.logger.error(f"Audio file is empty: {input_data}")
                raise ValueError(f"Audio file is empty: {input_data}")
            
            # Preprocess audio
            self.logger.info(f"Preprocessing audio file: {input_data}")
            try:
                features = self.preprocess_audio(input_data)
                # Handle dictionary features
                if isinstance(features, dict):
                    # Use the first feature in the dictionary (typically 'mel_spectrogram')
                    feature_key = list(features.keys())[0]
                    self.logger.info(f"Using feature '{feature_key}' from dictionary")
                    features = features[feature_key]
                self.logger.info(f"Audio preprocessing complete. Feature shape: {features.shape}")
            except Exception as e:
                self.logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to preprocess audio file: {str(e)}")
        # If input_data is a numpy array or dictionary, assume it's already preprocessed features
        elif isinstance(input_data, (np.ndarray, dict)):
            if isinstance(input_data, dict):
                # Handle dictionary features
                feature_key = list(input_data.keys())[0]
                self.logger.info(f"Using feature '{feature_key}' from dictionary")
                features = input_data[feature_key]
                self.logger.info(f"Starting prediction with feature from dictionary, shape: {features.shape}")
                
                # If the feature is a 2D array, reshape it to match model input
                if len(features.shape) == 2:
                    self.logger.info(f"Reshaping 2D features {features.shape} to match CNN model input")
                    
                    # Check if this is the (32, 1) or (128, 1) shape that's causing issues
                    if features.shape == (32, 1) or features.shape == (128, 1):
                        # Create a new array with the expected shape (1, 128, 173, 1)
                        new_features = np.zeros((1, 128, 173, 1))
                        
                        # Fill the first 32 rows with the values from features
                        # Create a pattern that repeats the frequency values across time
                        for i in range(min(32, features.shape[0])):
                            freq_value = features[i, 0] if features.shape[1] > 0 else 0
                            # Repeat this value across all time steps for this frequency bin
                            new_features[0, i, :, 0] = freq_value
                        
                        features = new_features
                        self.logger.info(f"Reshaped features from (32, 1) to {features.shape}")
                    else:
                        # Get the height from the feature shape
                        height = features.shape[0]
                        
                        # Create a new array with the right shape (1, height, 173, 1)
                        new_features = np.zeros((1, height, 173, 1))
                        
                        # Copy the values from the original features and broadcast across width
                        for i in range(height):
                            for j in range(173):
                                # Use the first column value if available, otherwise use 0
                                if features.shape[1] > 0:
                                    new_features[0, i, j, 0] = features[i, 0]
                        
                        features = new_features
                        self.logger.info(f"Reshaped features to {features.shape}")
                    
                    # Force the model to use this reshaped input
                    return self._make_predictions_with_models(features, top_k)
            else:
                # Input is a numpy array
                self.logger.info(f"Starting prediction with provided feature array, shape: {input_data.shape}")
                features = input_data
                
            # If features are 2D (e.g., (32, 1)), reshape to match model input
            if len(features.shape) == 2:
                self.logger.info(f"Reshaping 2D features {features.shape} to match CNN model input")
                
                # Check if this is the (32, 1) or (128, 1) shape that's causing issues
                if features.shape == (32, 1) or features.shape == (128, 1):
                    # Create a new array with the expected shape (1, 128, 173, 1)
                    new_features = np.zeros((1, 128, 173, 1))
                    
                    # Fill the first 32 rows with the values from features
                    # Create a pattern that repeats the frequency values across time
                    for i in range(min(32, features.shape[0])):
                        freq_value = features[i, 0] if features.shape[1] > 0 else 0
                        # Repeat this value across all time steps for this frequency bin
                        new_features[0, i, :, 0] = freq_value
                    
                    features = new_features
                    self.logger.info(f"Reshaped features from (32, 1) to {features.shape}")
                else:
                    # Get the height from the feature shape
                    height = features.shape[0]
                    
                    # Create a new array with the right shape (1, height, 173, 1)
                    new_features = np.zeros((1, height, 173, 1))
                    
                    # Copy the values from the original features and broadcast across width
                    for i in range(height):
                        for j in range(173):
                            # Use the first column value if available, otherwise use 0
                            if features.shape[1] > 0:
                                new_features[0, i, j, 0] = features[i, 0]
                    
                    features = new_features
                    self.logger.info(f"Reshaped features to {features.shape}")
                
                # Force the model to use this reshaped input
                return self._make_predictions_with_models(features, top_k)
            
            # Validate numpy array
            if features is None:
                self.logger.error("Input features are None")
                raise ValueError("Input features are None")
                
            if features.size == 0:
                self.logger.error("Input feature array is empty")
                raise ValueError("Input feature array is empty")
            
            # Handle scalar values (including numpy scalar types)
            if np.isscalar(features) or (hasattr(features, 'ndim') and features.ndim == 0):
                self.logger.debug(f"Converting scalar input value {features} (type: {type(features)}) to numpy array")
                # For numpy scalar types, we need to explicitly create a new array
                if isinstance(features, np.number):
                    # Create a new array with the scalar value
                    features = np.array([float(features)], dtype=np.float32)
                else:
                    # For other scalar types
                    features = np.atleast_1d(features).astype(np.float32)
                
            if np.isnan(features).any() or np.isinf(features).any():
                self.logger.warning("Input features contain NaN or Inf values, replacing with zeros")
                features = np.nan_to_num(features)
            
            # Ensure features have the right shape for the model
            if len(self.models) > 0:
                model = self.models[0]
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                    self.logger.info(f"Model expects input shape: {input_shape}, current shape: {features.shape}")
                    
                    # If we have a 2D or 3D array but need a 4D array (for CNN)
                    if len(input_shape) == 4:
                        # Extract dimensions from model's input shape
                        _, height, width, channels = input_shape
                        self.logger.info(f"Model expects shape: (batch, {height}, {width}, {channels})")
                        
                        # Handle different input shapes
                        if len(features.shape) == 2:
                            # For 2D input (e.g., (128, 1))
                            feat_height, feat_width = features.shape
                            self.logger.info(f"Reshaping 2D features {features.shape} to match model input")
                            
                            # If the feature dimensions match height and width of expected input
                            if feat_height == height and feat_width == 1:
                                # Reshape to (1, height, 1, 1) and then broadcast to (1, height, width, channels)
                                features = features.reshape(1, feat_height, 1, 1)
                                # Create a new array with the right shape and fill it
                                new_features = np.zeros((1, height, width, channels))
                                # Broadcast the values across width dimension
                                for i in range(width):
                                    new_features[0, :, i, 0] = features[0, :, 0, 0]
                                features = new_features
                            else:
                                self.logger.warning(f"Feature dimensions {features.shape} don't match expected {height}x{width}")
                                # Try to resize or pad the features
                                try:
                                    # Resize features to match expected dimensions
                                    resized = np.zeros((1, height, width, channels))
                                    # Copy as much data as possible
                                    h = min(feat_height, height)
                                    w = min(feat_width, width)
                                    for i in range(h):
                                        for j in range(w):
                                            resized[0, i, j, 0] = features[i, j] if j < feat_width else 0
                                    features = resized
                                except Exception as e:
                                    self.logger.error(f"Failed to resize features: {e}")
                                    self.logger.warning("Using default Error prediction")
                                    return [{'label': 'Error', 'probability': 0.0}]
                        elif len(features.shape) == 3:
                            # For 3D input (e.g., (1, 128, 1))
                            batch, feat_height, feat_width = features.shape
                            self.logger.info(f"Reshaping 3D features {features.shape} to 4D")
                            features = features.reshape(batch, feat_height, feat_width, 1)
                            
                            # If dimensions don't match, try to resize
                            if feat_height != height or feat_width != width:
                                self.logger.warning(f"Feature dimensions {feat_height}x{feat_width} don't match expected {height}x{width}")
                                try:
                                    # Resize features to match expected dimensions
                                    resized = np.zeros((batch, height, width, channels))
                                    # Copy as much data as possible
                                    h = min(feat_height, height)
                                    w = min(feat_width, width)
                                    for i in range(h):
                                        for j in range(w):
                                            resized[0, i, j, 0] = features[0, i, j, 0] if j < feat_width else 0
                                    features = resized
                                except Exception as e:
                                    self.logger.error(f"Failed to resize features: {e}")
                                    self.logger.warning("Using default Error prediction")
                                    return [{'label': 'Error', 'probability': 0.0}]
                        
                        self.logger.info(f"Final feature shape: {features.shape}")
                        if features.shape != (1, height, width, channels):
                            self.logger.warning(f"Feature shape {features.shape} still doesn't match expected {(1, height, width, channels)}")
                            self.logger.warning("Using default Error prediction")
                            return [{'label': 'Error', 'probability': 0.0}]
                    # If batch dimension is missing, add it
                    elif len(features.shape) == len(input_shape) - 1:
                        self.logger.info(f"Adding batch dimension to features")
                        features = np.expand_dims(features, axis=0)
        else:
            error_msg = f"Unsupported input type: {type(input_data)}. Must be a file path (str), feature array (np.ndarray), or feature dictionary (dict)"
            self.logger.error(error_msg)
            raise TypeError(f"Input must be a file path (str), feature array (np.ndarray), or feature dictionary (dict), got {type(input_data)}")
        
        # Make predictions with each model
        all_predictions = []
        
        # Get a thread-safe copy of the models
        with self._lock:
            models_copy = self.models.copy()
        
        for i, model in enumerate(models_copy):
            try:
                self.logger.info(f"Making prediction with model {i+1}/{len(models_copy)}")
                # Make prediction
                if hasattr(model, 'predict'):
                    self.logger.info(f"Using model.predict() method")
                    preds = model.predict(features)
                else:
                    # For custom model wrappers like TFLite models
                    self.logger.info(f"Using model as callable")
                    preds = model(features)
                
                self.logger.info(f"Prediction shape from model {i+1}: {preds.shape}")
                all_predictions.append(preds)
            except Exception as e:
                self.logger.error(f"Error making prediction with model {i+1}: {str(e)}", exc_info=True)
        
        # Average predictions from all models (ensemble)
        self.logger.info(f"Processing ensemble predictions from {len(all_predictions)} models")
        
        # Check if we have any predictions
        if not all_predictions:
            self.logger.error("No predictions available from any model")
            return [{'label': 'Error', 'probability': 0.0, 'index': -1}]
        
        if len(all_predictions) > 1:
            # Ensure all predictions have the same shape
            shapes = [p.shape for p in all_predictions]
            self.logger.info(f"Prediction shapes from models: {shapes}")
            
            # Check for NaN or Inf values
            for i, preds in enumerate(all_predictions):
                if np.isnan(preds).any() or np.isinf(preds).any():
                    self.logger.warning(f"Model {i+1} has NaN or Inf values in predictions")
                    # Replace NaN/Inf with zeros
                    all_predictions[i] = np.nan_to_num(preds)
            
            if len(set(shapes)) > 1:
                self.logger.warning(f"Models have different output shapes: {shapes}")
                # Reshape predictions to match
                # This is a simple approach - more sophisticated approaches might be needed
                target_shape = max(shapes, key=lambda x: x[1])  # Use the shape with the most classes
                self.logger.info(f"Reshaping all predictions to target shape: {target_shape}")
                
                for i in range(len(all_predictions)):
                    if all_predictions[i].shape != target_shape:
                        # Pad with zeros if needed
                        self.logger.info(f"Reshaping model {i+1} predictions from {all_predictions[i].shape} to {target_shape}")
                        padded = np.zeros(target_shape)
                        padded[:, :all_predictions[i].shape[1]] = all_predictions[i]
                        all_predictions[i] = padded
            
            # Average predictions
            self.logger.info("Averaging predictions from all models")
            ensemble_preds = np.mean(all_predictions, axis=0)
        else:
            self.logger.info("Using predictions from single model")
            ensemble_preds = all_predictions[0]
        
        # Get top-k predictions
        if ensemble_preds.ndim > 1:
            # For batch predictions, take the first one (we only processed one audio file)
            self.logger.info(f"Ensemble predictions shape: {ensemble_preds.shape}, taking first item")
            ensemble_preds = ensemble_preds[0]
        
        self.logger.info(f"Final prediction vector shape: {ensemble_preds.shape}")
        
        # Check for NaN or Inf values in final predictions
        if np.isnan(ensemble_preds).any() or np.isinf(ensemble_preds).any():
            self.logger.warning("Final ensemble predictions contain NaN or Inf values")
            ensemble_preds = np.nan_to_num(ensemble_preds)
        
        # Get top-k indices and probabilities
        top_indices = np.argsort(ensemble_preds)[-top_k:][::-1]
        top_probs = ensemble_preds[top_indices]
        
        self.logger.info(f"Top {top_k} indices: {top_indices}")
        self.logger.info(f"Top {top_k} probabilities: {top_probs}")
        
        # Get a thread-safe copy of class labels
        with self._lock:
            class_labels_copy = self.class_labels.copy()
            self.logger.info(f"Using {len(class_labels_copy)} class labels: {class_labels_copy}")
        
        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(class_labels_copy):
                label = class_labels_copy[idx]
            else:
                self.logger.warning(f"Index {idx} is out of bounds for class labels (max: {len(class_labels_copy)-1})")
                label = f"Unknown Class {idx}"
                
            results.append({
                'label': label,
                'probability': float(top_probs[i]),
                'index': int(idx)
            })
            
        self.logger.info(f"Final prediction results: {results}")
        
        return results
    
    def predict_batch(self, audio_paths: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Make predictions for multiple audio files.
        
        Args:
            audio_paths (List[str]): List of paths to audio files
            top_k (int): Number of top predictions to return for each file
            
        Returns:
            List[List[Dict[str, Any]]]: List of prediction results for each audio file
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error predicting {audio_path}: {e}")
                # Add empty result for failed predictions
                results.append([{'label': 'Error', 'probability': 0.0, 'index': -1}])
        
        return results
    
    def predict_proba(self, audio_path: str) -> np.ndarray:
        """
        Get raw probability predictions for all classes.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Array of probabilities for each class
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        
        # Make predictions with each model
        all_predictions = []
        failed_models = 0
        for i, model in enumerate(self.models):
            try:
                # Make prediction
                if hasattr(model, 'predict'):
                    preds = model.predict(features)
                else:
                    # For custom model wrappers like TFLite models
                    preds = model(features)
                
                # Verify prediction shape and content
                if preds is None or (hasattr(preds, 'size') and preds.size == 0):
                    self.logger.warning(f"Model {i} returned empty predictions, skipping")
                    continue
                    
                # Check for NaN or Inf values
                if np.isnan(preds).any() or np.isinf(preds).any():
                    self.logger.warning(f"Model {i} returned NaN or Inf values, replacing with zeros")
                    preds = np.nan_to_num(preds)
                
                all_predictions.append(preds)
                self.logger.debug(f"Model {i} prediction shape: {preds.shape}")
            except Exception as e:
                self.logger.error(f"Error making prediction with model {i}: {e}")
                failed_models += 1
        
        if not all_predictions:
            raise ValueError(f"All models failed to make predictions. {failed_models} models attempted.")
        
        # Average predictions from all models (ensemble)
        if len(all_predictions) > 1:
            # Ensure all predictions have the same shape
            shapes = [p.shape for p in all_predictions]
            if len(set(str(s) for s in shapes)) > 1:
                self.logger.warning(f"Models have different output shapes: {shapes}")
                
                # Find the shape with the most classes (last dimension)
                max_classes = max(s[-1] if len(s) > 1 else s[0] for s in shapes)
                target_shape = (1, max_classes) if len(shapes[0]) > 1 else (max_classes,)
                
                # Reshape predictions to match
                for i in range(len(all_predictions)):
                    current_shape = all_predictions[i].shape
                    
                    # Skip if already the right shape
                    if current_shape == target_shape:
                        continue
                        
                    # Handle different dimensionality
                    if len(current_shape) != len(target_shape):
                        # Add or remove batch dimension as needed
                        if len(current_shape) < len(target_shape):
                            all_predictions[i] = np.expand_dims(all_predictions[i], axis=0)
                        else:
                            all_predictions[i] = all_predictions[i][0]
                    
                    # Pad with zeros if needed for class dimension
                    current_classes = all_predictions[i].shape[-1] if len(all_predictions[i].shape) > 1 else all_predictions[i].shape[0]
                    if current_classes < target_shape[-1]:
                        # Create padded array
                        if len(target_shape) > 1:
                            padded = np.zeros(target_shape)
                            padded[0, :current_classes] = all_predictions[i][0, :current_classes]
                        else:
                            padded = np.zeros(target_shape)
                            padded[:current_classes] = all_predictions[i][:current_classes]
                        all_predictions[i] = padded
                        self.logger.debug(f"Padded prediction from shape {current_shape} to {all_predictions[i].shape}")
            
            # Average predictions
            ensemble_preds = np.mean(all_predictions, axis=0)
            self.logger.debug(f"Ensemble prediction shape: {ensemble_preds.shape}")
        else:
            ensemble_preds = all_predictions[0]
        
        # Return raw probabilities
        if ensemble_preds.ndim > 1:
            # For batch predictions, take the first one (we only processed one audio file)
            ensemble_preds = ensemble_preds[0]
        
        return ensemble_preds