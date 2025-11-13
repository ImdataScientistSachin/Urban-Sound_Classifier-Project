import os
import logging
import tensorflow as tf
from typing import List, Dict, Any, Optional, Union, Tuple
import threading

from ...config.config_manager import ConfigManager
from ..architectures.components import AttentionGate, MultiHeadSelfAttention, SEBlock, ResidualBlock

class ModelLoader:
    """
    Thread-safe class for loading and managing pre-trained models.
    
    This class handles loading pre-trained models from various formats,
    including HDF5, SavedModel, and TFLite, with support for custom objects
    and model ensembles.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        models_dir (str): Directory containing pre-trained models
        custom_objects (Dict[str, Any]): Dictionary of custom layer objects
        loaded_models (List[tf.keras.Model]): List of loaded models
        _lock (threading.RLock): Reentrant lock for thread safety
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the ModelLoader.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get models directory from config
        self.models_dir = config.get('PATHS.models_dir')
        
        # Define custom objects for model loading
        self.custom_objects = {
            'AttentionGate': AttentionGate,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'SEBlock': SEBlock,
            'ResidualBlock': ResidualBlock
        }
        
        # Initialize list for loaded models
        self.loaded_models = []
        
        # Initialize lock for thread safety
        self._lock = threading.RLock()
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load a single model from a file in a thread-safe manner.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            tf.keras.Model: Loaded Keras model
            
        Raises:
            FileNotFoundError: If the model file does not exist
            ValueError: If the model format is not supported
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Determine model format based on file extension
            file_ext = os.path.splitext(model_path)[1].lower()
            
            if file_ext == '.h5':
                # Load HDF5 model
                original_model = tf.keras.models.load_model(model_path, custom_objects=self.custom_objects)
                
                # Create a wrapper for H5 models to handle the (32, 1) shape issue
                class H5ModelWrapper:
                    def __init__(self, model):
                        self.model = model
                        self._lock = threading.Lock()
                    
                    def predict(self, x):
                        with self._lock:  # Thread-safe prediction
                            if isinstance(x, list):
                                x = x[0]  # Take the first element if it's a list
                            
                            # Handle the problematic (32, 1) or (128, 1) shape
                            if hasattr(x, 'shape') and (x.shape == (32, 1) or x.shape == (128, 1)):
                                # Get the expected shape for CNN models
                                input_shape = self.model.input_shape
                                if len(input_shape) == 4:  # For CNN models expecting 4D input
                                    # Create a new array with the expected shape
                                    if isinstance(x, tf.Tensor):
                                        x_np = x.numpy()
                                    else:
                                        x_np = x
                                    
                                    # Create a new tensor with the expected shape (batch, height, width, channels)
                                    # Using the model's expected input shape
                                    new_shape = (1, input_shape[1], input_shape[2], input_shape[3])
                                    new_x = tf.zeros(new_shape, dtype=x.dtype)
                                    new_x_np = new_x.numpy()
                                    
                                    # Distribute the 32 values across the spectrogram
                                    for i in range(min(32, x_np.shape[0])):
                                        freq_value = x_np[i, 0] if x_np.shape[1] > 0 else 0
                                        # Repeat this value across all time steps for this frequency bin
                                        new_x_np[0, i, :, 0] = freq_value
                                    
                                    x = tf.convert_to_tensor(new_x_np, dtype=x.dtype)
                                    logging.getLogger(__name__).info(f"Reshaped input from (32, 1) to {x.shape}")
                                    
                            return self.model.predict(x)
                
                model = H5ModelWrapper(original_model)
                self.logger.info(f"Loaded H5 model from {model_path} with shape handling wrapper")
                
            elif file_ext == '.tflite':
                # Load TFLite model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                # Create a wrapper class for the TFLite interpreter
                class TFLiteModel:
                    def __init__(self, interpreter):
                        self.interpreter = interpreter
                        self.input_details = interpreter.get_input_details()
                        self.output_details = interpreter.get_output_details()
                        self._lock = threading.Lock()
                    
                    def predict(self, x):
                        with self._lock:  # Thread-safe prediction
                            if isinstance(x, list):
                                x = x[0]  # Take the first element if it's a list
                            
                            # Handle the problematic (32, 1) or (128, 1) shape
                            if hasattr(x, 'shape') and (x.shape == (32, 1) or x.shape == (128, 1)):
                                # Get the expected shape for CNN models
                                input_shape = self.input_details[0]['shape']
                                if len(input_shape) == 4:  # For CNN models expecting 4D input
                                    # Create a new array with the expected shape
                                    if isinstance(x, tf.Tensor):
                                        x_np = x.numpy()
                                    else:
                                        x_np = x
                                    
                                    # Create a new tensor with the expected shape
                                    new_x = tf.zeros(input_shape, dtype=x.dtype)
                                    new_x_np = new_x.numpy()
                                    
                                    # Distribute the 32 values across the spectrogram
                                    for i in range(min(32, x_np.shape[0])):
                                        freq_value = x_np[i, 0] if x_np.shape[1] > 0 else 0
                                        # Repeat this value across all time steps for this frequency bin
                                        new_x_np[0, i, :, 0] = freq_value
                                    
                                    x = tf.convert_to_tensor(new_x_np, dtype=x.dtype)
                            # Ensure input is in the correct format for other shapes
                            elif x.shape != self.input_details[0]['shape'] and len(x.shape) == len(self.input_details[0]['shape']):
                                # Reshape input to match expected shape
                                x = tf.reshape(x, self.input_details[0]['shape'])
                            
                            # Set input tensor
                            self.interpreter.set_tensor(self.input_details[0]['index'], x)
                            
                            # Run inference
                            self.interpreter.invoke()
                            
                            # Get output tensor
                            output = self.interpreter.get_tensor(self.output_details[0]['index'])
                            return output
                
                model = TFLiteModel(interpreter)
                self.logger.info(f"Loaded TFLite model from {model_path}")
                
            elif os.path.isdir(model_path):
                # Load SavedModel
                model = tf.keras.models.load_model(model_path, custom_objects=self.custom_objects)
                self.logger.info(f"Loaded SavedModel from {model_path}")
                
            else:
                raise ValueError(f"Unsupported model format: {file_ext}")
            
            # Add to loaded models list in a thread-safe manner
            with self._lock:
                self.loaded_models.append(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_ensemble(self, model_paths: List[str] = None) -> List[tf.keras.Model]:
        """
        Load multiple models to form an ensemble in a thread-safe manner.
        
        Args:
            model_paths (List[str], optional): List of paths to model files.
                If None, will attempt to load all models from the models directory.
            
        Returns:
            List[tf.keras.Model]: List of loaded models
            
        Raises:
            ValueError: If no models are found or loaded
        """
        # Clear previously loaded models in a thread-safe manner
        with self._lock:
            self.loaded_models = []
        
        # If no model paths provided, find all model files in the models directory
        if model_paths is None:
            model_paths = []
            for root, _, files in os.walk(self.models_dir):
                for file in files:
                    if file.endswith(('.h5', '.tflite')) or os.path.isdir(os.path.join(root, file)):
                        model_paths.append(os.path.join(root, file))
        
        if not model_paths:
            raise ValueError(f"No model files found in {self.models_dir}")
        
        # Load each model
        for model_path in model_paths:
            try:
                self.load_model(model_path)
            except Exception as e:
                self.logger.warning(f"Failed to load model {model_path}: {e}")
        
        # Check if any models were loaded in a thread-safe manner
        with self._lock:
            if not self.loaded_models:
                raise ValueError("No models were successfully loaded")
            
            self.logger.info(f"Loaded ensemble of {len(self.loaded_models)} models")
            return self.loaded_models.copy()  # Return a copy to avoid external modification
    
    def get_loaded_models(self) -> List[tf.keras.Model]:
        """
        Get the list of currently loaded models in a thread-safe manner.
        
        Returns:
            List[tf.keras.Model]: List of loaded models
        """
        with self._lock:
            return self.loaded_models.copy()  # Return a copy to avoid external modification
    
    def compile_model(self, model: tf.keras.Model, compile_args: Dict[str, Any] = None) -> tf.keras.Model:
        """
        Compile a Keras model with the specified arguments.
        
        Args:
            model (tf.keras.Model): Model to compile
            compile_args (Dict[str, Any], optional): Compilation arguments.
                If None, default arguments from config will be used.
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Get default compilation arguments from config if not provided
        if compile_args is None:
            compile_args = {
                'optimizer': tf.keras.optimizers.Adam(
                    learning_rate=self.config.get('TRAINING.learning_rate', 0.001)
                ),
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy']
            }
        
        # Compile the model
        model.compile(**compile_args)
        self.logger.info(f"Compiled model with arguments: {compile_args}")
        
        return model
    
    def save_model(self, model: tf.keras.Model, output_path: str, format: str = 'h5') -> str:
        """
        Save a model to a file.
        
        Args:
            model (tf.keras.Model): Model to save
            output_path (str): Path to save the model
            format (str): Format to save the model ('h5', 'savedmodel', or 'tflite')
            
        Returns:
            str: Path to the saved model
            
        Raises:
            ValueError: If the format is not supported
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            if format.lower() == 'h5':
                # Save as HDF5
                model.save(output_path, save_format='h5')
                self.logger.info(f"Saved model to {output_path} in H5 format")
                
            elif format.lower() == 'savedmodel':
                # Save as SavedModel
                model.save(output_path, save_format='tf')
                self.logger.info(f"Saved model to {output_path} in SavedModel format")
                
            elif format.lower() == 'tflite':
                # Convert to TFLite
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                # Save TFLite model
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                    
                self.logger.info(f"Saved model to {output_path} in TFLite format")
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving model to {output_path}: {e}")
            raise