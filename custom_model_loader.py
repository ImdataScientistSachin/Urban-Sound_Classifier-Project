import tensorflow as tf
import numpy as np
import os
import sys

# Add the src directory to the path
src_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(src_dir) != 'src':
    src_dir = os.path.join(src_dir, 'src')

sys.path.append(src_dir)
from src.config import CLASS_LABELS

# Define the custom AttentionGate layer
class AttentionGate(tf.keras.layers.Layer):
    """
    Attention Gate implementation for UNet architecture.
    
    This layer implements the attention mechanism described in the paper
    "Attention U-Net: Learning Where to Look for the Pancreas"
    by Oktay et al. (2018).
    """
    def __init__(self, filters, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.gating_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')
        self.input_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')
        self.attention_conv = tf.keras.layers.Conv2D(1, (1, 1), padding='same')
        self.activation = tf.keras.layers.Activation('relu')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
    
    def call(self, inputs):
        # Unpack inputs: [input_tensor, gating_tensor]
        input_tensor, gating_tensor = inputs
        
        # Process gating signal
        gating = self.gating_conv(gating_tensor)
        
        # Process input signal
        input_signal = self.input_conv(input_tensor)
        
        # Combine signals
        combined = self.activation(gating + input_signal)
        
        # Generate attention coefficients
        attention = self.attention_conv(combined)
        attention = self.sigmoid(attention)
        
        # Apply attention to input tensor
        return input_tensor * attention
    
    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({"filters": self.filters})
        return config

def load_model(model_path):
    """
    Load the trained TensorFlow model from the specified path with custom objects.
    
    Args:
        model_path (str): Path to the saved model file (.h5)
        
    Returns:
        model: Loaded TensorFlow model
    """
    try:
        # Define custom objects dictionary
        custom_objects = {
            'AttentionGate': AttentionGate
        }
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def predict(model, features):
    """
    Make a prediction using the loaded model.
    
    Args:
        model: Loaded TensorFlow model
        features (numpy.ndarray): Preprocessed audio features
        
    Returns:
        tuple: (predicted_class_name, confidence, all_confidences)
    """
    try:
        # Validate input features
        if features is None:
            raise ValueError("Features cannot be None")
            
        if not isinstance(features, np.ndarray):
            print(f"Warning: Features are not a numpy array. Converting from {type(features)}")
            features = np.array(features)
            
        # Ensure features are in the right shape for the model
        # Check the current shape of features
        print(f"Original features shape: {features.shape}")
        
        # The model expects shape=(None, 128, 173, 1)
        # Handle different input shapes more robustly
        if len(features.shape) == 2:  # (freq_bins, time_steps)
            # Add batch and channel dimensions
            features = features.reshape(1, features.shape[0], features.shape[1], 1)
            print(f"Reshaped features to {features.shape}")
        elif len(features.shape) == 3:  # (freq_bins, time_steps, channels) or (batch, freq_bins, time_steps)
            if features.shape[2] == 1:  # (freq_bins, time_steps, channels)
                # Add batch dimension
                features = features.reshape(1, features.shape[0], features.shape[1], features.shape[2])
                print(f"Reshaped features to {features.shape}")
            else:  # (batch, freq_bins, time_steps)
                # Add channel dimension
                features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
                print(f"Reshaped features to {features.shape}")
        
        # Check if we need to transpose dimensions
        if len(features.shape) == 4 and features.shape[1] != 128 and features.shape[2] == 128:
            print(f"Transposing features from {features.shape}")
            features = np.transpose(features, (0, 2, 1, 3))
            print(f"Transposed features to {features.shape}")
        
        # Adjust time dimension if needed
        if len(features.shape) == 4 and features.shape[2] != 173:
            print(f"Adjusting time dimension from {features.shape[2]} to 173")
            if features.shape[2] < 173:
                # Pad with zeros
                pad_width = ((0, 0), (0, 0), (0, 173 - features.shape[2]), (0, 0))
                features = np.pad(features, pad_width, 'constant')
                print(f"Padded features to {features.shape}")
            else:
                # Trim to 173 time steps
                features = features[:, :, :173, :]
                print(f"Trimmed features to {features.shape}")
        
        # Convert to float32
        features = features.astype(np.float32)
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            print("Warning: NaN or Inf values detected in features. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0, 1] range if needed
        if np.max(features) > 1.0 or np.min(features) < 0.0:
            print("Normalizing features to [0, 1] range")
            features = (features - np.min(features)) / (np.max(features) - np.min(features))
        
        # Try different prediction approaches
        approaches = [
            ("Standard predict", lambda: model.predict(features, verbose=0)),
            ("Direct call", lambda: model(features, training=False).numpy()),
            ("Eager execution", lambda: tf.keras.backend.get_value(model(features, training=False))),
            ("tf.function", lambda: tf.function(lambda x: model(x, training=False))(features).numpy())
        ]
        
        predictions = None
        last_error = None
        for approach_name, approach_func in approaches:
            try:
                print(f"Trying prediction approach: {approach_name}")
                predictions = approach_func()
                print(f"Prediction successful with {approach_name}")
                break
            except Exception as e:
                print(f"Error with {approach_name} approach: {e}")
                last_error = e
        
        if predictions is None:
            raise ValueError(f"All prediction approaches failed. Last error: {last_error}")
        
        # Verify prediction shape and content
        print(f"Prediction shape: {predictions.shape}")
        
        # Check for NaN or Inf values in predictions
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: NaN or Inf values detected in predictions. Replacing with zeros.")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Get the predicted class index and confidence
        if len(predictions.shape) == 2:  # (batch_size, num_classes)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
        else:  # (num_classes,)
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[predicted_class_index])
        
        # Map the index to a class name
        if predicted_class_index < len(CLASS_LABELS):
            predicted_class_name = CLASS_LABELS[predicted_class_index]
        else:
            print(f"Warning: Predicted class index {predicted_class_index} is out of range for CLASS_LABELS")
            predicted_class_name = f"unknown_class_{predicted_class_index}"
        
        # Create a dictionary of all class confidences
        all_confidences = {}
        for i, class_name in enumerate(CLASS_LABELS):
            if len(predictions.shape) == 2:  # (batch_size, num_classes)
                if i < predictions.shape[1]:
                    all_confidences[class_name] = float(predictions[0][i])
                else:
                    all_confidences[class_name] = 0.0
            else:  # (num_classes,)
                if i < len(predictions):
                    all_confidences[class_name] = float(predictions[i])
                else:
                    all_confidences[class_name] = 0.0
        
        # Apply softmax if confidences don't sum to approximately 1
        confidence_sum = sum(all_confidences.values())
        if abs(confidence_sum - 1.0) > 0.01:
            print(f"Confidences sum to {confidence_sum}, applying softmax normalization")
            # Apply softmax to raw predictions
            if len(predictions.shape) == 2:  # (batch_size, num_classes)
                softmax_predictions = tf.nn.softmax(predictions[0]).numpy()
            else:  # (num_classes,)
                softmax_predictions = tf.nn.softmax(predictions).numpy()
            
            # Update confidences
            for i, class_name in enumerate(CLASS_LABELS):
                if i < len(softmax_predictions):
                    all_confidences[class_name] = float(softmax_predictions[i])
            
            # Update predicted class and confidence
            predicted_class_index = np.argmax(softmax_predictions)
            if predicted_class_index < len(CLASS_LABELS):
                predicted_class_name = CLASS_LABELS[predicted_class_index]
            confidence = float(softmax_predictions[predicted_class_index])
        
        print(f"Predicted class: {predicted_class_name} with confidence {confidence:.4f}")
        return predicted_class_name, confidence, all_confidences
    
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Provide a fallback with diagnostic information
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'feature_shape': getattr(features, 'shape', None),
            'feature_min_max': (getattr(np, 'min', lambda x: None)(features), getattr(np, 'max', lambda x: None)(features)) if 'features' in locals() else None,
            'prediction_shape': getattr(predictions, 'shape', None) if 'predictions' in locals() else None
        }
        
        print(f"Error diagnostics: {error_info}")
        
        # Return a fallback prediction
        return 'error', 0.0, {class_name: 0.0 for class_name in CLASS_LABELS}