import tensorflow as tf
import numpy as np
from config import CLASS_LABELS

def load_model(model_path):
    """
    Load the trained TensorFlow model from the specified path.
    
    Args:
        model_path (str): Path to the saved model file (.h5)
        
    Returns:
        model: Loaded TensorFlow model
    """
    try:
        model = tf.keras.models.load_model(model_path)
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
            features = np.expand_dims(np.expand_dims(features, axis=0), axis=-1)
            print(f"Added batch and channel dimensions. New shape: {features.shape}")
        elif len(features.shape) == 3:
            if features.shape[-1] == 1:  # (freq_bins, time_steps, 1) - just add batch dimension
                features = np.expand_dims(features, axis=0)
                print(f"Added batch dimension. New shape: {features.shape}")
            else:  # (batch, freq_bins, time_steps) or other 3D shape - add channel dimension
                features = np.expand_dims(features, axis=-1)
                print(f"Added channel dimension. New shape: {features.shape}")
        
        # Ensure the frequency dimension (mel bands) is correct
        # Model expects (batch, 128, time_steps, 1)
        if features.shape[1] != 128 and features.shape[2] == 128:
            print(f"Transposing dimensions to match model requirements")
            features = np.transpose(features, (0, 2, 1, 3))
            print(f"After transpose. New shape: {features.shape}")
        
        # Ensure time dimension is correct (173 time steps)
        expected_time_steps = 173
        if features.shape[2] != expected_time_steps:
            print(f"Adjusting time dimension from {features.shape[2]} to {expected_time_steps}")
            if features.shape[2] < expected_time_steps:
                # Pad if too short
                pad_width = ((0, 0), (0, 0), (0, expected_time_steps - features.shape[2]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            else:
                # Trim if too long
                features = features[:, :, :expected_time_steps, :]
            print(f"After time dimension adjustment. New shape: {features.shape}")
        
        # Ensure features are float32 for TensorFlow
        features = features.astype(np.float32)
        
        # Verify features have valid values and normalize if needed
        if np.isnan(features).any():
            print("Warning: Features contain NaN values. Replacing with zeros.")
            features = np.nan_to_num(features)
            
        if np.isinf(features).any():
            print("Warning: Features contain infinite values. Replacing with large finite values.")
            features = np.nan_to_num(features)
        
        # Check for proper normalization (values should be between 0 and 1)
        feature_min = np.min(features)
        feature_max = np.max(features)
        if feature_min < 0 or feature_max > 1:
            print(f"Features not properly normalized. Min: {feature_min}, Max: {feature_max}. Normalizing...")
            features = (features - feature_min) / (feature_max - feature_min + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Make prediction with multiple fallback approaches
        predictions = None
        error_messages = []
        
        # Approach 1: Standard predict method with batch size control
        try:
            print("Attempting prediction with model.predict() and controlled batch size")
            predictions = model.predict(features, batch_size=1, verbose=0)
        except Exception as e:
            error_message = f"Error during model.predict(): {str(e)}"
            print(error_message)
            error_messages.append(error_message)
        
        # Approach 2: Direct call if predict failed
        if predictions is None:
            try:
                print("Attempting prediction with model direct call")
                predictions = model(features, training=False)
                if isinstance(predictions, tf.Tensor):
                    predictions = predictions.numpy()
                else:
                    print(f"Warning: Prediction result is not a tensor but {type(predictions)}")
            except Exception as e:
                error_message = f"Error during model direct call: {str(e)}"
                print(error_message)
                error_messages.append(error_message)
        
        # Approach 3: Try with eager execution if both methods failed
        if predictions is None:
            try:
                print("Attempting prediction with eager execution")
                with tf.GradientTape() as tape:
                    predictions = model(features, training=False)
                if isinstance(predictions, tf.Tensor):
                    predictions = predictions.numpy()
                else:
                    print(f"Warning: Eager execution result is not a tensor but {type(predictions)}")
            except Exception as e:
                error_message = f"Error during eager execution: {str(e)}"
                print(error_message)
                error_messages.append(error_message)
        
        # Approach 4: Try with TensorFlow function if all previous methods failed
        if predictions is None:
            try:
                print("Attempting prediction with tf.function")
                @tf.function
                def predict_fn(x):
                    return model(x, training=False)
                
                predictions = predict_fn(tf.convert_to_tensor(features))
                if isinstance(predictions, tf.Tensor):
                    predictions = predictions.numpy()
                else:
                    print(f"Warning: tf.function result is not a tensor but {type(predictions)}")
            except Exception as e:
                error_message = f"Error during tf.function prediction: {str(e)}"
                print(error_message)
                error_messages.append(error_message)
        
        # If all approaches failed, raise an exception with all error messages
        if predictions is None:
            raise RuntimeError(f"All prediction approaches failed: {'; '.join(error_messages)}")
        
        # Verify predictions shape and content
        print(f"Raw prediction shape: {predictions.shape}")
        if len(predictions.shape) < 2 or predictions.shape[1] != len(CLASS_LABELS):
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}. Expected (batch_size, {len(CLASS_LABELS)})")
        
        # Check for NaN or Inf in predictions
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: Predictions contain NaN or Inf values. Replacing with zeros.")
            predictions = np.nan_to_num(predictions)
        
        # Get the predicted class index and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])  # Convert to Python float
        
        # Verify the predicted class index is valid
        if predicted_class_idx < 0 or predicted_class_idx >= len(CLASS_LABELS):
            print(f"Warning: Invalid predicted class index {predicted_class_idx}. Using fallback.")
            # Use the highest valid index as fallback
            valid_indices = np.arange(len(CLASS_LABELS))
            valid_probs = predictions[0][valid_indices]
            predicted_class_idx = valid_indices[np.argmax(valid_probs)]
            confidence = float(predictions[0][predicted_class_idx])
        
        # Map the index to class name
        predicted_class_name = CLASS_LABELS[predicted_class_idx]
        
        # Create a dictionary of all class confidences
        all_confidences = {}
        for i, class_name in enumerate(CLASS_LABELS):
            all_confidences[class_name] = float(predictions[0][i])  # Convert to Python float
        
        # Apply softmax if confidences don't sum to approximately 1
        confidence_sum = sum(all_confidences.values())
        if abs(confidence_sum - 1.0) > 0.01:  # If sum is not close to 1
            print(f"Confidences don't sum to 1 (sum={confidence_sum}). Applying softmax normalization.")
            # Apply softmax
            exp_preds = np.exp(predictions[0] - np.max(predictions[0]))  # Subtract max for numerical stability
            softmax_preds = exp_preds / exp_preds.sum()
            
            # Update confidences
            for i, class_name in enumerate(CLASS_LABELS):
                all_confidences[class_name] = float(softmax_preds[i])
            
            # Update top prediction
            predicted_class_idx = np.argmax(softmax_preds)
            predicted_class_name = CLASS_LABELS[predicted_class_idx]
            confidence = float(softmax_preds[predicted_class_idx])
        
        print(f"Final prediction: {predicted_class_name} with confidence {confidence:.4f}")
        return predicted_class_name, confidence, all_confidences
    except Exception as e:
        print(f"Error in predict function: {e}")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        
        # Attempt to provide a fallback prediction with diagnostic information
        try:
            # Create a fallback prediction with diagnostic information
            fallback_class = CLASS_LABELS[0]  # Use first class as fallback
            fallback_confidence = 0.0
            
            # Create diagnostic information in the confidences dictionary
            diagnostic_confidences = {}
            for i, class_name in enumerate(CLASS_LABELS):
                diagnostic_confidences[class_name] = 0.0
            
            # Add error information to the confidences
            diagnostic_confidences['_error_type'] = str(type(e).__name__)
            diagnostic_confidences['_error_message'] = str(e)
            diagnostic_confidences['_features_shape'] = str(features.shape) if 'features' in locals() and features is not None else 'unknown'
            diagnostic_confidences['_features_min'] = float(np.min(features)) if 'features' in locals() and features is not None else 0.0
            diagnostic_confidences['_features_max'] = float(np.max(features)) if 'features' in locals() and features is not None else 0.0
            diagnostic_confidences['_predictions_shape'] = str(predictions.shape) if 'predictions' in locals() and predictions is not None else 'none'
            
            print(f"Returning fallback prediction with diagnostic information")
            return fallback_class, fallback_confidence, diagnostic_confidences
        except Exception as fallback_error:
            print(f"Even fallback error handling failed: {fallback_error}")
            # If all else fails, re-raise the original exception
            raise Exception(f"Prediction failed: {str(e)}\n{error_trace}")

def get_model_summary(model):
    """
    Get a string representation of the model architecture.
    
    Args:
        model: Loaded TensorFlow model
        
    Returns:
        str: Model summary
    """
    # Redirect model.summary() output to a string
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return '\n'.join(stringlist)

def evaluate_model(model, test_features, test_labels):
    """
    Evaluate the model on test data.
    
    Args:
        model: Loaded TensorFlow model
        test_features: Test features
        test_labels: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Ensure test_features are in the right shape for the model
    if len(test_features.shape) == 3:  # (time_steps, freq_bins, channels) or (freq_bins, time_steps, channels)
        # First add batch dimension if not present
        test_features = np.expand_dims(test_features, axis=0)  # Add batch dimension
        
        # Check if we need to transpose dimensions
        if test_features.shape[1] != 128 and test_features.shape[2] == 128:
            # If features are (batch, time_steps, freq_bins, channels), transpose to (batch, freq_bins, time_steps, channels)
            test_features = np.transpose(test_features, (0, 2, 1, 3))
    
    # Evaluate the model
    results = model.evaluate(test_features, test_labels, verbose=1)
    
    # Create a dictionary of metrics
    metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        metrics[metric_name] = results[i]
    
    return metrics