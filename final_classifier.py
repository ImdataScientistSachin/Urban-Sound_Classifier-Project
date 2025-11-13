import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import time
import pickle
import librosa
import soundfile as sf
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.config import CLASS_LABELS, SAMPLE_RATE, DURATION, N_MELS, N_FFT, HOP_LENGTH

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class UrbanSoundClassifier:
    """A robust classifier for urban sound classification using TFLite models."""
    
    def __init__(self, model_dir='tflite_models', use_cached_features=True, features_dir='extracted_features',
                 use_advanced_features=False, use_advanced_ensemble=False, use_advanced_postprocessing=False):
        """
        Initialize the classifier.
        
        Args:
            model_dir (str): Directory containing TFLite model files
            use_cached_features (bool): Whether to use cached features
            features_dir (str): Directory containing cached features
            use_advanced_features (bool): Whether to use advanced feature extraction
            use_advanced_ensemble (bool): Whether to use advanced ensemble strategy
            use_advanced_postprocessing (bool): Whether to use advanced post-processing
        """
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
        self.use_cached_features = use_cached_features
        self.features_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), features_dir)
        self.interpreters = []
        self.model_weights = []
        self.cached_features = {}
        
        # Advanced feature flags
        self.use_advanced_features = use_advanced_features
        self.use_advanced_ensemble = use_advanced_ensemble
        self.use_advanced_postprocessing = use_advanced_postprocessing
        
        # Initialize advanced components if needed
        if self.use_advanced_features:
            self.feature_extractor = AdvancedFeatureExtractor()
        
        # For temporal smoothing in advanced post-processing
        self.prediction_history = []
        
        # Define class-specific thresholds for post-processing
        # These thresholds are used to adjust predictions based on observed model behavior
        self.class_thresholds = {
            'air_conditioner': 0.30,  # Higher threshold for air_conditioner to reduce false positives
            'car_horn': 0.35,        # Higher threshold for car_horn to reduce false positives
            'children_playing': 0.15, # Lower threshold for children_playing
            'dog_bark': 0.35,        # Higher threshold for dog_bark to reduce false positives
            'drilling': 0.35,        # Higher threshold for drilling to reduce false positives
            'engine_idling': 0.15,   # Lower threshold for engine_idling
            'gun_shot': 0.40,        # Higher threshold for gun_shot to reduce false positives
            'jackhammer': 0.35,      # Higher threshold for jackhammer to reduce false positives
            'siren': 0.40,           # Higher threshold for siren to reduce false positives
            'street_music': 0.15     # Lower threshold for street_music
        }
        
        # Define class bias adjustments to counteract model tendencies
        # Positive values increase likelihood, negative values decrease likelihood
        self.class_bias = {
            'air_conditioner': -0.20,   # Negative bias for air_conditioner to reduce false positives
            'car_horn': 0.15,        # Slight positive bias for car_horn
            'children_playing': 0.15, # Slight positive bias for children_playing
            'dog_bark': 0.15,        # Slight positive bias for dog_bark
            'drilling': 0.15,        # Slight positive bias for drilling
            'engine_idling': 0.15,    # Slight positive bias for engine_idling
            'gun_shot': 0.15,        # Slight positive bias for gun_shot
            'jackhammer': -0.30,      # Strong negative bias for jackhammer to reduce false positives
            'siren': 0.15,          # Slight positive bias for siren
            'street_music': 0.15      # Slight positive bias for street_music
        }
        
        # Create features directory if it doesn't exist
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Load cached features if available and requested
        if self.use_cached_features and os.path.exists(os.path.join(self.features_dir, 'all_features.pkl')):
            try:
                with open(os.path.join(self.features_dir, 'all_features.pkl'), 'rb') as f:
                    self.cached_features = pickle.load(f)
                print(f"Loaded cached features for {len(self.cached_features)} files")
            except Exception as e:
                print(f"Error loading cached features: {e}")
                self.cached_features = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load TFLite models from model files."""
        # Get all model files
        if not os.path.exists(self.model_dir):
            print(f"Model directory {self.model_dir} does not exist")
            return
        
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.tflite')]
        
        if not model_files:
            print(f"No TFLite model files found in {self.model_dir}")
            return
        
        print(f"Loading {len(model_files)} TFLite models from {self.model_dir}")
        
        # Load each model
        for model_file in tqdm(model_files, desc="Loading models"):
            model_path = os.path.join(self.model_dir, model_file)
            
            try:
                # Create an interpreter for the model
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Store the interpreter and its details
                self.interpreters.append({
                    'interpreter': interpreter,
                    'input_details': input_details,
                    'output_details': output_details,
                    'file': model_file
                })
                
                # Extract model type and fold from filename
                model_info = {
                    'file': model_file,
                    'type': model_file.split('_')[-1].split('.')[0] if '_' in model_file else 'Unknown',
                    'fold': int(model_file.split('fold')[1][0]) if 'fold' in model_file else 0
                }
                
                # Assign weight based on model type and fold
                # Adjust weights to improve accuracy
                if 'UNet' in model_info['type']:
                    weight = 3.0  # Significantly increase UNet model weight
                elif 'CNN' in model_info['type']:
                    weight = 1.0  # Standard weight for CNN model weight
                else:
                    weight = 0.5  # Decrease weight for other models
                
                self.model_weights.append(weight)
                
                print(f"Successfully loaded {model_file}")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
        
        # Normalize weights
        if self.model_weights:
            self.model_weights = np.array(self.model_weights) / sum(self.model_weights)
        
        print(f"Successfully loaded {len(self.interpreters)} models")
        if not self.interpreters:
            print("WARNING: No models were successfully loaded!")
    
    def load_audio(self, file_path, target_sr=SAMPLE_RATE, duration=DURATION):
        """
        Load and preprocess audio file.
        
        Args:
            file_path (str): Path to the audio file
            target_sr (int): Target sample rate
            duration (float): Target duration in seconds
            
        Returns:
            numpy.ndarray: Audio data
        """
        # Load audio file
        audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(audio_data) or (hasattr(audio_data, 'ndim') and audio_data.ndim == 0):
            print(f"Converting scalar value {audio_data} (type: {type(audio_data)}) to numpy array")
            audio_data = np.atleast_1d(audio_data).astype(np.float32)
        
        # Trim silence
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        # Ensure consistent duration
        target_length = int(duration * target_sr)
        if len(audio_data) < target_length:
            # Pad with zeros if audio is shorter than target duration
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
        else:
            # Trim if audio is longer than target duration
            audio_data = audio_data[:target_length]
        
        return audio_data
    
    def extract_features(self, audio_data, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fixed_time_dim=173):
        """
        Extract Mel spectrogram features from audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio data
            sr (int): Sample rate
            n_mels (int): Number of Mel bands
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            fixed_time_dim (int): Fixed time dimension for the output
            
        Returns:
            numpy.ndarray: Mel spectrogram features
        """
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))
        
        # Ensure consistent time dimension
        current_time_dim = mel_spec_norm.shape[1]
        if current_time_dim < fixed_time_dim:
            # Pad with zeros if shorter than fixed time dimension
            pad_width = ((0, 0), (0, fixed_time_dim - current_time_dim))
            mel_spec_norm = np.pad(mel_spec_norm, pad_width, 'constant')
        elif current_time_dim > fixed_time_dim:
            # Trim if longer than fixed time dimension
            mel_spec_norm = mel_spec_norm[:, :fixed_time_dim]
        
        return mel_spec_norm
    
    def process_audio_file(self, audio_file_path, apply_augmentation=False):
        """
        Process an audio file and extract features.
        
        Args:
            audio_file_path (str): Path to the audio file
            apply_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            dict: Dictionary with different feature representations
        """
        # Check if features are cached and augmentation is not requested
        file_name = os.path.basename(audio_file_path)
        if self.use_cached_features and file_name in self.cached_features and not apply_augmentation:
            return self.cached_features[file_name]['features']
        
        # Load audio
        audio_data = self.load_audio(audio_file_path)
        
        # Apply data augmentation if requested
        if apply_augmentation:
            audio_data = DataAugmentation.apply_random_augmentation(audio_data, SAMPLE_RATE)
        
        # Extract features using advanced extractor if enabled
        if self.use_advanced_features:
            # Extract advanced features
            features_data = self.feature_extractor.extract_all_features(audio_data)
            
            # Prepare different feature representations for different models
            features = {
                'mel_spec': features_data,
                'mel_spec_expanded': features_data.reshape(1, *features_data.shape),  # Add batch dimension
                'mel_spec_with_channel': features_data.reshape(features_data.shape[0], features_data.shape[1], 1),  # Add channel dimension
                'mel_spec_with_batch_and_channel': features_data.reshape(1, features_data.shape[0], features_data.shape[1], 1)  # Add batch and channel dimensions
            }
        else:
            # Extract standard features
            mel_spec = self.extract_features(audio_data)
            
            # Prepare different feature representations for different models
            features = {
                'mel_spec': mel_spec,
                'mel_spec_expanded': mel_spec.reshape(1, *mel_spec.shape),  # Add batch dimension
                'mel_spec_with_channel': mel_spec.reshape(mel_spec.shape[0], mel_spec.shape[1], 1),  # Add channel dimension
                'mel_spec_with_batch_and_channel': mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)  # Add batch and channel dimensions
            }
        
        return features
    
    def _prepare_features_for_interpreter(self, features, input_details):
        """Prepare features for a specific TFLite interpreter based on its input shape."""
        # Get the input shape of the model
        input_shape = input_details[0]['shape']
        
        # Select appropriate feature representation based on input shape
        if len(input_shape) == 4:  # (batch, height, width, channels)
            # Use the batch and channel dimensions feature
            mel_spec = features['mel_spec']
            reshaped_features = np.reshape(mel_spec, (1, mel_spec.shape[0], mel_spec.shape[1], 1))
        elif len(input_shape) == 3:  # (height, width, channels)
            # Use the channel dimension feature
            mel_spec = features['mel_spec']
            reshaped_features = np.reshape(mel_spec, (mel_spec.shape[0], mel_spec.shape[1], 1))
        else:  # Default case
            # Use the expanded feature
            reshaped_features = features['mel_spec_expanded']
        
        # Convert to the required data type
        input_dtype = input_details[0]['dtype']
        if input_dtype == np.float32:
            return reshaped_features.astype(np.float32)
        elif input_dtype == np.uint8:
            # Quantize the features to uint8
            scale, zero_point = input_details[0]['quantization']
            return np.uint8((reshaped_features / scale) + zero_point) if scale != 0 else np.uint8(reshaped_features * 255)
        else:
            return reshaped_features
    
    def _predict_with_interpreter(self, interpreter_info, features):
        """Make a prediction with a single TFLite interpreter."""
        interpreter = interpreter_info['interpreter']
        input_details = interpreter_info['input_details']
        output_details = interpreter_info['output_details']
        
        # Prepare features for this interpreter
        model_features = self._prepare_features_for_interpreter(features, input_details)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], model_features)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # If the output is quantized, dequantize it
        if output_details[0]['dtype'] != np.float32:
            scale, zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale if scale != 0 else output_data.astype(np.float32) / 255
        
        # Return the prediction
        return output_data[0] if output_data.shape[0] == 1 else output_data
    
    def predict(self, audio_file, confidence_threshold=0.3, apply_augmentation=False):
        """Make a prediction with the ensemble model.
        
        Args:
            audio_file (str): Path to the audio file
            confidence_threshold (float): Minimum confidence threshold for predictions
            apply_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (predicted_class, weighted_predictions, uncertainty)
        """
        if not self.interpreters:
            print("No models available for prediction")
            return None, None, 1.0
        
        # Prepare features
        features = self.process_audio_file(audio_file, apply_augmentation=apply_augmentation)
        
        # Make predictions with each model
        all_predictions = []
        all_confidences = []
        model_types = []
        raw_predictions = []  # Store raw predictions for uncertainty estimation
        
        for i, interpreter_info in enumerate(self.interpreters):
            try:
                # Get prediction from this model
                prediction = self._predict_with_interpreter(interpreter_info, features)
                raw_predictions.append(prediction)  # Store raw prediction
                
                # Calculate confidence (max probability)
                confidence = np.max(prediction)
                
                # Extract model type
                model_file = interpreter_info['file']
                model_type = model_file.split('_')[-1].split('.')[0] if '_' in model_file else 'Unknown'
                
                # Only include predictions with confidence above threshold
                if confidence >= confidence_threshold:
                    # Store prediction, confidence, and model type
                    all_predictions.append(prediction)
                    all_confidences.append(confidence)
                    model_types.append(model_type)
                else:
                    print(f"Model {i} ({model_type}) prediction discarded due to low confidence: {confidence:.4f}")
            except Exception as e:
                print(f"Error predicting with model {i}: {e}")
        
        if not all_predictions:
            print("No models made predictions with sufficient confidence")
            # Fall back to using all predictions if none meet the threshold
            for i, interpreter_info in enumerate(self.interpreters):
                try:
                    prediction = self._predict_with_interpreter(interpreter_info, features)
                    all_predictions.append(prediction)
                    all_confidences.append(np.max(prediction))
                    model_file = interpreter_info['file']
                    model_types.append(model_file.split('_')[-1].split('.')[0] if '_' in model_file else 'Unknown')
                except Exception:
                    pass
            
            if not all_predictions:
                print("All models failed to make predictions")
                return None, None, 1.0
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_confidences = np.array(all_confidences)
        raw_predictions = np.array(raw_predictions)
        
        # Use advanced ensemble strategy if enabled
        if self.use_advanced_ensemble:
            # Get model weights for the selected models
            model_priors = []
            for model_type in model_types:
                # Assign prior based on model type
                if 'UNet' in model_type:
                    prior = 0.6  # Higher prior for UNet models
                elif 'CNN' in model_type:
                    prior = 0.3  # Medium prior for CNN models
                else:
                    prior = 0.1  # Lower prior for other models
                model_priors.append(prior)
            
            # Use Bayesian model averaging
            weighted_predictions = AdvancedEnsembleStrategy.bayesian_model_averaging(
                all_predictions, model_priors=model_priors)
        else:
            # Use standard weighted ensemble
            # Get model weights for the selected models
            selected_weights = []
            for i, interpreter_info in enumerate(self.interpreters):
                model_file = interpreter_info['file']
                model_type = model_file.split('_')[-1].split('.')[0] if '_' in model_file else 'Unknown'
                if model_type in model_types:
                    # Assign weight based on model type
                    if 'UNet' in model_type:
                        weight = 3.0  # Much higher weight for UNet models
                    elif 'CNN' in model_type:
                        weight = 1.0  # Standard weight for CNN models
                    else:
                        weight = 0.5  # Lower weight for other models
                    selected_weights.append(weight)
            
            selected_weights = np.array(selected_weights)
            selected_weights = selected_weights / np.sum(selected_weights)  # Normalize
            
            # Calculate weighted average of predictions
            # Weight by both model weight and prediction confidence
            combined_weights = selected_weights * all_confidences
            combined_weights = combined_weights / np.sum(combined_weights)  # Normalize
            
            # Apply weights to predictions
            weighted_predictions = np.zeros_like(all_predictions[0])
            for i, pred in enumerate(all_predictions):
                weighted_predictions += pred * combined_weights[i]
        
        # Apply class bias adjustments to counteract model tendencies
        adjusted_predictions = weighted_predictions.copy()
        for i, class_name in enumerate(CLASS_LABELS):
            if class_name in self.class_bias:
                adjusted_predictions[i] += self.class_bias[class_name]
        
        # Apply advanced post-processing if enabled
        uncertainty = 0.0
        if self.use_advanced_postprocessing:
            # Store prediction in history for temporal smoothing
            self.prediction_history.append(adjusted_predictions)
            if len(self.prediction_history) > 5:  # Keep only last 5 predictions
                self.prediction_history.pop(0)
            
            # Apply temporal smoothing if we have enough history
            if len(self.prediction_history) >= 3:
                adjusted_predictions = AdvancedPostProcessing.temporal_smoothing(
                    self.prediction_history, window_size=min(3, len(self.prediction_history)))
            
            # Apply confidence calibration with temperature scaling
            adjusted_predictions = AdvancedPostProcessing.confidence_calibration(
                adjusted_predictions, temperature=1.5)  # Higher temperature = softer predictions
            
            # Apply class-specific thresholds with adaptive adjustment
            class_thresholds = [self.class_thresholds.get(class_name, 0.15) for class_name in CLASS_LABELS]
            adjusted_predictions = AdvancedPostProcessing.class_specific_thresholding(
                adjusted_predictions, class_thresholds)
            
            # Estimate uncertainty from ensemble variance
            _, uncertainty = AdvancedPostProcessing.ensemble_uncertainty_estimation(raw_predictions)
        else:
            # Apply standard class-specific thresholds
            # If a class doesn't meet its threshold, reduce its score
            for i, class_name in enumerate(CLASS_LABELS):
                if class_name in self.class_thresholds:
                    if adjusted_predictions[i] < self.class_thresholds[class_name]:
                        adjusted_predictions[i] *= 0.5  # Reduce score for classes below threshold
        
        # Get the predicted class
        predicted_class_idx = np.argmax(adjusted_predictions)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        
        # Print prediction details
        print(f"\nPrediction details:")
        print(f"  Predicted class: {predicted_class} (confidence: {adjusted_predictions[predicted_class_idx]:.4f})")
        print(f"  Original confidence: {weighted_predictions[predicted_class_idx]:.4f}, Adjusted: {adjusted_predictions[predicted_class_idx]:.4f}")
        if self.use_advanced_postprocessing:
            print(f"  Prediction uncertainty: {uncertainty:.4f}")
        print(f"  Top 3 classes:")
        top_indices = np.argsort(adjusted_predictions)[-3:]
        for idx in reversed(top_indices):
            print(f"    {CLASS_LABELS[idx]}: {adjusted_predictions[idx]:.4f} (original: {weighted_predictions[idx]:.4f})")
        
        return predicted_class, adjusted_predictions, uncertainty
    
    def extract_and_cache_features(self, audio_dir, file_extension='.wav'):
        """Extract features from all audio files in a directory and cache them."""
        # Get all audio files
        audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(file_extension)]
        
        if not audio_files:
            print(f"No {file_extension} files found in {audio_dir}")
            return
        
        print(f"Extracting features from {len(audio_files)} files in {audio_dir}")
        
        # Extract features from each file
        features_dict = {}
        for audio_file in tqdm(audio_files, desc="Extracting features"):
            file_path = os.path.join(audio_dir, audio_file)
            
            try:
                # Get the class name from the file name
                class_name = audio_file.split('.')[0]
                
                # Process the audio file
                features = self.process_audio_file(file_path)
                
                # Add to features dictionary
                features_dict[audio_file] = {
                    'features': features,
                    'class_name': class_name
                }
                
                # Save individual feature file
                output_file = os.path.join(self.features_dir, f"{os.path.splitext(audio_file)[0]}.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(features, f)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        
        # Save all features to a single file
        all_features_file = os.path.join(self.features_dir, 'all_features.pkl')
        with open(all_features_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        # Update cached features
        self.cached_features = features_dict
        
        print(f"Extracted features from {len(features_dict)} files")
        print(f"Saved all features to {all_features_file}")

def test_on_samples(classifier, test_dir, confidence_threshold=0.3, file_extension='.wav', 
apply_augmentation=False):
    """Test the classifier on a directory of audio samples.
    
    Args:
        classifier (UrbanSoundClassifier): The classifier to use
        test_dir (str): Directory containing test audio files
        confidence_threshold (float): Minimum confidence threshold for predictions
        file_extension (str): File extension of audio files
        apply_augmentation (bool): Whether to apply data augmentation during testing
        
    Returns:
        tuple: (accuracy, avg_processing_time, results)
    """
    # Get all audio files
    audio_files = [f for f in os.listdir(test_dir) if f.lower().endswith(file_extension)]
    
    if not audio_files:
        print(f"No {file_extension} files found in {test_dir}")
        return
    
    print(f"Testing on {len(audio_files)} files from {test_dir} with confidence threshold {confidence_threshold}")
    print(f"Advanced features: {classifier.use_advanced_features}, Advanced ensemble: {classifier.use_advanced_ensemble}, Advanced post-processing: {classifier.use_advanced_postprocessing}")
    if apply_augmentation:
        print(f"Data augmentation: Enabled")
    
    # Track results
    correct = 0
    total = 0
    processing_times = []
    results = []
    class_correct = {label: 0 for label in CLASS_LABELS}
    class_total = {label: 0 for label in CLASS_LABELS}
    confusion_matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=int)
    uncertainties = []
    
    # Test each file
    for audio_file in audio_files:
        file_path = os.path.join(test_dir, audio_file)
        
        # Get the true class from the file name
        true_class = audio_file.split('.')[0]
        
        # Time the prediction
        start_time = time.time()
        predicted_class, probabilities, uncertainty = classifier.predict(
            file_path, confidence_threshold=confidence_threshold, apply_augmentation=apply_augmentation)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        processing_times.append(processing_time)
        uncertainties.append(uncertainty)
        
        # Check if prediction is correct
        is_correct = predicted_class == true_class
        if is_correct:
            correct += 1
        total += 1
        
        # Update class-specific metrics
        if true_class in CLASS_LABELS:
            true_idx = CLASS_LABELS.index(true_class)
            pred_idx = CLASS_LABELS.index(predicted_class) if predicted_class in CLASS_LABELS else -1
            
            class_total[true_class] += 1
            if is_correct:
                class_correct[true_class] += 1
            
            if pred_idx >= 0:
                confusion_matrix[true_idx, pred_idx] += 1
        
        # Store result
        result = {
            'file': audio_file,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'is_correct': is_correct,
            'processing_time': processing_time,
            'confidence': float(np.max(probabilities)) if probabilities is not None else 0.0,
            'uncertainty': float(uncertainty),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        results.append(result)
        
        # Print result
        print(f"File: {audio_file}, True: {true_class}, Predicted: {predicted_class}, Confidence: {result['confidence']:.4f}, Uncertainty: {uncertainty:.4f}, Correct: {is_correct}, Time: {processing_time:.2f} ms")
    
    # Calculate accuracy and average processing time
    accuracy = correct / total if total > 0 else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Average processing time: {avg_processing_time:.2f} ms")
    print(f"Average prediction uncertainty: {avg_uncertainty:.4f}")
    
    # Print class-specific accuracy
    print("\nClass-specific accuracy:")
    for label in CLASS_LABELS:
        if class_total[label] > 0:
            class_acc = class_correct[label] / class_total[label]
            print(f"  {label}: {class_acc:.2%} ({class_correct[label]}/{class_total[label]})")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("  " + " ".join(f"{label[:3]:>5}" for label in CLASS_LABELS))
    for i, row in enumerate(confusion_matrix):
        print(f"{CLASS_LABELS[i][:7]:>7}: " + " ".join(f"{count:>5}" for count in row))
    
    # Print prediction details
    print("\nPrediction details:")
    for result in results:
        print(f"  {result['file']}: {result['true_class']} -> {result['predicted_class']} (conf: {result['confidence']:.4f}, unc: {result['uncertainty']:.4f}) ({'✓' if result['is_correct'] else '✗'})")
    
    # Calculate correlation between uncertainty and correctness
    correct_uncertainties = [r['uncertainty'] for r in results if r['is_correct']]
    incorrect_uncertainties = [r['uncertainty'] for r in results if not r['is_correct']]
    
    if correct_uncertainties and incorrect_uncertainties:
        avg_correct_uncertainty = sum(correct_uncertainties) / len(correct_uncertainties)
        avg_incorrect_uncertainty = sum(incorrect_uncertainties) / len(incorrect_uncertainties)
        print(f"\nUncertainty analysis:")
        print(f"  Average uncertainty for correct predictions: {avg_correct_uncertainty:.4f}")
        print(f"  Average uncertainty for incorrect predictions: {avg_incorrect_uncertainty:.4f}")
    
    return accuracy, avg_processing_time, results

# Recommendations for Further Improvement

class AdvancedFeatureExtractor:
    """Advanced feature extraction for improved audio classification."""
    
    def __init__(self, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
        """Initialize the feature extractor with audio parameters."""
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, audio_data, n_mfcc=20):
        """Extract MFCC features from audio data."""
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sr, n_mfcc=n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        return mfccs
    
    def extract_spectral_contrast(self, audio_data, n_bands=6):
        """Extract spectral contrast features."""
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sr, n_bands=n_bands, n_fft=self.n_fft, hop_length=self.hop_length)
        return contrast
    
    def extract_chroma(self, audio_data, n_chroma=12):
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sr, n_chroma=n_chroma, n_fft=self.n_fft, hop_length=self.hop_length)
        return chroma
    
    def extract_tonnetz(self, audio_data):
        """Extract tonnetz (tonal centroid) features."""
        y_harm, y_perc = librosa.effects.hpss(audio_data)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=self.sr)
        return tonnetz
    
    def extract_onset_strength(self, audio_data):
        """Extract onset strength."""
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sr, hop_length=self.hop_length)
        return onset_env.reshape(1, -1)  # Reshape to 2D
    
    def extract_tempo_features(self, audio_data):
        """Extract tempo and beat-related features."""
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sr, hop_length=self.hop_length)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length)
        return np.array([[tempo]])  # Return as 2D array
    
    def extract_all_features(self, audio_data, fixed_time_dim=173):
        """Extract all features and combine them."""
        # Extract individual features
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = self.extract_mfcc(audio_data)
        contrast = self.extract_spectral_contrast(audio_data)
        chroma = self.extract_chroma(audio_data)
        tonnetz = self.extract_tonnetz(audio_data)
        
        # Ensure all features have the same time dimension
        features = []
        for feat in [mel_spec_db, mfccs, contrast, chroma, tonnetz]:
            # Adjust time dimension
            current_time_dim = feat.shape[1]
            if current_time_dim < fixed_time_dim:
                pad_width = ((0, 0), (0, fixed_time_dim - current_time_dim))
                feat_padded = np.pad(feat, pad_width, 'constant')
                features.append(feat_padded)
            elif current_time_dim > fixed_time_dim:
                feat_trimmed = feat[:, :fixed_time_dim]
                features.append(feat_trimmed)
            else:
                features.append(feat)
        
        # Normalize each feature to [0, 1] range
        normalized_features = []
        for feat in features:
            feat_min, feat_max = np.min(feat), np.max(feat)
            if feat_max > feat_min:  # Avoid division by zero
                feat_norm = (feat - feat_min) / (feat_max - feat_min)
            else:
                feat_norm = np.zeros_like(feat)
            normalized_features.append(feat_norm)
        
        # Stack all features along the feature dimension
        combined_features = np.vstack(normalized_features)
        
        return combined_features

class DataAugmentation:
    """Audio data augmentation techniques for model training."""
    
    @staticmethod
    def add_noise(audio_data, noise_level=0.005):
        """Add random noise to audio."""
        noise = np.random.normal(0, noise_level, len(audio_data))
        return audio_data + noise
    
    @staticmethod
    def time_shift(audio_data, shift_range=0.2):
        """Randomly shift audio in time."""
        shift = int(np.random.uniform(-shift_range, shift_range) * len(audio_data))
        if shift > 0:
            shifted_data = np.pad(audio_data, (0, shift), 'constant')
            shifted_data = shifted_data[:len(audio_data)]
        else:
            shifted_data = np.pad(audio_data, (-shift, 0), 'constant')
            shifted_data = shifted_data[-shift:]
        return shifted_data
    
    @staticmethod
    def pitch_shift(audio_data, sr, n_steps=2):
        """Shift the pitch of audio."""
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(audio_data, rate=1.2):
        """Time stretch audio without changing pitch."""
        return librosa.effects.time_stretch(audio_data, rate=rate)
    
    @staticmethod
    def apply_random_augmentation(audio_data, sr):
        """Apply a random augmentation technique."""
        augmentation_type = np.random.choice(['noise', 'time_shift', 'pitch_shift', 'time_stretch', 'none'])
        
        if augmentation_type == 'noise':
            return DataAugmentation.add_noise(audio_data)
        elif augmentation_type == 'time_shift':
            return DataAugmentation.time_shift(audio_data)
        elif augmentation_type == 'pitch_shift':
            return DataAugmentation.pitch_shift(audio_data, sr, n_steps=np.random.uniform(-4, 4))
        elif augmentation_type == 'time_stretch':
            return DataAugmentation.time_stretch(audio_data, rate=np.random.uniform(0.8, 1.2))
        else:  # 'none'
            return audio_data

class AdvancedEnsembleStrategy:
    """Advanced ensemble strategies for improved prediction."""
    
    @staticmethod
    def stacked_generalization(base_predictions, meta_model=None):
        """Implement stacked generalization (stacking) ensemble method."""
        # This is a placeholder for actual stacking implementation
        # In a real implementation, you would train a meta-model on base model predictions
        if meta_model is None:
            # Simple weighted average as fallback
            return np.mean(base_predictions, axis=0)
        else:
            # Use meta-model to combine predictions
            return meta_model.predict(base_predictions)
    
    @staticmethod
    def bayesian_model_averaging(predictions, model_priors=None):
        """Implement Bayesian model averaging."""
        if model_priors is None:
            # Equal priors if not specified
            model_priors = np.ones(len(predictions)) / len(predictions)
        
        # Normalize priors
        model_priors = np.array(model_priors) / sum(model_priors)
        
        # Weighted average based on priors
        weighted_predictions = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_predictions += pred * model_priors[i]
        
        return weighted_predictions
    
    @staticmethod
    def dynamic_weighting(predictions, confidences):
        """Dynamically weight models based on their confidence."""
        # Convert confidences to weights
        weights = np.array(confidences) / sum(confidences) if sum(confidences) > 0 else np.ones(len(confidences)) / len(confidences)
        
        # Apply weights to predictions
        weighted_predictions = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_predictions += pred * weights[i]
        
        return weighted_predictions

class AdvancedPostProcessing:
    """Advanced post-processing techniques for improved predictions."""
    
    @staticmethod
    def temporal_smoothing(predictions, window_size=3, weights=None):
        """Apply temporal smoothing to a sequence of predictions."""
        if len(predictions) < window_size:
            return predictions[-1]  # Return the most recent prediction if not enough history
        
        # Use exponential weights if not specified
        if weights is None:
            weights = np.exp(np.linspace(0, 1, window_size))
            weights = weights / sum(weights)  # Normalize
        
        # Apply weighted average over the window
        smoothed = np.zeros_like(predictions[0])
        for i in range(window_size):
            smoothed += predictions[-(i+1)] * weights[i]
        
        return smoothed
    
    @staticmethod
    def confidence_calibration(predictions, temperature=1.0):
        """Calibrate prediction confidences using temperature scaling."""
        # Apply temperature scaling
        scaled_predictions = np.exp(np.log(predictions) / temperature)
        # Normalize to get valid probabilities
        return scaled_predictions / np.sum(scaled_predictions)
    
    @staticmethod
    def class_specific_thresholding(predictions, class_thresholds):
        """Apply class-specific thresholds with adaptive adjustment."""
        adjusted_predictions = predictions.copy()
        
        # Apply thresholds
        for i, threshold in enumerate(class_thresholds):
            if predictions[i] < threshold:
                # Reduce confidence for predictions below threshold
                adjusted_predictions[i] *= (predictions[i] / threshold)
        
        return adjusted_predictions
    
    @staticmethod
    def ensemble_uncertainty_estimation(predictions_list):
        """Estimate prediction uncertainty from ensemble variance."""
        # Calculate mean prediction across ensemble
        mean_prediction = np.mean(predictions_list, axis=0)
        
        # Calculate variance for each class
        variance = np.var(predictions_list, axis=0)
        
        # Calculate uncertainty as average variance
        uncertainty = np.mean(variance)
        
        return mean_prediction, uncertainty

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Urban Sound Classifier')
    parser.add_argument('--mode', choices=['predict', 'test', 'extract', 'augment', 'advanced_features'], default='test', 
                        help='Mode: predict a single file, test on multiple files, extract features, augment data, or extract advanced features')
    parser.add_argument('--file', help='Audio file to predict (for predict mode)')
    parser.add_argument('--test_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_audio_samples'), 
                        help='Directory containing test audio files (for test mode)')
    parser.add_argument('--model_dir', default='tflite_models', help='Directory containing TFLite model files')
    parser.add_argument('--use_cached_features', action='store_true', help='Use cached features if available')
    parser.add_argument('--features_dir', default='extracted_features', help='Directory containing cached features')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Minimum confidence threshold for predictions')
    parser.add_argument('--use_advanced_features', action='store_true', help='Use advanced feature extraction')
    parser.add_argument('--use_advanced_ensemble', action='store_true', help='Use advanced ensemble strategy')
    parser.add_argument('--use_advanced_postprocessing', action='store_true', help='Use advanced post-processing')
    parser.add_argument('--augment_data', action='store_true', help='Apply data augmentation')
    
    args = parser.parse_args()
    
    # Create classifier with advanced features if requested
    classifier = UrbanSoundClassifier(
        model_dir=args.model_dir,
        use_cached_features=args.use_cached_features,
        features_dir=args.features_dir,
        use_advanced_features=args.use_advanced_features,
        use_advanced_ensemble=args.use_advanced_ensemble,
        use_advanced_postprocessing=args.use_advanced_postprocessing
    )
    
    # Run in appropriate mode
    if args.mode == 'predict':
        if not args.file:
            print("Error: --file is required for predict mode")
            return
        
        # Make prediction
        predicted_class, probabilities, uncertainty = classifier.predict(
            args.file, 
            confidence_threshold=args.confidence_threshold,
            apply_augmentation=args.augment_data
        )
        
        # Print result
        if predicted_class is None:
            print(f"\nNo prediction could be made for {args.file} with sufficient confidence")
        else:
            print(f"\nPrediction for {args.file}:")
            print(f"Predicted class: {predicted_class}")
            print(f"Uncertainty: {uncertainty:.4f}")
            print("Class probabilities:")
            for i, label in enumerate(CLASS_LABELS):
                print(f"  {label}: {probabilities[i]:.4f}")
    
    elif args.mode == 'extract':
        # Extract and cache features
        classifier.extract_and_cache_features(args.test_dir)
    
    elif args.mode == 'augment':
        # Demonstrate data augmentation
        if not args.file:
            print("Error: --file is required for augment mode")
            return
            
        print(f"Demonstrating data augmentation on {args.file}...")
        audio_data, sr = librosa.load(args.file, sr=SAMPLE_RATE, mono=True)
        
        # Apply different augmentations
        augmentations = {
            "original": audio_data,
            "noise": DataAugmentation.add_noise(audio_data),
            "time_shift": DataAugmentation.time_shift(audio_data),
            "pitch_shift": DataAugmentation.pitch_shift(audio_data, sr=sr, n_steps=2),
            "time_stretch": DataAugmentation.time_stretch(audio_data, rate=1.2)
        }
        
        # Save augmented samples
        output_dir = os.path.join(os.path.dirname(args.file), "augmented")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, audio in augmentations.items():
            output_path = os.path.join(output_dir, f"{name}_{os.path.basename(args.file)}")
            sf.write(output_path, audio, sr)
            print(f"Saved {name} augmentation to {output_path}")
    
    else:  # test mode
        # Test on samples
        test_on_samples(
            classifier, 
            args.test_dir, 
            confidence_threshold=args.confidence_threshold,
            apply_augmentation=args.augment_data
        )

if __name__ == "__main__":
    main()