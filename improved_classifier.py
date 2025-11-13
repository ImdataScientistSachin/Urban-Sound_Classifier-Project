import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
from collections import defaultdict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom model loader
from custom_model_loader import load_model, predict

# Import project modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.config import CLASS_LABELS
from standardized_feature_extraction import process_audio_file

class ImprovedClassifier:
    """
    Improved classifier that uses standardized feature extraction and ensemble model approach.
    """
    def __init__(self, model_weights_dir=None, use_best_models_only=False, top_k_models=3):
        """
        Initialize the improved classifier.
        
        Args:
            model_weights_dir (str): Directory containing model weight files
            use_best_models_only (bool): Whether to use only the best performing models
            top_k_models (int): Number of top models to use if use_best_models_only is True
        """
        if model_weights_dir is None:
            model_weights_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models',
                'unetropolis-hybrid-unet-urbansound_96.63_weights'
            )
        
        self.models = []
        self.model_paths = []
        self.model_weights_dir = model_weights_dir
        self.use_best_models_only = use_best_models_only
        self.top_k_models = top_k_models
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """
        Load models from the weights directory.
        """
        print(f"Loading models from {self.model_weights_dir}...")
        
        if not os.path.exists(self.model_weights_dir):
            print(f"Model weights directory {self.model_weights_dir} does not exist")
            return
        
        # Get all model files
        model_files = [f for f in os.listdir(self.model_weights_dir) if f.endswith('.h5')]
        
        if not model_files:
            print(f"No model files found in {self.model_weights_dir}")
            return
        
        print(f"Found {len(model_files)} model files: {model_files}")
        
        # If using best models only, select specific models based on architecture
        if self.use_best_models_only:
            # Prioritize UNet models which typically perform better for audio classification
            unet_models = [f for f in model_files if 'UNet' in f]
            cnn_models = [f for f in model_files if 'CNN' in f and 'Simple' not in f]
            simple_cnn_models = [f for f in model_files if 'SimpleCNN' in f]
            
            # Select top models from each architecture type
            selected_models = []
            selected_models.extend(unet_models[:min(self.top_k_models, len(unet_models))])
            
            # Add CNN models if we need more
            if len(selected_models) < self.top_k_models:
                selected_models.extend(cnn_models[:min(self.top_k_models - len(selected_models), len(cnn_models))])
            
            # Add SimpleCNN models if we still need more
            if len(selected_models) < self.top_k_models:
                selected_models.extend(simple_cnn_models[:min(self.top_k_models - len(selected_models), len(simple_cnn_models))])
            
            model_files = selected_models
            print(f"Using {len(model_files)} best models: {model_files}")
        
        # Load each model
        for model_file in model_files:
            model_path = os.path.join(self.model_weights_dir, model_file)
            try:
                print(f"Loading model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                
                # Compile the model with metrics
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                )
                
                self.models.append(model)
                self.model_paths.append(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
    
    def predict(self, features):
        """
        Make a prediction using the ensemble of models.
        
        Args:
            features (numpy.ndarray): Preprocessed audio features
            
        Returns:
            tuple: (predicted_class_name, confidence, all_confidences)
        """
        if not self.models:
            raise ValueError("No models loaded in the classifier")
        
        # Initialize aggregated confidences
        aggregated_confidences = {class_name: 0.0 for class_name in CLASS_LABELS}
        
        # Get predictions from each model
        model_predictions = []
        for i, model in enumerate(self.models):
            try:
                print(f"Getting prediction from model {i+1}/{len(self.models)}")
                
                # Ensure features are in the right shape for the model
                input_features = self._prepare_features_for_model(features, model)
                
                # Make prediction
                predictions = self._predict_with_model(model, input_features)
                
                # Apply softmax to get probabilities
                if len(predictions.shape) == 2:  # (batch_size, num_classes)
                    probabilities = tf.nn.softmax(predictions, axis=-1).numpy()[0]
                else:  # (num_classes,)
                    probabilities = tf.nn.softmax(predictions).numpy()
                
                # Create confidences dictionary
                confidences = {}
                for j, class_name in enumerate(CLASS_LABELS):
                    if j < len(probabilities):
                        confidences[class_name] = float(probabilities[j])
                    else:
                        confidences[class_name] = 0.0
                
                model_predictions.append(confidences)
                
                # Log individual model predictions
                print(f"Model {i+1} prediction:")
                for class_name, conf in confidences.items():
                    print(f"  {class_name}: {conf:.4f}")
            except Exception as e:
                print(f"Error getting prediction from model {i+1}: {e}")
                import traceback
                print(traceback.format_exc())
        
        # If no valid predictions, raise an error
        if not model_predictions:
            raise ValueError("No valid predictions from any model")
        
        # Aggregate confidences (weighted average based on model confidence)
        total_weight = 0
        for confidences in model_predictions:
            # Calculate the max confidence as a weight
            max_conf = max(confidences.values())
            weight = max_conf  # Higher confidence gets higher weight
            
            for class_name, conf in confidences.items():
                aggregated_confidences[class_name] += conf * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for class_name in aggregated_confidences:
                aggregated_confidences[class_name] /= total_weight
        
        # Get the predicted class and confidence
        predicted_class = max(aggregated_confidences.items(), key=lambda x: x[1])[0]
        confidence = aggregated_confidences[predicted_class]
        
        print(f"Ensemble prediction: {predicted_class} with confidence {confidence:.4f}")
        print("All confidences:")
        for class_name, conf in aggregated_confidences.items():
            print(f"  {class_name}: {conf:.4f}")
        
        return predicted_class, confidence, aggregated_confidences
    
    def _prepare_features_for_model(self, features, model):
        """
        Prepare features for the model by ensuring correct shape.
        
        Args:
            features (numpy.ndarray): Input features
            model: TensorFlow model
            
        Returns:
            numpy.ndarray: Prepared features
        """
        # Get the expected input shape from the model
        input_shape = model.input_shape
        
        # Convert to numpy array if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Print shapes for debugging
        print(f"Original features shape: {features.shape}")
        print(f"Model expects input shape: {input_shape}")
        
        # Handle different input shapes
        if len(features.shape) == 3:  # (n_mels, time_steps, 1)
            # Add batch dimension
            prepared_features = features.reshape(1, *features.shape)
        elif len(features.shape) == 2:  # (n_mels, time_steps)
            # Add batch and channel dimensions
            prepared_features = features.reshape(1, *features.shape, 1)
        else:
            prepared_features = features
        
        # Ensure the feature dimensions match the model's expected input
        if len(input_shape) > 1 and len(prepared_features.shape) > 1:
            # Check if we need to transpose dimensions
            if input_shape[1] != prepared_features.shape[1] and input_shape[1] == prepared_features.shape[2]:
                print(f"Transposing features from {prepared_features.shape} to match model input")
                prepared_features = np.transpose(prepared_features, (0, 2, 1, 3))
            
            # Adjust time dimension if needed
            if len(input_shape) > 2 and len(prepared_features.shape) > 2:
                if input_shape[2] != prepared_features.shape[2]:
                    print(f"Adjusting time dimension from {prepared_features.shape[2]} to {input_shape[2]}")
                    if prepared_features.shape[2] < input_shape[2]:
                        # Pad with zeros
                        pad_width = ((0, 0), (0, 0), (0, input_shape[2] - prepared_features.shape[2]), (0, 0))
                        prepared_features = np.pad(prepared_features, pad_width, 'constant')
                    else:
                        # Trim to expected length
                        prepared_features = prepared_features[:, :, :input_shape[2], :]
        
        # Convert to float32
        prepared_features = prepared_features.astype(np.float32)
        
        # Check for NaN or Inf values
        if np.isnan(prepared_features).any() or np.isinf(prepared_features).any():
            print("Warning: NaN or Inf values detected in features. Replacing with zeros.")
            prepared_features = np.nan_to_num(prepared_features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize to [0, 1] range if needed
        if np.max(prepared_features) > 1.0 or np.min(prepared_features) < 0.0:
            print("Normalizing features to [0, 1] range")
            prepared_features = (prepared_features - np.min(prepared_features)) / (np.max(prepared_features) - np.min(prepared_features))
        
        print(f"Prepared features shape: {prepared_features.shape}")
        return prepared_features
    
    def _predict_with_model(self, model, features):
        """
        Make a prediction with a model, with multiple fallback approaches.
        
        Args:
            model: TensorFlow model
            features (numpy.ndarray): Prepared features
            
        Returns:
            numpy.ndarray: Raw model predictions
        """
        # Try different prediction approaches
        approaches = [
            ("Standard predict", lambda: model.predict(features, verbose=0)),
            ("Direct call", lambda: model(features, training=False).numpy()),
            ("Eager execution", lambda: tf.keras.backend.get_value(model(features, training=False))),
            ("tf.function", lambda: tf.function(lambda x: model(x, training=False))(features).numpy())
        ]
        
        last_error = None
        for approach_name, approach_func in approaches:
            try:
                print(f"Trying prediction approach: {approach_name}")
                predictions = approach_func()
                print(f"Prediction successful with {approach_name}")
                return predictions
            except Exception as e:
                print(f"Error with {approach_name} approach: {e}")
                last_error = e
        
        # If all approaches fail, raise the last error
        raise ValueError(f"All prediction approaches failed. Last error: {last_error}")

def process_audio_and_predict(file_path, use_best_models_only=True, top_k_models=3):
    """
    Process an audio file and make a prediction using the improved classifier.
    
    Args:
        file_path (str): Path to the audio file
        use_best_models_only (bool): Whether to use only the best performing models
        top_k_models (int): Number of top models to use if use_best_models_only is True
        
    Returns:
        tuple: (predicted_class_name, confidence, all_confidences)
    """
    try:
        # Initialize the improved classifier
        classifier = ImprovedClassifier(use_best_models_only=use_best_models_only, top_k_models=top_k_models)
        
        if not classifier.models:
            raise ValueError("No models loaded in the classifier")
        
        print(f"Processing audio file: {file_path}")
        
        # Process the audio file with standardized feature extraction
        features = process_audio_file(file_path)
        
        # Make prediction using the classifier
        return classifier.predict(features)
    except Exception as e:
        print(f"Error in process_audio_and_predict: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def test_on_samples(test_dir, use_best_models_only=True, top_k_models=3):
    """
    Test the improved classifier on all samples in the test directory.
    
    Args:
        test_dir (str): Directory containing test audio samples
        use_best_models_only (bool): Whether to use only the best performing models
        top_k_models (int): Number of top models to use if use_best_models_only is True
        
    Returns:
        dict: Test results
    """
    try:
        # Get all WAV files in the test directory
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        
        if not test_files:
            print(f"No WAV files found in {test_dir}")
            return None
        
        print(f"Testing on {len(test_files)} files in {test_dir}")
        
        # Initialize results
        results = {
            'predictions': [],
            'correct_predictions': 0,
            'total_predictions': len(test_files),
            'processing_times': [],
            'total_times': [],
            'class_distribution': defaultdict(int)
        }
        
        # Initialize the improved classifier
        classifier = ImprovedClassifier(use_best_models_only=use_best_models_only, top_k_models=top_k_models)
        
        if not classifier.models:
            raise ValueError("No models loaded in the classifier")
        
        # Test each file
        for test_file in test_files:
            file_path = os.path.join(test_dir, test_file)
            expected_class = test_file.split('.')[0]  # Assuming filename is the class name
            
            # Process the audio file and make prediction
            start_time = time.time()
            try:
                # Process the audio file with standardized feature extraction
                features = process_audio_file(file_path)
                
                # Make prediction using the classifier
                predicted_class, confidence, all_confidences = classifier.predict(features)
                
                # Calculate times
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update results
                results['predictions'].append({
                    'file': test_file,
                    'expected_class': expected_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_confidences': all_confidences,
                    'processing_time': processing_time
                })
                
                results['processing_times'].append(processing_time)
                results['total_times'].append(processing_time)
                results['class_distribution'][predicted_class] += 1
                
                if predicted_class == expected_class:
                    results['correct_predictions'] += 1
                
                print(f"  {test_file}: Expected {expected_class}, Predicted {predicted_class}, Confidence {confidence:.4f}, Time {processing_time:.2f} ms")
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
        
        # Calculate accuracy
        results['accuracy'] = results['correct_predictions'] / results['total_predictions'] if results['total_predictions'] > 0 else 0
        
        # Calculate average times
        results['avg_processing_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
        results['avg_total_time'] = np.mean(results['total_times']) if results['total_times'] else 0
        
        # Print summary
        print(f"\nAccuracy: {results['accuracy']:.2%} ({results['correct_predictions']}/{results['total_predictions']})")
        print(f"Average processing time: {results['avg_processing_time']:.2f} ms")
        print(f"Average total time: {results['avg_total_time']:.2f} ms")
        print("Class distribution:")
        for class_name, count in results['class_distribution'].items():
            percentage = count / results['total_predictions'] * 100 if results['total_predictions'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Print detailed prediction results
        print("\nDetailed prediction results:")
        for pred in results['predictions']:
            print(f"  {pred['file']}: Expected {pred['expected_class']}, Predicted {pred['predicted_class']}, Confidence {pred['confidence']:.4f}")
        
        return results
    except Exception as e:
        print(f"Error testing on samples: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Improved Urban Sound Classifier')
    parser.add_argument('--mode', choices=['predict', 'test'], default='predict', help='Mode: predict a single file or test on all samples')
    parser.add_argument('--file', help='Path to the audio file for prediction')
    parser.add_argument('--test_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_audio_samples'), help='Directory containing test audio samples')
    parser.add_argument('--use_best_models', action='store_true', help='Use only the best performing models')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top models to use if use_best_models is True')
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        if not args.file:
            print("Error: --file is required for predict mode")
            return
        
        # Process the audio file and make a prediction
        try:
            predicted_class, confidence, all_confidences = process_audio_and_predict(
                args.file, 
                use_best_models_only=args.use_best_models, 
                top_k_models=args.top_k
            )
            
            print(f"\nPrediction result:")
            print(f"  Class: {predicted_class}")
            print(f"  Confidence: {confidence:.4f}")
            
            print("\nAll confidences:")
            for class_name, conf in all_confidences.items():
                print(f"  {class_name}: {conf:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.mode == 'test':
        # Test on all samples in the test directory
        test_on_samples(
            args.test_dir, 
            use_best_models_only=args.use_best_models, 
            top_k_models=args.top_k
        )

if __name__ == "__main__":
    main()