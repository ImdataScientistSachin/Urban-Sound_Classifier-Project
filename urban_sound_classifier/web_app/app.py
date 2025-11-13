import os
import time
import logging
import numpy as np
import threading
import platform
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from ..config.config_manager import ConfigManager
from ..models.prediction import Predictor
from ..feature_extraction import MelSpectrogramExtractor
from ..utils.audio import AudioUtils
from ..utils.file import FileUtils

class WebApp:
    """
    Thread-safe web application for audio classification.
    
    This class handles the web interface for audio classification,
    including file uploads, predictions, and visualization.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        predictor (Predictor): Predictor instance
        feature_extractor (MelSpectrogramExtractor): Feature extractor instance
        app (Flask): Flask application instance
        _plot_lock (threading.Lock): Lock for matplotlib operations
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the WebApp.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize predictor
        self.predictor = Predictor(config)
        
        # Initialize feature extractor
        self.feature_extractor = MelSpectrogramExtractor(config)
        
        # Load models during initialization
        self._load_models()
        
        # Initialize Flask app
        self.app = Flask(
            __name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates')
        )
        
        # Enable CORS
        CORS(self.app)
        
        # Configure upload folder
        self.app.config['UPLOAD_FOLDER'] = config.get('PATHS.upload_dir', 'uploads')
        self.app.config['MAX_CONTENT_LENGTH'] = config.get('WEB.max_upload_size', 16) * 1024 * 1024  # MB to bytes
        
        # Configure request timeout
        self.app.config['TIMEOUT'] = config.get('WEB.request_timeout', 60)  # seconds
        
        # Create upload folder if it doesn't exist
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Initialize lock for matplotlib operations (not thread-safe)
        self._plot_lock = threading.Lock()
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """
        Register Flask routes.
        """
        # Home page
        @self.app.route('/')
        def home():
            return render_template('index.html')
            
        # Debug page for predict_sample
        @self.app.route('/debug')
        def debug_predict_sample():
            return render_template('debug_predict_sample.html')
        
        # Upload and predict endpoint is defined below
                
        # Predict from sample audio files
        @self.app.route('/predict_sample', methods=['POST'])
        def predict_sample():
            # Check if sample path exists in request
            if 'sample_path' not in request.form:
                return jsonify({'error': 'No sample path provided'}), 400
            
            url_path = request.form['sample_path']
            self.logger.info(f"Received sample path: {url_path}")
            
            # Convert URL path to file path
            if url_path.startswith('/audio_samples/'):
                # Extract filename from URL path
                filename = os.path.basename(url_path)
                sample_path = os.path.join('test_audio_samples', filename)
                self.logger.info(f"Converted to test_audio_samples path: {sample_path}")
            elif url_path.startswith('/urbansound_samples/'):
                # Extract fold and filename from URL path
                # Format: /urbansound_samples/fold{num}/{filename}
                parts = url_path.split('/')
                if len(parts) >= 4:
                    fold = parts[2]  # e.g., fold1
                    filename = parts[3]  # e.g., 101415-3-0-2.wav
                    sample_path = os.path.join('data/UrbanSound8K', fold, filename)
                    self.logger.info(f"Converted to UrbanSound8K path: {sample_path}")
                else:
                    self.logger.error(f"Invalid sample path format: {url_path}")
                    return jsonify({'error': 'Invalid sample path format'}), 400
            else:
                # Try to use the path directly (for backward compatibility)
                sample_path = url_path
                self.logger.info(f"Using direct path: {sample_path}")
            
            # Validate the sample path (ensure it exists)
            if not os.path.exists(sample_path):
                self.logger.error(f"Sample file not found: {sample_path}")
                return jsonify({'error': f'Sample file not found: {sample_path}'}), 404
                
            # Check if the sample path is in an allowed directory
            allowed_dirs = [
                os.path.abspath('data/UrbanSound8K'),
                os.path.abspath('test_audio_samples')
            ]
            
            abs_path = os.path.abspath(sample_path)
            self.logger.info(f"Absolute path: {abs_path}")
            self.logger.info(f"Allowed directories: {allowed_dirs}")
            
            if not any(abs_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                self.logger.error(f"Invalid sample path (not in allowed directories): {abs_path}")
                return jsonify({'error': 'Invalid sample path'}), 403
                
            try:
                # Convert audio to WAV format if needed
                if not sample_path.lower().endswith('.wav'):
                    self.logger.info(f"Converting audio to WAV format: {sample_path}")
                    temp_path = os.path.join(
                        self.app.config['UPLOAD_FOLDER'],
                        f"temp_{int(time.time())}_{os.path.basename(sample_path)}.wav"
                    )
                    sample_rate = self.config.get('AUDIO.sample_rate', 22050)
                    temp_path = AudioUtils.convert_audio_to_wav(sample_path, output_path=temp_path, sample_rate=sample_rate)
                    sample_path = temp_path
                    self.logger.info(f"Converted to WAV: {sample_path}")
                
                # Make prediction
                self.logger.info(f"Making prediction for: {sample_path}")
                results = self.predictor.predict(sample_path, top_k=5)
                
                # Generate spectrogram
                spectrogram_path = self._generate_spectrogram(sample_path)
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'label': result['label'],
                        'probability': float(result['probability']),
                        'percentage': f"{float(result['probability']) * 100:.2f}%"
                    })
                
                self.logger.info(f"Prediction successful for: {sample_path}")
                return jsonify({
                    'predictions': formatted_results,
                    'spectrogram': os.path.basename(spectrogram_path),
                    'audio_filename': os.path.basename(sample_path)
                })
            except ValueError as e:
                if "Audio data must be of type numpy.ndarray" in str(e):
                    self.logger.error(f"Error processing audio file: {e}")
                    return jsonify({'error': 'Unable to process audio file. The file may be corrupted or in an unsupported format.'}), 400
                raise
            except Exception as e:
                self.logger.error(f"Error processing sample file: {e}")
                return jsonify({'error': str(e)}), 500
                
        # Serve uploaded audio files
        @self.app.route('/audio/<filename>')
        def serve_audio(filename):
            return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)
            
        # Handle prediction endpoint
        @self.app.route('/predict', methods=['POST'])
        def predict():
            # Check if file is in request
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            
            # Check if file is empty
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            # Check file extension
            allowed_extensions = self.config.get('AUDIO.valid_extensions', ['.wav', '.mp3', '.ogg', '.flac'])
            if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                return jsonify({'error': f'File type not allowed. Allowed types: {allowed_extensions}'}), 400
            
            # Check file size before processing
            max_size_mb = self.config.get('WEB.max_upload_size', 16)
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer
            
            if file_size > max_size_mb * 1024 * 1024:
                return jsonify({
                    'error': f'File too large. Maximum size is {max_size_mb} MB. Please upload a smaller file.'
                }), 413
                
            try:
                # Save file with a unique name to prevent conflicts
                filename = secure_filename(file.filename)
                timestamp = int(time.time())
                unique_id = os.urandom(4).hex()  # Add random component for uniqueness
                filename = f"{timestamp}_{unique_id}_{filename}"
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                self.logger.debug(f"Processing uploaded file: {file_path}")
                
                # Make prediction with detailed error handling
                try:
                    results = self.predictor.predict(file_path, top_k=5)
                except ValueError as ve:
                    self.logger.error(f"ValueError during prediction: {ve}")
                    return jsonify({'error': f"Invalid audio data: {str(ve)}"}), 400
                except TypeError as te:
                    self.logger.error(f"TypeError during prediction: {te}")
                    return jsonify({'error': f"Type error: {str(te)}"}), 400
                except Exception as pred_error:
                    self.logger.error(f"Prediction error: {pred_error}")
                    return jsonify({'error': f"Failed to process audio: {str(pred_error)}"}), 500
                
                # Generate spectrogram
                try:
                    spectrogram_path = self._generate_spectrogram(file_path)
                except Exception as spec_error:
                    self.logger.error(f"Spectrogram generation error: {spec_error}")
                    # Continue without spectrogram if it fails
                    spectrogram_path = None
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'label': result['label'],
                        'probability': float(result['probability']),
                        'percentage': f"{float(result['probability']) * 100:.2f}%"
                    })
                
                response_data = {
                    'predictions': formatted_results,
                    'audio_filename': os.path.basename(file_path)
                }
                
                if spectrogram_path:
                    response_data['spectrogram'] = os.path.basename(spectrogram_path)
                
                return jsonify(response_data)
            except Exception as e:
                self.logger.error(f"Error processing file: {e}")
                return jsonify({'error': f"Server error: {str(e)}"}), 500
        
        # Serve spectrograms
        @self.app.route('/spectrograms/<filename>')
        def serve_spectrogram(filename):
            return send_from_directory(os.path.join(self.app.config['UPLOAD_FOLDER'], 'spectrograms'), filename)
        
        # Sample sounds route
        @self.app.route('/samples/<category>')
        def get_samples(category):
            # Define the base directory for samples
            samples_dir = os.path.abspath('test_audio_samples')
            urbansound_dir = os.path.abspath('data/UrbanSound8K')
            
            # Validate category name (prevent directory traversal)
            valid_categories = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                               'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
            
            if category not in valid_categories:
                return jsonify({'error': 'Invalid category'}), 400
                
            # First check test_audio_samples for the specific category file
            sample_files = []
            
            # Check for direct matches in test_audio_samples
            for file in os.listdir(samples_dir):
                if file.endswith('.wav') and category in file.lower():
                    sample_files.append({
                        'name': file,
                        'path': f'/audio_samples/{file}'
                    })
            
            # If we have enough samples, return them
            if len(sample_files) >= 3:
                return jsonify({'samples': sample_files})
                
            # Otherwise, look in UrbanSound8K dataset
            # We'll search through fold1 to fold10 for matching files
            for fold_num in range(1, 11):
                fold_dir = os.path.join(urbansound_dir, f'fold{fold_num}')
                if os.path.exists(fold_dir):
                    # Read the CSV file to match class names with filenames
                    csv_path = os.path.join(urbansound_dir, 'UrbanSound8K.csv')
                    if os.path.exists(csv_path):
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path)
                            # Filter by class name matching the category
                            class_id = valid_categories.index(category)
                            matching_files = df[df['class'] == class_id]['slice_file_name'].tolist()
                            
                            # Add matching files from this fold
                            for file in os.listdir(fold_dir):
                                if file in matching_files and len(sample_files) < 5:
                                    sample_files.append({
                                        'name': file,
                                        'path': f'/urbansound_samples/fold{fold_num}/{file}'
                                    })
                        except Exception as e:
                            self.logger.error(f"Error reading CSV: {e}")
                    
                    # If we don't have CSV data, just look for files with matching names
                    if len(sample_files) < 5:
                        for file in os.listdir(fold_dir):
                            if file.endswith('.wav') and len(sample_files) < 5:
                                sample_files.append({
                                    'name': file,
                                    'path': f'/urbansound_samples/fold{fold_num}/{file}'
                                })
            
            return jsonify({'samples': sample_files})
        
        # Serve audio samples
        @self.app.route('/audio_samples/<filename>')
        def serve_audio_sample(filename):
            return send_from_directory('test_audio_samples', filename)
            
        # Serve UrbanSound8K samples
        @self.app.route('/urbansound_samples/<fold>/<filename>')
        def serve_urbansound_sample(fold, filename):
            return send_from_directory(os.path.join('data/UrbanSound8K', fold), filename)
        
        # About page
        @self.app.route('/about')
        def about():
            return render_template('about.html')
    
    def predict_audio(self, file_path: str, top_k: int = 5) -> dict:
        """
        Predict the class of an audio file.
        
        Args:
            file_path (str): Path to the audio file
            top_k (int, optional): Number of top predictions to return
            
        Returns:
            dict: Dictionary containing prediction results
        """
        self.logger.info(f"Starting prediction for file: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"Audio file not found: {file_path}")
                return {
                    'error': f'Audio file not found: {file_path}',
                    'predictions': [],
                    'spectrogram': None
                }
            
            # Check if models are loaded, if not, try to load them
            if not self.predictor.models:
                self.logger.warning("No models loaded. Attempting to load models...")
                # Use the existing _load_models method to ensure consistent model loading
                self._load_models()
                
                # Verify models were loaded successfully
                if not self.predictor.models:
                    self.logger.error("Failed to load any models. Cannot make predictions.")
                    return {
                        'error': 'No models available for prediction',
                        'predictions': [],
                        'spectrogram': None
                    }
            
            # Generate spectrogram first (this will validate the audio file)
            try:
                spectrogram_path = self._generate_spectrogram(file_path)
                self.logger.info(f"Generated spectrogram at: {spectrogram_path}")
            except Exception as spec_error:
                self.logger.error(f"Error generating spectrogram: {spec_error}")
                # Continue with prediction even if spectrogram generation fails
                spectrogram_path = None
            
            # Make prediction using direct audio loading instead of relying on predictor's method
            try:
                # Load audio directly
                import librosa
                import numpy as np
                
                self.logger.info(f"Loading audio for prediction: {file_path}")
                try:
                    y, sr = librosa.load(file_path, sr=22050, mono=True)
                except Exception as load_error:
                    self.logger.error(f"Error loading audio file: {load_error}")
                    raise ValueError(f"Failed to load audio file: {str(load_error)}")
                
                # Validate audio data
                if y is None:
                    raise ValueError(f"No audio data loaded from {file_path}")
                
                # Handle scalar values (including numpy scalar types)
                if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
                    self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
                    # For numpy scalar types, we need to explicitly create a new array
                    if isinstance(y, np.number):
                        # Create a new array with the scalar value
                        y = np.array([float(y)], dtype=np.float32)
                    else:
                        # For other scalar types
                        y = np.atleast_1d(y).astype(np.float32)
                
                if len(y) == 0:
                    raise ValueError(f"Empty audio data loaded from {file_path}")
                
                # Check for NaN or Inf values
                if np.isnan(y).any() or np.isinf(y).any():
                    self.logger.error(f"Audio data contains NaN or Inf values")
                    raise ValueError("Audio data contains invalid values (NaN or Inf)")
                
                # Extract features
                self.logger.info("Extracting features for prediction")
                try:
                    mel_spec = librosa.feature.melspectrogram(
                        y=y, 
                        sr=sr, 
                        n_fft=2048, 
                        hop_length=512, 
                        n_mels=128
                    )
                except Exception as mel_error:
                    self.logger.error(f"Error extracting mel spectrogram: {mel_error}")
                    raise ValueError(f"Failed to extract features: {str(mel_error)}")
                
                # Validate mel spectrogram
                if mel_spec is None or mel_spec.size == 0:
                    raise ValueError("Generated empty mel spectrogram")
                
                # Check for NaN or Inf values in mel spectrogram
                if np.isnan(mel_spec).any() or np.isinf(mel_spec).any():
                    self.logger.error("Mel spectrogram contains NaN or Inf values")
                    raise ValueError("Feature extraction produced invalid values (NaN or Inf)")
                
                # Convert to dB scale
                try:
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                except Exception as db_error:
                    self.logger.error(f"Error converting to dB scale: {db_error}")
                    raise ValueError(f"Failed to convert to dB scale: {str(db_error)}")
                
                # Reshape for model input if needed
                if len(self.predictor.models) > 0:
                    model = self.predictor.models[0]
                    if hasattr(model, 'input_shape'):
                        input_shape = model.input_shape
                        if input_shape and input_shape[0] is None:
                            # Add batch dimension
                            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
                
                # Make prediction
                self.logger.info("Making prediction")
                try:
                    results = self.predictor.predict(mel_spec_db, top_k=top_k)
                except Exception as pred_error:
                    self.logger.error(f"Error in predictor.predict: {pred_error}")
                    raise ValueError(f"Prediction failed: {str(pred_error)}")
                
                # Format results
                formatted_results = {
                    'predictions': results,
                    'spectrogram': os.path.basename(spectrogram_path) if spectrogram_path else None
                }
                
                return formatted_results
                
            except Exception as pred_error:
                self.logger.error(f"Error in prediction processing: {pred_error}")
                return {
                    'error': str(pred_error),
                    'predictions': [],
                    'spectrogram': os.path.basename(spectrogram_path) if spectrogram_path else None
                }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            # Return error information instead of raising exception
            return {
                'error': str(e),
                'predictions': [],
                'spectrogram': None
            }
            
    def _load_models(self):
        """
        Load models for prediction.
        """
        if not self.predictor.models:
            self.logger.info("Loading models for prediction...")
            model_path = self.config.get('MODEL.path', None)
            models_dir = self.config.get('PATHS.models_dir', 'models')
            
            self.logger.info(f"Looking for models in: {models_dir}")
            if os.path.exists(models_dir):
                # List available models for debugging
                model_files = [f for f in os.listdir(models_dir) 
                              if f.endswith('.h5') or f.endswith('.tflite') 
                              or os.path.isdir(os.path.join(models_dir, f))]
                if model_files:
                    self.logger.info(f"Found {len(model_files)} model files: {', '.join(model_files)}")
                    
                    # If there's a specific model file, use it
                    if 'best_model.h5' in model_files:
                        specific_model = os.path.join(models_dir, 'best_model.h5')
                        self.logger.info(f"Found best_model.h5, loading from: {specific_model}")
                        try:
                            self.predictor.load_models([specific_model])
                            self.logger.info(f"Successfully loaded model: {specific_model}")
                            return
                        except Exception as e:
                            self.logger.error(f"Error loading specific model {specific_model}: {e}")
                    
                    # Try to load models from directory
                    try:
                        self.logger.info(f"Loading models from directory: {models_dir}")
                        self.predictor.load_models()
                        self.logger.info(f"Successfully loaded {len(self.predictor.models)} models from directory")
                        return
                    except Exception as e:
                        self.logger.error(f"Error loading models from directory {models_dir}: {e}")
                else:
                    self.logger.warning(f"No model files found in {models_dir}")
            else:
                self.logger.warning(f"Models directory does not exist: {models_dir}")
            
            # Try to load from model_path if specified
            if model_path and os.path.exists(model_path):
                try:
                    self.logger.info(f"Attempting to load model from specific path: {model_path}")
                    self.predictor.load_models([model_path])
                    self.logger.info(f"Successfully loaded model from: {model_path}")
                    return
                except Exception as e:
                    self.logger.error(f"Error loading model from {model_path}: {e}")
            
            self.logger.warning("No models were loaded. Predictions will not work until models are loaded.")
        else:
            self.logger.info(f"Models already loaded: {len(self.predictor.models)} models")
    
    def _generate_spectrogram(self, audio_path: str) -> str:
        """
        Generate a spectrogram image for an audio file in a thread-safe manner.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Path to the generated spectrogram image
        """
        # Create spectrograms directory if it doesn't exist
        spectrograms_dir = os.path.join(self.app.config['UPLOAD_FOLDER'], 'spectrograms')
        os.makedirs(spectrograms_dir, exist_ok=True)
        
        try:
            # Load audio file directly using librosa
            self.logger.info(f"Loading audio file for spectrogram: {audio_path}")
            import librosa
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            if y is None:
                self.logger.error(f"No audio data loaded from {audio_path}")
                raise ValueError(f"No audio data loaded from {audio_path}")
            
            # Handle scalar values (including numpy scalar types)
            if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
                self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
                # For numpy scalar types, we need to explicitly create a new array
                if isinstance(y, np.number):
                    # Create a new array with the scalar value
                    y = np.array([float(y)], dtype=np.float32)
                else:
                    # For other scalar types
                    y = np.atleast_1d(y).astype(np.float32)
            
            if len(y) == 0:
                self.logger.error(f"Empty audio data loaded from {audio_path}")
                raise ValueError(f"Empty audio data loaded from {audio_path}")
            
            # Generate mel spectrogram
            self.logger.info("Generating mel spectrogram")
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=2048, 
                hop_length=512, 
                n_mels=128
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Generate a unique filename for the spectrogram
            filename = os.path.basename(audio_path)
            unique_id = os.urandom(4).hex()  # Add random component for uniqueness
            spectrogram_filename = f"{os.path.splitext(filename)[0]}_{unique_id}_spectrogram.png"
            spectrogram_path = os.path.join(spectrograms_dir, spectrogram_filename)
            
            # Use lock for matplotlib operations (not thread-safe)
            with self._plot_lock:
                plt.figure(figsize=(10, 4))
                plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel Spectrogram')
                plt.xlabel('Time')
                plt.ylabel('Mel Frequency')
                plt.tight_layout()
                
                # Save image
                plt.savefig(spectrogram_path, dpi=100)
                plt.close()
            
            self.logger.info(f"Generated spectrogram at: {spectrogram_path}")
            return spectrogram_path
            
        except Exception as e:
            self.logger.error(f"Error generating spectrogram: {e}")
            # Create a placeholder image for failed spectrogram generation
            placeholder_path = os.path.join(spectrograms_dir, f"error_{os.urandom(4).hex()}.png")
            with self._plot_lock:
                plt.figure(figsize=(10, 4))
                plt.text(0.5, 0.5, f"Error generating spectrogram: {str(e)}", 
                         horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                plt.savefig(placeholder_path)
                plt.close()
            return placeholder_path
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """
        Run the Flask application.
        
        Args:
            host (str, optional): Host to run the app on.
                If None, use value from config.
            port (int, optional): Port to run the app on.
                If None, use value from config.
            debug (bool, optional): Whether to run in debug mode.
                If None, use value from config.
        """
        # Get parameters from config if not provided
        if host is None:
            host = self.config.get('WEB.host', '0.0.0.0')
        
        if port is None:
            port = self.config.get('WEB.port', 5000)
        
        if debug is None:
            debug = self.config.get('WEB.debug', False)
        
        # Load models before starting the app
        if not self.predictor.models:
            self.logger.info("No models loaded yet. Attempting to load models before starting the server...")
            self._load_models()
            
            # Log model loading status
            if self.predictor.models:
                self.logger.info(f"Successfully loaded {len(self.predictor.models)} models before starting the server")
            else:
                self.logger.warning("No models were loaded. The prediction functionality will not work until models are loaded.")
        else:
            self.logger.info(f"Models already loaded: {len(self.predictor.models)} models")
        
        # Run the app
        self.logger.info(f"Starting web app on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def create_app(config_path=None):
    """
    Create a Flask application instance.
    
    Args:
        config_path: Either a path to the configuration file (str) or
            a ConfigManager instance. If None, default configuration will be used.
            
    Returns:
        Flask: Flask application instance
    """
    # Initialize configuration
    if isinstance(config_path, ConfigManager):
        config = config_path
    else:
        config = ConfigManager()
        if config_path and os.path.exists(config_path):
            config.load(config_path)
    
    # Create web app
    web_app = WebApp(config)
    
    return web_app.app


if __name__ == '__main__':
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run app
    app = create_app()
    app.run(debug=True)