from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import sys
import tensorflow as tf
from model import load_model, predict
from utils import process_audio_file
from config import MODEL_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, STATIC_FOLDER, TEMPLATES_FOLDER, CLASS_LABELS

# Initialize Flask app with correct template and static folders
app = Flask(__name__, 
           template_folder=TEMPLATES_FOLDER,
           static_folder=STATIC_FOLDER)
CORS(app)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model at startup if it exists
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        
        # Compile the model with metrics to address the warning
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        print("Model compiled with metrics successfully")
        print(f"Model metrics: {model.metrics_names}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("The application will run, but prediction functionality will be disabled.")
        print("Please ensure the model file exists before using the prediction endpoint.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("The application will run, but prediction functionality will be disabled.")
    print("Please check the model file and restart the application.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_sound():
    # Record the start time of the request
    import datetime
    request_start_time = datetime.datetime.now()
    
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Request-ID'
        return response
        
    print(f"\n{'='*50}\nPrediction request received\n{'='*50}")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    # Get request ID for tracking - check both headers and form data
    request_id = request.headers.get('X-Request-ID') or request.form.get('request_id', f"req-{os.urandom(4).hex()}")
    print(f"Request ID: {request_id}")
    
    # Get user signature if provided
    user_signature = request.form.get('user_signature', 'Anonymous User')
    
    # Check if we're using small chunks mode
    use_small_chunks = request.form.get('use_small_chunks') == 'true'
    
    # Log all request details for debugging
    print(f"Request details:\n  - ID: {request_id}\n  - User: {user_signature}\n  - Remote IP: {request.remote_addr}\n  - User Agent: {request.user_agent}\n  - Small chunks mode: {use_small_chunks}")
    print(f"Request form data: {dict(request.form)}")
    print(f"Request files: {list(request.files.keys())}")
    
    # Set response headers for CORS and cache control
    response_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, X-Request-ID',
        'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    
    # Check if model is loaded
    if model is None:
        error_msg = "Model not loaded. Please ensure the model file exists and restart the application."
        print(f"Error: {error_msg}")
        return create_error_response(error_msg, 503, request_id)
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        error_msg = "No file part in the request"
        print(f"Error: {error_msg}")
        return create_error_response(error_msg, 400, request_id)
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        error_msg = "No selected file"
        print(f"Error: {error_msg}")
        return create_error_response(error_msg, 400, request_id)
    
    if not allowed_file(file.filename):
        error_msg = f"File type not allowed - {file.filename}. Supported formats: WAV, MP3, OGG, FLAC, M4A"
        print(f"Error: {error_msg}")
        return create_error_response(error_msg, 400, request_id)
    
    # Save the file to a temporary location to ensure it's properly loaded
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{request_id}_{file.filename}")
    
    try:
        file.save(temp_path)
        
        # Check if the file was saved correctly and has content
        file_size = os.path.getsize(temp_path)
        print(f"File saved to {temp_path}, Size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = "Uploaded file is empty"
            print(f"Error: {error_msg}")
            return create_error_response(error_msg, 400, request_id)
        
        # Create a new file object for processing
        with open(temp_path, 'rb') as f:
            # Create a new file-like object with the necessary attributes
            from io import BytesIO
            file_content = BytesIO(f.read())
            file_content.filename = os.path.basename(temp_path)
            file_content.content_type = file.content_type if hasattr(file, 'content_type') else 'audio/wav'
            
            print(f"File received: {file_content.filename}, Size: {file_size} bytes, Content type: {file_content.content_type}")
            
            # Replace the original file object with our new one
            file = file_content
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"Error: {error_msg}")
        import traceback
        print(traceback.format_exc())
        return create_error_response(error_msg, 500, request_id)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up temporary file at {temp_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {str(e)}")
                # If we can't remove it now, schedule it for removal when the program exits
                import atexit
                atexit.register(lambda path=temp_path: os.remove(path) if os.path.exists(path) else None)
    
    # Process the audio file
    try:
        print(f"Processing file: {file.filename}")
        
        # Check if we're using small chunks mode
        use_small_chunks = request.form.get('use_small_chunks') == 'true'
        print(f"Using small chunks mode: {use_small_chunks}")
        
        # Use a timeout-compatible approach for Windows
        import threading
        import concurrent.futures
        
        # Set a timeout for processing (30 seconds for normal mode, 60 for small chunks)
        timeout_duration = 60 if use_small_chunks else 30
        
        # Define a function to process the audio file with timeout
        def process_with_timeout(file_obj, use_small_chunks_flag):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_audio_file, file_obj, use_small_chunks_flag)
                try:
                    return future.result(timeout=timeout_duration)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError("Processing took too long. Try using small chunks mode.")
        
        try:
            features = process_with_timeout(file, use_small_chunks)
            print(f"Features extracted successfully. Shape: {features.shape}")
        except TimeoutError as te:
            print(f"Timeout error during processing: {str(te)}")
            return create_error_response(str(te), 408, request_id)
        except Exception as proc_error:
            print(f"Error during audio processing: {str(proc_error)}")
            raise proc_error
        
        # Make prediction with enhanced error handling
        print("Making prediction...")
        try:
            prediction, confidence, all_confidences = predict(model, features)
            print(f"Prediction result: {prediction}, confidence: {confidence}")
            
            # Validate and normalize prediction results
            prediction, confidence, serializable_confidences = validate_prediction_results(prediction, confidence, all_confidences)
        except Exception as pred_error:
            print(f"Error during prediction: {str(pred_error)}")
            import traceback
            print(traceback.format_exc())
            
            # Provide a fallback prediction with error information
            prediction = CLASS_LABELS[0]  # Use first class as fallback
            confidence = 0.0
            
            # Create diagnostic information in the confidences dictionary
            serializable_confidences = {}
            for class_name in CLASS_LABELS:
                serializable_confidences[class_name] = 0.0
            
            # Add error information to the confidences
            serializable_confidences['_error_type'] = str(type(pred_error).__name__)
            serializable_confidences['_error_message'] = str(pred_error)
            serializable_confidences['_features_shape'] = str(features.shape) if 'features' in locals() and features is not None else 'unknown'
            
            # Add more feature statistics if available
            if 'features' in locals() and features is not None:
                try:
                    serializable_confidences['_features_min'] = float(np.min(features))
                    serializable_confidences['_features_max'] = float(np.max(features))
                    serializable_confidences['_features_mean'] = float(np.mean(features))
                    serializable_confidences['_features_std'] = float(np.std(features))
                    serializable_confidences['_features_has_nan'] = bool(np.isnan(features).any())
                    serializable_confidences['_features_has_inf'] = bool(np.isinf(features).any())
                except Exception as stat_error:
                    serializable_confidences['_feature_stats_error'] = str(stat_error)
            
            # Add model diagnostic information
            try:
                if model is not None:
                    serializable_confidences['_model_input_shape'] = str(model.input_shape)
                    serializable_confidences['_model_output_shape'] = str(model.output_shape)
                    serializable_confidences['_model_layers_count'] = len(model.layers)
            except Exception as model_error:
                serializable_confidences['_model_error'] = str(model_error)
            
            # Add a flag to indicate this is a fallback prediction
            serializable_confidences['_is_fallback'] = True
            
            # Use validate_prediction_results to ensure consistent output format
            prediction, confidence, serializable_confidences = validate_prediction_results(prediction, confidence, serializable_confidences)
        
        # Get current timestamp using datetime directly
        import datetime
        current_time = datetime.datetime.now()
        
        # Determine if this is a fallback/error prediction
        is_fallback = '_is_fallback' in serializable_confidences and serializable_confidences['_is_fallback']
        
        # Create the result dictionary with enhanced diagnostics
        result = {
            'class': prediction,
            'confidence': float(confidence),
            'all_confidences': serializable_confidences,
            'user_signature': user_signature,
            'request_id': request_id,
            'status': 'success' if not is_fallback else 'partial_success',
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'small_chunks_used': use_small_chunks
        }
        
        # Calculate processing time (from start of request to now)
        end_time = datetime.datetime.now()
        response_time_ms = int((end_time - current_time).total_seconds() * 1000)
        total_processing_time_ms = int((end_time - request_start_time).total_seconds() * 1000)
        
        # Add diagnostic information
        diagnostics = {
            'model_path': MODEL_PATH,
            'features_shape': str(features.shape) if 'features' in locals() and features is not None else 'unknown',
            'response_time_ms': response_time_ms,
            'total_processing_time_ms': total_processing_time_ms,
        }
        
        # Add feature statistics if available
        if 'features' in locals() and features is not None:
            try:
                diagnostics['features_min'] = float(np.min(features))
                diagnostics['features_max'] = float(np.max(features))
                diagnostics['features_mean'] = float(np.mean(features))
                diagnostics['features_std'] = float(np.std(features))
                # Check for NaN or Inf values
                diagnostics['features_has_nan'] = bool(np.isnan(features).any())
                diagnostics['features_has_inf'] = bool(np.isinf(features).any())
            except Exception as stat_error:
                print(f"Error calculating feature statistics: {str(stat_error)}")
                diagnostics['feature_stats_error'] = str(stat_error)
        
        # Include diagnostics in the result
        result['diagnostics'] = diagnostics
        
        # Log prediction metrics for monitoring and debugging
        log_prediction_metrics(result, features, is_fallback)
        
        print(f"Returning result: {result}")
        
        # Create response with aggressive cache prevention
        response = jsonify(result)
        
        # Set no-cache headers
        set_no_cache_headers(response)
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = request_id
        
        # Add diagnostic headers
        response.headers['X-Prediction-Status'] = 'fallback' if is_fallback else 'normal'
        response.headers['X-Response-Time-Ms'] = str(diagnostics['response_time_ms'])
        response.headers['X-Total-Processing-Time-Ms'] = str(diagnostics['total_processing_time_ms'])
        
        # Add CORS headers if they were set earlier
        if 'response_headers' in locals():
            for header, value in response_headers.items():
                response.headers[header] = value
        
        print(f"Response headers: {dict(response.headers)}")
        return response
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        print(f"Traceback: {error_traceback}")
        return create_error_response(error_msg, 500, request_id, error_traceback)

@app.route('/classes', methods=['GET'])
def get_classes():
    # Return the list of classes the model can predict
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
               'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    return jsonify({'classes': classes})

def set_no_cache_headers(response):
    """Set headers to prevent caching"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

def validate_prediction_results(prediction, confidence, all_confidences):
    """Validate and normalize prediction results to ensure consistent output"""
    # Validate prediction class
    if prediction not in CLASS_LABELS:
        print(f"Warning: Predicted class '{prediction}' is not in CLASS_LABELS. Using highest confidence class.")
        # Find the class with the highest confidence as fallback
        try:
            highest_conf_class = max(all_confidences.items(), key=lambda x: x[1])[0]
            if highest_conf_class in CLASS_LABELS:
                prediction = highest_conf_class
                confidence = all_confidences[highest_conf_class]
                print(f"Using fallback class: {prediction}, confidence: {confidence}")
            else:
                # If even that fails, use the first class
                prediction = CLASS_LABELS[0]
                confidence = 0.0
                print(f"Using default class: {prediction}")
        except (ValueError, KeyError) as e:
            # If we can't determine the highest confidence class, use the first class
            print(f"Error finding highest confidence class: {str(e)}")
            prediction = CLASS_LABELS[0]
            confidence = 0.0
    
    # Ensure confidence is a valid float between 0 and 1
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        print(f"Warning: Invalid confidence value: {confidence}. Normalizing.")
        if isinstance(confidence, (int, float)):
            confidence = max(0.0, min(1.0, float(confidence)))  # Clamp between 0 and 1
        else:
            confidence = 0.0  # Default if not a number
    
    # Ensure all values are JSON serializable
    serializable_confidences = {}
    
    # First, normalize all provided confidence values
    for class_name, conf_value in all_confidences.items():
        # Ensure each confidence is a valid float between 0 and 1
        if isinstance(conf_value, (int, float)):
            serializable_confidences[class_name] = max(0.0, min(1.0, float(conf_value)))
        else:
            print(f"Warning: Non-numeric confidence for {class_name}: {conf_value}. Setting to 0.")
            serializable_confidences[class_name] = 0.0
    
    # Verify that all classes have confidence values
    for class_name in CLASS_LABELS:
        if class_name not in serializable_confidences:
            print(f"Warning: Missing confidence for class {class_name}. Setting to 0.")
            serializable_confidences[class_name] = 0.0
    
    # Check if confidences sum to approximately 1.0 (allowing for floating point error)
    confidence_sum = sum(serializable_confidences.values())
    if abs(confidence_sum - 1.0) > 0.01 and confidence_sum > 0:
        print(f"Warning: Confidence values don't sum to 1.0 (sum: {confidence_sum}). Normalizing.")
        # Normalize confidences to sum to 1.0
        for class_name in serializable_confidences:
            serializable_confidences[class_name] /= confidence_sum
    
    return prediction, confidence, serializable_confidences


def log_prediction_metrics(metrics, error=None):
    """Log prediction metrics for monitoring and analysis"""
    try:
        # Add timestamp
        import datetime
        metrics['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add error information if present
        if error is not None:
            metrics['error'] = str(error)
        
        # Log the metrics to console
        print(f"PREDICTION METRICS: {metrics}")
        
        # In a production environment, log these metrics to a file or database
        log_to_file = app.config.get('LOG_PREDICTIONS_TO_FILE', False)
        log_to_db = app.config.get('LOG_PREDICTIONS_TO_DB', False)
        
        if log_to_file:
            import json
            import os
            
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            # Log to file with date-based rotation
            log_date = datetime.datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(logs_dir, f'prediction_metrics_{log_date}.log')
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        
        if log_to_db:
            # Example implementation for database logging
            # This would be replaced with actual database code in production
            try:
                # Import database module only when needed
                from database_logger import log_to_database
                log_to_database('prediction_metrics', metrics)
            except ImportError:
                print("Database logging module not available")
            except Exception as db_error:
                print(f"Error logging to database: {str(db_error)}")
    except Exception as e:
        print(f"Error logging prediction metrics: {str(e)}")


def create_error_response(message, status_code, request_id, traceback=None):
    """Create a standardized error response"""
    import datetime
    current_time = datetime.datetime.now()
    
    error_data = {
        'error': message,
        'status': 'error',
        'request_id': request_id,
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Log prediction metrics for the error
    log_prediction_metrics({'request_id': request_id}, error=message)
    
    if traceback and app.debug:
        error_data['traceback'] = traceback
    
    # Log the error for server-side debugging
    print(f"ERROR [{request_id}] {status_code}: {message}")
    if traceback:
        print(f"Traceback for {request_id}:\n{traceback}")
        
    # Create response with headers
    response = jsonify(error_data)
    response.status_code = status_code
    
    # Set no-cache headers
    set_no_cache_headers(response)
    
    # Add request ID to response headers
    response.headers['X-Request-ID'] = request_id
    
    return response

@app.after_request
def add_header(response):
    """Add headers to all responses"""
    return set_no_cache_headers(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)