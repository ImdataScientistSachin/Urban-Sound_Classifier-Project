import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# First try to use the model in the log directory (which is the most recent training)
LOG_MODEL_PATH = os.path.join(BASE_DIR, 'log', 'urban_sound_run_20250805_123220', 'models', 'best_model.h5')
# Fallback to the models directory
MODELS_DIR_PATH = os.path.join(BASE_DIR, 'models', 'best_model.h5')

# Use the first model that exists
if os.path.exists(LOG_MODEL_PATH):
    MODEL_PATH = LOG_MODEL_PATH
else:
    MODEL_PATH = MODELS_DIR_PATH
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'src', 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, 'src', 'static')

# Audio processing parameters
SAMPLE_RATE = 22050  # Sample rate in Hz
DURATION = 4.0      # Duration in seconds to analyze
N_MELS = 128        # Number of Mel bands
N_FFT = 2048        # FFT window size
HOP_LENGTH = 512    # Hop length for STFT

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

# Class labels (in order of model output indices)
CLASS_LABELS = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]

# API settings
API_VERSION = 'v1'
DEBUG_MODE = True
HOST = '0.0.0.0'
PORT = 5000