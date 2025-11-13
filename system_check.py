import os
import sys
import importlib
import platform
import subprocess
import shutil
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from src.config import MODEL_PATH, UPLOAD_FOLDER, STATIC_FOLDER, TEMPLATE_FOLDER, CLASS_LABELS
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_status(message, status):
    """Print a status message with colored output."""
    status_str = "[PASS]" if status else "[FAIL]"
    color = "\033[92m" if status else "\033[91m"  # Green for pass, red for fail
    reset = "\033[0m"
    
    # Check if running in a terminal that supports colors
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        print(f"{message.ljust(70)} {color}{status_str}{reset}")
    else:
        print(f"{message.ljust(70)} {status_str}")
    
    return status

def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    required_version = (3, 6)
    
    version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    required_str = f"{required_version[0]}.{required_version[1]}"
    
    message = f"Python version (>= {required_str}, current: {version_str})"
    return print_status(message, python_version >= required_version)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "flask", "flask_cors", "tensorflow", "numpy", "pandas", 
        "librosa", "scikit-learn", "pydub", "matplotlib", "pyaudio",
        "soundfile"
    ]
    
    optional_packages = ["pydot", "graphviz"]
    
    all_required_installed = True
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_status(f"  {package}", True)
        except ImportError:
            print_status(f"  {package}", False)
            all_required_installed = False
    
    print("\nChecking optional packages:")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print_status(f"  {package}", True)
        except ImportError:
            print_status(f"  {package} (optional)", False)
    
    return all_required_installed

def check_directories():
    """Check if required directories exist and are accessible."""
    directories = [
        ("Models directory", os.path.dirname(MODEL_PATH)),
        ("Upload directory", UPLOAD_FOLDER),
        ("Static directory", STATIC_FOLDER),
        ("Templates directory", TEMPLATE_FOLDER)
    ]
    
    all_dirs_exist = True
    
    for name, path in directories:
        exists = os.path.isdir(path)
        if not exists and name == "Upload directory":
            # Create upload directory if it doesn't exist
            try:
                os.makedirs(path, exist_ok=True)
                exists = True
                print_status(f"{name} ({path}) - created", True)
                continue
            except Exception as e:
                print(f"Error creating directory {path}: {e}")
        
        all_dirs_exist = print_status(f"{name} ({path})", exists) and all_dirs_exist
    
    return all_dirs_exist

def check_model_file():
    """Check if the model file exists."""
    model_exists = os.path.isfile(MODEL_PATH)
    return print_status(f"Model file ({MODEL_PATH})", model_exists)

def check_tensorflow():
    """Check TensorFlow installation and GPU availability."""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        print_status(f"TensorFlow version: {tf_version}", True)
        print_status("GPU available for TensorFlow", gpu_available)
        
        # Print additional TensorFlow info
        print("\nTensorFlow devices:")
        for device in tf.config.list_physical_devices():
            print(f"  {device.device_type}: {device.name}")
        
        return True
    except Exception as e:
        print_status("TensorFlow check", False)
        print(f"Error: {e}")
        return False

def check_audio_processing():
    """Check if audio processing libraries are working."""
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        # Create a simple test signal
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create a temporary directory for test files
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_test")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save and load the test signal
        test_file = os.path.join(temp_dir, "test_audio.wav")
        sf.write(test_file, test_signal, sr)
        
        # Load with librosa
        y, sr = librosa.load(test_file, sr=None)
        
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            print(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
            y = np.atleast_1d(y).astype(np.float32)
        
        # Extract features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return print_status("Audio processing libraries", True)
    except Exception as e:
        print_status("Audio processing libraries", False)
        print(f"Error: {e}")
        return False

def check_flask_app():
    """Check if Flask app can be imported and initialized."""
    try:
        from src.app import app
        return print_status("Flask application", True)
    except Exception as e:
        print_status("Flask application", False)
        print(f"Error: {e}")
        return False

def check_system_info():
    """Display system information."""
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
    except ImportError:
        print("Memory: psutil not installed, cannot check memory")

def run_system_check():
    """Run all system checks."""
    print_header("Urban Sound Classifier - System Check")
    
    print("\nSystem Information:")
    check_system_info()
    
    print_header("Environment Checks")
    python_check = check_python_version()
    dependencies_check = check_dependencies()
    directories_check = check_directories()
    model_check = check_model_file()
    
    print_header("Functionality Checks")
    tensorflow_check = check_tensorflow()
    audio_check = check_audio_processing()
    flask_check = check_flask_app()
    
    # Summary
    print_header("Check Summary")
    checks = [
        ("Python version", python_check),
        ("Required dependencies", dependencies_check),
        ("Directories", directories_check),
        ("Model file", model_check),
        ("TensorFlow", tensorflow_check),
        ("Audio processing", audio_check),
        ("Flask application", flask_check)
    ]
    
    all_passed = True
    for name, result in checks:
        all_passed = all_passed and result
        print_status(name, result)
    
    print_header("Final Result")
    if all_passed:
        print("\n✅ All checks passed! The system is ready to run.\n")
        print("You can start the application with:")
        print("  python src/app.py")
        print("\nOr run the test suite with:")
        print("  python run_all_tests.bat  (Windows)")
        print("  ./run_all_tests.sh        (Linux/macOS)")
    else:
        print("\n❌ Some checks failed. Please fix the issues before running the application.\n")
        print("For troubleshooting, refer to the DOCUMENTATION.md file.")
    
    return all_passed

if __name__ == "__main__":
    run_system_check()