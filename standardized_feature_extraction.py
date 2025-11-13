import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tempfile
import time
from pydub import AudioSegment
import warnings

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
warnings.filterwarnings('ignore')

def convert_audio_to_wav(file_path, target_sr=22050):
    """
    Convert any audio file to WAV format with consistent parameters.
    
    Args:
        file_path (str): Path to the audio file
        target_sr (int): Target sample rate
        
    Returns:
        str: Path to the converted WAV file
    """
    try:
        # Create a temporary directory for the WAV file
        temp_dir = tempfile.mkdtemp()
        temp_wav_path = os.path.join(temp_dir, 'converted.wav')
        
        print(f"Converting {file_path} to WAV format at {temp_wav_path}")
        
        # Load the audio file using pydub
        try:
            audio = AudioSegment.from_file(file_path)
            
            # Set consistent parameters
            audio = audio.set_frame_rate(target_sr)
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Export to WAV
            audio.export(temp_wav_path, format="wav")
        except Exception as e:
            print(f"Error converting with pydub: {e}. Trying with librosa...")
            
            # Fallback to librosa
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            
            # Handle scalar values (including numpy scalar types)
            if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
                print(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
                y = np.atleast_1d(y).astype(np.float32)
                
            sf.write(temp_wav_path, y, target_sr, subtype='PCM_16')
        
        # Verify the WAV file exists and is not empty
        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            raise ValueError(f"Failed to create valid WAV file at {temp_wav_path}")
        
        return temp_wav_path
    except Exception as e:
        print(f"Error converting audio to WAV: {e}")
        raise

def extract_features(wav_path, target_sr=22050, n_mels=128, n_fft=2048, hop_length=512, fixed_length=173):
    """
    Extract Mel spectrogram features from a WAV file with consistent parameters.
    
    Args:
        wav_path (str): Path to the WAV file
        target_sr (int): Target sample rate
        n_mels (int): Number of Mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        fixed_length (int): Fixed length for the time dimension
        
    Returns:
        numpy.ndarray: Mel spectrogram features
    """
    try:
        print(f"Extracting features from {wav_path}")
        
        # Load the audio file
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            print(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
            y = np.atleast_1d(y).astype(np.float32)
        
        # Ensure consistent duration by padding or trimming
        target_duration = fixed_length * hop_length / target_sr
        target_samples = int(target_duration * target_sr)
        
        if len(y) < target_samples:
            # Pad with zeros
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        elif len(y) > target_samples:
            # Trim to target length
            y = y[:target_samples]
        
        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, 
            sr=target_sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize to [0, 1] range
        normalized_mel = (mel_spectrogram_db - np.min(mel_spectrogram_db)) / (np.max(mel_spectrogram_db) - np.min(mel_spectrogram_db))
        
        # Ensure the time dimension is fixed_length
        if normalized_mel.shape[1] < fixed_length:
            # Pad with zeros
            pad_width = fixed_length - normalized_mel.shape[1]
            normalized_mel = np.pad(normalized_mel, ((0, 0), (0, pad_width)), 'constant')
        elif normalized_mel.shape[1] > fixed_length:
            # Trim to fixed_length
            normalized_mel = normalized_mel[:, :fixed_length]
        
        # Reshape to (n_mels, fixed_length, 1) for model input
        features = normalized_mel.reshape(n_mels, fixed_length, 1)
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            print("Warning: NaN or Inf values detected in features. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"Features extracted with shape {features.shape}")
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        raise

def process_audio_file(file_path, target_sr=22050, n_mels=128, n_fft=2048, hop_length=512, fixed_length=173):
    """
    Process an audio file and extract features with consistent parameters.
    
    Args:
        file_path (str): Path to the audio file
        target_sr (int): Target sample rate
        n_mels (int): Number of Mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        fixed_length (int): Fixed length for the time dimension
        
    Returns:
        numpy.ndarray: Mel spectrogram features
    """
    try:
        start_time = time.time()
        
        # Convert to WAV format
        wav_path = convert_audio_to_wav(file_path, target_sr)
        
        # Extract features
        features = extract_features(wav_path, target_sr, n_mels, n_fft, hop_length, fixed_length)
        
        # Clean up temporary WAV file
        try:
            os.remove(wav_path)
            os.rmdir(os.path.dirname(wav_path))
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        print(f"Audio processing completed in {processing_time:.2f} ms")
        
        return features
    except Exception as e:
        print(f"Error processing audio file: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def main():
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python standardized_feature_extraction.py <audio_file_path>")
        return
    
    # Get the file path
    file_path = sys.argv[1]
    
    # Process the audio file
    try:
        features = process_audio_file(file_path)
        print(f"Features shape: {features.shape}")
        print(f"Features min: {np.min(features)}, max: {np.max(features)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()