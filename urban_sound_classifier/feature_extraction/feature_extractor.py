import os
import numpy as np
import librosa
import logging
import warnings
import psutil
import threading
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
from ..config.config_manager import ConfigManager
from ..utils.audio.audio_utils import AudioUtils

class FeatureExtractor(ABC):
    """
    Thread-safe base class for audio feature extraction.
    
    This class provides a framework for extracting features from audio files
    with support for caching, batch processing, and memory optimization.
    
    Attributes:
        config (ConfigManager): Configuration manager instance
        sample_rate (int): Audio sample rate
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        n_mels (int): Number of mel bands
        cache_dir (str): Directory for caching extracted features
        use_cache (bool): Whether to use feature caching
        _file_locks (Dict[str, threading.Lock]): Dictionary of locks for file operations
        _cache_lock (threading.Lock): Lock for cache operations
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the feature extractor.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audio parameters
        self.sample_rate = config.get('AUDIO.sample_rate', 22050)
        self.duration = config.get('AUDIO.duration', 4.0)
        self.mono = config.get('AUDIO.mono', True)
        self.normalize_audio = config.get('AUDIO.normalize_audio', True)
        
        # Feature parameters
        self.n_fft = config.get('FEATURES.mel_spectrogram.n_fft', 2048)
        self.hop_length = config.get('FEATURES.mel_spectrogram.hop_length', 512)
        self.n_mels = config.get('FEATURES.mel_spectrogram.n_mels', 128)
        
        # Caching parameters
        self.cache_dir = config.get('PATHS.cache_dir', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache'))
        self.use_cache = config.get('SYSTEM.use_cache', True)
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"Created cache directory: {self.cache_dir}")
        
        # Suppress warnings from librosa
        if not config.get('SYSTEM.debug', False):
            warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
        
        # Initialize locks for thread safety
        self._file_locks = {}
        self._cache_lock = threading.Lock()
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """
        Get or create a lock for a specific file path.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            threading.Lock: Lock for the file
        """
        with self._cache_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            return self._file_locks[file_path]
    
    def extract_features_from_file(self, file_path: str, force_recompute: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract features from an audio file in a thread-safe manner.
        
        Args:
            file_path (str): Path to the audio file
            force_recompute (bool): Whether to force recomputation even if cached
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Get lock for this file
        file_lock = self._get_file_lock(file_path)
        
        # Use the lock for all operations on this file
        with file_lock:
            # Check if features are cached
            cache_path = self._get_cache_path(file_path)
            
            if self.use_cache and os.path.exists(cache_path) and not force_recompute:
                try:
                    features = np.load(cache_path, allow_pickle=True).item()
                    self.logger.debug(f"Loaded cached features for {os.path.basename(file_path)}")
                    return features
                except Exception as e:
                    self.logger.warning(f"Failed to load cached features: {e}. Recomputing...")
            
            # Load and preprocess audio
            try:
                self.logger.debug(f"Starting feature extraction for file: {file_path}")
                
                # Convert audio to WAV format with the correct sample rate
                try:
                    wav_file = AudioUtils.convert_audio_to_wav(file_path, sample_rate=self.sample_rate)
                    self.logger.debug(f"Converted audio to WAV format: {wav_file}")
                except Exception as e:
                    self.logger.error(f"Error converting audio to WAV format: {e}")
                    raise ValueError(f"Failed to convert audio file {file_path} to WAV format: {e}")
                
                # Check if wav_file exists
                if not os.path.exists(wav_file):
                    self.logger.error(f"Converted WAV file not found: {wav_file}")
                    raise FileNotFoundError(f"Converted WAV file not found: {wav_file}")
                    
                # Load audio
                try:
                    self.logger.debug(f"Loading audio file: {wav_file}")
                    y, sr = librosa.load(
                        wav_file, 
                        sr=self.sample_rate, 
                        mono=self.mono, 
                        duration=self.duration
                    )
                    self.logger.debug(f"Loaded audio with shape: {getattr(y, 'shape', 'N/A')}, sample rate: {sr}")
                except Exception as e:
                    self.logger.error(f"Error loading audio file {wav_file}: {e}")
                    raise ValueError(f"Failed to load audio file {wav_file}: {e}")
                
                # Verify audio data is valid
                if y is None or len(y) == 0:
                    self.logger.error(f"No audio data loaded from {wav_file}")
                    raise ValueError(f"No audio data loaded from {wav_file}")
                    
                # Apply preprocessing
                try:
                    self.logger.debug("Applying audio preprocessing")
                    y = self._preprocess_audio(y)
                except Exception as e:
                    self.logger.error(f"Error preprocessing audio: {e}, audio data type: {type(y)}")
                    raise ValueError(f"Failed to preprocess audio data: {e}")
                
                # Extract features
                try:
                    self.logger.debug("Extracting features from preprocessed audio")
                    features = self._extract_features(y, sr)
                    self.logger.debug(f"Features extracted successfully with keys: {list(features.keys())}")
                except Exception as e:
                    self.logger.error(f"Error extracting features from {file_path}: {e}, audio data type: {type(y)}")
                    raise ValueError(f"Failed to extract features: {e}")
                
                # Cache features if enabled
                if self.use_cache:
                    cache_dir = os.path.dirname(cache_path)
                    # Create cache directory if it doesn't exist
                    if not os.path.exists(cache_dir):
                        with self._cache_lock:
                            os.makedirs(cache_dir, exist_ok=True)
                    
                    np.save(cache_path, features)
                    self.logger.debug(f"Cached features for {os.path.basename(file_path)}")
                
                return features
                
            except Exception as e:
                # Enhanced error logging with more context
                audio_info = "unknown"
                try:
                    if 'y' in locals() and y is not None:
                        audio_info = f"type: {type(y)}, shape: {getattr(y, 'shape', 'N/A')}, sr: {sr}"
                except:
                    pass
                
                self.logger.error(f"Error extracting features from {file_path}: {e}, audio data: {audio_info}")
                # Re-raise with more context
                raise type(e)(f"{str(e)} (while processing {os.path.basename(file_path)}, audio data: {audio_info})") from e
    
    def _get_cache_path(self, file_path: str) -> str:
        """
        Generate a unique cache path for a file based on its path and extraction parameters.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            str: Path to the cache file
        """
        # Create a hash of the file path and extraction parameters
        params_str = f"{file_path}_{self.sample_rate}_{self.n_fft}_{self.hop_length}_{self.n_mels}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        # Create a directory structure based on the hash to avoid too many files in one directory
        hash_dir = os.path.join(self.cache_dir, params_hash[:2], params_hash[2:4])
        
        # Use the original filename with the hash to make it more readable
        filename = os.path.splitext(os.path.basename(file_path))[0]
        cache_filename = f"{filename}_{params_hash}.npy"
        
        return os.path.join(hash_dir, cache_filename)
    
    def extract_features_batch(self, file_paths: List[str], batch_size: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from multiple audio files in batches.
        
        Args:
            file_paths (List[str]): List of audio file paths
            batch_size (Optional[int]): Batch size for processing, if None, estimated based on memory
            
        Returns:
            List[Dict[str, np.ndarray]]: List of feature dictionaries
        """
        if batch_size is None:
            batch_size = self._estimate_batch_size(len(file_paths))
        
        self.logger.info(f"Processing {len(file_paths)} files in batches of {batch_size}")
        
        results = []
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths)-1)//batch_size + 1} ({len(batch_files)} files)")
            
            batch_results = []
            for file_path in batch_files:
                try:
                    features = self.extract_features_from_file(file_path)
                    batch_results.append(features)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    # Add None for failed files to maintain alignment with input list
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def _preprocess_audio(self, y) -> np.ndarray:
        """
        Preprocess audio signal.
        
        Args:
            y: Audio signal (can be numpy.ndarray, scalar, or other types)
            
        Returns:
            np.ndarray: Preprocessed audio signal
        """
        # Add debug logging for input data type and shape
        self.logger.debug(f"Audio data type before preprocessing: {type(y)}, shape: {getattr(y, 'shape', 'N/A')}")
        
        # Ensure y is a numpy array, not a scalar or None
        if y is None:
            raise ValueError("Audio data must be of type numpy.ndarray, not None")
        
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
            try:
                # Always convert to a numpy array with at least one dimension
                # This handles both Python scalar types and numpy scalar types
                # For numpy scalar types, we need to explicitly create a new array
                if isinstance(y, np.number):
                    # Create a new array with the scalar value
                    y = np.array([float(y)], dtype=np.float32)
                else:
                    # For other scalar types
                    y = np.atleast_1d(y).astype(np.float32)
                
                self.logger.debug(f"After conversion: type={type(y)}, shape={y.shape}")
                
                # Verify the conversion worked
                if y.size == 0 or not isinstance(y, np.ndarray):
                    raise ValueError(f"Conversion resulted in invalid array: size={getattr(y, 'size', 'N/A')}, type={type(y)}")
                    
            except Exception as e:
                self.logger.error(f"Error converting scalar to array: {e}")
                # Create a minimal valid array as fallback
                self.logger.warning("Using fallback empty array due to conversion error")
                y = np.array([0.0], dtype=np.float32)
        
        # For non-scalar, non-ndarray types (like lists)
        elif not isinstance(y, np.ndarray):
            self.logger.debug(f"Converting {type(y)} to numpy array")
            try:
                y = np.array(y, dtype=np.float32)
                if y.size == 0:
                    raise ValueError("Conversion resulted in empty array")
            except Exception as e:
                raise ValueError(f"Failed to convert audio data to numpy array: {e}")
        
        # Final validation
        if not isinstance(y, np.ndarray):
            raise ValueError(f"Audio data must be of type numpy.ndarray, got {type(y)}")
        
        # Check for stereo to mono conversion if needed
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.logger.debug(f"Converting stereo audio with shape {y.shape} to mono")
            y = np.mean(y, axis=1)
            
        # Apply DC offset removal if configured
        if self.config.get('AUDIO.remove_dc_offset', True):
            # Fix for numpy 2.1.3 compatibility with librosa 0.11.0
            try:
                self.logger.debug("Applying DC offset removal")
                y = librosa.util.normalize(y, axis=0)
            except TypeError as e:
                # Handle the 'numpy.float64' object does not support item assignment error
                if "'numpy.float64' object does not support item assignment" in str(e):
                    # Manual normalization as a workaround
                    self.logger.debug("Using manual normalization as workaround for numpy/librosa compatibility issue")
                    if np.max(np.abs(y)) > 0:
                        y = y / np.max(np.abs(y))
                else:
                    self.logger.error(f"Error during normalization: {e}")
                    raise
        
        # Apply high-pass filter if configured
        if self.config.get('AUDIO.high_pass_filter', False):
            cutoff = self.config.get('AUDIO.high_pass_cutoff', 20.0)
            self.logger.debug(f"Applying high-pass filter with cutoff {cutoff} Hz")
            y = librosa.effects.preemphasis(y, coef=0.97, zi=[0])[0]
            
        # Normalize audio if configured
        if self.normalize_audio:
            self.logger.debug("Applying audio normalization")
            # Fix for numpy 2.1.3 compatibility with librosa 0.11.0
            try:
                y = librosa.util.normalize(y)
            except TypeError as e:
                # Handle the 'numpy.float64' object does not support item assignment error
                if "'numpy.float64' object does not support item assignment" in str(e):
                    # Manual normalization as a workaround
                    self.logger.debug("Using manual normalization as workaround")
                    if np.max(np.abs(y)) > 0:
                        y = y / np.max(np.abs(y))
                else:
                    self.logger.error(f"Error during normalization: {e}")
                    raise
        
        # Final verification of audio data
        if y.size == 0:
            raise ValueError("Preprocessed audio data is empty")
            
        self.logger.debug(f"Audio data after preprocessing: shape={y.shape}, min={np.min(y)}, max={np.max(y)}, mean={np.mean(y)}")
        return y
    
    def _get_cache_path(self, file_path: str) -> str:
        """
        Get the cache path for an audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            str: Path to the cached features
        """
        # Create a unique cache filename based on the audio file path and extractor parameters
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        
        # Include key parameters in the cache filename to ensure uniqueness
        params_str = f"sr{self.sample_rate}_nfft{self.n_fft}_hop{self.hop_length}_mel{self.n_mels}"
        
        cache_file = f"{file_name_no_ext}_{params_str}.npy"
        return os.path.join(self.cache_dir, cache_file)
    
    def _estimate_batch_size(self, total_files: int) -> int:
        """
        Estimate an appropriate batch size based on available memory.
        
        Args:
            total_files (int): Total number of files to process
            
        Returns:
            int: Estimated batch size
        """
        # Default batch size - use a more conservative default
        default_batch_size = 16
        min_batch_size = 1
        max_batch_size = 64
        
        # If memory optimization is disabled, use default batch size
        if not self.config.get('SYSTEM.memory_optimization', True):
            return default_batch_size
        
        try:
            # Get available memory in bytes
            available_memory = psutil.virtual_memory().available
            total_memory = psutil.virtual_memory().total
            
            # More conservative memory usage - use at most 30% of available memory
            usable_memory = min(available_memory * 0.3, total_memory * 0.1)
            
            # More accurate estimation of memory per file
            # Audio data: sample_rate * duration * 4 bytes (float32)
            # Mel spectrogram: n_mels * n_frames * 4 bytes (float32)
            # STFT: n_fft/2+1 * n_frames * 8 bytes (complex64)
            n_frames = int(self.duration * self.sample_rate / self.hop_length) + 1
            audio_memory = self.sample_rate * self.duration * 4
            mel_memory = self.n_mels * n_frames * 4
            stft_memory = (self.n_fft // 2 + 1) * n_frames * 8
            
            # Add overhead for processing and temporary arrays
            estimated_memory_per_file = (audio_memory + mel_memory + stft_memory) * 2.5
            
            # Calculate batch size based on memory estimate
            memory_based_batch_size = int(usable_memory / estimated_memory_per_file)
            
            # Ensure batch size is within reasonable bounds
            batch_size = max(min_batch_size, min(memory_based_batch_size, max_batch_size))
            
            # For very large files, further reduce batch size
            if self.duration > 10.0 or self.n_mels > 128:
                batch_size = max(min_batch_size, batch_size // 2)
            
            self.logger.debug(f"Memory estimation: Available={available_memory/(1024*1024):.2f}MB, "
                             f"Per file={estimated_memory_per_file/(1024*1024):.2f}MB, "
                             f"Batch size={batch_size}")
            return batch_size
            
        except Exception as e:
            self.logger.warning(f"Error estimating batch size: {e}. Using default: {default_batch_size}")
            return default_batch_size
    
    @abstractmethod
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract features from preprocessed audio signal.
        
        Args:
            y (np.ndarray): Preprocessed audio signal
            sr (int): Sample rate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        # Validate input types
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Audio data must be a numpy array, got {type(y)}")
            
        if not isinstance(sr, int):
            raise TypeError(f"Sample rate must be an integer, got {type(sr)}")
            
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array in _extract_features")
            # For numpy scalar types, we need to explicitly create a new array
            if isinstance(y, np.number):
                # Create a new array with the scalar value
                y = np.array([float(y)], dtype=np.float32)
            else:
                # For other scalar types
                y = np.atleast_1d(y).astype(np.float32)
            
        # Check for empty or invalid audio data
        if y.size == 0:
            raise ValueError("Cannot extract features from empty audio data")
        
        pass


class MelSpectrogramExtractor(FeatureExtractor):
    """
    Mel spectrogram feature extractor.
    
    This class extracts mel spectrogram features from audio files.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the mel spectrogram extractor.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        super().__init__(config)
        
        # Additional mel spectrogram parameters
        self.fmin = config.get('FEATURES.mel_spectrogram.fmin', 0)
        self.fmax = config.get('FEATURES.mel_spectrogram.fmax', None)
        self.multi_resolution = config.get('FEATURES.mel_spectrogram.multi_resolution', False)
        
        if self.multi_resolution:
            self.n_fft_list = [1024, 2048, 4096]
            self.hop_length_list = [256, 512, 1024]
        else:
            self.n_fft_list = [self.n_fft]
            self.hop_length_list = [self.hop_length]
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract mel spectrogram features from preprocessed audio signal.
        
        Args:
            y (np.ndarray): Preprocessed audio signal
            sr (int): Sample rate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with mel spectrogram features
        """
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array in MelSpectrogramExtractor")
            # For numpy scalar types, we need to explicitly create a new array
            if isinstance(y, np.number):
                # Create a new array with the scalar value
                y = np.array([float(y)], dtype=np.float32)
            else:
                # For other scalar types
                y = np.atleast_1d(y).astype(np.float32)
        
        features = {}
        
        # Extract mel spectrograms with different resolutions if multi-resolution is enabled
        for i, (n_fft, hop_length) in enumerate(zip(self.n_fft_list, self.hop_length_list)):
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1] range
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            if self.multi_resolution:
                features[f'mel_spectrogram_{i}'] = mel_spec_norm
            else:
                features['mel_spectrogram'] = mel_spec_norm
        
        return features


class MFCCExtractor(FeatureExtractor):
    """
    MFCC feature extractor.
    
    This class extracts MFCC features from audio files.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the MFCC extractor.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        super().__init__(config)
        
        # MFCC parameters
        self.n_mfcc = config.get('FEATURES.mfcc.n_mfcc', 40)
        self.include_deltas = config.get('FEATURES.mfcc.include_deltas', True)
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features from preprocessed audio signal.
        
        Args:
            y (np.ndarray): Preprocessed audio signal
            sr (int): Sample rate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with MFCC features
        """
        # Verify input types and shapes
        if not isinstance(y, np.ndarray):
            error_msg = f"Expected numpy array for audio data, got {type(y)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
            
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array in MFCCExtractor")
            # For numpy scalar types, we need to explicitly create a new array
            if isinstance(y, np.number):
                # Create a new array with the scalar value
                y = np.array([float(y)], dtype=np.float32)
            else:
                # For other scalar types
                y = np.atleast_1d(y).astype(np.float32)
            
        if len(y) == 0:
            error_msg = "Empty audio data provided to feature extractor"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.logger.debug(f"Extracting MFCC features from audio with shape {y.shape}, sr={sr}")
        features = {}
        
        try:
            # Compute MFCCs
            self.logger.debug(f"Computing MFCCs with n_mfcc={self.n_mfcc}, n_fft={self.n_fft}, hop_length={self.hop_length}")
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Verify MFCC is valid
            if mfcc.size == 0 or np.isnan(mfcc).any():
                error_msg = "Invalid MFCC features generated (contains NaN or empty)"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Normalize MFCCs with safe division
            mfcc_mean = mfcc.mean(axis=1, keepdims=True)
            mfcc_std = mfcc.std(axis=1, keepdims=True)
            
            # Check for zero standard deviation
            if np.any(mfcc_std < 1e-8):
                self.logger.warning("Some MFCC features have near-zero standard deviation, using epsilon for stability")
                mfcc_std = np.maximum(mfcc_std, 1e-8)
                
            mfcc_norm = (mfcc - mfcc_mean) / mfcc_std
            features['mfcc'] = mfcc_norm
            
            # Compute delta and delta-delta features if enabled
            if self.include_deltas:
                self.logger.debug("Computing delta and delta-delta features")
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                
                # Normalize delta features with safe division
                delta_mean = mfcc_delta.mean(axis=1, keepdims=True)
                delta_std = mfcc_delta.std(axis=1, keepdims=True)
                delta_std = np.maximum(delta_std, 1e-8)  # Prevent division by zero
                
                delta2_mean = mfcc_delta2.mean(axis=1, keepdims=True)
                delta2_std = mfcc_delta2.std(axis=1, keepdims=True)
                delta2_std = np.maximum(delta2_std, 1e-8)  # Prevent division by zero
                
                mfcc_delta_norm = (mfcc_delta - delta_mean) / delta_std
                mfcc_delta2_norm = (mfcc_delta2 - delta2_mean) / delta2_std
                
                features['mfcc_delta'] = mfcc_delta_norm
                features['mfcc_delta2'] = mfcc_delta2_norm
            
            self.logger.debug(f"Successfully extracted MFCC features with shape {mfcc_norm.shape}")
            
        except Exception as e:
            error_msg = f"Error extracting MFCC features: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        return features


class CompleteFeatureExtractor(FeatureExtractor):
    """
    Complete feature extractor that combines multiple feature types.
    
    This class extracts multiple types of features from audio files, including
    mel spectrograms, MFCCs, harmonic-percussive separation, and tonal features.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the complete feature extractor.
        
        Args:
            config (ConfigManager): Configuration manager instance
        """
        super().__init__(config)
        
        # Feature flags
        self.extract_mel = config.get('FEATURES.mel_spectrogram.enabled', True)
        self.extract_mfcc = config.get('FEATURES.mfcc.enabled', True)
        self.extract_harmonic_percussive = config.get('FEATURES.harmonic_percussive.enabled', False)
        self.extract_tonal = config.get('FEATURES.tonal.enabled', False)
        
        # MFCC parameters
        self.n_mfcc = config.get('FEATURES.mfcc.n_mfcc', 40)
        self.include_deltas = config.get('FEATURES.mfcc.include_deltas', True)
        
        # Harmonic-percussive parameters
        self.hp_margin = config.get('FEATURES.harmonic_percussive.margin', 1.0)
        
        # Tonal parameters
        self.extract_chroma = config.get('FEATURES.tonal.chroma', True)
        self.extract_tonnetz = config.get('FEATURES.tonal.tonnetz', True)
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract multiple features from preprocessed audio signal.
        
        Args:
            y (np.ndarray): Preprocessed audio signal
            sr (int): Sample rate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with multiple features
        """
        # Validate input data
        if y is None:
            error_msg = "Input audio data is None"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
            
        if not isinstance(y, np.ndarray):
            error_msg = f"Input audio data must be a numpy array, got {type(y)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
            
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            self.logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array in CompleteFeatureExtractor")
            # For numpy scalar types, we need to explicitly create a new array
            if isinstance(y, np.number):
                # Create a new array with the scalar value
                y = np.array([float(y)], dtype=np.float32)
            else:
                # For other scalar types
                y = np.atleast_1d(y).astype(np.float32)
            
        if y.size == 0:
            error_msg = "Input audio data is empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.logger.debug(f"Extracting complete features from audio with shape {y.shape}, sr={sr}")
        features = {}
        
        # Extract mel spectrogram if enabled
        if self.extract_mel:
            try:
                self.logger.debug(f"Computing mel spectrogram with n_fft={self.n_fft}, hop_length={self.hop_length}, n_mels={self.n_mels}")
                # Compute mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                
                # Check for invalid mel spectrogram
                if mel_spec.size == 0 or np.isnan(mel_spec).any():
                    self.logger.warning("Invalid mel spectrogram generated (contains NaN or empty), skipping")
                else:
                    # Convert to dB scale
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Check for valid range for normalization
                    mel_min = mel_spec_db.min()
                    mel_max = mel_spec_db.max()
                    mel_range = mel_max - mel_min
                    
                    if mel_range < 1e-8:
                        self.logger.warning("Mel spectrogram has very small dynamic range, using epsilon for stability")
                        mel_range = 1e-8
                    
                    # Normalize to [0, 1] range
                    mel_spec_norm = (mel_spec_db - mel_min) / mel_range
                    features['mel_spectrogram'] = mel_spec_norm
                    self.logger.debug(f"Successfully extracted mel spectrogram with shape {mel_spec_norm.shape}")
            except Exception as e:
                self.logger.error(f"Error extracting mel spectrogram: {e}")
                # Continue with other features instead of failing completely
        
        # Extract MFCCs if enabled
        if self.extract_mfcc:
            try:
                self.logger.debug(f"Computing MFCCs with n_mfcc={self.n_mfcc}, n_fft={self.n_fft}, hop_length={self.hop_length}")
                # Compute MFCCs
                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Check for invalid MFCCs
                if mfcc.size == 0 or np.isnan(mfcc).any():
                    self.logger.warning("Invalid MFCC features generated (contains NaN or empty), skipping")
                else:
                    # Normalize MFCCs with safe division
                    mfcc_mean = mfcc.mean(axis=1, keepdims=True)
                    mfcc_std = mfcc.std(axis=1, keepdims=True)
                    
                    # Check for zero standard deviation
                    if np.any(mfcc_std < 1e-8):
                        self.logger.warning("Some MFCC features have near-zero standard deviation, using epsilon for stability")
                        mfcc_std = np.maximum(mfcc_std, 1e-8)
                    
                    mfcc_norm = (mfcc - mfcc_mean) / mfcc_std
                    features['mfcc'] = mfcc_norm
                    self.logger.debug(f"Successfully extracted MFCC features with shape {mfcc_norm.shape}")
                    
                    # Compute delta and delta-delta features if enabled
                    if self.include_deltas:
                        try:
                            self.logger.debug("Computing delta and delta-delta features")
                            mfcc_delta = librosa.feature.delta(mfcc)
                            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                            
                            # Check for invalid delta features
                            if np.isnan(mfcc_delta).any() or np.isnan(mfcc_delta2).any():
                                self.logger.warning("Invalid delta features generated (contains NaN), skipping")
                            else:
                                # Normalize delta features with safe division
                                delta_mean = mfcc_delta.mean(axis=1, keepdims=True)
                                delta_std = mfcc_delta.std(axis=1, keepdims=True)
                                delta_std = np.maximum(delta_std, 1e-8)  # Prevent division by zero
                                
                                delta2_mean = mfcc_delta2.mean(axis=1, keepdims=True)
                                delta2_std = mfcc_delta2.std(axis=1, keepdims=True)
                                delta2_std = np.maximum(delta2_std, 1e-8)  # Prevent division by zero
                                
                                mfcc_delta_norm = (mfcc_delta - delta_mean) / delta_std
                                mfcc_delta2_norm = (mfcc_delta2 - delta2_mean) / delta2_std
                                
                                features['mfcc_delta'] = mfcc_delta_norm
                                features['mfcc_delta2'] = mfcc_delta2_norm
                                self.logger.debug(f"Successfully extracted delta features with shapes {mfcc_delta_norm.shape}, {mfcc_delta2_norm.shape}")
                        except Exception as e:
                            self.logger.error(f"Error computing delta features: {e}")
                            # Continue without delta features
            except Exception as e:
                self.logger.error(f"Error extracting MFCC features: {e}")
                # Continue with other features instead of failing completely
        
        # Extract harmonic and percussive components if enabled
        if self.extract_harmonic_percussive:
            try:
                self.logger.debug(f"Separating harmonic and percussive components with margin={self.hp_margin}")
                # Separate harmonic and percussive components
                y_harmonic, y_percussive = librosa.effects.hpss(y, margin=self.hp_margin)
                
                # Check for valid harmonic and percussive components
                if y_harmonic.size == 0 or y_percussive.size == 0 or np.isnan(y_harmonic).any() or np.isnan(y_percussive).any():
                    self.logger.warning("Invalid harmonic/percussive separation (contains NaN or empty), skipping")
                else:
                    # Process harmonic component
                    try:
                        self.logger.debug("Computing mel spectrogram for harmonic component")
                        # Compute mel spectrograms for harmonic component
                        mel_harmonic = librosa.feature.melspectrogram(
                            y=y_harmonic,
                            sr=sr,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            n_mels=self.n_mels
                        )
                        
                        if mel_harmonic.size == 0 or np.isnan(mel_harmonic).any():
                            self.logger.warning("Invalid harmonic mel spectrogram (contains NaN or empty), skipping")
                        else:
                            # Convert to dB scale
                            mel_harmonic_db = librosa.power_to_db(mel_harmonic, ref=np.max)
                            
                            # Check for valid range for normalization
                            h_min = mel_harmonic_db.min()
                            h_max = mel_harmonic_db.max()
                            h_range = h_max - h_min
                            
                            if h_range < 1e-8:
                                self.logger.warning("Harmonic mel spectrogram has very small dynamic range, using epsilon for stability")
                                h_range = 1e-8
                            
                            # Normalize
                            mel_harmonic_norm = (mel_harmonic_db - h_min) / h_range
                            features['mel_harmonic'] = mel_harmonic_norm
                            self.logger.debug(f"Successfully extracted harmonic mel spectrogram with shape {mel_harmonic_norm.shape}")
                    except Exception as e:
                        self.logger.error(f"Error processing harmonic component: {e}")
                    
                    # Process percussive component
                    try:
                        self.logger.debug("Computing mel spectrogram for percussive component")
                        # Compute mel spectrograms for percussive component
                        mel_percussive = librosa.feature.melspectrogram(
                            y=y_percussive,
                            sr=sr,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            n_mels=self.n_mels
                        )
                        
                        if mel_percussive.size == 0 or np.isnan(mel_percussive).any():
                            self.logger.warning("Invalid percussive mel spectrogram (contains NaN or empty), skipping")
                        else:
                            # Convert to dB scale
                            mel_percussive_db = librosa.power_to_db(mel_percussive, ref=np.max)
                            
                            # Check for valid range for normalization
                            p_min = mel_percussive_db.min()
                            p_max = mel_percussive_db.max()
                            p_range = p_max - p_min
                            
                            if p_range < 1e-8:
                                self.logger.warning("Percussive mel spectrogram has very small dynamic range, using epsilon for stability")
                                p_range = 1e-8
                            
                            # Normalize
                            mel_percussive_norm = (mel_percussive_db - p_min) / p_range
                            features['mel_percussive'] = mel_percussive_norm
                            self.logger.debug(f"Successfully extracted percussive mel spectrogram with shape {mel_percussive_norm.shape}")
                    except Exception as e:
                        self.logger.error(f"Error processing percussive component: {e}")
            except Exception as e:
                self.logger.error(f"Error in harmonic-percussive separation: {e}")
                # Continue with other features instead of failing completely
        
        # Extract tonal features if enabled
        if self.extract_tonal:
            self.logger.debug("Extracting tonal features")
            
            # Compute chroma features if enabled
            if self.extract_chroma:
                try:
                    self.logger.debug("Computing chroma features")
                    chroma = librosa.feature.chroma_stft(
                        y=y,
                        sr=sr,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length
                    )
                    
                    # Check for valid chroma features
                    if chroma.size == 0 or np.isnan(chroma).any():
                        self.logger.warning("Invalid chroma features generated (contains NaN or empty), skipping")
                    else:
                        # Normalize chroma features if needed
                        if np.max(chroma) > 1.0 or np.min(chroma) < 0.0:
                            self.logger.debug("Normalizing chroma features to [0,1] range")
                            chroma_min = chroma.min()
                            chroma_max = chroma.max()
                            chroma_range = chroma_max - chroma_min
                            
                            if chroma_range < 1e-8:
                                self.logger.warning("Chroma features have very small dynamic range, using epsilon for stability")
                                chroma_range = 1e-8
                                
                            chroma = (chroma - chroma_min) / chroma_range
                            
                        features['chroma'] = chroma
                        self.logger.debug(f"Successfully extracted chroma features with shape {chroma.shape}")
                except Exception as e:
                    self.logger.error(f"Error extracting chroma features: {e}")
            
            # Compute tonnetz features if enabled
            if self.extract_tonnetz:
                try:
                    self.logger.debug("Computing tonnetz features")
                    # Check if we already have chroma features we can use
                    if 'chroma' in features and features['chroma'] is not None and features['chroma'].size > 0:
                        self.logger.debug("Using existing chroma features for tonnetz computation")
                        chroma_for_tonnetz = features['chroma']
                    else:
                        self.logger.debug("Computing new chroma_cqt for tonnetz computation")
                        # Compute chromagram
                        chroma_for_tonnetz = librosa.feature.chroma_cqt(y=y, sr=sr)
                    
                    # Check for valid chroma for tonnetz
                    if chroma_for_tonnetz.size == 0 or np.isnan(chroma_for_tonnetz).any():
                        self.logger.warning("Invalid chroma for tonnetz computation (contains NaN or empty), skipping")
                    else:
                        # Compute tonnetz
                        tonnetz = librosa.feature.tonnetz(chroma=chroma_for_tonnetz, sr=sr)
                        
                        # Check for valid tonnetz features
                        if tonnetz.size == 0 or np.isnan(tonnetz).any():
                            self.logger.warning("Invalid tonnetz features generated (contains NaN or empty), skipping")
                        else:
                            # Normalize tonnetz features if needed
                            if np.max(tonnetz) > 1.0 or np.min(tonnetz) < -1.0:
                                self.logger.debug("Normalizing tonnetz features to [-1,1] range")
                                tonnetz_min = tonnetz.min()
                                tonnetz_max = tonnetz.max()
                                tonnetz_range = tonnetz_max - tonnetz_min
                                
                                if tonnetz_range < 1e-8:
                                    self.logger.warning("Tonnetz features have very small dynamic range, using epsilon for stability")
                                    tonnetz_range = 1e-8
                                    
                                tonnetz = 2.0 * ((tonnetz - tonnetz_min) / tonnetz_range) - 1.0
                                
                            features['tonnetz'] = tonnetz
                            self.logger.debug(f"Successfully extracted tonnetz features with shape {tonnetz.shape}")
                except Exception as e:
                    self.logger.error(f"Error extracting tonnetz features: {e}")
        
        # Check if we extracted at least one feature
        if not features:
            error_msg = "Failed to extract any valid features from the audio"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.logger.debug(f"Successfully extracted {len(features)} feature types: {list(features.keys())}")
        return features