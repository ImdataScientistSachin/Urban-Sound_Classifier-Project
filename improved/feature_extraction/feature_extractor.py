import os
import logging
import numpy as np
import librosa
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import time
import gc

from improved.utils.exceptions import FeatureExtractionError
from improved.utils.audio_utils import AudioUtils
from improved.utils.memory_optimizer import MemoryOptimizer
from improved.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Base class for feature extraction with robust error handling and memory optimization.
    
    This class provides methods for extracting features from audio files with proper
    error handling, logging, and memory optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary containing feature extraction parameters
        """
        self.config = config
        self.memory_optimizer = MemoryOptimizer()
        
        # Extract common parameters from config
        self.sample_rate = config.get('sample_rate', 22050)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.n_mels = config.get('n_mels', 128)
        self.duration = config.get('duration', None)
        self.fixed_length = config.get('fixed_length', None)
        
        # Feature cache to avoid recomputing features
        self.feature_cache = {}
        self.max_cache_size = config.get('max_cache_size', 100)
        
        logger.info(f"Initialized {self.__class__.__name__} with parameters: "
                  f"sr={self.sample_rate}, n_fft={self.n_fft}, "
                  f"hop_length={self.hop_length}, n_mels={self.n_mels}")
    
    def extract_features(self, audio_file: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Extract features from an audio file.
        
        Args:
            audio_file: Path to audio file or audio data
            
        Returns:
            np.ndarray: Extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        raise NotImplementedError("Subclasses must implement extract_features method")
    
    def extract_features_batch(self, 
                             audio_files: List[Union[str, Path]],
                             batch_size: Optional[int] = None,
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, np.ndarray]:
        """
        Extract features from a batch of audio files with memory optimization.
        
        Args:
            audio_files: List of paths to audio files
            batch_size: Batch size for processing (if None, estimated based on memory)
            progress_callback: Callback function for reporting progress
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping file paths to extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            # Estimate batch size if not provided
            if batch_size is None:
                # Estimate based on a sample file
                if audio_files:
                    sample_file = audio_files[0]
                    try:
                        # Extract features from sample file to estimate memory usage
                        sample_features = self.extract_features(sample_file)
                        sample_size_bytes = sample_features.nbytes
                        batch_size = self.memory_optimizer.estimate_batch_size(sample_size_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to estimate batch size from sample: {e}, using default")
                        batch_size = 16
                else:
                    batch_size = 16
            
            logger.info(f"Extracting features from {len(audio_files)} files with batch size {batch_size}")
            
            # Initialize results dictionary
            results = {}
            
            # Process files in batches
            for i in range(0, len(audio_files), batch_size):
                batch = audio_files[i:i + batch_size]
                batch_start_time = time.time()
                
                logger.debug(f"Processing batch {i // batch_size + 1}/{(len(audio_files) + batch_size - 1) // batch_size} "
                           f"({len(batch)} files)")
                
                # Process each file in the batch
                for j, file_path in enumerate(batch):
                    try:
                        # Extract features
                        features = self.extract_features(file_path)
                        
                        # Store results
                        results[str(file_path)] = features
                        
                        # Report progress if callback provided
                        if progress_callback:
                            progress_callback(i + j + 1, len(audio_files))
                    
                    except Exception as e:
                        logger.error(f"Failed to extract features from {file_path}: {str(e)}")
                        # Continue with next file
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                logger.debug(f"Batch processed in {batch_duration:.2f}s "
                           f"({batch_duration / len(batch):.2f}s per file)")
                
                # Check memory usage and clean up if necessary
                if self.memory_optimizer.is_memory_critical():
                    logger.warning("Memory usage critical, cleaning up")
                    self.memory_optimizer.clear_memory()
                    # Reduce batch size for next iterations if memory is still critical
                    if self.memory_optimizer.is_memory_critical() and batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        logger.warning(f"Reduced batch size to {batch_size} due to memory constraints")
            
            logger.info(f"Feature extraction completed for {len(results)} files")
            return results
        
        except Exception as e:
            if not isinstance(e, FeatureExtractionError):
                error_msg = f"Batch feature extraction failed: {str(e)}"
                logger.error(error_msg)
                raise FeatureExtractionError(error_msg) from e
            raise
    
    def extract_and_save(self, 
                        audio_files: List[Union[str, Path]],
                        output_dir: Union[str, Path],
                        file_format: str = 'npy',
                        batch_size: Optional[int] = None,
                        progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, str]:
        """
        Extract features from audio files and save them to disk.
        
        Args:
            audio_files: List of paths to audio files
            output_dir: Directory to save extracted features
            file_format: Format to save features ('npy' or 'csv')
            batch_size: Batch size for processing
            progress_callback: Callback function for reporting progress
            
        Returns:
            Dict[str, str]: Dictionary mapping original file paths to saved feature paths
            
        Raises:
            FeatureExtractionError: If feature extraction or saving fails
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_dir)
            FileUtils.ensure_directory(output_path)
            
            logger.info(f"Extracting and saving features to {output_path} in {file_format} format")
            
            # Initialize results dictionary
            results = {}
            
            # Process files in batches
            for i in range(0, len(audio_files), batch_size or 16):
                batch = audio_files[i:i + (batch_size or 16)]
                
                # Extract features for the batch
                batch_features = self.extract_features_batch(
                    batch, 
                    batch_size=batch_size,
                    progress_callback=progress_callback
                )
                
                # Save features to disk
                for file_path, features in batch_features.items():
                    try:
                        # Generate output file path
                        rel_path = Path(file_path).stem
                        if file_format.lower() == 'npy':
                            out_file = output_path / f"{rel_path}.npy"
                            np.save(out_file, features)
                        elif file_format.lower() == 'csv':
                            out_file = output_path / f"{rel_path}.csv"
                            pd.DataFrame(features).to_csv(out_file, index=False)
                        else:
                            raise FeatureExtractionError(f"Unsupported file format: {file_format}")
                        
                        # Store mapping
                        results[file_path] = str(out_file)
                        
                        logger.debug(f"Saved features for {file_path} to {out_file}")
                    
                    except Exception as e:
                        logger.error(f"Failed to save features for {file_path}: {str(e)}")
                        # Continue with next file
                
                # Clean up memory
                self.memory_optimizer.clear_memory()
            
            logger.info(f"Feature extraction and saving completed for {len(results)} files")
            return results
        
        except Exception as e:
            if not isinstance(e, FeatureExtractionError):
                error_msg = f"Feature extraction and saving failed: {str(e)}"
                logger.error(error_msg)
                raise FeatureExtractionError(error_msg) from e
            raise
    
    def clear_cache(self):
        """
        Clear the feature cache.
        """
        self.feature_cache.clear()
        logger.debug("Feature cache cleared")


class MelSpectrogramExtractor(FeatureExtractor):
    """
    Feature extractor for mel spectrogram features.
    """
    
    def extract_features(self, audio_file: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Extract mel spectrogram features from an audio file.
        
        Args:
            audio_file: Path to audio file or audio data
            
        Returns:
            np.ndarray: Mel spectrogram features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            # Check cache for file path
            if isinstance(audio_file, (str, Path)) and str(audio_file) in self.feature_cache:
                logger.debug(f"Using cached features for {audio_file}")
                return self.feature_cache[str(audio_file)]
            
            # Extract features using AudioUtils
            features = AudioUtils.extract_features(
                audio_file,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                duration=self.duration,
                fixed_length=self.fixed_length
            )
            
            # Cache features if it's a file path
            if isinstance(audio_file, (str, Path)):
                # Limit cache size
                if len(self.feature_cache) >= self.max_cache_size:
                    # Remove oldest item
                    self.feature_cache.pop(next(iter(self.feature_cache)))
                
                self.feature_cache[str(audio_file)] = features
            
            return features
        
        except Exception as e:
            if not isinstance(e, FeatureExtractionError):
                error_msg = f"Mel spectrogram extraction failed: {str(e)}"
                logger.error(error_msg)
                raise FeatureExtractionError(error_msg) from e
            raise


class MFCCExtractor(FeatureExtractor):
    """
    Feature extractor for MFCC features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MFCC extractor.
        
        Args:
            config: Configuration dictionary containing feature extraction parameters
        """
        super().__init__(config)
        self.n_mfcc = config.get('n_mfcc', 13)
        logger.info(f"Initialized MFCCExtractor with n_mfcc={self.n_mfcc}")
    
    def extract_features(self, audio_file: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Extract MFCC features from an audio file.
        
        Args:
            audio_file: Path to audio file or audio data
            
        Returns:
            np.ndarray: MFCC features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            # Check cache for file path
            if isinstance(audio_file, (str, Path)) and str(audio_file) in self.feature_cache:
                logger.debug(f"Using cached features for {audio_file}")
                return self.feature_cache[str(audio_file)]
            
            # Load audio if file path is provided
            if isinstance(audio_file, (str, Path)):
                y, sr = AudioUtils.load_audio(
                    audio_file,
                    sample_rate=self.sample_rate,
                    duration=self.duration
                )
            else:
                # Use provided audio data
                y = audio_file
                sr = self.sample_rate
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Pad or trim to fixed length if specified
            if self.fixed_length is not None:
                if mfcc.shape[1] < self.fixed_length:
                    # Pad with zeros
                    pad_width = self.fixed_length - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
                elif mfcc.shape[1] > self.fixed_length:
                    # Trim
                    mfcc = mfcc[:, :self.fixed_length]
            
            # Cache features if it's a file path
            if isinstance(audio_file, (str, Path)):
                # Limit cache size
                if len(self.feature_cache) >= self.max_cache_size:
                    # Remove oldest item
                    self.feature_cache.pop(next(iter(self.feature_cache)))
                
                self.feature_cache[str(audio_file)] = mfcc
            
            logger.debug(f"Extracted MFCC features with shape {mfcc.shape}")
            return mfcc
        
        except Exception as e:
            if not isinstance(e, FeatureExtractionError):
                error_msg = f"MFCC extraction failed: {str(e)}"
                logger.error(error_msg)
                raise FeatureExtractionError(error_msg) from e
            raise


class MultiFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that combines multiple feature types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-feature extractor.
        
        Args:
            config: Configuration dictionary containing feature extraction parameters
        """
        super().__init__(config)
        
        # Configure feature types to extract
        self.feature_types = config.get('feature_types', ['mel_spectrogram'])
        
        # Initialize individual extractors
        self.extractors = {}
        if 'mel_spectrogram' in self.feature_types:
            self.extractors['mel_spectrogram'] = MelSpectrogramExtractor(config)
        if 'mfcc' in self.feature_types:
            self.extractors['mfcc'] = MFCCExtractor(config)
        
        logger.info(f"Initialized MultiFeatureExtractor with feature types: {self.feature_types}")
    
    def extract_features(self, audio_file: Union[str, Path, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract multiple feature types from an audio file.
        
        Args:
            audio_file: Path to audio file or audio data
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping feature types to extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            # Check cache for file path
            if isinstance(audio_file, (str, Path)) and str(audio_file) in self.feature_cache:
                logger.debug(f"Using cached features for {audio_file}")
                return self.feature_cache[str(audio_file)]
            
            # Extract features using each extractor
            features = {}
            for feature_type, extractor in self.extractors.items():
                try:
                    features[feature_type] = extractor.extract_features(audio_file)
                except Exception as e:
                    logger.error(f"Failed to extract {feature_type} features: {str(e)}")
                    # Continue with other feature types
            
            # Cache features if it's a file path and at least one feature type was extracted
            if isinstance(audio_file, (str, Path)) and features:
                # Limit cache size
                if len(self.feature_cache) >= self.max_cache_size:
                    # Remove oldest item
                    self.feature_cache.pop(next(iter(self.feature_cache)))
                
                self.feature_cache[str(audio_file)] = features
            
            if not features:
                raise FeatureExtractionError(f"Failed to extract any features from {audio_file}")
            
            logger.debug(f"Extracted multiple features: {', '.join(features.keys())}")
            return features
        
        except Exception as e:
            if not isinstance(e, FeatureExtractionError):
                error_msg = f"Multi-feature extraction failed: {str(e)}"
                logger.error(error_msg)
                raise FeatureExtractionError(error_msg) from e
            raise


class AdvancedFeatureExtractor(FeatureExtractor):
    """
    Advanced feature extractor with additional preprocessing and feature engineering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced feature extractor.
        
        Args:
            config: Configuration dictionary containing feature extraction parameters
        """
        super().__init__(config)
        
        # Additional parameters
        self.normalize = config.get('normalize', True)
        self.augment = config.get('augment', False)
        self.trim_silence = config.get('trim_silence', False)
        self.delta_order = config.get('delta_order', 0)  # 0 means no delta features
        
        # Initialize base extractor
        feature_type = config.get('feature_type', 'mel_spectrogram')
        if feature_type == 'mel_spectrogram':
            self.base_extractor = MelSpectrogramExtractor(config)
        elif feature_type == 'mfcc':
            self.base_extractor = MFCCExtractor(config)
        elif feature_type == 'multi':
            self.base_extractor = MultiFeatureExtractor(config)
        else:
            raise FeatureExtractionError(f"Unsupported feature type: {feature_type}")
        
        logger.info(f"Initialized AdvancedFeatureExtractor with feature_type={feature_type}, "
                  f"normalize={self.normalize}, augment={self.augment}, "
                  f"trim_silence={self.trim_silence}, delta_order={self.delta_order}")
    
    def extract_features(self, audio_file: Union[str, Path, np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract features with advanced preprocessing and feature engineering.
        
        Args:
            audio_file: Path to audio file or audio data
            
        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            # Check cache for file path
            if isinstance(audio_file, (str, Path)) and str(audio_file) in self.feature_cache:
                logger.debug(f"Using cached features for {audio_file}")
                return self.feature_cache[str(audio_file)]
            
            # Preprocess audio if it's a file path
            if isinstance(audio_file, (str, Path)):
                # Load audio
                y, sr = AudioUtils.load_audio(
                    audio_file,
                    sample_rate=self.sample_rate,
                    duration=self.duration
                )
                
                # Trim silence if enabled
                if self.trim_silence:
                    y = AudioUtils.trim_silence(y)
                
                # Apply augmentation if enabled
                if self.augment:
                    y = self._augment_audio(y, sr)
                
                # Extract base features from preprocessed audio
                features = self.base_extractor.extract_features(y)
            else:
                # Extract base features directly from provided audio data
                features = self.base_extractor.extract_features(audio_file)
            
            # Apply post-processing to features
            if isinstance(features, dict):
                # Process each feature type
                processed_features = {}
                for feature_type, feature_data in features.items():
                    processed_features[feature_type] = self._postprocess_features(feature_data)
                features = processed_features
            else:
                # Process single feature type
                features = self._postprocess_features(features)
            
            # Cache features if it's a file path
            if isinstance(audio_file, (str, Path)):
                # Limit cache size
                if len(self.feature_cache) >= self.max_cache_size:
                    # Remove oldest item
                    self.feature_cache.pop(next(iter(self.feature_cache)))
                
                self.feature_cache[str(audio_file)] = features
            
            return features
        
        except Exception as e:
            if not isinstance(e, FeatureExtractionError):
                error_msg = f"Advanced feature extraction failed: {str(e)}"
                logger.error(error_msg)
                raise FeatureExtractionError(error_msg) from e
            raise
    
    def _augment_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply audio augmentation techniques.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            np.ndarray: Augmented audio signal
        """
        try:
            # Randomly select an augmentation technique
            import random
            augmentation = random.choice(['pitch_shift', 'time_stretch', 'noise', 'none'])
            
            if augmentation == 'pitch_shift':
                # Pitch shift by -2 to 2 semitones
                n_steps = random.uniform(-2, 2)
                y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                logger.debug(f"Applied pitch shift augmentation: {n_steps} semitones")
            
            elif augmentation == 'time_stretch':
                # Time stretch by 0.8 to 1.2 factor
                rate = random.uniform(0.8, 1.2)
                y_aug = librosa.effects.time_stretch(y, rate=rate)
                logger.debug(f"Applied time stretch augmentation: {rate} rate")
            
            elif augmentation == 'noise':
                # Add random noise
                noise_level = random.uniform(0.001, 0.005)
                noise = np.random.randn(len(y))
                y_aug = y + noise_level * noise
                logger.debug(f"Applied noise augmentation: {noise_level} level")
            
            else:  # 'none'
                # No augmentation
                y_aug = y
            
            return y_aug
        
        except Exception as e:
            logger.warning(f"Audio augmentation failed: {str(e)}, using original audio")
            return y
    
    def _postprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to extracted features.
        
        Args:
            features: Extracted features
            
        Returns:
            np.ndarray: Processed features
        """
        try:
            # Normalize if enabled
            if self.normalize:
                # Standardize features (zero mean, unit variance)
                mean = np.mean(features, axis=1, keepdims=True)
                std = np.std(features, axis=1, keepdims=True) + 1e-10  # Avoid division by zero
                features = (features - mean) / std
            
            # Compute delta features if enabled
            if self.delta_order > 0:
                # Compute first-order delta
                delta1 = librosa.feature.delta(features, order=1)
                
                if self.delta_order > 1:
                    # Compute second-order delta
                    delta2 = librosa.feature.delta(features, order=2)
                    
                    # Stack original features with delta features
                    features = np.vstack([features, delta1, delta2])
                else:
                    # Stack original features with first-order delta
                    features = np.vstack([features, delta1])
            
            return features
        
        except Exception as e:
            logger.warning(f"Feature post-processing failed: {str(e)}, using original features")
            return features