import os
import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path

from pydub import AudioSegment
from improved.utils.exceptions import AudioProcessingError
from improved.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class AudioUtils:
    """
    Utility class for audio processing operations with robust error handling.
    
    This class provides methods for common audio operations like loading, converting,
    and processing audio files with proper error handling and logging.
    """
    
    @staticmethod
    def convert_audio_to_wav(input_file: Union[str, Path, Any], 
                           output_file: Optional[Union[str, Path]] = None,
                           sample_rate: int = 22050,
                           use_small_chunks: bool = False,
                           delete_original: bool = False) -> str:
        """
        Convert audio file to WAV format with robust error handling.
        
        Args:
            input_file: Path to input audio file or file-like object
            output_file: Path to output WAV file (if None, a temporary file is created)
            sample_rate: Target sample rate for the output WAV file
            use_small_chunks: Whether to process the file in small chunks (for large files)
            delete_original: Whether to delete the original file after conversion
            
        Returns:
            str: Path to the converted WAV file
            
        Raises:
            AudioProcessingError: If audio conversion fails
        """
        # Handle file-like objects (e.g., Flask FileStorage)
        temp_input_file = None
        temp_output_file = None
        original_filename = None
        
        try:
            # Check if input is a file-like object
            if hasattr(input_file, 'read') and callable(input_file.read):
                # Get original filename if available
                if hasattr(input_file, 'filename'):
                    original_filename = input_file.filename
                
                # Create a temporary file for the input
                fd, temp_input_file = tempfile.mkstemp(suffix=f"_{original_filename}" if original_filename else '.tmp')
                os.close(fd)
                
                # Save the file content to the temporary file
                if hasattr(input_file, 'save'):
                    input_file.save(temp_input_file)
                else:
                    with open(temp_input_file, 'wb') as f:
                        f.write(input_file.read())
                
                input_file = temp_input_file
            
            # Convert input_file to Path object
            input_path = Path(input_file)
            
            # Create output file path if not provided
            if output_file is None:
                fd, temp_output_file = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                output_path = Path(temp_output_file)
            else:
                output_path = Path(output_file)
                # Ensure output directory exists
                FileUtils.ensure_directory(output_path.parent)
            
            # Check if input file exists
            if not input_path.exists():
                raise AudioProcessingError(f"Input audio file not found: {input_path}")
            
            # Get file extension
            file_ext = input_path.suffix.lower()
            
            # If already a WAV file with correct sample rate, just copy it
            if file_ext == '.wav':
                try:
                    # Check sample rate
                    info = sf.info(str(input_path))
                    if int(info.samplerate) == sample_rate:
                        # Just copy the file
                        FileUtils.safe_copy_file(input_path, output_path, overwrite=True)
                        logger.debug(f"Input already in WAV format with correct sample rate, copied to {output_path}")
                        return str(output_path)
                except Exception as e:
                    logger.warning(f"Failed to check WAV file info: {e}, proceeding with conversion")
            
            # Convert audio file using pydub
            try:
                logger.debug(f"Converting audio file {input_path} to WAV format")
                
                if use_small_chunks:
                    # Process in small chunks for large files
                    audio = AudioSegment.from_file(str(input_path), format=file_ext[1:] if file_ext else None)
                    audio = audio.set_frame_rate(sample_rate)
                    audio.export(str(output_path), format='wav')
                else:
                    # Standard processing
                    audio = AudioSegment.from_file(str(input_path), format=file_ext[1:] if file_ext else None)
                    audio = audio.set_frame_rate(sample_rate)
                    audio.export(str(output_path), format='wav')
                
                logger.debug(f"Successfully converted audio to WAV: {output_path}")
            except Exception as e:
                logger.warning(f"Pydub conversion failed: {e}, trying ffmpeg fallback")
                
                # Fallback to ffmpeg directly
                try:
                    import subprocess
                    cmd = [
                        'ffmpeg', '-y', '-i', str(input_path),
                        '-ar', str(sample_rate),
                        '-ac', '1',  # Convert to mono
                        '-sample_fmt', 's16',  # 16-bit PCM
                        str(output_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    logger.debug(f"Successfully converted audio to WAV using ffmpeg: {output_path}")
                except Exception as ffmpeg_error:
                    raise AudioProcessingError(f"Audio conversion failed: {str(e)}. FFmpeg fallback also failed: {str(ffmpeg_error)}") from e
            
            return str(output_path)
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to convert audio file: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
        
        finally:
            # Clean up temporary files
            if temp_input_file and os.path.exists(temp_input_file) and temp_input_file != input_file:
                try:
                    os.unlink(temp_input_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary input file: {e}")
            
            # Delete original file if requested
            if delete_original and input_file != temp_input_file and isinstance(input_file, (str, Path)):
                try:
                    os.unlink(input_file)
                    logger.debug(f"Deleted original audio file: {input_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete original audio file: {e}")
    
    @staticmethod
    def load_audio(file_path: Union[str, Path], 
                 sample_rate: int = 22050,
                 mono: bool = True,
                 duration: Optional[float] = None,
                 offset: float = 0.0,
                 res_type: str = 'kaiser_best') -> Tuple[np.ndarray, int]:
        """
        Load audio file with robust error handling.
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate
            mono: Whether to convert to mono
            duration: Duration to load in seconds (None for full file)
            offset: Start reading after this time (in seconds)
            res_type: Resampling type
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
            
        Raises:
            AudioProcessingError: If audio loading fails
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise AudioProcessingError(f"Audio file not found: {path}")
            
            logger.debug(f"Loading audio file: {path} (sr={sample_rate}, mono={mono}, duration={duration})")
            
            # Check file size to avoid memory issues
            file_size = path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100 MB
                logger.warning(f"Large audio file detected ({file_size / (1024*1024):.2f} MB), loading with optimized parameters")
                # For large files, use a more memory-efficient approach
                y, sr = librosa.load(
                    path, 
                    sr=sample_rate, 
                    mono=mono, 
                    duration=duration, 
                    offset=offset,
                    res_type='kaiser_fast'  # Faster resampling
                )
            else:
                # Standard loading for normal-sized files
                y, sr = librosa.load(
                    path, 
                    sr=sample_rate, 
                    mono=mono, 
                    duration=duration, 
                    offset=offset,
                    res_type=res_type
                )
            
            # Handle scalar values (including numpy scalar types)
            if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
                logger.debug(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
                y = np.atleast_1d(y).astype(np.float32)
            
            logger.debug(f"Successfully loaded audio: {path.name}, shape={y.shape}, sr={sr}")
            return y, sr
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to load audio file {file_path}: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
    
    @staticmethod
    def extract_features(audio_file: Union[str, Path, np.ndarray],
                        sample_rate: int = 22050,
                        n_fft: int = 2048,
                        hop_length: int = 512,
                        n_mels: int = 128,
                        duration: Optional[float] = None,
                        fixed_length: Optional[int] = None) -> np.ndarray:
        """
        Extract mel spectrogram features from audio file with robust error handling.
        
        Args:
            audio_file: Path to audio file or audio data
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            duration: Duration to load in seconds (None for full file)
            fixed_length: Fixed number of time steps (None for variable length)
            
        Returns:
            np.ndarray: Mel spectrogram features
            
        Raises:
            AudioProcessingError: If feature extraction fails
        """
        try:
            # Load audio if file path is provided
            if isinstance(audio_file, (str, Path)):
                path = Path(audio_file)
                
                if not path.exists():
                    raise AudioProcessingError(f"Audio file not found: {path}")
                
                # Check file size to optimize parameters for large files
                file_size = path.stat().st_size
                if file_size > 50 * 1024 * 1024:  # 50 MB
                    logger.warning(f"Large audio file detected ({file_size / (1024*1024):.2f} MB), using optimized parameters")
                    # Adjust parameters for large files
                    y, sr = librosa.load(
                        path, 
                        sr=sample_rate, 
                        mono=True, 
                        duration=duration,
                        res_type='kaiser_fast'
                    )
                else:
                    y, sr = librosa.load(
                        path, 
                        sr=sample_rate, 
                        mono=True, 
                        duration=duration
                    )
            else:
                # Use provided audio data
                y = audio_file
                sr = sample_rate
            
            # Ensure audio is not empty
            if len(y) == 0:
                raise AudioProcessingError("Audio data is empty")
            
            # Pad or trim audio to expected duration if specified
            if duration is not None:
                expected_length = int(duration * sr)
                if len(y) < expected_length:
                    # Pad with zeros
                    y = np.pad(y, (0, expected_length - len(y)))
                elif len(y) > expected_length:
                    # Trim
                    y = y[:expected_length]
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                n_mels=n_mels
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Pad or trim to fixed length if specified
            if fixed_length is not None:
                if log_mel_spec.shape[1] < fixed_length:
                    # Pad with zeros
                    pad_width = fixed_length - log_mel_spec.shape[1]
                    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)))
                elif log_mel_spec.shape[1] > fixed_length:
                    # Trim
                    log_mel_spec = log_mel_spec[:, :fixed_length]
            
            logger.debug(f"Extracted mel spectrogram features with shape {log_mel_spec.shape}")
            return log_mel_spec
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to extract features: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
    
    @staticmethod
    def save_audio(y: np.ndarray, 
                  file_path: Union[str, Path], 
                  sample_rate: int = 22050,
                  subtype: str = 'PCM_16') -> str:
        """
        Save audio data to file with robust error handling.
        
        Args:
            y: Audio data
            file_path: Path to output audio file
            sample_rate: Sample rate
            subtype: Audio subtype (e.g., 'PCM_16', 'FLOAT', etc.)
            
        Returns:
            str: Path to the saved audio file
            
        Raises:
            AudioProcessingError: If audio saving fails
        """
        try:
            path = Path(file_path)
            
            # Ensure directory exists
            FileUtils.ensure_directory(path.parent)
            
            # Ensure audio data is valid
            if not isinstance(y, np.ndarray):
                raise AudioProcessingError(f"Audio data must be a numpy array, got {type(y)}")
            
            if np.isnan(y).any() or np.isinf(y).any():
                raise AudioProcessingError("Audio data contains NaN or Inf values")
            
            # Normalize audio if needed
            if np.abs(y).max() > 1.0:
                logger.warning("Audio data exceeds [-1.0, 1.0] range, normalizing")
                y = y / np.abs(y).max()
            
            # Save audio file
            sf.write(str(path), y, sample_rate, subtype=subtype)
            logger.debug(f"Successfully saved audio to {path}")
            
            return str(path)
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to save audio to {file_path}: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
    
    @staticmethod
    def get_audio_duration(file_path: Union[str, Path]) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            float: Duration in seconds
            
        Raises:
            AudioProcessingError: If duration retrieval fails
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise AudioProcessingError(f"Audio file not found: {path}")
            
            # Use soundfile for efficient duration retrieval
            info = sf.info(str(path))
            duration = info.duration
            
            logger.debug(f"Audio duration: {duration:.2f}s for {path.name}")
            return duration
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to get audio duration for {file_path}: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
    
    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict[str, Any]: Audio information including duration, sample rate, channels, etc.
            
        Raises:
            AudioProcessingError: If information retrieval fails
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise AudioProcessingError(f"Audio file not found: {path}")
            
            # Get basic info using soundfile
            info = sf.info(str(path))
            
            # Create info dictionary
            audio_info = {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'file_size': path.stat().st_size,
                'file_name': path.name,
                'file_path': str(path),
                'file_extension': path.suffix.lower()
            }
            
            logger.debug(f"Retrieved audio info for {path.name}: {audio_info}")
            return audio_info
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to get audio info for {file_path}: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
    
    @staticmethod
    def trim_silence(y: np.ndarray, 
                    top_db: float = 60,
                    frame_length: int = 2048,
                    hop_length: int = 512) -> np.ndarray:
        """
        Trim leading and trailing silence from an audio signal.
        
        Args:
            y: Audio signal
            top_db: Threshold (in decibels) below reference to consider as silence
            frame_length: Length of analysis frame
            hop_length: Number of samples between frames
            
        Returns:
            np.ndarray: Trimmed audio signal
            
        Raises:
            AudioProcessingError: If trimming fails
        """
        try:
            # Ensure audio data is valid
            if not isinstance(y, np.ndarray):
                raise AudioProcessingError(f"Audio data must be a numpy array, got {type(y)}")
            
            if len(y) == 0:
                raise AudioProcessingError("Audio data is empty")
            
            # Trim silence
            y_trimmed, _ = librosa.effects.trim(
                y, 
                top_db=top_db, 
                frame_length=frame_length, 
                hop_length=hop_length
            )
            
            logger.debug(f"Trimmed silence: {len(y)} → {len(y_trimmed)} samples")
            return y_trimmed
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to trim silence: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise
    
    @staticmethod
    def split_audio(y: np.ndarray, 
                   sr: int,
                   segment_duration: float = 3.0,
                   overlap: float = 0.0) -> List[np.ndarray]:
        """
        Split audio into segments of specified duration with optional overlap.
        
        Args:
            y: Audio signal
            sr: Sample rate
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List[np.ndarray]: List of audio segments
            
        Raises:
            AudioProcessingError: If splitting fails
        """
        try:
            # Ensure audio data is valid
            if not isinstance(y, np.ndarray):
                raise AudioProcessingError(f"Audio data must be a numpy array, got {type(y)}")
            
            if len(y) == 0:
                raise AudioProcessingError("Audio data is empty")
            
            # Calculate segment and hop lengths in samples
            segment_length = int(segment_duration * sr)
            hop_length = int((segment_duration - overlap) * sr)
            
            if segment_length <= 0:
                raise AudioProcessingError(f"Invalid segment duration: {segment_duration}")
            
            if hop_length <= 0:
                raise AudioProcessingError(f"Invalid overlap: {overlap}")
            
            # Calculate number of segments
            num_segments = max(1, 1 + (len(y) - segment_length) // hop_length)
            
            # Split audio into segments
            segments = []
            for i in range(num_segments):
                start = i * hop_length
                end = start + segment_length
                
                if end <= len(y):
                    segments.append(y[start:end])
                else:
                    # Pad the last segment if needed
                    last_segment = np.zeros(segment_length)
                    last_segment[:len(y) - start] = y[start:]
                    segments.append(last_segment)
            
            logger.debug(f"Split audio into {len(segments)} segments of {segment_duration}s with {overlap}s overlap")
            return segments
        
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                error_msg = f"Failed to split audio: {str(e)}"
                logger.error(error_msg)
                raise AudioProcessingError(error_msg) from e
            raise