import os
import tempfile
import logging
import shutil
from typing import Union, Optional, BinaryIO
from pydub import AudioSegment

class AudioUtils:
    """
    Utility class for audio processing operations.
    
    This class provides static methods for common audio processing tasks
    such as format conversion, resampling, and audio manipulation.
    """
    
    @staticmethod
    def convert_audio_to_wav(audio_input: Union[str, BinaryIO], output_path: Optional[str] = None, sample_rate: int = 22050) -> str:
        """
        Convert an audio file to WAV format with the specified sample rate.
        
        This method handles various audio formats and ensures the output is in WAV format
        with the correct sample rate. If the input is already a WAV file with the correct
        sample rate, it will be returned as is or copied to a temporary location.
        
        Args:
            audio_input (Union[str, BinaryIO]): Path to audio file or file-like object
            output_path (Optional[str]): Path to save the converted WAV file. If None, a temporary file will be created.
            sample_rate (int): Target sample rate for the output WAV file
            
        Returns:
            str: Path to the WAV file (original or converted)
            
        Raises:
            ValueError: If the audio input is invalid or conversion fails
        """
        logger = logging.getLogger(__name__)
        
        # Validate input
        if audio_input is None:
            raise ValueError("Audio input cannot be None")
        
        # Handle file path input
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise ValueError(f"Audio file not found: {audio_input}")
                
            # Check if file is empty
            if os.path.getsize(audio_input) == 0:
                raise ValueError(f"Audio file is empty: {audio_input}")
                
            # Check if the file is already a WAV file with the correct sample rate
            if audio_input.lower().endswith('.wav') and output_path is None:
                try:
                    audio = AudioSegment.from_wav(audio_input)
                    if audio.frame_rate == sample_rate:
                        logger.debug(f"Audio file is already in WAV format with correct sample rate: {audio_input}")
                        return audio_input
                except Exception as e:
                    logger.warning(f"Error checking WAV file: {e}. Will convert anyway.")
            
            # Load the audio file using pydub
            try:
                file_ext = os.path.splitext(audio_input)[1].lower()
                if file_ext == '.wav':
                    audio = AudioSegment.from_wav(audio_input)
                elif file_ext == '.mp3':
                    audio = AudioSegment.from_mp3(audio_input)
                elif file_ext == '.ogg':
                    audio = AudioSegment.from_ogg(audio_input)
                elif file_ext == '.flac':
                    audio = AudioSegment.from_file(audio_input, format="flac")
                else:
                    # Try to load using the file extension as format
                    format_name = file_ext[1:] if file_ext.startswith('.') else file_ext
                    audio = AudioSegment.from_file(audio_input, format=format_name)
            except Exception as e:
                raise ValueError(f"Failed to load audio file {audio_input}: {e}")
        
        # Handle file-like object input (including Flask FileStorage)
        else:
            try:
                # Check if file-like object has read method
                if not hasattr(audio_input, 'read'):
                    raise ValueError("File-like object must have 'read' method")
                
                # For Flask FileStorage objects, save to a temporary file first
                if hasattr(audio_input, 'save'):
                    temp_input = tempfile.NamedTemporaryFile(delete=False)
                    temp_input.close()
                    audio_input.save(temp_input.name)
                    
                    # Get format from filename
                    format_name = None
                    if hasattr(audio_input, 'filename'):
                        file_ext = os.path.splitext(audio_input.filename)[1].lower()
                        format_name = file_ext[1:] if file_ext.startswith('.') else file_ext
                    
                    # Load from the temporary file
                    if format_name:
                        audio = AudioSegment.from_file(temp_input.name, format=format_name)
                    else:
                        # Try to guess format
                        audio = AudioSegment.from_file(temp_input.name)
                    
                    # Clean up temporary file
                    os.unlink(temp_input.name)
                else:
                    # Try to determine format from the object's name if available
                    format_name = None
                    if hasattr(audio_input, 'name'):
                        file_ext = os.path.splitext(audio_input.name)[1].lower()
                        format_name = file_ext[1:] if file_ext.startswith('.') else file_ext
                    
                    # If format couldn't be determined, default to WAV
                    if not format_name:
                        format_name = 'wav'
                    
                    # Reset file pointer if possible
                    if hasattr(audio_input, 'seek'):
                        audio_input.seek(0)
                    
                    # Load audio from file-like object
                    audio = AudioSegment.from_file(audio_input, format=format_name)
            except Exception as e:
                raise ValueError(f"Failed to load audio from file-like object: {e}")
        
        # Verify audio data is valid
        if len(audio) == 0:
            raise ValueError("Audio data is empty")
        
        # Set the sample rate if needed
        if audio.frame_rate != sample_rate:
            logger.debug(f"Converting audio from {audio.frame_rate}Hz to {sample_rate}Hz")
            audio = audio.set_frame_rate(sample_rate)
        
        # Determine output path
        is_temp_file = False
        if output_path is None:
            # Create a temporary file for the output
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
            is_temp_file = True
        
        # Create directory for output_path if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Export the audio to the output file
        try:
            audio.export(output_path, format='wav')
            logger.debug(f"Converted audio to WAV format: {output_path}")
            
            # Verify the output file exists and is not empty
            if not os.path.exists(output_path):
                raise ValueError(f"Exported WAV file not found: {output_path}")
                
            if os.path.getsize(output_path) == 0:
                raise ValueError(f"Exported WAV file is empty: {output_path}")
                
            return output_path
        except Exception as e:
            # Clean up the output file if export fails and we created it
            if is_temp_file and os.path.exists(output_path):
                os.unlink(output_path)
            raise ValueError(f"Failed to convert audio to WAV format: {e}")
    
    @staticmethod
    def trim_audio(audio_path: str, start_time: float, end_time: float) -> str:
        """
        Trim an audio file to the specified start and end times.
        
        Args:
            audio_path (str): Path to the audio file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            
        Returns:
            str: Path to the trimmed audio file
            
        Raises:
            ValueError: If the audio file is invalid or trimming fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Ensure end time is not beyond the audio duration
            if end_ms > len(audio):
                logger.warning(f"End time {end_time}s exceeds audio duration {len(audio)/1000}s. Using full duration.")
                end_ms = len(audio)
            
            # Ensure start time is not negative
            if start_ms < 0:
                logger.warning(f"Start time {start_time}s is negative. Using 0s.")
                start_ms = 0
            
            # Ensure start time is before end time
            if start_ms >= end_ms:
                raise ValueError(f"Start time {start_time}s must be less than end time {end_time}s")
            
            # Trim the audio
            trimmed_audio = audio[start_ms:end_ms]
            
            # Create a temporary file for the output
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Export the trimmed audio
            trimmed_audio.export(temp_file.name, format='wav')
            logger.debug(f"Trimmed audio from {start_time}s to {end_time}s: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            raise ValueError(f"Failed to trim audio: {e}")
    
    @staticmethod
    def adjust_volume(audio_path: str, gain_db: float) -> str:
        """
        Adjust the volume of an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            gain_db (float): Gain in decibels (positive for amplification, negative for attenuation)
            
        Returns:
            str: Path to the volume-adjusted audio file
            
        Raises:
            ValueError: If the audio file is invalid or volume adjustment fails
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Adjust the volume
            adjusted_audio = audio + gain_db
            
            # Create a temporary file for the output
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Export the adjusted audio
            adjusted_audio.export(temp_file.name, format='wav')
            logger.debug(f"Adjusted audio volume by {gain_db}dB: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            raise ValueError(f"Failed to adjust audio volume: {e}")
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            float: Duration of the audio file in seconds
            
        Raises:
            ValueError: If the audio file is invalid or duration calculation fails
        """
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Get the duration in seconds
            duration_seconds = len(audio) / 1000.0
            
            return duration_seconds
            
        except Exception as e:
            raise ValueError(f"Failed to get audio duration: {e}")
    
    @staticmethod
    def mix_audio(audio_paths: list, weights: Optional[list] = None) -> str:
        """
        Mix multiple audio files with optional weights.
        
        Args:
            audio_paths (list): List of paths to audio files
            weights (Optional[list]): List of weights for each audio file (default: equal weights)
            
        Returns:
            str: Path to the mixed audio file
            
        Raises:
            ValueError: If the audio files are invalid or mixing fails
        """
        logger = logging.getLogger(__name__)
        
        if not audio_paths:
            raise ValueError("No audio files provided for mixing")
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(audio_paths)] * len(audio_paths)
        
        # Ensure weights and audio_paths have the same length
        if len(weights) != len(audio_paths):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of audio files ({len(audio_paths)})")
        
        try:
            # Load all audio files
            audios = []
            max_length_ms = 0
            
            for path in audio_paths:
                audio = AudioSegment.from_file(path)
                audios.append(audio)
                max_length_ms = max(max_length_ms, len(audio))
            
            # Ensure all audio files have the same length by padding with silence
            for i in range(len(audios)):
                if len(audios[i]) < max_length_ms:
                    audios[i] = audios[i] + AudioSegment.silent(duration=max_length_ms - len(audios[i]))
            
            # Mix the audio files with weights
            mixed_audio = AudioSegment.silent(duration=max_length_ms)
            
            for audio, weight in zip(audios, weights):
                # Apply weight by adjusting the volume
                gain_db = 20 * (weight if weight > 0 else 0)  # Convert weight to dB gain
                weighted_audio = audio + gain_db
                
                # Overlay the weighted audio
                mixed_audio = mixed_audio.overlay(weighted_audio)
            
            # Create a temporary file for the output
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Export the mixed audio
            mixed_audio.export(temp_file.name, format='wav')
            logger.debug(f"Mixed {len(audio_paths)} audio files: {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            raise ValueError(f"Failed to mix audio files: {e}")