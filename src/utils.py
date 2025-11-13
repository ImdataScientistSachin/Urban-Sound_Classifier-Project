import librosa
import numpy as np
import os
import tempfile
from pydub import AudioSegment
from config import SAMPLE_RATE, DURATION, N_MELS, N_FFT, HOP_LENGTH

def convert_audio_to_wav(file, use_small_chunks=False):
    """
    Convert uploaded audio file to WAV format if needed.
    
    Args:
        file: Flask file object from request or file-like object
        use_small_chunks: Whether to process the file in smaller chunks for large files
        
    Returns:
        str: Path to the WAV file
    """
    def convert_audio_to_wav(file, use_small_chunks=False):
        """
        Convert uploaded audio file to WAV format if needed.
        
        Args:
            file: Flask file object from request or file-like object
            use_small_chunks: Whether to process the file in smaller chunks for large files
            
        Returns:
            str: Path to the WAV file
        """
        temp_files = []  # Track all temporary files for cleanup
        
        try:
            # Create a temporary file to save the uploaded file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, 'temp_audio')
            
            # Save the uploaded file
            original_path = f"{temp_path}_original"
            temp_files.append(original_path)  # Track for cleanup
            
            # Handle different types of file objects
            if hasattr(file, 'filename') and hasattr(file, 'save'):
                # This is a Flask FileStorage object
                print(f"Saving uploaded file to {original_path}")
                file.save(original_path)
                filename = file.filename
            else:
                # This is likely a file-like object (already opened file)
                print(f"Copying file-like object to {original_path}")
                with open(original_path, 'wb') as f:
                    if hasattr(file, 'read'):
                        # If using small chunks mode, read in smaller chunks to avoid memory issues
                        if use_small_chunks:
                            print("Reading file in small chunks")
                            chunk_size = 1024 * 1024  # 1MB chunks
                            while True:
                                chunk = file.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                        else:
                            # Standard read for normal mode
                            f.write(file.read())
                        
                        # Reset file pointer if it's a file-like object
                        if hasattr(file, 'seek'):
                            file.seek(0)
                    else:
                        raise TypeError("File object doesn't have 'read' method")
                
                # Try to get filename if available
                filename = getattr(file, 'name', 'unknown_file') if hasattr(file, 'name') else 'unknown_file'
            
            print(f"File saved successfully. Size: {os.path.getsize(original_path)} bytes")
            
            # Get file info
            file_extension = os.path.splitext(filename)[1].lower() if hasattr(file, 'filename') else ''
            print(f"File extension: {file_extension}")
            
            # Convert to WAV if needed
            wav_path = f"{temp_path}.wav"
            temp_files.append(wav_path)  # Track for cleanup
            print(f"Converting to WAV format at {wav_path}")
            
            # Use pydub to convert the file
            try:
                # If using small chunks mode and file is large, use a more memory-efficient approach
                file_size = os.path.getsize(original_path)
                if use_small_chunks and file_size > 10 * 1024 * 1024:  # 10MB
                    print(f"Large file detected ({file_size} bytes), using optimized conversion")
                    # Force garbage collection before loading audio
                    import gc
                    gc.collect()
                
                audio = AudioSegment.from_file(original_path)
                print(f"Audio loaded successfully. Duration: {len(audio)/1000}s, Channels: {audio.channels}, Sample width: {audio.sample_width}, Frame rate: {audio.frame_rate}")
                
                # Export with appropriate settings
                if use_small_chunks:
                    # Use lower quality settings for large files in small chunks mode
                    audio.export(wav_path, format="wav", parameters=["-ac", "1"])  # Convert to mono
                else:
                    audio.export(wav_path, format="wav")
                    
                print(f"Audio converted successfully to WAV. Size: {os.path.getsize(wav_path)} bytes")
            except Exception as e:
                print(f"Error during audio conversion: {str(e)}")
                # If conversion fails with standard approach, try a more conservative approach
                if use_small_chunks:
                    print("Trying alternative conversion method")
                    import subprocess
                    try:
                        # Try using ffmpeg directly with conservative settings
                        subprocess.run(["ffmpeg", "-i", original_path, "-ac", "1", "-ar", "22050", wav_path], 
                                      check=True, capture_output=True)
                        print(f"Alternative conversion successful. Size: {os.path.getsize(wav_path)} bytes")
                    except Exception as ffmpeg_error:
                        print(f"Alternative conversion failed: {str(ffmpeg_error)}")
                        raise e  # Re-raise the original error if the alternative also fails
                else:
                    raise  # Re-raise the exception if not in small chunks mode
            
            # Remove original_path from temp_files list since we're returning wav_path
            # and don't want to delete it in the finally block
            if wav_path in temp_files:
                temp_files.remove(wav_path)
                
            return wav_path
        except Exception as e:
            print(f"Error in convert_audio_to_wav: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Error converting audio: {str(e)}")
        finally:
            # Clean up all temporary files except the returned WAV file
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"Cleaned up temporary file: {temp_file}")
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file {temp_file}: {str(e)}")


def extract_features(audio_path, use_small_chunks=False):
    """
    Extract mel spectrogram features from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        use_small_chunks: Whether to process the file in smaller chunks for large files
        
    Returns:
        numpy.ndarray: Mel spectrogram features
    """
    try:
        print(f"Starting feature extraction from {audio_path}")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")
            
        file_size = os.path.getsize(audio_path)
        print(f"Audio file exists. Size: {file_size} bytes")
        
        # If using small chunks mode and file is large, optimize memory usage
        if use_small_chunks and file_size > 10 * 1024 * 1024:  # 10MB
            print("Large file detected, using memory-optimized feature extraction")
            # Force garbage collection before loading audio
            import gc
            gc.collect()
            
            # Use a lower sample rate for very large files to reduce memory usage
            effective_sample_rate = 22050 if file_size > 20 * 1024 * 1024 else SAMPLE_RATE
            print(f"Using sample rate: {effective_sample_rate} Hz for large file")
            
            # Load audio file with resampling if needed, using a shorter duration for very large files
            effective_duration = min(DURATION, 5.0) if file_size > 30 * 1024 * 1024 else DURATION
            print(f"Loading audio with librosa. Sample rate: {effective_sample_rate}, Duration: {effective_duration}s")
            y, sr = librosa.load(audio_path, sr=effective_sample_rate, duration=effective_duration, res_type='kaiser_fast')
        else:
            # Standard loading for normal files
            print(f"Loading audio with librosa. Sample rate: {SAMPLE_RATE}, Duration: {DURATION}s")
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Handle scalar values (including numpy scalar types)
        if np.isscalar(y) or (hasattr(y, 'ndim') and y.ndim == 0):
            print(f"Converting scalar value {y} (type: {type(y)}) to numpy array")
            y = np.atleast_1d(y).astype(np.float32)
            
        print(f"Audio loaded successfully. Length: {len(y)} samples, Sample rate: {sr}Hz")
        
        # Pad or trim the audio to the expected duration
        expected_length = int(DURATION * sr)
        print(f"Expected length: {expected_length} samples, Actual length: {len(y)} samples")
        
        if len(y) < expected_length:
            print(f"Audio is shorter than expected. Padding with zeros.")
            y = np.pad(y, (0, expected_length - len(y)), 'constant')
        else:
            print(f"Audio is longer than expected. Trimming to {expected_length} samples.")
            y = y[:expected_length]
        
        # Extract mel spectrogram with optimized parameters for small chunks mode
        if use_small_chunks and file_size > 10 * 1024 * 1024:
            # Use smaller n_fft and hop_length for large files to reduce memory usage
            effective_n_fft = 1024  # Smaller window size
            effective_hop_length = 512  # Smaller hop length
            print(f"Using optimized parameters for large file. N_FFT: {effective_n_fft}, HOP_LENGTH: {effective_hop_length}")
            
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=effective_n_fft, 
                hop_length=effective_hop_length, 
                n_mels=N_MELS
            )
        else:
            # Standard parameters for normal files
            print(f"Extracting mel spectrogram. N_FFT: {N_FFT}, HOP_LENGTH: {HOP_LENGTH}, N_MELS: {N_MELS}")
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                n_mels=N_MELS
            )
            
        print(f"Mel spectrogram extracted. Shape: {mel_spectrogram.shape}")
        
        # Free memory after spectrogram extraction if in small chunks mode
        if use_small_chunks:
            del y
            import gc
            gc.collect()
        
        # Convert to decibels
        print("Converting to decibels")
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        print(f"Converted to decibels. Min: {mel_spectrogram_db.min()}, Max: {mel_spectrogram_db.max()}")
        
        # Free memory after conversion if in small chunks mode
        if use_small_chunks:
            del mel_spectrogram
            import gc
            gc.collect()
        
        # Normalize
        print("Normalizing spectrogram")
        mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
        print(f"Normalized. Shape: {mel_spectrogram_normalized.shape}, Min: {mel_spectrogram_normalized.min()}, Max: {mel_spectrogram_normalized.max()}")
        
        # Free memory after normalization if in small chunks mode
        if use_small_chunks:
            del mel_spectrogram_db
            import gc
            gc.collect()
        
        # Reshape for the model (the model expects shape=(None, 128, 173, 1))
        print("Reshaping for model input")
        
        # Ensure we have the correct number of mel bands (frequency bins)
        if mel_spectrogram_normalized.shape[0] != N_MELS:
            print(f"Warning: Number of mel bands is {mel_spectrogram_normalized.shape[0]}, expected {N_MELS}")
            if mel_spectrogram_normalized.shape[1] == N_MELS:
                # Dimensions are swapped, transpose them
                print("Transposing dimensions to correct mel bands")
                mel_spectrogram_normalized = mel_spectrogram_normalized.T
            else:
                # Resize to correct number of mel bands
                print(f"Resizing to {N_MELS} mel bands")
                from scipy.ndimage import zoom
                zoom_factor = N_MELS / mel_spectrogram_normalized.shape[0]
                mel_spectrogram_normalized = zoom(mel_spectrogram_normalized, (zoom_factor, 1), order=1)
        
        # Ensure we have the correct number of time steps (173)
        expected_time_steps = 173
        if mel_spectrogram_normalized.shape[1] != expected_time_steps:
            print(f"Adjusting time steps from {mel_spectrogram_normalized.shape[1]} to {expected_time_steps}")
            if mel_spectrogram_normalized.shape[1] < expected_time_steps:
                # Pad if too short
                pad_width = ((0, 0), (0, expected_time_steps - mel_spectrogram_normalized.shape[1]))
                mel_spectrogram_normalized = np.pad(mel_spectrogram_normalized, pad_width, mode='constant')
            else:
                # Trim if too long
                mel_spectrogram_normalized = mel_spectrogram_normalized[:, :expected_time_steps]
        
        # Add channel dimension for the model
        features = np.expand_dims(mel_spectrogram_normalized, axis=-1)  # Shape: (n_mels, time_steps, 1)
        print(f"Final features shape: {features.shape}")
        
        # Final verification of shape
        if features.shape[0] != N_MELS or features.shape[1] != expected_time_steps:
            print(f"Warning: Features shape {features.shape} doesn't match expected ({N_MELS}, {expected_time_steps}, 1)")
            print("Forcing correct dimensions")
            # Create a new array with the correct dimensions
            correct_features = np.zeros((N_MELS, expected_time_steps, 1), dtype=np.float32)
            # Copy as much data as possible
            src_mel = min(features.shape[0], N_MELS)
            src_time = min(features.shape[1], expected_time_steps)
            correct_features[:src_mel, :src_time, 0] = features[:src_mel, :src_time, 0]
            features = correct_features
            print(f"Corrected features shape: {features.shape}")
        
        # Ensure no NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            print("Warning: Features contain NaN or Inf values. Replacing with zeros.")
            features = np.nan_to_num(features)
        
        return features
    
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise Exception(f"Error extracting features: {str(e)}")
    finally:
        # Clean up the temporary file only if it's not being used elsewhere
        # We'll let the calling function handle cleanup to avoid file in use errors
        pass


def process_audio_file(file, use_small_chunks=False):
    """
    Process an uploaded audio file: convert to WAV and extract features.
    
    Args:
        file: Flask file object from request or file-like object
        use_small_chunks: Whether to process the file in smaller chunks for large files
        
    Returns:
        numpy.ndarray: Processed audio features ready for prediction
    """
    wav_path = None
    original_file = file  # Keep a reference to the original file
    file_info = {}
    
    try:
        # Validate input file
        if file is None:
            raise ValueError("Input file cannot be None")
        
        # Handle different types of file objects and collect diagnostic information
        if hasattr(file, 'filename') and hasattr(file, 'content_type'):
            # This is a Flask FileStorage object
            file_info['type'] = 'Flask FileStorage'
            file_info['filename'] = getattr(file, 'filename', 'unknown')
            file_info['content_type'] = getattr(file, 'content_type', 'unknown')
            
            if hasattr(file, 'content_length'):
                file_info['size'] = getattr(file, 'content_length', 0)
                print(f"Processing audio file: {file_info['filename']}, Content type: {file_info['content_type']}, Size: {file_info['size']} bytes")
            else:
                print(f"Processing audio file: {file_info['filename']}, Content type: {file_info['content_type']}")
        elif hasattr(file, 'name'):
            # This is a regular file object
            file_info['type'] = 'File object'
            file_info['filename'] = os.path.basename(file.name)
            file_info['size'] = os.path.getsize(file.name)
            print(f"Processing audio file: {file_info['filename']}, Size: {file_info['size']} bytes")
        else:
            # This is some other file-like object
            file_info['type'] = 'File-like object'
            print("Processing audio from file-like object")
        
        # Reset file pointer if it's a file-like object
        if hasattr(file, 'seek'):
            try:
                file.seek(0)
                print("Reset file pointer to beginning")
            except Exception as e:
                print(f"Warning: Could not reset file pointer: {str(e)}")
                file_info['seek_error'] = str(e)
                
                # If we can't reset the file pointer, try to create a new file-like object
                if hasattr(file, 'read'):
                    try:
                        from io import BytesIO
                        content = file.read()
                        file_info['content_length'] = len(content)
                        file = BytesIO(content)
                        if hasattr(original_file, 'filename'):
                            file.filename = original_file.filename
                        if hasattr(original_file, 'content_type'):
                            file.content_type = original_file.content_type
                        print(f"Created new file-like object from content ({file_info['content_length']} bytes)")
                    except Exception as inner_e:
                        error_msg = f"Error creating new file-like object: {str(inner_e)}"
                        print(error_msg)
                        file_info['read_error'] = error_msg
        
        # Convert to WAV format if needed
        print("Converting to WAV format...")
        
        # If using small chunks mode, set a memory limit for processing
        if use_small_chunks:
            print("Using small chunks mode for audio processing")
            # Set a smaller buffer size for audio processing
            import gc
            # Force garbage collection before processing
            gc.collect()
            file_info['small_chunks'] = True
        else:
            file_info['small_chunks'] = False
            
        try:
            wav_path = convert_audio_to_wav(file, use_small_chunks)
            file_info['wav_path'] = wav_path
            print(f"Conversion to WAV complete. Path: {wav_path}")
            
            # Verify the WAV file exists and has content
            if not os.path.exists(wav_path):
                raise Exception(f"WAV file not found at {wav_path}")
            
            file_size = os.path.getsize(wav_path)
            file_info['wav_size'] = file_size
            if file_size == 0:
                raise Exception(f"WAV file is empty: {wav_path}")
            
            print(f"WAV file verified: {wav_path}, Size: {file_size} bytes")
        except Exception as wav_error:
            error_msg = f"Error during WAV conversion: {str(wav_error)}"
            print(error_msg)
            file_info['wav_error'] = error_msg
            raise Exception(error_msg)
        
        # Extract features
        print("Extracting features...")
        
        # If file is very large and small chunks mode is enabled, use a different approach
        if use_small_chunks and file_size > 10 * 1024 * 1024:  # 10MB
            print("Large file detected, using optimized processing")
            file_info['large_file_optimization'] = True
            # Force garbage collection before feature extraction
            import gc
            gc.collect()
        else:
            file_info['large_file_optimization'] = False
        
        try:
            features = extract_features(wav_path, use_small_chunks)
            if features is not None:
                file_info['features_shape'] = features.shape
                print(f"Feature extraction complete. Features shape: {features.shape}")
            else:
                raise Exception("Feature extraction returned None")
            
            # Verify features are valid
            if features.size == 0:
                raise Exception("Feature extraction produced empty features")
            
            # Ensure features are in the correct format
            if not isinstance(features, np.ndarray):
                print(f"Warning: Features are not a numpy array. Converting from {type(features)}")
                features = np.array(features)
                file_info['features_converted'] = True
            
            # Check for NaN or Inf values
            has_nan = np.isnan(features).any()
            has_inf = np.isinf(features).any()
            if has_nan or has_inf:
                print("Warning: Features contain NaN or Inf values. Replacing with finite values.")
                file_info['features_had_nan'] = has_nan
                file_info['features_had_inf'] = has_inf
                features = np.nan_to_num(features)
            
            # Verify feature ranges
            feature_min = np.min(features)
            feature_max = np.max(features)
            file_info['features_min'] = float(feature_min)
            file_info['features_max'] = float(feature_max)
            
            # Ensure features are normalized between 0 and 1
            if feature_min < 0 or feature_max > 1:
                print(f"Features not properly normalized. Min: {feature_min}, Max: {feature_max}. Normalizing...")
                features = (features - feature_min) / (feature_max - feature_min + 1e-10)  # Add small epsilon to avoid division by zero
                file_info['features_normalized'] = True
            
            return features
        except Exception as feature_error:
            error_msg = f"Error during feature extraction: {str(feature_error)}"
            print(error_msg)
            file_info['feature_error'] = error_msg
            import traceback
            file_info['feature_traceback'] = traceback.format_exc()
            raise Exception(error_msg)
    except Exception as e:
        print(f"Error in process_audio_file: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        
        # Create a diagnostic error message with all collected information
        error_details = f"Error processing audio file: {str(e)}\n"
        error_details += f"File info: {str(file_info)}\n"
        error_details += f"Traceback: {error_trace}"
        
        raise Exception(error_details)
    finally:
        # Clean up temporary files
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                print(f"Cleaned up WAV file at {wav_path}")
            except Exception as e:
                print(f"Warning: Could not remove WAV file: {str(e)}")
                # If we can't remove it now, schedule it for removal when the program exits
                import atexit
                atexit.register(lambda path=wav_path: os.remove(path) if os.path.exists(path) else None)