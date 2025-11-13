import tensorflow as tf
import numpy as np
import os
import sys
import pyaudio
import wave
import time
import threading
import queue
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.model import load_model, predict
from src.utils import extract_features
from src.config import MODEL_PATH, CLASS_LABELS, SAMPLE_RATE, DURATION

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = SAMPLE_RATE
RECORD_SECONDS = DURATION
OVERLAP_SECONDS = 2  # 50% overlap for 4-second windows

class AudioClassifier:
    def __init__(self, model_path=MODEL_PATH):
        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        
        # Compile the model with metrics
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Create a queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Create a queue for results
        self.results_queue = queue.Queue()
        
        # Flag to control recording
        self.is_recording = False
        
        # Create output directory for saving audio clips
        self.output_dir = 'recorded_audio'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_recording(self):
        """
        Start recording audio from the microphone.
        """
        self.is_recording = True
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Recording started. Press Ctrl+C to stop.")
    
    def stop_recording(self):
        """
        Stop recording audio.
        """
        self.is_recording = False
        
        # Wait for threads to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1)
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1)
        
        print("Recording stopped.")
    
    def _record_audio(self):
        """
        Record audio from the microphone and add chunks to the queue.
        """
        # Open stream
        stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("Recording audio...")
        
        # Record audio in chunks and add to queue
        try:
            while self.is_recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
        except KeyboardInterrupt:
            pass
        finally:
            # Stop and close the stream
            stream.stop_stream()
            stream.close()
    
    def _process_audio(self):
        """
        Process audio chunks from the queue and classify them.
        """
        # Buffer to store audio chunks
        audio_buffer = []
        samples_per_window = int(RATE * RECORD_SECONDS)
        samples_per_overlap = int(RATE * OVERLAP_SECONDS)
        
        # Calculate how many chunks we need for a full window
        chunks_per_window = samples_per_window // CHUNK
        
        print(f"Processing audio (window size: {RECORD_SECONDS}s, overlap: {OVERLAP_SECONDS}s)")
        
        try:
            while self.is_recording or not self.audio_queue.empty():
                # Get audio chunk from queue
                if not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    audio_buffer.append(data)
                    
                    # If we have enough chunks for a window, process it
                    if len(audio_buffer) >= chunks_per_window:
                        # Process the current window
                        self._classify_audio_window(audio_buffer[:chunks_per_window])
                        
                        # Remove chunks that are no longer needed (keeping overlap)
                        chunks_to_remove = chunks_per_window - (samples_per_overlap // CHUNK)
                        audio_buffer = audio_buffer[chunks_to_remove:]
                else:
                    # If queue is empty but we're still recording, wait a bit
                    if self.is_recording:
                        time.sleep(0.1)
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def _classify_audio_window(self, audio_chunks):
        """
        Classify an audio window.
        
        Args:
            audio_chunks: List of audio chunks forming a window
        """
        try:
            # Save audio to a temporary WAV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_wav_path = os.path.join(self.output_dir, f"temp_{timestamp}.wav")
            
            # Save as WAV file
            with wave.open(temp_wav_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_chunks))
            
            # Extract features
            features = extract_features(temp_wav_path)
            
            # Make prediction
            prediction, confidence, all_confidences = predict(self.model, features)
            
            # Add result to queue
            self.results_queue.put((prediction, confidence, all_confidences, temp_wav_path))
            
            # Print result
            print(f"\nDetected sound: {prediction} (confidence: {confidence:.2f})")
            
            # Print top 3 predictions
            top_predictions = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top 3 predictions:")
            for class_name, conf in top_predictions:
                print(f"  {class_name}: {conf:.2f}")
        except Exception as e:
            print(f"Error classifying audio window: {e}")
    
    def get_latest_result(self):
        """
        Get the latest classification result.
        
        Returns:
            tuple: (prediction, confidence, all_confidences, audio_path) or None if no results
        """
        if not self.results_queue.empty():
            return self.results_queue.get()
        return None
    
    def close(self):
        """
        Clean up resources.
        """
        self.stop_recording()
        self.p.terminate()
        print("Audio classifier closed.")

def main():
    # Create audio classifier
    classifier = AudioClassifier()
    
    try:
        # Start recording and classifying
        classifier.start_recording()
        
        # Keep the program running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Clean up
        classifier.close()
        print("\nReal-time classification ended.")

if __name__ == "__main__":
    main()