# Urban Sound Classifier - Technical Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Project Architecture](#project-architecture)
3. [Audio Processing Pipeline](#audio-processing-pipeline)
4. [Model Architecture](#model-architecture)
5. [Feature Extraction](#feature-extraction)
6. [Training Process](#training-process)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Web Application](#web-application)
9. [Real-time Classification](#real-time-classification)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

## Introduction

The Urban Sound Classifier is a deep learning application designed to identify and classify urban environmental sounds. This document provides a comprehensive technical overview of the system's architecture, components, and implementation details.

## Project Architecture

The project follows a modular architecture with clear separation of concerns:

```
Urban-Sound_Classifier-Project/
├── src/                           # Core source code
│   ├── app.py                     # Flask web application
│   ├── config.py                  # Configuration parameters
│   ├── model.py                   # Model loading and prediction
│   ├── utils.py                   # Audio processing utilities
│   ├── test_model.py              # Model testing functions
│   └── templates/                 # HTML templates
├── models/                        # Trained model storage
├── evaluation_results/            # Model evaluation outputs
├── model_visualization/           # Model architecture visualizations
├── test_audio_samples/            # Generated test audio files
└── recorded_audio/                # Real-time recorded audio clips
```

### Key Components

1. **Audio Processing Module**: Handles audio file conversion, normalization, and feature extraction
2. **Model Module**: Manages model loading, prediction, and evaluation
3. **Web Application**: Provides a user interface for audio classification
4. **Real-time Classifier**: Enables live audio classification from microphone input
5. **Visualization Tools**: Generate model architecture and activation visualizations
6. **Evaluation Tools**: Assess model performance with various metrics

## Audio Processing Pipeline

The audio processing pipeline consists of several stages:

1. **Audio Loading**: Supports multiple formats (WAV, MP3, OGG, FLAC, M4A)
2. **Format Conversion**: Converts all audio to WAV format using pydub
3. **Resampling**: Standardizes to 22050Hz sample rate using librosa
4. **Duration Normalization**: Adjusts audio to a fixed 4-second duration
   - Shorter audio is padded with silence
   - Longer audio is trimmed to the first 4 seconds
5. **Feature Extraction**: Converts audio to mel-spectrogram representation
6. **Normalization**: Scales features for optimal model performance

### Code Implementation

The audio processing pipeline is implemented in `utils.py` with the following key functions:

- `convert_audio_to_wav`: Converts various audio formats to WAV
- `extract_features`: Extracts mel-spectrogram features from WAV files
- `process_audio_file`: Orchestrates the complete processing pipeline

## Model Architecture

The Urban Sound Classifier uses a Convolutional Neural Network (CNN) architecture optimized for audio classification tasks. The model processes mel-spectrograms as 2D images, extracting hierarchical features that capture both temporal and frequency patterns.

### Network Structure

The model architecture consists of:

1. **Input Layer**: Accepts mel-spectrograms of shape (128, 173, 1)
2. **Convolutional Blocks**: Multiple blocks of:
   - 2D Convolutional layers with ReLU activation
   - Batch normalization
   - Max pooling
3. **Global Average Pooling**: Reduces spatial dimensions
4. **Dense Layers**: Fully connected layers with dropout for regularization
5. **Output Layer**: Softmax activation for 10-class classification

### Model Parameters

- **Input Shape**: (128, 173, 1) - representing (mel_bands, time_steps, channels)
- **Number of Classes**: 10 (corresponding to the urban sound categories)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall

## Feature Extraction

The feature extraction process converts raw audio waveforms into mel-spectrogram representations, which are more suitable for neural network processing.

### Mel-Spectrogram Generation

Mel-spectrograms are generated using the following parameters:

- **Sample Rate**: 22050 Hz
- **FFT Window Size**: 2048 samples
- **Hop Length**: 512 samples (75% overlap)
- **Number of Mel Bands**: 128

### Feature Normalization

The mel-spectrograms undergo several transformations:

1. **Conversion to Decibels**: Using librosa's `power_to_db` function
2. **Normalization**: Scaling to the range [0, 1]
3. **Reshaping**: Adding a channel dimension for CNN input

### Code Implementation

The feature extraction process is implemented in the `extract_features` function in `utils.py`:

```python
def extract_features(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Load audio file
    y, sr = librosa.load(file_path, sr=sr)
    
    # Ensure consistent duration
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    else:
        y = y[:target_length]
    
    # Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Normalize
    mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
    
    # Reshape for model input
    features = mel_spectrogram_normalized.reshape((n_mels, -1, 1))
    
    return features
```

## Training Process

The model was trained on the UrbanSound8K dataset, which contains 8,732 labeled sound excerpts of urban sounds from 10 classes.

### Dataset

- **Source**: UrbanSound8K dataset
- **Size**: 8,732 audio clips
- **Classes**: 10 urban sound categories
- **Duration**: Each clip is ≤ 4 seconds
- **Format**: WAV files

### Training Strategy

1. **Data Augmentation**: Applied to increase dataset diversity
   - Time stretching
   - Pitch shifting
   - Adding background noise
   - Random time masking

2. **Cross-Validation**: 10-fold cross-validation using the predefined folds in UrbanSound8K

3. **Hyperparameter Tuning**: Optimized for:
   - Learning rate
   - Batch size
   - Network depth
   - Regularization strength

4. **Early Stopping**: To prevent overfitting

## Evaluation Metrics

The model is evaluated using several metrics to assess its performance:

1. **Accuracy**: Overall classification accuracy
2. **Precision**: Measure of exactness (TP / (TP + FP))
3. **Recall**: Measure of completeness (TP / (TP + FN))
4. **F1 Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Visualizes class-wise performance
6. **Classification Report**: Detailed per-class metrics

### Evaluation Code

The model evaluation is implemented in `evaluate_model.py`, which:

1. Loads the trained model
2. Processes test data
3. Computes evaluation metrics
4. Generates visualizations
5. Saves results to the `evaluation_results` directory

## Web Application

The web application provides a user-friendly interface for audio classification.

### Backend

- **Framework**: Flask
- **API Endpoints**:
  - `/`: Main page
  - `/predict`: Audio file classification
  - `/classes`: List available classes

### Frontend

- **Technologies**: HTML5, CSS3, JavaScript
- **Features**:
  - File upload with drag-and-drop
  - Audio playback
  - Results visualization
  - Confidence scores display

### Deployment

The web application can be deployed locally or on a server:

```bash
python src/app.py
```

This starts the Flask development server on http://localhost:5000.

## Real-time Classification

The real-time classification module enables live audio processing from a microphone input.

### Implementation

The real-time classifier is implemented in `realtime_classifier.py` using:

- **PyAudio**: For microphone input capture
- **Threading**: For concurrent audio recording and processing
- **Queue**: For thread-safe data exchange

### Processing Strategy

1. **Windowing**: Processes audio in 4-second windows
2. **Overlapping**: Uses 50% overlap between consecutive windows
3. **Buffering**: Maintains a buffer of recent audio chunks
4. **Continuous Classification**: Classifies each window as it becomes available

### Usage

To start real-time classification:

```bash
python realtime_classifier.py
```

## Troubleshooting

### Common Issues

1. **Audio Format Errors**:
   - Ensure audio files are in supported formats (WAV, MP3, OGG, FLAC, M4A)
   - Check that required audio libraries are installed

2. **Model Loading Errors**:
   - Verify model file exists at the specified path
   - Ensure TensorFlow version compatibility

3. **Feature Extraction Issues**:
   - Check librosa installation
   - Verify audio file integrity

4. **Real-time Classification Problems**:
   - Ensure microphone access is granted
   - Check PyAudio installation

### Debugging

For detailed debugging, enable debug mode in `config.py`:

```python
DEBUG_MODE = True
```

This increases logging verbosity throughout the application.

## References

1. J. Salamon, C. Jacoby, and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

2. McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference (Vol. 8).

3. Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural networks for music classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 2392-2396). IEEE.

4. Hershey, S., Chaudhuri, S., Ellis, D. P., Gemmeke, J. F., Jansen, A., Moore, R. C., ... & Wilson, K. (2017). CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 131-135). IEEE.