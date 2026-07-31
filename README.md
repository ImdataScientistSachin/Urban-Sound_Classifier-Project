# Urban Sound Classifier

![Urban Sound Classifier](https://img.shields.io/badge/AI-Urban%20Sound%20Classification-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-96.63%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A state-of-the-art deep learning application that classifies urban environmental sounds using a hybrid U-Net architecture. This project leverages the UrbanSound8K dataset to identify common urban sounds with exceptional accuracy.

## 🔊 Project Overview

The Urban Sound Classifier is an advanced audio classification system designed to identify and categorize urban environmental sounds. Using deep learning techniques and a hybrid U-Net architecture, the system achieves remarkable accuracy in distinguishing between different urban sound categories.

### Key Achievements

- **High Accuracy**: 96.63% classification accuracy on test data
- **Real-time Processing**: Supports both file uploads and real-time microphone recording
- **Robust Feature Extraction**: Advanced mel-spectrogram feature extraction pipeline
- **User-friendly Interface**: Modern, responsive web interface with real-time feedback
- **Versatile Audio Support**: Handles multiple audio formats (WAV, MP3, OGG, FLAC, M4A)

## 🎯 Features

### Audio Classification

- **Multi-class Classification**: Identifies 10 distinct urban sound categories
- **Confidence Scoring**: Provides confidence metrics for each prediction
- **Spectrogram Analysis**: Converts audio to mel-spectrograms for neural network processing

### User Interface

- **Drag-and-Drop**: Easy file upload with drag-and-drop functionality
- **Real-time Microphone Recording**: Record and analyze sounds directly from your microphone
- **Audio Visualization**: Visual feedback during recording with waveform display
- **Responsive Design**: Works seamlessly across desktop and mobile devices

### Technical Features

- **Hybrid U-Net Architecture**: Combines feature extraction capabilities of CNNs with context-preserving properties of U-Net
- **Audio Preprocessing Pipeline**: Automatic conversion, normalization, and feature extraction
- **RESTful API**: Endpoints for classification and retrieving available classes
- **Modular Design**: Well-structured codebase with separation of concerns

## 🧠 Model Architecture

The classifier uses a sophisticated hybrid U-Net architecture that combines:

1. **Feature Extraction**: Convolutional layers extract hierarchical features from mel-spectrograms
2. **Context Preservation**: U-Net's skip connections maintain spatial context information
3. **Multi-scale Analysis**: Captures both fine-grained details and broader patterns in audio signals

## 🚀 Using Pre-trained Models

This project includes pre-trained models that can be used directly for urban sound classification. The models are located in the `models/unetropolis-hybrid-unet-urbansound_96.63_weights` directory.

### Available Models

- UNet models for each fold (best_model_fold1_UNet.h5, best_model_fold2_UNet.h5, etc.)
- SimpleCNN models for each fold (best_model_fold1_SimpleCNN.h5, etc.)

These models are used together in an ensemble to achieve higher accuracy.

### Quick Start

#### 1. Using the main script

Run the main script to load the models and see information about them:

```bash
python double_unet_audio_classifier.py
```

This will load all available models and display information about them. If the UrbanSound8K dataset is available, it will also evaluate the models on a sample of the dataset.

#### 2. Using the prediction script

To classify a single audio file, use the prediction script:

```bash
python predict_sample.py
```

This script will prompt you to enter the path to an audio file, and then it will classify the sound using the pre-trained models.

### Using the Models in Your Own Code

To use the pre-trained models in your own code, follow these steps:

```python
from double_unet_audio_classifier import load_pretrained_models, predict_with_models

# Load models
models = load_pretrained_models('path/to/models', model_types=['UNet', 'SimpleCNN'])

# Process your audio features
# features = ...

# Make predictions
ensemble_preds, ensemble_classes = predict_with_models(models, features)
```

This architecture significantly outperforms traditional CNN models for audio classification tasks, achieving 96.63% accuracy on the UrbanSound8K dataset.

## 📊 Dataset

This project uses the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), which contains 8,732 labeled sound excerpts (≤4s) of urban sounds from 10 classes:

1. 🌬️ Air conditioner
2. 🚗 Car horn
3. 👶 Children playing
4. 🐕 Dog bark
5. 🔨 Drilling
6. 🚘 Engine idling
7. 🔫 Gun shot
8. 🏗️ Jackhammer
9. 🚨 Siren
10. 🎵 Street music

### Dataset Setup

The dataset should be placed in the following directory structure:

```
D:\Urban-Sound-Classifier-Project\data\UrbanSound8K\
├── UrbanSound8K.csv
├── fold1\
├── fold2\
...
└── fold10\
```

The code is configured to look for the dataset in this location. If you have the dataset in a different location, you can modify the path in the `double_unet_audio_classifier.py` file.

## 🚀 Installation

1. Clone the repository

```bash

git clone https://github.com/ImdataScientistSachin/Urban-Sound-Classifier-Project.git
cd Urban-Sound-Classifier-Project
```

2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Web Application

```bash
python src/app.py
```

Then open your browser and navigate to `http://localhost:5000`

### Using the Interface

1. **File Upload**: Click "Select Audio File" or drag and drop an audio file
2. **Microphone Recording**: Click "Use Microphone" to record a sound for classification
3. **Analysis**: Click "Analyze Sound" to process the audio and view results

### API Endpoints

- `POST /predict` - Upload an audio file for classification
- `GET /classes` - Get a list of all sound classes the model can predict

## 🔧 Technical Implementation

### Audio Processing Pipeline

1. **Audio Conversion**: Converts various audio formats to WAV
2. **Resampling**: Standardizes to 22050Hz sample rate
3. **Duration Normalization**: Adjusts to 4-second segments
4. **Feature Extraction**: Generates mel-spectrograms with 128 mel bands
5. **Normalization**: Scales features for optimal model performance

### Web Application

- **Backend**: Flask-based RESTful API
- **Frontend**: Modern HTML5/CSS3/JavaScript interface
- **Real-time Processing**: Asynchronous audio handling with Web Audio API

## 🔍 Advanced Features

### Real-time Classification

The project now includes real-time audio classification capabilities:

```bash
python realtime_classifier.py
```

This will start recording from your microphone and classify sounds in real-time. Press Ctrl+C to stop.

### Model Visualization

Generate visualizations of the model architecture and layer activations:

```bash
python visualize_model.py
```

The visualizations will be saved in the `model_visualization` directory.

### Model Evaluation

Run comprehensive evaluation of the model:

```bash
python evaluate_model.py
```

This will generate evaluation metrics, confusion matrix, and classification report in the `evaluation_results` directory.

## 🔍 Future Enhancements

- **Model Fine-tuning**: Further optimization for specific urban environments
- **Continuous Learning**: Implementation of feedback mechanisms for model improvement
- **Multi-label Classification**: Detection of overlapping sound categories
- **Mobile Application**: Native mobile apps for iOS and Android
- **Edge Deployment**: Optimization for edge devices and IoT applications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgements

- [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) - J. Salamon, C. Jacoby, and J. P. Bello
- [TensorFlow](https://www.tensorflow.org/) - For the deep learning framework
- [Flask](https://flask.palletsprojects.com/) - For the web application framework
- [Librosa](https://librosa.org/) - For audio processing capabilities

## 👨‍💻 Developer

Developed by **Sachin Paunikar**

---

© 2025 Sachin Paunikar. All Rights Reserved.
