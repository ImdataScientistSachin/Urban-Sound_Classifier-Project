# Urban Sound Classifier - Technical Flowchart

## Data Processing and Model Architecture

```mermaid
flowchart TD
    %% Audio Processing Pipeline
    subgraph "Audio Processing Pipeline"
        A1[Raw Audio Input] --> A2[File Format Check]
        A2 --> A3[Convert to WAV Format]
        A3 --> A4[Resample to 22050Hz]
        A4 --> A5[Duration Check]
        
        A5 -->|Too Short| A6[Pad with Zeros]
        A5 -->|Too Long| A7[Trim to 4s]
        A5 -->|Correct Length| A8[Continue]
        
        A6 --> A9[Extract Mel Spectrogram]
        A7 --> A9
        A8 --> A9
        
        A9 --> A10[Convert to dB Scale]
        A10 --> A11[Normalize Features]
        A11 --> A12[Reshape: Time x Frequency x Channel]
    end
    
    %% Model Architecture
    subgraph "Hybrid U-Net Architecture"
        B1[Input Layer 128x173x1] --> B2[Conv2D + ReLU]
        B2 --> B3[MaxPooling]
        B3 --> B4[Conv2D + ReLU]
        B4 --> B5[MaxPooling]
        
        %% Bottleneck
        B5 --> B6[Conv2D + ReLU Bottleneck]
        
        %% Upsampling path with skip connections
        B6 --> B7[UpSampling2D]
        B4 -->|Skip Connection| B7
        B7 --> B8[Conv2D + ReLU]
        
        B8 --> B9[UpSampling2D]
        B2 -->|Skip Connection| B9
        B9 --> B10[Conv2D + ReLU]
        
        %% Classification head
        B10 --> B11[Global Average Pooling]
        B11 --> B12[Dense Layer]
        B12 --> B13[Dropout]
        B13 --> B14[Output Layer - 10 Classes]
    end
    
    %% Prediction Flow
    subgraph "Prediction Process"
        C1[Processed Features] --> C2[Model Forward Pass]
        C2 --> C3[Softmax Probabilities]
        C3 --> C4[Argmax - Class Index]
        C4 --> C5[Map to Class Label]
        C5 --> C6[Return Prediction + Confidence]
    end
    
    %% Connect the subgraphs
    A12 --> C1
    B14 --> C2
```

## API Endpoints and Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Flask_Server
    participant Audio_Processor
    participant Model
    
    User->>Frontend: Upload Audio File / Record Audio
    Frontend->>Flask_Server: POST /predict with audio file
    Flask_Server->>Audio_Processor: process_audio_file(file)
    Audio_Processor->>Audio_Processor: convert_audio_to_wav(file)
    Audio_Processor->>Audio_Processor: extract_features(wav_path)
    Audio_Processor->>Flask_Server: Return processed features
    Flask_Server->>Model: predict(model, features)
    Model->>Flask_Server: Return (class_name, confidence)
    Flask_Server->>Frontend: JSON response with prediction
    Frontend->>User: Display results and confidence
```

## System Components Interaction

```mermaid
flowchart LR
    %% Main components
    A[config.py] --> B[app.py]
    C[model.py] --> B
    D[utils.py] --> B
    
    %% Config details
    subgraph "Configuration"
        A1[Paths] --> A
        A2[Audio Parameters] --> A
        A3[Class Labels] --> A
        A4[API Settings] --> A
    end
    
    %% Model details
    subgraph "Model Functions"
        C1[load_model] --> C
        C2[predict] --> C
        C3[get_model_summary] --> C
    end
    
    %% Utils details
    subgraph "Utility Functions"
        D1[convert_audio_to_wav] --> D
        D2[extract_features] --> D
        D3[process_audio_file] --> D
    end
    
    %% App details
    subgraph "Flask Application"
        B1[index Route] --> B
        B2[predict Route] --> B
        B3[classes Route] --> B
    end
    
    %% External components
    E[index.html] --> B1
    F[static assets] --> E
    G[Pre-trained Model] --> C1
```

## Cross-Validation and Model Training

```mermaid
flowchart TD
    %% Dataset preparation
    A[UrbanSound8K Dataset] --> B[10 Folds Cross-Validation]
    
    %% Training process
    B --> C1[Train on Folds 2-10, Test on Fold 1]
    B --> C2[Train on Folds 1,3-10, Test on Fold 2]
    B --> C3[Train on Folds 1-2,4-10, Test on Fold 3]
    B --> C4[Train on Folds 1-3,5-10, Test on Fold 4]
    B --> C5[Train on Folds 1-4,6-10, Test on Fold 5]
    
    %% Model selection
    C1 --> D1[Model Fold 1]
    C2 --> D2[Model Fold 2]
    C3 --> D3[Model Fold 3]
    C4 --> D4[Model Fold 4]
    C5 --> D5[Model Fold 5]
    
    %% Ensemble or best model selection
    D1 --> E[Select Best Model]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    
    E --> F[Final Model]
    F --> G[Deploy to Production]
```

## Feature Extraction Detail

```mermaid
flowchart TD
    %% Audio input
    A[Audio Signal] --> B[Short-Time Fourier Transform]
    B --> C[Power Spectrogram]
    
    %% Mel spectrogram generation
    C --> D[Apply Mel Filterbank]
    D --> E[Mel Spectrogram]
    E --> F[Convert to dB Scale]
    
    %% Feature normalization
    F --> G[Min-Max Normalization]
    G --> H[Reshape for Model Input]
    
    %% Parameters
    I[Parameters] --> B
    I --> D
    
    %% Parameter details
    subgraph "Audio Parameters"
        I1[Sample Rate: 22050Hz] --> I
        I2[Duration: 4.0s] --> I
        I3[N_FFT: 2048] --> I
        I4[Hop Length: 512] --> I
        I5[N_Mels: 128] --> I
    end
```

These flowcharts provide a comprehensive technical overview of the Urban Sound Classifier system, illustrating the data processing pipeline, model architecture, API interactions, cross-validation approach, and feature extraction details. The diagrams help visualize the complex relationships between different components and the flow of data through the system.