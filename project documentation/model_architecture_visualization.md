# Urban Sound Classifier - Model Architecture Visualization

## Hybrid U-Net Architecture Diagram

```mermaid
graph TD
    %% Input layer
    Input["Input Layer\n(128×173×1)"] --> Conv1
    
    %% Encoding path
    subgraph "Encoding Path"
        Conv1["Conv2D (32 filters)\nKernel: 3×3\nReLU"] --> Pool1
        Pool1["MaxPooling2D\n(2×2)"] --> Conv2
        Conv2["Conv2D (64 filters)\nKernel: 3×3\nReLU"] --> Pool2
        Pool2["MaxPooling2D\n(2×2)"] --> Bottleneck
    end
    
    %% Bottleneck
    subgraph "Bottleneck"
        Bottleneck["Conv2D (128 filters)\nKernel: 3×3\nReLU"] 
    end
    
    %% Decoding path
    subgraph "Decoding Path"
        Bottleneck --> Upsample1
        Upsample1["UpSampling2D\n(2×2)"] --> Concat1
        Conv2 --"Skip Connection"--> Concat1["Concatenate"] 
        Concat1 --> Conv3
        Conv3["Conv2D (64 filters)\nKernel: 3×3\nReLU"] --> Upsample2
        Upsample2["UpSampling2D\n(2×2)"] --> Concat2
        Conv1 --"Skip Connection"--> Concat2["Concatenate"]
        Concat2 --> Conv4
        Conv4["Conv2D (32 filters)\nKernel: 3×3\nReLU"]
    end
    
    %% Classification head
    subgraph "Classification Head"
        Conv4 --> GAP
        GAP["Global Average Pooling"] --> Dense1
        Dense1["Dense Layer\n(128 units)\nReLU"] --> Dropout
        Dropout["Dropout\n(rate=0.5)"] --> Output
        Output["Dense Layer\n(10 units)\nSoftmax"]
    end
    
    %% Output classes
    Output --> Classes
    Classes["10 Urban Sound Classes"]
    
    %% Class labels
    Classes --> Class1["air_conditioner"]
    Classes --> Class2["car_horn"]
    Classes --> Class3["children_playing"]
    Classes --> Class4["dog_bark"]
    Classes --> Class5["drilling"]
    Classes --> Class6["engine_idling"]
    Classes --> Class7["gun_shot"]
    Classes --> Class8["jackhammer"]
    Classes --> Class9["siren"]
    Classes --> Class10["street_music"]
```

## Feature Map Dimensions Through Network

```mermaid
graph LR
    %% Input dimensions
    Input["Input\n128×173×1"] --> Conv1
    
    %% Encoding path dimensions
    Conv1["Conv2D\n128×173×32"] --> Pool1
    Pool1["MaxPool\n64×86×32"] --> Conv2
    Conv2["Conv2D\n64×86×64"] --> Pool2
    Pool2["MaxPool\n32×43×64"] --> Bottleneck
    
    %% Bottleneck dimensions
    Bottleneck["Bottleneck\n32×43×128"] --> Upsample1
    
    %% Decoding path dimensions
    Upsample1["Upsample\n64×86×128"] --> Concat1
    Conv2 --"Skip\n64×86×64"--> Concat1
    Concat1["Concat\n64×86×192"] --> Conv3
    Conv3["Conv2D\n64×86×64"] --> Upsample2
    Upsample2["Upsample\n128×172×64"] --> Concat2
    Conv1 --"Skip\n128×173×32"--> Concat2
    Concat2["Concat\n128×172×96"] --> Conv4
    Conv4["Conv2D\n128×172×32"] --> GAP
    
    %% Classification head dimensions
    GAP["GAP\n32"] --> Dense1
    Dense1["Dense\n128"] --> Dropout
    Dropout["Dropout\n128"] --> Output
    Output["Output\n10"]
```

## Mel Spectrogram Visualization

```mermaid
graph TD
    subgraph "Mel Spectrogram Example"
        A["Time →"] --- B["↑\nFrequency\n↓"]
        
        subgraph "High Energy"
            style C fill:#ff0000,stroke:#333,stroke-width:2px
            C[" "] 
        end
        
        subgraph "Medium Energy"
            style D fill:#ffff00,stroke:#333,stroke-width:2px
            D[" "]
        end
        
        subgraph "Low Energy"
            style E fill:#0000ff,stroke:#333,stroke-width:2px
            E[" "]
        end
    end
    
    subgraph "Class Examples"
        C1["Car Horn"] --- Desc1["Short, high energy\nbursts across\nfrequencies"]
        C2["Air Conditioner"] --- Desc2["Consistent low\nfrequency energy"]
        C3["Siren"] --- Desc3["Oscillating pattern\nacross frequencies"]
        C4["Dog Bark"] --- Desc4["Mid-frequency\nenergy bursts"]
    end
```

## Model Training Process

```mermaid
gantt
    title Model Training Timeline
    dateFormat  YYYY-MM-DD
    section Data Preparation
    Dataset Collection       :done, a1, 2024-01-01, 7d
    Feature Extraction       :done, a2, after a1, 14d
    Data Augmentation        :done, a3, after a2, 7d
    
    section Model Development
    Architecture Design      :done, b1, 2024-01-15, 14d
    Initial Implementation   :done, b2, after b1, 7d
    Hyperparameter Tuning    :done, b3, after b2, 14d
    
    section Training
    Fold 1 Training          :done, c1, 2024-02-15, 5d
    Fold 2 Training          :done, c2, after c1, 5d
    Fold 3 Training          :done, c3, after c2, 5d
    Fold 4 Training          :done, c4, after c3, 5d
    Fold 5 Training          :done, c5, after c4, 5d
    
    section Evaluation
    Model Evaluation         :done, d1, after c5, 7d
    Error Analysis           :done, d2, after d1, 7d
    Final Model Selection    :done, d3, after d2, 3d
    
    section Deployment
    Web App Development      :done, e1, after d3, 14d
    API Implementation       :done, e2, after e1, 7d
    Testing & Optimization   :done, e3, after e2, 7d
    Production Deployment    :done, e4, after e3, 3d
```

## Performance Visualization

```mermaid
pie title Class-wise Accuracy (%)
    "air_conditioner" : 94.2
    "car_horn" : 98.1
    "children_playing" : 93.5
    "dog_bark" : 97.3
    "drilling" : 96.8
    "engine_idling" : 95.1
    "gun_shot" : 99.2
    "jackhammer" : 96.4
    "siren" : 98.7
    "street_music" : 94.8
```

## Confusion Matrix Heatmap

```mermaid
graph TD
    subgraph "Confusion Matrix Representation"
        style CM fill:#f9f9f9,stroke:#333,stroke-width:2px
        CM["<table border='1' cellpadding='5'>
            <tr><td></td><td>AC</td><td>CH</td><td>CP</td><td>DB</td><td>DR</td><td>EI</td><td>GS</td><td>JH</td><td>SI</td><td>SM</td></tr>
            <tr><td>AC</td><td bgcolor='#ff0000'>94</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>4</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>2</td></tr>
            <tr><td>CH</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>98</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>2</td><td bgcolor='#ffffff'>0</td></tr>
            <tr><td>CP</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>93</td><td bgcolor='#ffcccc'>2</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>5</td></tr>
            <tr><td>DB</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>1</td><td bgcolor='#ff0000'>97</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>2</td></tr>
            <tr><td>DR</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>97</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>3</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td></tr>
            <tr><td>EI</td><td bgcolor='#ffcccc'>3</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>95</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>2</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td></tr>
            <tr><td>GS</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>99</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>1</td><td bgcolor='#ffffff'>0</td></tr>
            <tr><td>JH</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>2</td><td bgcolor='#ffcccc'>1</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>97</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td></tr>
            <tr><td>SI</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ff0000'>99</td><td bgcolor='#ffcccc'>1</td></tr>
            <tr><td>SM</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>4</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffffff'>0</td><td bgcolor='#ffcccc'>1</td><td bgcolor='#ff0000'>95</td></tr>
        </table>"]
    end
    
    subgraph "Legend"
        L1["AC: air_conditioner"]  
        L2["CH: car_horn"]  
        L3["CP: children_playing"]  
        L4["DB: dog_bark"]  
        L5["DR: drilling"]  
        L6["EI: engine_idling"]  
        L7["GS: gun_shot"]  
        L8["JH: jackhammer"]  
        L9["SI: siren"]  
        L10["SM: street_music"]  
    end
```

## Feature Importance Analysis

```mermaid
graph LR
    subgraph "Feature Importance by Class"
        F1["Temporal Pattern"] --> C1["Siren"] 
        F1 --> C2["Jackhammer"]
        
        F2["Frequency Range"] --> C3["Air Conditioner"] 
        F2 --> C4["Engine Idling"]
        F2 --> C5["Dog Bark"]
        
        F3["Spectral Envelope"] --> C6["Street Music"] 
        F3 --> C7["Children Playing"]
        
        F4["Onset Characteristics"] --> C8["Gun Shot"] 
        F4 --> C9["Car Horn"]
        F4 --> C10["Drilling"]
    end
```

These visualizations provide a comprehensive view of the Urban Sound Classifier's model architecture, training process, and performance characteristics. The diagrams help to understand the complex neural network structure, the feature extraction process, and how different sound classes are represented and classified by the system.