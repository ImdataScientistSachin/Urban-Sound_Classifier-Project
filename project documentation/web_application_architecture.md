# Urban Sound Classifier - Web Application Architecture

## Application Architecture Overview

```mermaid
flowchart TD
    subgraph "Client Side"
        A1[User Interface] --> A2[File Upload/Microphone Recording]
        A2 --> A3[Audio Capture]
        A3 --> A4[HTTP Request]
    end
    
    subgraph "Server Side"
        B1[Flask Web Server] --> B2[Request Handling]
        B2 --> B3[File Processing]
        B3 --> B4[Audio Conversion]
        B4 --> B5[Feature Extraction]
        B5 --> B6[Model Prediction]
        B6 --> B7[Response Formatting]
        B7 --> B8[HTTP Response]
    end
    
    subgraph "Model Components"
        C1[Pre-trained Model] --> B6
        C2[Class Labels] --> B7
    end
    
    %% Connect the subgraphs
    A4 --> B1
    B8 --> A1
```

## User Interaction Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Web Interface
    participant API as Flask API
    participant Processor as Audio Processor
    participant Model as Neural Network
    
    User->>UI: Access web application
    UI->>User: Display upload interface
    
    alt File Upload
        User->>UI: Select audio file
        UI->>UI: Display file details
    else Microphone Recording
        User->>UI: Click "Use Microphone"
        UI->>User: Request microphone permission
        User->>UI: Grant permission
        UI->>UI: Start recording
        User->>UI: Stop recording
        UI->>UI: Create audio file
    end
    
    User->>UI: Click "Analyze Sound"
    UI->>UI: Show loading animation
    UI->>API: POST /predict with audio file
    
    API->>Processor: process_audio_file(file)
    Processor->>Processor: convert_audio_to_wav()
    Processor->>Processor: extract_features()
    Processor-->>API: Return features
    
    API->>Model: predict(features)
    Model-->>API: Return prediction and confidence
    
    API-->>UI: JSON response with results
    UI->>UI: Process results
    UI->>User: Display prediction and confidence
```

## Component Architecture

```mermaid
flowchart LR
    subgraph "Frontend Components"
        F1["index.html"] --> F2["CSS Styles"]
        F1 --> F3["JavaScript"]
        F3 --> F4["File Upload Handler"]
        F3 --> F5["Microphone Recorder"]
        F3 --> F6["API Client"]
        F3 --> F7["Results Renderer"]
    end
    
    subgraph "Backend Components"
        B1["app.py"] --> B2["Flask Routes"]
        B2 --> B3["/predict Endpoint"]
        B2 --> B4["/classes Endpoint"]
        B3 --> B5["File Processing"]
        B5 --> B6["Audio Conversion"]
        B6 --> B7["Feature Extraction"]
        B7 --> B8["Model Prediction"]
    end
    
    subgraph "Utility Components"
        U1["config.py"] --> B1
        U2["utils.py"] --> B5
        U3["model.py"] --> B8
    end
    
    %% Connect the subgraphs
    F6 --> B3
    B4 --> F7
```

## API Endpoints

```mermaid
classDiagram
    class PredictEndpoint {
        URL: /predict
        Method: POST
        Input: audio file (multipart/form-data)
        Output: JSON with class and confidence
        Process: Convert audio, extract features, predict
    }
    
    class ClassesEndpoint {
        URL: /classes
        Method: GET
        Input: None
        Output: JSON with available classes
        Process: Return list of class labels
    }
    
    class ErrorHandling {
        400: Bad Request (invalid file)
        415: Unsupported Media Type
        500: Internal Server Error
        503: Service Unavailable (model not loaded)
    }
    
    PredictEndpoint -- ErrorHandling
    ClassesEndpoint -- ErrorHandling
```

## File Processing Pipeline

```mermaid
flowchart TD
    A["File Upload"] --> B{"Valid File?"}
    B -->|"Yes"| C["Save to Temporary Location"]
    B -->|"No"| D["Return 400 Error"]
    
    C --> E["Check File Size"]
    E -->|"Empty"| F["Return 400 Error"]
    E -->|"Non-empty"| G["Create File Object"]
    
    G --> H["Convert to WAV"]
    H --> I["Verify WAV File"]
    I -->|"Invalid"| J["Return 500 Error"]
    I -->|"Valid"| K["Extract Features"]
    
    K --> L["Make Prediction"]
    L --> M["Format Response"]
    M --> N["Return JSON Result"]
    
    %% Cleanup steps
    N --> O["Clean Up Temporary Files"]
```

## User Interface Components

```mermaid
flowchart TD
    subgraph "Header Section"
        H1["Logo"] --> H2["Title"]
        H2 --> H3["Tagline"]
        H3 --> H4["Developer Info"]
        H4 --> H5["Supported Formats"]
    end
    
    subgraph "Upload Section"
        U1["Upload Container"] --> U2["Drag & Drop Area"]
        U2 --> U3["Upload Icon"]
        U3 --> U4["Instructions"]
        U4 --> U5["File Selection Button"]
        U5 --> U6["Microphone Button"]
        U6 --> U7["User Signature Input"]
        U7 --> U8["File Info Display"]
    end
    
    subgraph "Results Section"
        R1["Results Container"] --> R2["Result Card"]
        R2 --> R3["Result Icon"]
        R3 --> R4["Detected Sound Class"]
        R4 --> R5["User Signature"]
        R5 --> R6["Confidence Display"]
        R6 --> R7["Confidence Bar"]
        R7 --> R8["Confidence Scale"]
    end
    
    subgraph "Action Buttons"
        A1["Analyze Sound Button"] --> A2["Try Again Button"]
    end
    
    %% Connect the sections
    H5 --> U1
    U8 --> A1
    A1 --> R1
    R8 --> A2
    A2 --> U1
```

## Responsive Design

```mermaid
flowchart LR
    subgraph "Desktop Layout"
        D1["Header"] --> D2["Upload Section"]
        D2 --> D3["Results Section"]
    end
    
    subgraph "Tablet Layout"
        T1["Header"] --> T2["Upload Section"]
        T2 --> T3["Results Section"]
    end
    
    subgraph "Mobile Layout"
        M1["Header"] --> M2["Upload Section"]
        M2 --> M3["Results Section"]
    end
    
    %% CSS Media Queries
    CSS["CSS Media Queries"] --> D1
    CSS --> T1
    CSS --> M1
```

## Error Handling Flow

```mermaid
flowchart TD
    A["User Action"] --> B{"Error Type?"}
    
    B -->|"Invalid File"| C["Display 'File type not allowed' message"]
    B -->|"Empty File"| D["Display 'File is empty' message"]
    B -->|"Processing Error"| E["Display technical error details"]
    B -->|"Model Error"| F["Display 'Model not loaded' message"]
    B -->|"Server Error"| G["Display 'Server error' message"]
    
    C --> H["Enable retry"]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I["User retries"]
    I --> A
```

## Deployment Architecture

```mermaid
flowchart TD
    subgraph "Development Environment"
        D1["Local Flask Server"] --> D2["Local Browser"]
    end
    
    subgraph "Production Environment"
        P1["Web Server"] --> P2["WSGI Server"]
        P2 --> P3["Flask Application"]
        P3 --> P4["Model"]
        P3 --> P5["Static Files"]
        P3 --> P6["Templates"]
    end
    
    subgraph "Client"
        C1["Web Browser"] --> C2["HTML/CSS/JS"]
        C2 --> C3["API Requests"]
    end
    
    %% Connect the subgraphs
    C3 --> P1
    P5 --> C2
```

These diagrams provide a comprehensive view of the Urban Sound Classifier's web application architecture, showing how the different components interact, the user flow through the application, and the technical implementation details of the frontend and backend systems. The visualizations help to understand the complete system from user interaction to model prediction and result presentation.