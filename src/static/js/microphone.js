// Microphone handling for real-time audio classification

class MicrophoneHandler {
    constructor() {
        this.audioContext = null;
        this.microphone = null;
        this.analyser = null;
        this.isRecording = false;
        this.recordingStartTime = null;
        this.audioChunks = [];
        this.mediaRecorder = null;
        this.recordingDuration = 4000; // Default 4 seconds (in milliseconds)
        this.visualizerCanvas = null;
        this.visualizerContext = null;
        this.animationFrame = null;
    }

    async initialize(visualizerCanvas) {
        try {
            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Set up visualizer if canvas is provided
            if (visualizerCanvas) {
                this.visualizerCanvas = visualizerCanvas;
                this.visualizerContext = visualizerCanvas.getContext('2d');
                this.setupCanvasSize();
                window.addEventListener('resize', () => this.setupCanvasSize());
            }

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Create media recorder for capturing audio
            this.mediaRecorder = new MediaRecorder(stream);
            
            // Set up event handlers for the media recorder
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.isRecording = false;
                this.processRecording();
            };

            // Connect microphone to audio context for visualization
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.microphone.connect(this.analyser);

            return true;
        } catch (error) {
            console.error('Error initializing microphone:', error);
            return false;
        }
    }

    setupCanvasSize() {
        if (!this.visualizerCanvas) return;
        
        // Make canvas size match its display size
        const rect = this.visualizerCanvas.getBoundingClientRect();
        this.visualizerCanvas.width = rect.width;
        this.visualizerCanvas.height = rect.height;
    }

    startRecording() {
        if (!this.mediaRecorder || this.isRecording) return false;
        
        this.audioChunks = [];
        this.isRecording = true;
        this.recordingStartTime = Date.now();
        
        // Start the media recorder
        this.mediaRecorder.start();
        
        // Start visualizing if canvas is available
        if (this.visualizerCanvas) {
            this.startVisualizer();
        }
        
        // Set a timeout to stop recording after the specified duration
        setTimeout(() => {
            if (this.isRecording) {
                this.stopRecording();
            }
        }, this.recordingDuration);
        
        return true;
    }
    
    setRecordingDuration(seconds) {
        // Convert seconds to milliseconds
        this.recordingDuration = seconds * 1000;
        return this.recordingDuration;
    }

    stopRecording() {
        if (!this.mediaRecorder || !this.isRecording) return false;
        
        this.mediaRecorder.stop();
        this.isRecording = false;
        
        // Stop visualizer
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        return true;
    }

    processRecording() {
        // Create a Blob from the recorded audio chunks
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        
        // Create a File object from the Blob
        const fileName = `recording_${new Date().toISOString()}.wav`;
        const audioFile = new File([audioBlob], fileName, { type: 'audio/wav' });
        
        // Get user signature if available
        const userSignature = document.getElementById('user-signature');
        const signatureValue = userSignature && userSignature.value.trim() !== '' ? userSignature.value.trim() : null;
        
        // Trigger the file handling function from main.js
        if (window.handleFile) {
            window.handleFile(audioFile, signatureValue);
        } else {
            console.error('handleFile function not found in global scope');
        }
    }

    startVisualizer() {
        if (!this.visualizerCanvas || !this.analyser) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            this.animationFrame = requestAnimationFrame(draw);
            
            this.analyser.getByteTimeDomainData(dataArray);
            
            // Clear the canvas
            this.visualizerContext.fillStyle = 'rgba(34, 34, 34, 0.2)';
            this.visualizerContext.fillRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
            
            // Draw the waveform
            this.visualizerContext.lineWidth = 2;
            this.visualizerContext.strokeStyle = 'rgb(0, 217, 255)';
            this.visualizerContext.beginPath();
            
            const sliceWidth = this.visualizerCanvas.width / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * this.visualizerCanvas.height / 2;
                
                if (i === 0) {
                    this.visualizerContext.moveTo(x, y);
                } else {
                    this.visualizerContext.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            this.visualizerContext.lineTo(this.visualizerCanvas.width, this.visualizerCanvas.height / 2);
            this.visualizerContext.stroke();
            
            // Draw recording progress if recording
            if (this.isRecording) {
                const elapsed = Date.now() - this.recordingStartTime;
                const progress = Math.min(elapsed / this.recordingDuration, 1);
                
                this.visualizerContext.fillStyle = 'rgba(255, 0, 0, 0.3)';
                this.visualizerContext.fillRect(
                    0, 
                    this.visualizerCanvas.height - 5, 
                    this.visualizerCanvas.width * progress, 
                    5
                );
            }
        };
        
        draw();
    }

    getRemainingTime() {
        if (!this.isRecording) return 0;
        
        const elapsed = Date.now() - this.recordingStartTime;
        return Math.max(0, this.recordingDuration - elapsed);
    }

    getRecordingProgress() {
        if (!this.isRecording) return 0;
        
        const elapsed = Date.now() - this.recordingStartTime;
        return Math.min(elapsed / this.recordingDuration, 1);
    }
}

// Export the class
window.MicrophoneHandler = MicrophoneHandler;