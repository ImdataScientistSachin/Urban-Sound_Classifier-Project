/**
 * Audio Recording and Playback Functionality
 */

class AudioHandler {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.audioUrl = null;
        this.audioElement = null;
        this.isRecording = false;
        this.visualizerCanvas = document.getElementById('visualizer-canvas');
        this.visualizerContext = this.visualizerCanvas ? this.visualizerCanvas.getContext('2d') : null;
        this.analyser = null;
        this.dataArray = null;
        this.animationId = null;
        this.recordingTimer = null;
        this.recordingTime = 0;
        this.maxRecordingTime = 10; // Maximum recording time in seconds
        
        // Initialize audio elements
        this.initAudioElements();
        
        // Initialize event listeners
        this.initEventListeners();
    }
    
    initAudioElements() {
        // Create audio element for playback
        this.audioElement = document.createElement('audio');
        this.audioElement.setAttribute('controls', 'controls');
        this.audioElement.style.display = 'none';
        document.body.appendChild(this.audioElement);
    }
    
    initEventListeners() {
        // Record button
        const recordBtn = document.getElementById('record-btn');
        if (recordBtn) {
            recordBtn.addEventListener('click', () => this.startRecording());
        }
        
        // Stop button
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopRecording());
        }
        
        // Play button
        const playBtn = document.getElementById('play-btn');
        if (playBtn) {
            playBtn.addEventListener('click', () => this.playRecording());
        }
        
        // Predict button for recorded audio
        const recordPredictBtn = document.getElementById('record-predict-btn');
        if (recordPredictBtn) {
            recordPredictBtn.addEventListener('click', () => this.predictRecordedAudio());
        }
        
        // Sample sound items
        const soundItems = document.querySelectorAll('.sound-item');
        soundItems.forEach(item => {
            item.addEventListener('click', (e) => this.handleSampleSelection(e));
        });
        
        // Play result audio button
        const playResultBtn = document.getElementById('play-result-btn');
        if (playResultBtn) {
            playResultBtn.addEventListener('click', () => this.playResultAudio());
        }
    }
    
    async startRecording() {
        try {
            // Reset recording state
            this.audioChunks = [];
            this.isRecording = true;
            this.recordingTime = 0;
            
            // Update UI
            this.updateRecordingStatus('Recording...');
            document.getElementById('record-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('play-btn').disabled = true;
            document.getElementById('record-predict-btn').disabled = true;
            
            // Get user media
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Set up audio context for visualizer
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            this.analyser = audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
            
            // Set up data array for visualizer
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
            // Start visualizer
            this.startVisualizer();
            
            // Set up media recorder
            this.mediaRecorder = new MediaRecorder(stream);
            
            // Event handler for data available
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            // Event handler for recording stop
            this.mediaRecorder.onstop = () => {
                // Create audio blob and URL
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.audioUrl = URL.createObjectURL(this.audioBlob);
                
                // Set audio source
                this.audioElement.src = this.audioUrl;
                
                // Update UI
                this.updateRecordingStatus('Recording complete');
                document.getElementById('record-btn').disabled = false;
                document.getElementById('play-btn').disabled = false;
                document.getElementById('record-predict-btn').disabled = false;
                
                // Stop visualizer
                this.stopVisualizer();
                
                // Stop recording timer
                clearInterval(this.recordingTimer);
            };
            
            // Start recording
            this.mediaRecorder.start();
            
            // Start recording timer
            this.startRecordingTimer();
            
            console.log('Recording started');
        } catch (error) {
            console.error('Error starting recording:', error);
            this.updateRecordingStatus('Error: ' + error.message);
            this.isRecording = false;
            document.getElementById('record-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.isRecording = false;
            this.mediaRecorder.stop();
            document.getElementById('stop-btn').disabled = true;
            console.log('Recording stopped');
        }
    }
    
    playRecording() {
        if (this.audioUrl) {
            this.audioElement.play();
            console.log('Playing recording');
        } else {
            console.log('No recording to play');
        }
    }
    
    predictRecordedAudio() {
        if (!this.audioBlob) {
            alert('Please record audio first');
            return;
        }
        
        // Show loading overlay
        document.querySelector('.loading-overlay').classList.add('show');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', this.audioBlob, 'recorded_audio.wav');
        
        // Send to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Process prediction results
            processPredictionResults(data);
        })
        .catch(error => {
            console.error('Error predicting recorded audio:', error);
            alert('Error predicting recorded audio: ' + error.message);
            document.querySelector('.loading-overlay').classList.remove('show');
        });
    }
    
    handleSampleSelection(event) {
        const soundItem = event.currentTarget;
        const soundPath = soundItem.getAttribute('data-sound-path');
        
        if (!soundPath) {
            console.error('No sound path found');
            return;
        }
        
        // Highlight selected sound
        document.querySelectorAll('.sound-item').forEach(item => {
            item.classList.remove('selected');
        });
        soundItem.classList.add('selected');
        
        // Update UI
        const selectedSoundNameElement = document.getElementById('selected-sound-name');
        if (selectedSoundNameElement) {
            selectedSoundNameElement.textContent = soundItem.querySelector('.sound-item-name').textContent;
        }
        
        const samplePredictBtn = document.getElementById('sample-predict-btn');
        if (samplePredictBtn) {
            samplePredictBtn.disabled = false;
        }
        
        // Play the sample sound
        this.playSampleSound(soundPath);
    }
    
    playSampleSound(soundPath) {
        // For URL paths, use as is
        // For relative or absolute paths, we need to convert to a URL
        if (soundPath.startsWith('/')) {
            // Already a URL path, use as is
            this.audioElement.src = soundPath;
        } else if (soundPath.startsWith('test_audio_samples/')) {
            // Convert relative path to URL path
            this.audioElement.src = '/audio_samples/' + soundPath.substring('test_audio_samples/'.length);
        } else if (soundPath.includes('UrbanSound8K')) {
            // Handle UrbanSound8K paths
            const parts = soundPath.split('/');
            const foldName = parts[parts.length - 2]; // e.g., fold1
            const fileName = parts[parts.length - 1]; // e.g., 101415-3-0-2.wav
            this.audioElement.src = `/urbansound_samples/${foldName}/${fileName}`;
        } else {
            // Use as is, but log a warning
            console.warn('Unknown path format:', soundPath);
            this.audioElement.src = soundPath;
        }
        
        console.log('Playing audio from:', this.audioElement.src);
        this.audioElement.play();
    }
    
    playResultAudio() {
        // This function will play the audio file that was used for prediction
        const audioPath = document.getElementById('play-result-btn').getAttribute('data-audio-path');
        if (audioPath) {
            this.audioElement.src = audioPath;
            this.audioElement.play();
        } else {
            console.log('No audio to play');
        }
    }
    
    startVisualizer() {
        if (!this.visualizerContext || !this.analyser) return;
        
        // Clear placeholder text
        const placeholder = document.querySelector('.visualizer-placeholder');
        if (placeholder) placeholder.style.display = 'none';
        
        // Resize canvas to fit container
        const visualizerContainer = document.querySelector('.visualizer');
        if (visualizerContainer) {
            this.visualizerCanvas.width = visualizerContainer.offsetWidth;
            this.visualizerCanvas.height = visualizerContainer.offsetHeight;
        }
        
        // Animation function
        const draw = () => {
            this.animationId = requestAnimationFrame(draw);
            
            // Get frequency data
            this.analyser.getByteFrequencyData(this.dataArray);
            
            // Clear canvas
            this.visualizerContext.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
            
            // Draw visualization
            const barWidth = (this.visualizerCanvas.width / this.dataArray.length) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < this.dataArray.length; i++) {
                barHeight = this.dataArray[i] / 2;
                
                // Create gradient
                const gradient = this.visualizerContext.createLinearGradient(
                    0, this.visualizerCanvas.height - barHeight, 0, this.visualizerCanvas.height
                );
                gradient.addColorStop(0, '#6200ea');
                gradient.addColorStop(1, '#b388ff');
                
                this.visualizerContext.fillStyle = gradient;
                this.visualizerContext.fillRect(
                    x, this.visualizerCanvas.height - barHeight, barWidth, barHeight
                );
                
                x += barWidth + 1;
            }
        };
        
        draw();
    }
    
    stopVisualizer() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
            
            // Clear canvas
            if (this.visualizerContext) {
                this.visualizerContext.clearRect(
                    0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height
                );
            }
            
            // Show placeholder text
            const placeholder = document.querySelector('.visualizer-placeholder');
            if (placeholder) placeholder.style.display = 'block';
        }
    }
    
    startRecordingTimer() {
        const timerElement = document.querySelector('.recording-time');
        if (!timerElement) return;
        
        this.recordingTimer = setInterval(() => {
            this.recordingTime++;
            
            // Format time as MM:SS
            const minutes = Math.floor(this.recordingTime / 60);
            const seconds = this.recordingTime % 60;
            const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Update timer display
            timerElement.textContent = formattedTime;
            
            // Check if max recording time reached
            if (this.recordingTime >= this.maxRecordingTime) {
                this.stopRecording();
            }
        }, 1000);
    }
    
    updateRecordingStatus(status) {
        const statusElement = document.querySelector('.record-status');
        if (statusElement) {
            statusElement.textContent = status;
        }
    }
}

/**
 * Tab Navigation Functionality
 */
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Hide all tab contents
            tabContents.forEach(content => {
                content.classList.add('hidden');
                content.classList.remove('active');
            });
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Show corresponding content
            const tabId = button.getAttribute('data-tab') + 'Tab';
            const tabContent = document.getElementById(tabId);
            if (tabContent) {
                tabContent.classList.remove('hidden');
                tabContent.classList.add('active');
            }
        });
    });
}

/**
 * Sample Sounds Prediction
 */
function predictSampleSound() {
    const selectedSound = document.querySelector('.sound-item.selected');
    if (!selectedSound) {
        alert('Please select a sound first');
        return;
    }
    
    const soundPath = selectedSound.getAttribute('data-sound-path');
    if (!soundPath) {
        alert('Invalid sound selection');
        return;
    }
    
    // Show loading overlay
    document.querySelector('.loading-overlay').classList.add('show');
    
    // Create form data with the path to the sample sound
    const formData = new FormData();
    formData.append('sample_path', soundPath);
    
    console.log('Sending sample path:', soundPath);
    
    // Send to server
    fetch('/predict_sample', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // First check if the response is OK (status 200-299)
        if (!response.ok) {
            return response.text().then(text => {
                console.error(`Server error (${response.status}):`, text.substring(0, 200));
                throw new Error(`Server error (${response.status}): ${response.statusText}`);
            });
        }
        
        // Then check if the response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            // Not JSON, handle as text to see the error
            return response.text().then(text => {
                console.error('Non-JSON response:', text.substring(0, 200));
                // Show a more detailed error message
                throw new Error('Server returned non-JSON response. The server might be returning an HTML error page instead of JSON data.');
            });
        }
        return response.json();
    })
    .then(data => {
        // Check if data has the expected structure
        if (!data) {
            console.error('Empty response received');
            throw new Error('Server returned an empty response');
        }
        
        if (!data.predictions || !Array.isArray(data.predictions)) {
            console.error('Invalid response format:', data);
            throw new Error('Server returned an invalid response format. Expected predictions array.');
        }
        
        // Process prediction results
        processPredictionResults(data);
    })
    .catch(error => {
        console.error('Error predicting sample sound:', error);
        // Make sure loading overlay is hidden
        document.querySelector('.loading-overlay').classList.remove('show');
        // Only show alert if it's not already handled
        if (!error.message.includes('Server returned non-JSON response')) {
            alert('Error predicting sample sound: ' + error.message);
        }
    });
}

/**
 * Load Sample Sounds for a Category
 */
function loadSampleSounds(event) {
    const button = event.currentTarget;
    const category = button.getAttribute('data-category');
    
    if (!category) {
        console.error('No category specified');
        return;
    }
    
    // Show loading state
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    button.disabled = true;
    
    // Fetch samples for this category
    fetch(`/samples/${category}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to load samples');
                });
            }
            return response.json();
        })
        .then(data => {
            // Get the container for sample files
            const sampleFilesContainer = document.getElementById('sampleFiles');
            
            // Clear previous samples
            sampleFilesContainer.innerHTML = '';
            
            // Check if we have samples
            if (!data.samples || data.samples.length === 0) {
                sampleFilesContainer.innerHTML = `<p class="no-samples">No sample sounds found for ${category.replace('_', ' ')}.</p>`;
                return;
            }
            
            // Create a heading for the samples
            const heading = document.createElement('h4');
            heading.textContent = `${category.replace(/_/g, ' ')} Samples`;
            heading.className = 'sample-files-heading';
            sampleFilesContainer.appendChild(heading);
            
            // Create a container for the sound items
            const soundItemsContainer = document.createElement('div');
            soundItemsContainer.className = 'sound-items';
            
            // Add each sample to the container
            data.samples.forEach(sample => {
                const soundItem = document.createElement('div');
                soundItem.className = 'sound-item';
                // Store the full path for prediction
                soundItem.setAttribute('data-sound-path', sample.path);
                
                const soundName = document.createElement('div');
                soundName.className = 'sound-item-name';
                soundName.textContent = sample.name;
                
                const playIcon = document.createElement('div');
                playIcon.className = 'sound-item-play';
                playIcon.innerHTML = '<i class="fas fa-play"></i>';
                
                soundItem.appendChild(soundName);
                soundItem.appendChild(playIcon);
                soundItemsContainer.appendChild(soundItem);
            });
            
            // Add the sound items to the container
            sampleFilesContainer.appendChild(soundItemsContainer);
            
            // Add event listeners to the new sound items
            const audioHandler = new AudioHandler();
            const newSoundItems = sampleFilesContainer.querySelectorAll('.sound-item');
            newSoundItems.forEach(item => {
                item.addEventListener('click', (e) => audioHandler.handleSampleSelection(e));
            });
            
            // Show the predict button
            const samplePredictBtn = document.getElementById('sample-predict-btn');
            if (samplePredictBtn) {
                samplePredictBtn.style.display = 'block';
                samplePredictBtn.disabled = true;
            }
            
            // Add a selected sound name display
            if (!document.getElementById('selected-sound-name')) {
                const selectedSoundDisplay = document.createElement('div');
                selectedSoundDisplay.className = 'selected-sound-display';
                selectedSoundDisplay.innerHTML = '<strong>Selected Sound:</strong> <span id="selected-sound-name">None</span>';
                sampleFilesContainer.insertBefore(selectedSoundDisplay, soundItemsContainer);
            }
        })
        .catch(error => {
            console.error('Error loading samples:', error);
            const sampleFilesContainer = document.getElementById('sampleFiles');
            sampleFilesContainer.innerHTML = `<p class="error-message">Error loading samples: ${error.message}</p>`;
        })
        .finally(() => {
            // Reset button state
            button.innerHTML = 'Load Samples';
            button.disabled = false;
        });
}

/**
 * Initialize Audio Functionality
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tabs
    initTabs();
    
    // Initialize audio handler
    const audioHandler = new AudioHandler();
    
    // Initialize sample predict button
    const samplePredictBtn = document.getElementById('sample-predict-btn');
    if (samplePredictBtn) {
        samplePredictBtn.addEventListener('click', predictSampleSound);
    }
    
    // Initialize sample category buttons
    const sampleBtns = document.querySelectorAll('.sample-btn');
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', loadSampleSounds);
    });
    
    // Set first tab as active by default
    const firstTabBtn = document.querySelector('.tab-btn');
    if (firstTabBtn) {
        try {
            firstTabBtn.click();
        } catch (error) {
            console.warn('Could not activate first tab:', error);
        }
    }
});