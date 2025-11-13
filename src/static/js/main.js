// Main JavaScript for Urban Sound Classifier

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    const classifyBtn = document.getElementById('classify-button');
    const resultsSection = document.getElementById('results-section');
    const loadingSection = document.getElementById('loading-section');
    const resultClass = document.getElementById('result-class');
    const resultSignature = document.getElementById('result-signature');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    const newClassificationBtn = document.getElementById('new-classification');
    const uploadButton = document.querySelector('.upload-button');
    const micButton = document.getElementById('mic-button');
    const microphoneModal = document.getElementById('microphone-modal');
    const microphoneCanvas = document.getElementById('microphone-canvas');
    const startRecordingBtn = document.getElementById('start-recording');
    const cancelRecordingBtn = document.getElementById('cancel-recording');
    const recordingStatus = document.getElementById('recording-status');
    const userSignature = document.getElementById('user-signature');
    const audioPlayer = document.getElementById('audio-player');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const playResultAudioBtn = document.getElementById('play-result-audio');
    const recordingDurationSlider = document.getElementById('recording-duration');
    const durationValue = document.getElementById('duration-value');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const allConfidencesContainer = document.getElementById('all-confidences-container');
    
    let selectedFile = null;
    let microphoneHandler = null;
    let recordingTimer = null;
    
    // Event Listeners
// Fix for the double-click issue: Only trigger file input click when clicking on the upload button
uploadButton.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent event bubbling to uploadArea
    // Prevent default to avoid any browser-specific double-opening behavior
    e.preventDefault();
    fileInput.click();
});
    
    // Add visual feedback when clicking on upload area
    uploadArea.addEventListener('click', function() {
        this.classList.add('upload-area-active');
        setTimeout(() => {
            this.classList.remove('upload-area-active');
        }, 200);
    });
    fileInput.addEventListener('change', handleFileSelect);
    removeFileBtn.addEventListener('click', removeFile);
    classifyBtn.addEventListener('click', classifySound);
    newClassificationBtn.addEventListener('click', resetClassification);
    
    // Play audio button in results section
    if (playResultAudioBtn) {
        playResultAudioBtn.addEventListener('click', playResultAudio);
    }
    
    // Microphone button event listener
    if (micButton) {
        micButton.addEventListener('click', openMicrophoneModal);
    }
    
    // Microphone modal event listeners
    if (startRecordingBtn) {
        startRecordingBtn.addEventListener('click', startRecording);
    }
    
    if (cancelRecordingBtn) {
        cancelRecordingBtn.addEventListener('click', closeMicrophoneModal);
    }
    
    // Recording duration slider event listener
    if (recordingDurationSlider) {
        recordingDurationSlider.addEventListener('input', updateRecordingDuration);
    }
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Functions
    function handleFileSelect(e) {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
            // Reset the file input value to ensure change event fires even if same file is selected again
            setTimeout(() => {
                e.target.value = '';
            }, 100);
        }
    }
    
    function handleFile(file, signatureValue = null) {
        // Check if file is audio
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/mp4'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid audio file (WAV, MP3, OGG, FLAC, M4A)');
            return;
        }
        
        selectedFile = file;
        
        // Create audio URL for playback
        // First check if there's an existing URL and properly clean it up
        if (window.audioURL) {
            try {
                // Store the current audio player source before revoking
                const currentPlayerSrc = audioPlayer ? audioPlayer.src : null;
                
                // Only revoke if it's not currently being used by the audio player
                if (!currentPlayerSrc || !currentPlayerSrc.includes(window.audioURL.split('/').pop())) {
                    // Store the URL to revoke in a local variable
                    const urlToRevoke = window.audioURL;
                    
                    // Clear the global reference first to prevent race conditions
                    window.audioURL = null;
                    
                    // Use setTimeout to ensure any pending operations with the URL are complete
                    setTimeout(() => {
                        try {
                            URL.revokeObjectURL(urlToRevoke);
                        } catch (error) {
                            console.warn('Error revoking previous URL:', error);
                        }
                    }, 100);
                }
            } catch (error) {
                console.warn('Error checking or revoking previous URL:', error);
            }
        }
        
        // Create a new object URL
        try {
            window.audioURL = URL.createObjectURL(file);
            console.log('Created new blob URL:', window.audioURL);
        } catch (error) {
            console.error('Error creating object URL:', error);
        }
        
        // Update UI
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        uploadArea.style.display = 'none';
        fileInfo.style.display = 'flex';
        classifyBtn.disabled = false;
        
        // Show audio player
        const audioPlayer = document.getElementById('audio-player-container');
        if (audioPlayer) {
            const player = document.getElementById('audio-player');
            if (player) {
                player.src = window.audioURL;
                audioPlayer.style.display = 'flex';
            }
        }
        
        // If signature value is provided from microphone, update the signature input
        if (signatureValue && userSignature) {
            userSignature.value = signatureValue;
        }
    }
    
    function removeFile() {
        selectedFile = null;
        fileInput.value = '';
        uploadArea.style.display = 'block';
        fileInfo.style.display = 'none';
        classifyBtn.disabled = true;
        
        // Hide audio player
        const audioPlayer = document.getElementById('audio-player-container');
        if (audioPlayer) {
            audioPlayer.style.display = 'none';
            const player = document.getElementById('audio-player');
            if (player) {
                player.pause();
                player.src = '';
            }
        }
        
        // Revoke object URL - but only after ensuring the audio player is no longer using it
        if (window.audioURL) {
            // Set a small timeout to ensure the audio player has released the resource
            setTimeout(() => {
                try {
                    URL.revokeObjectURL(window.audioURL);
                } catch (error) {
                    console.warn('Error revoking URL:', error);
                }
                window.audioURL = null;
            }, 100);
        }
    }
    
    function classifySound() {
        if (!selectedFile) return;
        
        console.log('Classifying sound with file:', selectedFile.name);
        
        // Show loading
        loadingSection.style.display = 'block';
        classifyBtn.disabled = true;
        
        // Store the file for waveform generation
        window.currentAudioFile = selectedFile;
        
        // Create form data
        const formData = new FormData();
        
        // Check if we need to use smaller chunks for the file
        let fileToSend = selectedFile;
        if (window.useSmallChunks) {
            console.log('Using smaller chunk size for file transfer');
            
            // If the file is larger than 1MB, create a smaller version
            if (selectedFile.size > 1024 * 1024) {
                // Get just the first 1MB of the file to reduce size
                fileToSend = selectedFile.slice(0, 1024 * 1024, selectedFile.type);
                console.log('Created smaller file slice:', fileToSend.size, 'bytes');
            }
        }
        
        formData.append('file', fileToSend);
        
        // Add user signature if provided
        if (userSignature && userSignature.value.trim()) {
            formData.append('user_signature', userSignature.value.trim());
            console.log('User signature included:', userSignature.value.trim());
        }
        
        // Add a unique request ID
        const requestId = 'req-' + Math.random().toString(36).substring(2, 15);
        formData.append('request_id', requestId);
        
        // Add flag for chunked upload
        if (window.useSmallChunks) {
            formData.append('use_small_chunks', 'true');
        }
        
        // Log the file being sent
        console.log('Sending file:', fileToSend.name || 'slice', 'Type:', fileToSend.type, 'Size:', fileToSend.size, 'Request ID:', requestId, 'Using small chunks:', !!window.useSmallChunks);
        
        // Send request to API with full URL
        const apiUrl = window.location.origin + '/predict';
        console.log('Sending request to:', apiUrl);
        
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        const urlWithTimestamp = `${apiUrl}?t=${timestamp}&nocache=${Math.random()}`;
        
        // Clear any previous error messages
        const errorContainer = document.getElementById('error-message-container');
        if (errorContainer) {
            errorContainer.innerHTML = '';
        }
        
        // Maximum number of retries
        const maxRetries = 3;
        let retryCount = 0;
        
        function attemptFetch() {
            console.log(`Attempt ${retryCount + 1} of ${maxRetries} for request ${requestId}`);
            
            // Create a new timestamp for each retry
            const retryTimestamp = new Date().getTime();
            const retryUrl = `${apiUrl}?t=${retryTimestamp}&retry=${retryCount}&nocache=${Math.random()}`;
            
            // Create an abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            fetch(retryUrl, {
                method: 'POST',
                body: formData,
                // Add these headers to prevent caching issues
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'X-Request-ID': requestId
                },
                // Use the abort controller's signal
                signal: controller.signal
            })
            .then(response => {
                clearTimeout(timeoutId); // Clear the timeout
                console.log(`Response status for request ${requestId}:`, response.status);
                if (!response.ok) {
                    return response.text().then(text => {
                        console.error(`Error response for request ${requestId}:`, text);
                        try {
                            // Try to parse the error as JSON
                            const errorJson = JSON.parse(text);
                            throw new Error(`Server responded with status ${response.status}: ${errorJson.error || text}`);
                        } catch (parseError) {
                            // If parsing fails, use the raw text
                            throw new Error(`Server responded with status ${response.status}: ${text}`);
                        }
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Prediction result:', data);
                // Hide loading
                loadingSection.style.display = 'none';
                
                // Show results
                // Check if data.class or data.prediction is used in the API response
                const predictionClass = data.class || data.prediction;
                console.log('Prediction class:', predictionClass);
                resultClass.textContent = formatClassName(predictionClass);
                
                // Update user signature if provided
                if (data.user_signature && data.user_signature !== 'Anonymous User') {
                    resultSignature.textContent = data.user_signature;
                } else {
                    resultSignature.textContent = 'Anonymous User';
                }
                
                // Update confidence bar and value
                const confidencePercent = (data.confidence * 100).toFixed(1);
                console.log('Confidence percent:', confidencePercent);
                confidenceBar.style.width = `${confidencePercent}%`;
                confidenceValue.textContent = `${confidencePercent}%`;
                
                // Generate waveform visualization if we have a file
                console.log('Checking for waveform generation:', { selectedFile, waveformCanvas });
                console.log('Audio visualization container:', document.getElementById('audio-visualization-container'));
                
                // Make sure the waveform section is visible
                const waveformSection = document.getElementById('audio-visualization-container');
                if (waveformSection) {
                    waveformSection.style.display = 'block';
                    console.log('Set waveform section display to block');
                }
                
                // Use the stored file if selectedFile is not available
                const audioFile = selectedFile || window.currentAudioFile;
                
                if (audioFile && waveformCanvas) {
                    console.log('Calling generateWaveform with:', audioFile.name);
                    generateWaveform(audioFile, waveformCanvas);
                } else {
                    console.error('Cannot generate waveform, missing:', { 
                        audioFile: audioFile ? 'present' : 'missing', 
                        waveformCanvas: waveformCanvas ? 'present' : 'missing' 
                    });
                }
                
                // Display all confidence scores if available
        if (data.all_confidences && allConfidencesContainer) {
            displayAllConfidences(data.all_confidences);
            console.log('Displayed all confidence scores:', data.all_confidences);
        } else {
            console.error('Could not display confidence scores:', { 
                all_confidences_exists: !!data.all_confidences, 
                container_exists: !!allConfidencesContainer 
            });
        }
                
                // Make sure results section is visible
                resultsSection.style.display = 'block';
                console.log('Results section displayed');
            })
            .catch(error => {
                console.error(`Error during classification for request ${requestId}:`, error);
                
                // Check if the error is due to timeout
                const isTimeout = error.name === 'AbortError' || error.message.includes('timeout');
                
                // Check if we should retry
                if (retryCount < maxRetries - 1) {
                    retryCount++;
                    // Exponential backoff: wait longer for each retry
                    const backoffTime = Math.pow(2, retryCount) * 1000; // 2s, 4s, 8s...
                    console.log(`Retrying request ${requestId} in ${backoffTime/1000} seconds... (${isTimeout ? 'timeout occurred' : 'error occurred'})`);
                    setTimeout(attemptFetch, backoffTime);
                } else {
                    // All retries failed
                    loadingSection.style.display = 'none';
                    
                    // Provide more specific error message based on error type
                    // Create error message element if it doesn't exist
                     const errorMessageContainer = document.getElementById('error-message-container') || (() => {
                         const container = document.createElement('div');
                         container.id = 'error-message-container';
                         container.style.color = 'red';
                         container.style.marginTop = '10px';
                         container.style.padding = '10px';
                         container.style.border = '1px solid red';
                         container.style.borderRadius = '5px';
                         container.style.backgroundColor = '#fff8f8';
                         document.querySelector('.upload-container').appendChild(container);
                         return container;
                     })();
                     
                     // Clear previous error messages
                     errorMessageContainer.innerHTML = '';
                     
                     let errorMessage = '';
                     
                     if (isTimeout) {
                         errorMessage = 'Request timed out. The server took too long to respond. Please try again with a smaller audio file.';
                     } else if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_RESET') || error.message.includes('ERR_ABORTED')) {
                         errorMessage = 'Connection error. The server may be processing a large file or experiencing high load. Please try again with a smaller audio file or wait a moment before retrying.';
                     } else {
                         errorMessage = `An error occurred while classifying the sound: ${error.message}. Please check the console for more details.`;
                     }
                     
                     // Display the error message
                     const errorText = document.createElement('p');
                     errorText.textContent = errorMessage;
                     errorMessageContainer.appendChild(errorText);
                     
                     // Add a retry button for connection errors
                     if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_RESET') || error.message.includes('ERR_ABORTED')) {
                         const retryButton = document.createElement('button');
                         retryButton.textContent = 'Retry with smaller chunk size';
                         retryButton.className = 'retry-button';
                         retryButton.style.marginTop = '10px';
                         retryButton.style.padding = '5px 10px';
                         retryButton.onclick = function() {
                             // Set a flag to use smaller chunk size
                             window.useSmallChunks = true;
                             // Reset and try again
                             errorMessageContainer.innerHTML = '<p>Retrying with smaller data chunks...</p>';
                             setTimeout(classifySound, 1000);
                         };
                         errorMessageContainer.appendChild(retryButton);
                     }
                    
                    classifyBtn.disabled = false;
                    
                    // Log detailed error information for debugging
                    console.error('Detailed error information:', {
                        requestId: requestId,
                        errorName: error.name,
                        errorMessage: error.message,
                        errorStack: error.stack,
                        timestamp: new Date().toISOString()
                    });
                }
            });
        }
        
        // Start the first attempt
        attemptFetch();
    }
    
    function resetClassification() {
        // Reset UI elements
        uploadArea.style.display = 'flex';
        resultsSection.style.display = 'none';
        loadingSection.style.display = 'none';
        resultClass.textContent = '';
        resultSignature.textContent = 'Anonymous User';
        confidenceBar.style.width = '0%';
        confidenceValue.textContent = '0%';
        
        // Reset file selection
        selectedFile = null;
        fileInfo.style.display = 'none';
        fileName.textContent = '';
        fileSize.textContent = '';
        classifyBtn.disabled = true;
        
        // Hide audio player
        const audioPlayer = document.getElementById('audio-player-container');
        if (audioPlayer) {
            audioPlayer.style.display = 'none';
            const player = document.getElementById('audio-player');
            if (player) {
                player.pause();
                player.src = '';
            }
        }
        
        // Revoke object URL - but only after ensuring the audio player is no longer using it
        if (window.audioURL) {
            // Store the URL in a temporary variable
            const urlToRevoke = window.audioURL;
            // Clear the global reference first
            window.audioURL = null;
            
            // Set a small timeout to ensure the audio player has released the resource
            setTimeout(() => {
                try {
                    URL.revokeObjectURL(urlToRevoke);
                } catch (error) {
                    console.warn('Error revoking URL:', error);
                }
            }, 100);
        }
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function formatClassName(className) {
        // Replace underscores with spaces and capitalize each word
        return className.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    function playResultAudio() {
        // Check if we have a selected file and audio URL
        if (!selectedFile) {
            console.error('No audio file available to play');
            return;
        }
        
        // Show audio player if it's hidden
        if (audioPlayerContainer) {
            audioPlayerContainer.style.display = 'flex';
        }
        
        // Set source and play
        if (audioPlayer) {
            try {
                // If the audio URL is missing or invalid, recreate it from the selected file
                if (!window.audioURL) {
                    console.log('Audio URL missing, recreating from selected file');
                    window.audioURL = URL.createObjectURL(selectedFile);
                }
                
                if (audioPlayer.src !== window.audioURL) {
                    console.log('Setting audio player source to:', window.audioURL);
                    audioPlayer.src = window.audioURL;
                }
                
                // Play the audio
                audioPlayer.play().catch(error => {
                    console.error('Error playing audio:', error);
                    
                    // If the blob URL is invalid, try to recreate it
                    if (selectedFile && (error.name === 'AbortError' || error.name === 'NotSupportedError')) {
                        console.log('Attempting to recreate audio URL due to error:', error.name);
                        // Recreate the blob URL
                        if (window.audioURL) {
                            try {
                                // Store the URL to revoke in a local variable
                                const urlToRevoke = window.audioURL;
                                // Clear the global reference first
                                window.audioURL = null;
                                
                                // Use setTimeout to ensure any pending operations with the URL are complete
                                setTimeout(() => {
                                    try {
                                        URL.revokeObjectURL(urlToRevoke);
                                    } catch (e) {
                                        console.warn('Error revoking old URL:', e);
                                    }
                                }, 100);
                            } catch (e) {
                                console.warn('Error handling old URL:', e);
                            }
                        }
                        
                        // Create a new URL and try to play again
                        try {
                            window.audioURL = URL.createObjectURL(selectedFile);
                            console.log('Recreated blob URL:', window.audioURL);
                            audioPlayer.src = window.audioURL;
                            audioPlayer.play().catch(e => {
                                console.error('Still unable to play audio after recreating URL:', e);
                            });
                        } catch (e) {
                            console.error('Failed to recreate blob URL:', e);
                        }
                    }
                });
            } catch (error) {
                console.error('Exception during audio playback setup:', error);
            }
        }
    }
    
    // Microphone handling functions
    async function openMicrophoneModal() {
        if (!microphoneModal) return;
        
        // Show the modal
        microphoneModal.style.display = 'flex';
        
        // Initialize microphone handler if not already done
        if (!microphoneHandler) {
            microphoneHandler = new MicrophoneHandler();
            const initialized = await microphoneHandler.initialize(microphoneCanvas);
            
            if (!initialized) {
                recordingStatus.textContent = 'Could not access microphone. Please check permissions.';
                recordingStatus.style.color = 'red';
                startRecordingBtn.disabled = true;
                return;
            }
        }
        
        // Reset UI
        recordingStatus.textContent = 'Ready to record';
        recordingStatus.style.color = '';
        startRecordingBtn.disabled = false;
    }
    
    function closeMicrophoneModal() {
        if (!microphoneModal) return;
        
        // Hide the modal
        microphoneModal.style.display = 'none';
        
        // Stop recording if in progress
        if (microphoneHandler && microphoneHandler.isRecording) {
            microphoneHandler.stopRecording();
        }
        
        // Clear timer if active
        if (recordingTimer) {
            clearInterval(recordingTimer);
            recordingTimer = null;
        }
    }
    
    function startRecording() {
        if (!microphoneHandler) return;
        
        // Get the selected recording duration from the slider (default to 4 if not available)
        const recordingDurationSeconds = recordingDurationSlider ? parseInt(recordingDurationSlider.value) : 4;
        
        // Update the microphone handler's recording duration
        if (recordingDurationSlider) {
            microphoneHandler.setRecordingDuration(recordingDurationSeconds);
        }
        
        const success = microphoneHandler.startRecording();
        
        if (success) {
            // Update UI
            startRecordingBtn.disabled = true;
            recordingStatus.textContent = `Recording... ${recordingDurationSeconds}s`;
            recordingStatus.style.color = 'red';
            
            // Start countdown timer
            let timeLeft = recordingDurationSeconds; // Use the selected duration
            recordingTimer = setInterval(() => {
                timeLeft--;
                if (timeLeft <= 0) {
                    clearInterval(recordingTimer);
                    recordingTimer = null;
                    recordingStatus.textContent = 'Processing recording...';
                } else {
                    recordingStatus.textContent = `Recording... ${timeLeft}s`;
                }
            }, 1000);
        } else {
            recordingStatus.textContent = 'Failed to start recording';
            recordingStatus.style.color = 'red';
        }
    }
    
    // Recording duration slider handler
    function updateRecordingDuration() {
        if (!recordingDurationSlider || !durationValue || !microphoneHandler) return;
        
        const seconds = parseInt(recordingDurationSlider.value);
        durationValue.textContent = seconds;
        
        // Update the microphone handler's recording duration
        microphoneHandler.setRecordingDuration(seconds);
    }
    
    // Generate waveform visualization from audio file
    function generateWaveform(audioFile, canvas) {
        console.log('Generating waveform for:', audioFile.name);
        if (!audioFile || !canvas) {
            console.error('Missing audioFile or canvas:', { audioFile, canvas });
            return;
        }
        
        // Force the canvas to be visible
        canvas.style.display = 'block';
        
        // Check if canvas is in the DOM
        if (!document.body.contains(canvas)) {
            console.error('Canvas is not in the DOM');
            // Try to find it by ID
            const canvasById = document.getElementById('waveform-canvas');
            if (canvasById) {
                console.log('Found canvas by ID, using that instead');
                canvas = canvasById;
            } else {
                console.error('Could not find canvas by ID either');
                return;
            }
        }
        
        console.log('Canvas dimensions:', canvas.clientWidth, canvas.clientHeight);
        
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const reader = new FileReader();
        const canvasContext = canvas.getContext('2d');
        
        // Set canvas dimensions - use fixed dimensions if clientWidth/Height are 0
        canvas.width = canvas.clientWidth || 600; // Default width if clientWidth is 0
        canvas.height = canvas.clientHeight || 120; // Default height if clientHeight is 0
        
        console.log('Set canvas dimensions to:', canvas.width, canvas.height);
        
        reader.onload = function(e) {
            console.log('FileReader loaded');
            // Clear canvas
            canvasContext.clearRect(0, 0, canvas.width, canvas.height);
            
            // Decode audio data
            audioContext.decodeAudioData(e.target.result, function(buffer) {
                console.log('Audio data decoded successfully');
                // Get the audio data
                const data = buffer.getChannelData(0);
                
                // Draw the waveform
                // Create a gradient background for better visual appeal
                const bgGradient = canvasContext.createLinearGradient(0, 0, 0, canvas.height);
                bgGradient.addColorStop(0, 'rgba(20, 20, 30, 0.9)');
                bgGradient.addColorStop(1, 'rgba(40, 40, 60, 0.9)');
                canvasContext.fillStyle = bgGradient;
                canvasContext.fillRect(0, 0, canvas.width, canvas.height);
                
                // Add title to the waveform
                canvasContext.font = 'bold 12px Arial';
                canvasContext.textAlign = 'left';
                canvasContext.fillStyle = 'rgba(0, 255, 255, 0.8)';
                canvasContext.fillText('Audio Waveform', 10, 20);
                
                // Add subtle header line
                canvasContext.fillStyle = 'rgba(0, 255, 255, 0.3)';
                canvasContext.fillRect(10, 25, 100, 1);
                
                // Add a subtle grid for better visualization
                canvasContext.strokeStyle = 'rgba(100, 100, 150, 0.2)';
                canvasContext.lineWidth = 1;
                
                // Draw horizontal grid lines
                const gridSpacing = canvas.height / 8;
                for (let y = gridSpacing; y < canvas.height; y += gridSpacing) {
                    canvasContext.beginPath();
                    canvasContext.moveTo(0, y);
                    canvasContext.lineTo(canvas.width, y);
                    canvasContext.stroke();
                }
                
                // Use a thicker line and brighter color for better visibility
                canvasContext.lineWidth = 3; // Thinner but clearer line
                canvasContext.strokeStyle = 'rgb(0, 255, 255)'; // Bright cyan color for better contrast
                canvasContext.beginPath();
                
                // We need to scale the data to fit the canvas
                const step = Math.ceil(data.length / canvas.width);
                const amp = canvas.height / 2;
                
                // Calculate all min/max values first for smoother drawing
                const minValues = [];
                const maxValues = [];
                
                for (let i = 0; i < canvas.width; i++) {
                    // Find the max value in this segment
                    let min = 1.0;
                    let max = -1.0;
                    
                    for (let j = 0; j < step; j++) {
                        const index = (i * step) + j;
                        if (index < data.length) {
                            const datum = data[index];
                            if (datum < min) min = datum;
                            if (datum > max) max = datum;
                        }
                    }
                    
                    // Apply some smoothing to avoid jagged edges
                    if (i > 0) {
                        min = (min + minValues[i-1]) / 2;
                        max = (max + maxValues[i-1]) / 2;
                    }
                    
                    minValues.push(min);
                    maxValues.push(max);
                }
                
                // First draw the outline with a smooth curve
                canvasContext.lineJoin = 'round';
                canvasContext.lineCap = 'round';
                
                for (let i = 0; i < canvas.width; i++) {
                    // Draw the min-max range for this segment
                    canvasContext.moveTo(i, (1 + minValues[i]) * amp);
                    canvasContext.lineTo(i, (1 + maxValues[i]) * amp);
                }
                
                // Add glow effect to the stroke
                canvasContext.shadowColor = 'rgba(0, 255, 255, 0.8)';
                canvasContext.shadowBlur = 10;
                canvasContext.stroke();
                
                // Reset shadow for the fill
                canvasContext.shadowColor = 'transparent';
                canvasContext.shadowBlur = 0;
                
                // Now draw a filled area with gradient for better visibility
                canvasContext.beginPath();
                
                // Create a gradient for the waveform fill
                const waveGradient = canvasContext.createLinearGradient(0, 0, 0, canvas.height);
                waveGradient.addColorStop(0, 'rgba(0, 255, 255, 0.9)'); // Bright cyan at the top
                waveGradient.addColorStop(0.5, 'rgba(64, 224, 208, 0.7)'); // Turquoise in the middle
                waveGradient.addColorStop(1, 'rgba(0, 150, 255, 0.4)'); // Blue at the bottom
                canvasContext.fillStyle = waveGradient;
                
                // Start at the bottom left
                canvasContext.moveTo(0, canvas.height);
                
                // Draw the top line using the pre-calculated max values for a smoother curve
                // First point
                canvasContext.lineTo(0, (1 + maxValues[0]) * amp);
                
                // Use bezier curves for smoother waveform
                for (let i = 1; i < canvas.width - 2; i += 2) {
                    const xc = (i + i + 1) / 2;
                    const yc = ((1 + maxValues[i]) * amp + (1 + maxValues[i + 1]) * amp) / 2;
                    canvasContext.quadraticCurveTo(i, (1 + maxValues[i]) * amp, xc, yc);
                }
                
                // Handle the last point
                if (canvas.width > 1) {
                    canvasContext.quadraticCurveTo(
                        canvas.width - 2, 
                        (1 + maxValues[canvas.width - 2]) * amp,
                        canvas.width - 1, 
                        (1 + maxValues[canvas.width - 1]) * amp
                    );
                }
                
                // Continue to the bottom right and close the path
                canvasContext.lineTo(canvas.width, canvas.height);
                canvasContext.closePath();
                canvasContext.fill();
                
                // Add time axis markers
                canvasContext.fillStyle = 'rgba(255, 255, 255, 0.7)';
                canvasContext.font = '10px Arial';
                canvasContext.textAlign = 'center';
                
                // Calculate duration in seconds
                const duration = buffer.duration;
                const numMarkers = 5; // Number of time markers to show
                
                for (let i = 0; i <= numMarkers; i++) {
                    const x = (canvas.width / numMarkers) * i;
                    const time = (duration / numMarkers) * i;
                    const timeFormatted = time.toFixed(1) + 's';
                    
                    // Draw time marker
                    canvasContext.fillText(timeFormatted, x, canvas.height - 5);
                    
                    // Draw tick mark
                    canvasContext.fillRect(x, canvas.height - 15, 1, 5);
                }
            });
        };
        
        reader.readAsArrayBuffer(audioFile);
    }
    
    // Display all confidence scores
    function displayAllConfidences(confidences) {
        console.log('displayAllConfidences called with:', confidences);
        if (!allConfidencesContainer) {
            console.error('allConfidencesContainer not found');
            return;
        }
        
        // Clear previous content
        allConfidencesContainer.innerHTML = '';
        
        // Filter out diagnostic keys (those starting with underscore)
        const classKeys = Object.keys(confidences).filter(key => !key.startsWith('_'));
        console.log('Filtered class keys:', classKeys);
        
        // Sort classes by confidence (descending)
        const sortedClasses = classKeys.sort((a, b) => {
            return confidences[b] - confidences[a];
        });
        
        // Create and append confidence bars for each class
        sortedClasses.forEach(className => {
            const confidence = confidences[className];
            const confidencePercent = (confidence * 100).toFixed(1);
            console.log(`Creating bar for ${className}: ${confidencePercent}%`);
            
            const confidenceItem = document.createElement('div');
            confidenceItem.className = 'confidence-item';
            
            const classNameElement = document.createElement('div');
            classNameElement.className = 'confidence-item-class';
            classNameElement.textContent = formatClassName(className);
            
            const barContainer = document.createElement('div');
            barContainer.className = 'confidence-item-bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'confidence-item-bar';
            bar.style.width = `${confidencePercent}%`;
            
            const valueElement = document.createElement('div');
            valueElement.className = 'confidence-item-value';
            valueElement.textContent = `${confidencePercent}%`;
            
            barContainer.appendChild(bar);
            
            confidenceItem.appendChild(classNameElement);
            confidenceItem.appendChild(barContainer);
            confidenceItem.appendChild(valueElement);
            
            allConfidencesContainer.appendChild(confidenceItem);
        });
        
        // Show the all confidences section
        const allConfidencesSection = document.querySelector('.all-confidences-section');
        if (allConfidencesSection) {
            allConfidencesSection.style.display = 'block';
            console.log('All confidences section displayed');
        }
    }
    
    // Make handleFile function available globally for the microphone handler
    window.handleFile = handleFile;
});