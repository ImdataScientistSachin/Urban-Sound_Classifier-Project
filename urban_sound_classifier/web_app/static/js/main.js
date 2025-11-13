document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const predictBtn = document.getElementById('predictBtn');
    const resultsSection = document.getElementById('resultsSection');
    const predictionsList = document.getElementById('predictionsList');
    const spectrogramImg = document.getElementById('spectrogramImg');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const newPredictionBtn = document.getElementById('newPredictionBtn');
    
    // File handling
    let selectedFile = null;
    let currentAudioPath = null;
    
    // Event listeners for drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('dragover');
    }
    
    function unhighlight() {
        dropArea.classList.remove('dragover');
    }
    
    // Handle file drop
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFiles(files[0]);
        }
    }
    
    // Handle file selection via input
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFiles(this.files[0]);
        }
    });
    
    // Process the selected file
    function handleFiles(file) {
        // Check if file is an audio file
        const validTypes = ['.wav', '.mp3', '.ogg', '.flac'];
        const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        
        if (!validTypes.includes(fileExtension)) {
            alert('Please select a valid audio file (WAV, MP3, OGG, or FLAC)');
            return;
        }
        
        // Update UI
        selectedFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        predictBtn.disabled = false;
        
        // Play the selected audio file
        playSelectedAudio(file);
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' bytes';
        } else if (bytes < 1048576) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / 1048576).toFixed(1) + ' MB';
        }
    }
    
    // Play the selected audio file
    function playSelectedAudio(file) {
        // Get or create audio player element
        let audioPlayer = document.getElementById('audioPlayer');
        let audioElement = document.getElementById('audioElement');
        
        if (!audioPlayer) {
            console.error('Audio player element not found');
            return;
        }
        
        if (!audioElement) {
            console.error('Audio element not found');
            return;
        }
        
        // Create object URL for the file
        const audioURL = URL.createObjectURL(file);
        
        // Set the audio source
        audioElement.src = audioURL;
        
        // Update audio info
        const audioInfo = document.getElementById('audioInfo');
        if (audioInfo) {
            audioInfo.textContent = `${file.name} (${formatFileSize(file.size)})`;
        }
        
        // Show audio player
        audioPlayer.classList.remove('hidden');
        
        // Play the audio
        audioElement.play();
    }
    
    // Handle prediction
    predictBtn.addEventListener('click', function() {
        if (!selectedFile) {
            alert('Please select an audio file first');
            return;
        }
        
        // Show loading overlay
        loadingOverlay.classList.add('active');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Send request to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || `Server error (${response.status}): Please try a different audio file`);
                });
            }
            return response.json();
        })
        .then(data => {
            // Process prediction results
            processPredictionResults(data);
            
            // Store the audio path for playback
            currentAudioPath = `/uploads/${data.audio_filename}`;
            const playResultBtn = document.getElementById('play-result-btn');
            if (playResultBtn) {
                playResultBtn.setAttribute('data-audio-path', currentAudioPath);
            }
        })
        .catch(error => {
            // Hide loading overlay
            loadingOverlay.classList.remove('active');
            
            // Show error
            console.error('Prediction error:', error);
            
            // Create error message element
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
            
            // Add to results section
            const resultsContainer = document.querySelector('.results-container');
            resultsContainer.innerHTML = '';
            resultsContainer.appendChild(errorDiv);
            
            // Show results section with error
            resultsSection.classList.remove('hidden');
            
            // Show new prediction button
            newPredictionBtn.classList.remove('hidden');
        });
    });
    
    // Handle new prediction button
    newPredictionBtn.addEventListener('click', function() {
        // Reset UI
        selectedFile = null;
        fileInfo.textContent = '';
        predictBtn.disabled = true;
        resultsSection.classList.add('hidden');
        
        // Scroll to upload section
        document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
    });
});

/**
 * Process prediction results
 * This function is used by both main.js and audio.js
 */
function processPredictionResults(data) {
    try {
        // Always hide loading overlays first to ensure UI is responsive
        // even if there's an error processing the data
        hideAllLoadingOverlays();
        
        const resultsSection = document.getElementById('resultsSection');
        const predictionsList = document.getElementById('predictionsList');
        const spectrogramImg = document.getElementById('spectrogramImg');
        const topPrediction = document.getElementById('topPrediction');
        const newPredictionBtn = document.getElementById('newPredictionBtn');
        
        if (!resultsSection || !predictionsList) {
            console.error('Required DOM elements not found');
            throw new Error('Required DOM elements for displaying results not found');
        }
        
        // Clear previous results
        predictionsList.innerHTML = '';
        if (topPrediction) topPrediction.innerHTML = '';
        
        // Validate data structure
        if (!data) {
            throw new Error('No data received from server');
        }
        
        if (!data.predictions || !Array.isArray(data.predictions)) {
            console.error('Invalid data format:', data);
            const errorItem = document.createElement('div');
            errorItem.className = 'prediction-item error';
            errorItem.textContent = 'Error: Invalid response format from server';
            predictionsList.appendChild(errorItem);
            resultsSection.classList.remove('hidden');
            // Show new prediction button
            if (newPredictionBtn) newPredictionBtn.classList.remove('hidden');
            return;
        }
        
        // Make sure we have predictions
        if (data.predictions.length === 0) {
            const noResultsItem = document.createElement('div');
            noResultsItem.className = 'prediction-item info';
            noResultsItem.textContent = 'No sound classifications found. Try a different audio file.';
            predictionsList.appendChild(noResultsItem);
            resultsSection.classList.remove('hidden');
            // Show new prediction button
            if (newPredictionBtn) newPredictionBtn.classList.remove('hidden');
            return;
        }
        
        // Display top prediction if available
        if (data.predictions.length > 0 && topPrediction) {
            const topResult = data.predictions[0];
            // Safely parse probability value with fallbacks
            let probValue = 0;
            try {
                probValue = parseFloat(topResult.probability);
                if (isNaN(probValue) || probValue < 0 || probValue > 1) {
                    console.warn('Invalid probability value:', topResult.probability);
                    probValue = 0;
                }
            } catch (e) {
                console.warn('Error parsing probability:', e);
            }
            
            const percentValue = (probValue * 100).toFixed(1);
            
            // Create top prediction display
            const topPredictionContent = document.createElement('div');
            topPredictionContent.className = 'top-prediction-content';
            
            // Add icon based on sound type - normalize label to handle different formats
            const normalizedLabel = (topResult.label || '').toLowerCase().replace(/[\s-_]/g, '_');
            
            const iconMap = {
                'air_conditioner': 'fa-fan',
                'car_horn': 'fa-car',
                'children_playing': 'fa-child',
                'dog_bark': 'fa-dog',
                'drilling': 'fa-tools',
                'engine_idling': 'fa-car-side',
                'gun_shot': 'fa-bullseye',
                'jackhammer': 'fa-hammer',
                'siren': 'fa-ambulance',
                'street_music': 'fa-music'
            };
            
            // Try to find icon by normalized label or parts of the label
            let iconClass = null;
            for (const [key, value] of Object.entries(iconMap)) {
                if (normalizedLabel.includes(key)) {
                    iconClass = value;
                    break;
                }
            }
            
            // Default icon if no match found
            if (!iconClass) iconClass = 'fa-volume-up';
            
            topPredictionContent.innerHTML = `
                <div class="top-prediction-icon">
                    <i class="fas ${iconClass}"></i>
                </div>
                <div class="top-prediction-details">
                    <div class="top-prediction-label">${topResult.label || 'Unknown'}</div>
                    <div class="top-prediction-confidence">
                        <div class="confidence-text">Confidence: ${percentValue}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            `;
            
            topPrediction.appendChild(topPredictionContent);
            
            // Animate the confidence bar
            setTimeout(() => {
                const confidenceBar = topPrediction.querySelector('.confidence-bar-fill');
                if (confidenceBar) {
                    confidenceBar.style.width = `${percentValue}%`;
                    
                    // Add color class based on confidence level
                    if (probValue >= 0.7) {
                        confidenceBar.classList.add('high-confidence');
                    } else if (probValue >= 0.4) {
                        confidenceBar.classList.add('medium-confidence');
                    } else {
                        confidenceBar.classList.add('low-confidence');
                    }
                }
            }, 100);
        }
        
        // Display all predictions
        data.predictions.forEach((prediction, index) => {
            if (!prediction || typeof prediction !== 'object') {
                console.warn('Invalid prediction item:', prediction);
                return;
            }
            
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item';
            predictionItem.style.animationDelay = `${index * 0.1}s`;
            
            const labelProb = document.createElement('div');
            labelProb.className = 'label-prob';
            
            const label = document.createElement('div');
            label.className = 'prediction-label';
            label.textContent = prediction.label || 'Unknown';
            
            const probability = document.createElement('div');
            probability.className = 'prediction-probability';
            // Format percentage with % symbol, handle NaN
            const probValue = parseFloat(prediction.probability) || 0;
            probability.textContent = (probValue * 100).toFixed(1) + '%';
            
            labelProb.appendChild(label);
            labelProb.appendChild(probability);
            
            const bar = document.createElement('div');
            bar.className = 'prediction-bar';
            
            const barFill = document.createElement('div');
            barFill.className = 'prediction-bar-fill';
            barFill.style.width = '0%';
            
            // Add color class based on probability
            if (probValue >= 0.7) {
                barFill.classList.add('high-confidence');
            } else if (probValue >= 0.4) {
                barFill.classList.add('medium-confidence');
            } else {
                barFill.classList.add('low-confidence');
            }
            
            bar.appendChild(barFill);
            
            predictionItem.appendChild(labelProb);
            predictionItem.appendChild(bar);
            
            predictionsList.appendChild(predictionItem);
            
            // Animate bar fill
            setTimeout(() => {
                barFill.style.width = (probValue * 100) + '%';
            }, 100 * index);
        });
        
        // Display spectrogram with error handling
        if (spectrogramImg) {
            if (data.spectrogram) {
                spectrogramImg.src = '/spectrograms/' + data.spectrogram;
                spectrogramImg.onerror = function() {
                    console.warn('Failed to load spectrogram image');
                    spectrogramImg.src = '/static/img/spectrogram-placeholder.png';
                    spectrogramImg.alt = 'Spectrogram unavailable';
                };
            } else {
                console.warn('No spectrogram data available');
                spectrogramImg.src = '/static/img/spectrogram-placeholder.png';
                spectrogramImg.alt = 'Spectrogram unavailable';
            }
        }
        
        // Show results section
        resultsSection.classList.remove('hidden');
        
        // Show new prediction button
        if (newPredictionBtn) newPredictionBtn.classList.remove('hidden');
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Error processing prediction results:', error);
        // Ensure loading overlays are hidden
        hideAllLoadingOverlays();
        
        // Display error in results section instead of alert
        const resultsContainer = document.querySelector('.results-container');
        if (resultsContainer) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
            
            resultsContainer.innerHTML = '';
            resultsContainer.appendChild(errorDiv);
            
            // Show results section with error
            if (resultsSection) resultsSection.classList.remove('hidden');
            
            // Show new prediction button
            if (newPredictionBtn) newPredictionBtn.classList.remove('hidden');
        } else {
            // Fallback to alert if results container not found
            alert('Error processing prediction results: ' + error.message);
        }
    }
}

/**
 * Helper function to hide all loading overlays
 * This ensures UI remains responsive even when errors occur
 */
function hideAllLoadingOverlays() {
    try {
        // Hide main loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('active');
        }
        
        // Hide any other loading overlays that might be present
        const allLoadingOverlays = document.querySelectorAll('.loading-overlay');
        allLoadingOverlays.forEach(overlay => {
            overlay.classList.remove('active');
        });
        
        // Also hide any spinners or loading indicators
        const loadingSpinners = document.querySelectorAll('.spinner, .loading-indicator');
        loadingSpinners.forEach(spinner => {
            spinner.style.display = 'none';
        });
    } catch (e) {
        console.error('Error hiding loading overlays:', e);
        // Last resort attempt to hide by class
        try {
            document.querySelectorAll('[class*="loading"]').forEach(el => {
                el.style.display = 'none';
            });
        } catch (innerError) {
            console.error('Failed final attempt to hide loading elements:', innerError);
        }
    }
    
    // Hide alternative loading overlay
    const altLoadingOverlay = document.querySelector('.loading-overlay');
    if (altLoadingOverlay) {
        altLoadingOverlay.classList.remove('show');
    }
}