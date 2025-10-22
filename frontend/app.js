// AI Navigation Assistant Frontend JavaScript
// 
// ARCHITECTURE NOTE:
// This frontend (port 3000) is a PREVIEW/CONTROL interface only.
// It does NOT request camera or microphone permissions.
// 
// Media capture is handled by the client device (port 3001).
// This frontend only:
// - Displays processed video stream from backend
// - Provides navigation controls (start/stop)
// - Shows system status and monitoring
// - Communicates via WebSocket for commands

// Authentication System
class AuthManager {
    constructor() {
        this.isAuthenticated = false;
        this.accessCode = null;
        this.encryptedCode = this.hashCode("GreenMean"); // Encrypted access code
    }

    // Simple hash function for code encryption
    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash).toString(36);
    }

    validateCode(inputCode) {
        const hashedInput = this.hashCode(inputCode);
        return hashedInput === this.encryptedCode;
    }

    authenticate(code) {
        if (this.validateCode(code)) {
            this.isAuthenticated = true;
            this.accessCode = code;
            localStorage.setItem('nav_auth', this.encryptedCode);
            return true;
        }
        return false;
    }

    checkStoredAuth() {
        const stored = localStorage.getItem('nav_auth');
        if (stored === this.encryptedCode) {
            this.isAuthenticated = true;
            return true;
        }
        return false;
    }

    logout() {
        this.isAuthenticated = false;
        this.accessCode = null;
        localStorage.removeItem('nav_auth');
    }
}

class WebSocketManager {
    constructor(url, onMessage, onStateChange, onConnectionChange) {
        this.url = url;
        this.websocket = null;
        this.onMessage = onMessage;
        this.onStateChange = onStateChange;
        this.onConnectionChange = onConnectionChange;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.isConnecting = false;
        this.shouldReconnect = true;
    }

    connect() {
        if (this.isConnecting || (this.websocket && this.websocket.readyState === WebSocket.CONNECTING)) {
            return;
        }

        this.isConnecting = true;
        this.onConnectionChange('connecting');

        try {
            this.websocket = new WebSocket(this.url);
            this.setupEventListeners();
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.handleConnectionError();
        }
    }

    setupEventListeners() {
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            this.onConnectionChange('connected');
        };

        this.websocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.onMessage(message);
                
                // Handle state changes
                if (message.state) {
                    this.onStateChange(message.state, message);
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnecting = false;
            this.onConnectionChange('disconnected');
            
            if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError();
        };
    }

    handleConnectionError() {
        this.isConnecting = false;
        this.onConnectionChange('disconnected');
        
        // Enhanced error logging with connection details
        console.error('WebSocket connection error occurred', {
            url: this.url,
            readyState: this.websocket ? this.websocket.readyState : 'null',
            reconnectAttempts: this.reconnectAttempts,
            timestamp: new Date().toISOString()
        });
        
        // Implement circuit breaker pattern for connection failures
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached - entering circuit breaker mode');
            this.onConnectionChange('failed');
            this.handleConnectionFailure();
        }
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        
        // Enhanced exponential backoff with jitter and circuit breaker
        const baseDelay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        const jitter = Math.random() * 1000; // Add randomness to prevent thundering herd
        const delay = Math.min(baseDelay + jitter, 30000); // Cap at 30 seconds
        
        console.log(`Scheduling reconnection in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.onConnectionChange('reconnecting');
        
        // Store timeout ID for potential cancellation
        this.reconnectTimeoutId = setTimeout(() => {
            if (this.shouldReconnect) {
                this.connect();
            }
        }, delay);
    }

    handleConnectionFailure() {
        console.error('WebSocket connection failed permanently - implementing emergency protocols');
        this.shouldReconnect = false;
        
        // Notify the application about permanent connection failure
        if (this.onConnectionChange) {
            this.onConnectionChange('failed');
        }
        
        // Implement emergency fallback protocols
        this.activateEmergencyMode();
        
        // Show comprehensive error message with recovery options
        this.showConnectionError('Connection to server lost. Navigation system is now in emergency mode.');
    }

    activateEmergencyMode() {
        console.warn('Activating emergency mode due to connection failure');
        
        // Store emergency state
        this.emergencyMode = true;
        
        // Attempt to preserve critical functionality
        try {
            // Stop any ongoing audio feedback to prevent confusion
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            
            // Provide emergency audio notification if possible
            if (window.speechSynthesis && window.speechSynthesis.speak) {
                const emergencyMessage = new SpeechSynthesisUtterance(
                    'Connection lost. Navigation system offline. Please stop and seek assistance.'
                );
                emergencyMessage.rate = 1.0;
                emergencyMessage.volume = 1.0;
                emergencyMessage.pitch = 1.2;
                window.speechSynthesis.speak(emergencyMessage);
            }
            
            // Log emergency state for debugging
            console.error('Emergency mode activated', {
                timestamp: new Date().toISOString(),
                reconnectAttempts: this.reconnectAttempts,
                lastError: this.lastError || 'Unknown'
            });
            
        } catch (error) {
            console.error('Error during emergency mode activation:', error);
        }
    }

    // Enhanced error recovery with health check
    async performHealthCheck() {
        try {
            // Extract base URL from WebSocket URL
            const baseUrl = this.url.replace('ws://', 'http://').replace('wss://', 'https://').replace('/ws', '');
            
            const response = await fetch(`${baseUrl}/health`, {
                method: 'GET',
                timeout: 5000,
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (response.ok) {
                const healthData = await response.json();
                console.log('Server health check passed:', healthData);
                return true;
            } else {
                console.warn('Server health check failed:', response.status);
                return false;
            }
        } catch (error) {
            console.error('Health check error:', error);
            return false;
        }
    }

    // Enhanced connection retry with health check
    async retryConnection() {
        console.log('Manual connection retry initiated with health check');
        
        // Cancel any pending reconnection attempts
        if (this.reconnectTimeoutId) {
            clearTimeout(this.reconnectTimeoutId);
            this.reconnectTimeoutId = null;
        }
        
        // Perform health check before attempting reconnection
        const serverHealthy = await this.performHealthCheck();
        
        if (serverHealthy) {
            this.reconnectAttempts = 0;
            this.shouldReconnect = true;
            this.emergencyMode = false;
            this.connect();
        } else {
            this.showConnectionError('Server is not responding. Please check server status and try again later.');
        }
    }

    showConnectionError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'connection-error-notification';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #f44336;
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            font-weight: bold;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        `;
        errorDiv.textContent = message;
        
        // Add retry button
        const retryBtn = document.createElement('button');
        retryBtn.textContent = 'Retry Connection';
        retryBtn.style.cssText = `
            margin-left: 15px;
            padding: 5px 10px;
            background: white;
            color: #f44336;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        `;
        retryBtn.onclick = () => {
            this.retryConnection();
            document.body.removeChild(errorDiv);
        };
        
        errorDiv.appendChild(retryBtn);
        document.body.appendChild(errorDiv);
    }

    retryConnection() {
        console.log('Retrying WebSocket connection...');
        this.reconnectAttempts = 0;
        this.shouldReconnect = true;
        this.connect();
    }

    sendCommand(command, data = {}) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            const message = { command, ...data };
            this.websocket.send(JSON.stringify(message));
            return true;
        } else {
            console.warn('WebSocket not connected, cannot send command:', command);
            return false;
        }
    }

    disconnect() {
        this.shouldReconnect = false;
        if (this.websocket) {
            this.websocket.close();
        }
    }

    isConnected() {
        return this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
}

class AudioFeedbackSystem {
    constructor() {
        this.speechSynthesis = window.speechSynthesis;
        this.currentLanguage = 'en-US';
        this.isSupported = 'speechSynthesis' in window;
        this.urgencyQueue = [];
        this.isSpeaking = false;
        
        if (!this.isSupported) {
            console.warn('Speech Synthesis API not supported in this browser');
        }
        
        this.init();
    }

    init() {
        // Wait for voices to be loaded
        if (this.isSupported) {
            this.speechSynthesis.onvoiceschanged = () => {
                this.voices = this.speechSynthesis.getVoices();
                console.log('Available voices loaded:', this.voices.length);
            };
        }
    }

    speak(text, urgency = 'normal') {
        if (!this.isSupported || !text) {
            console.warn('Cannot speak: API not supported or no text provided');
            this.handleTTSFailure(text, urgency);
            return false;
        }

        try {
            const utterance = new SpeechSynthesisUtterance(text);
            this.configureUtterance(utterance, urgency);

            // Enhanced error handling for TTS failures
            utterance.onerror = (event) => {
                console.error('Speech synthesis error:', event.error);
                this.handleTTSFailure(text, urgency, event.error);
                this.isSpeaking = false;
                this.processQueue();
            };

            // Handle urgency-based priority with fail-safe measures
            if (urgency === 'urgent' || urgency === 'emergency') {
                // Stop current speech for urgent messages
                this.speechSynthesis.cancel();
                this.urgencyQueue = []; // Clear queue for emergency
                this.isSpeaking = false;
            }

            if (urgency === 'emergency') {
                // Immediate playback for emergency messages with fail-safe
                this.speakWithFailSafe(utterance, text, urgency);
                this.isSpeaking = true;
            } else if (urgency === 'urgent') {
                // High priority - add to front of queue
                this.urgencyQueue.unshift({ utterance, urgency, originalText: text });
                this.processQueue();
            } else {
                // Normal priority - add to end of queue
                this.urgencyQueue.push({ utterance, urgency, originalText: text });
                this.processQueue();
            }

            return true;
            
        } catch (error) {
            console.error('Error in speech synthesis:', error);
            this.handleTTSFailure(text, urgency, error.message);
            return false;
        }
    }

    speakWithFailSafe(utterance, originalText, urgency) {
        try {
            // Set up timeout for TTS failure detection
            const ttsTimeout = setTimeout(() => {
                if (this.isSpeaking) {
                    console.error('TTS timeout detected - activating fail-safe');
                    this.handleTTSFailure(originalText, urgency, 'TTS timeout');
                }
            }, 5000); // 5 second timeout

            utterance.onstart = () => {
                clearTimeout(ttsTimeout);
                this.isSpeaking = true;
            };

            utterance.onend = () => {
                clearTimeout(ttsTimeout);
                this.isSpeaking = false;
                this.processQueue();
            };

            this.speechSynthesis.speak(utterance);
            
        } catch (error) {
            console.error('Error in fail-safe speech:', error);
            this.handleTTSFailure(originalText, urgency, error.message);
        }
    }

    handleTTSFailure(text, urgency, error = 'Unknown error') {
        console.error(`TTS failure for "${text}" (${urgency}): ${error}`);
        
        // Implement fail-safe audio message system
        this.activateFailSafeAudio(text, urgency);
        
        // Show visual alert as backup
        this.showVisualFailSafeAlert(text, urgency);
        
        // Log failure for monitoring
        this.logTTSFailure(text, urgency, error);
    }

    activateFailSafeAudio(text, urgency) {
        try {
            // Try alternative TTS approach
            if (window.speechSynthesis && window.speechSynthesis.getVoices().length > 0) {
                console.log('Attempting alternative TTS approach');
                
                // Use different voice or settings
                const fallbackUtterance = new SpeechSynthesisUtterance(text);
                fallbackUtterance.rate = 1.0;
                fallbackUtterance.pitch = 1.0;
                fallbackUtterance.volume = 1.0;
                
                // Try with first available voice
                const voices = window.speechSynthesis.getVoices();
                if (voices.length > 0) {
                    fallbackUtterance.voice = voices[0];
                }
                
                window.speechSynthesis.speak(fallbackUtterance);
                console.log('Alternative TTS activated');
                return;
            }
            
            // If TTS completely fails, use audio beeps for critical messages
            if (urgency === 'emergency' || urgency === 'urgent') {
                this.playEmergencyBeeps(urgency);
            }
            
        } catch (error) {
            console.error('Fail-safe audio activation failed:', error);
            // Last resort: just show visual alert
        }
    }

    playEmergencyBeeps(urgency) {
        try {
            // Create audio context for emergency beeps
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            const beepCount = urgency === 'emergency' ? 5 : 3;
            const frequency = urgency === 'emergency' ? 1000 : 800; // Hz
            
            for (let i = 0; i < beepCount; i++) {
                setTimeout(() => {
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.value = frequency;
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                    
                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + 0.3);
                    
                }, i * 400); // 400ms between beeps
            }
            
            console.log(`Emergency beeps activated (${beepCount} beeps)`);
            
        } catch (error) {
            console.error('Emergency beeps failed:', error);
        }
    }

    showVisualFailSafeAlert(text, urgency) {
        // Create prominent visual alert for TTS failures
        const alertDiv = document.createElement('div');
        alertDiv.className = `tts-failure-alert ${urgency}`;
        alertDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: ${urgency === 'emergency' ? '#f44336' : '#ff9800'};
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1.2rem;
            z-index: 2000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            text-align: center;
            max-width: 80%;
            animation: pulse 1s infinite;
        `;
        
        alertDiv.innerHTML = `
            <div style="margin-bottom: 10px;">⚠️ AUDIO SYSTEM FAILURE ⚠️</div>
            <div style="font-size: 1rem;">${text}</div>
            <div style="margin-top: 15px; font-size: 0.9rem;">
                Audio feedback unavailable - Please read this message
            </div>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Remove alert after extended time for critical messages
        const displayTime = urgency === 'emergency' ? 10000 : 5000;
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, displayTime);
        
        // Add CSS animation if not already present
        if (!document.getElementById('tts-failure-styles')) {
            const style = document.createElement('style');
            style.id = 'tts-failure-styles';
            style.textContent = `
                @keyframes pulse {
                    0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                    50% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.05); }
                    100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    logTTSFailure(text, urgency, error) {
        const failureLog = {
            timestamp: new Date().toISOString(),
            text: text,
            urgency: urgency,
            error: error,
            userAgent: navigator.userAgent,
            speechSynthesisSupported: 'speechSynthesis' in window,
            voicesAvailable: window.speechSynthesis ? window.speechSynthesis.getVoices().length : 0
        };
        
        console.error('TTS FAILURE LOG:', failureLog);
        
        // Could send to server for monitoring if needed
        // this.sendTTSFailureToServer(failureLog);
    }

    configureUtterance(utterance, urgency) {
        // Set language
        utterance.lang = this.currentLanguage;
        
        // Find appropriate voice for current language
        const voice = this.findVoiceForLanguage(this.currentLanguage);
        if (voice) {
            utterance.voice = voice;
        }

        // Configure based on urgency
        switch (urgency) {
            case 'emergency':
                utterance.rate = 1.1;
                utterance.pitch = 1.2;
                utterance.volume = 1.0;
                break;
            case 'urgent':
                utterance.rate = 1.0;
                utterance.pitch = 1.1;
                utterance.volume = 0.9;
                break;
            case 'normal':
            default:
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                utterance.volume = 0.8;
                break;
        }

        // Set up event handlers
        utterance.onstart = () => {
            this.isSpeaking = true;
        };

        utterance.onend = () => {
            this.isSpeaking = false;
            this.processQueue();
        };

        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event.error);
            this.isSpeaking = false;
            this.processQueue();
        };
    }

    processQueue() {
        if (this.isSpeaking || this.urgencyQueue.length === 0) {
            return;
        }

        const nextItem = this.urgencyQueue.shift();
        this.speechSynthesis.speak(nextItem.utterance);
    }

    findVoiceForLanguage(language) {
        if (!this.voices || this.voices.length === 0) {
            return null;
        }

        // Try to find exact match
        let voice = this.voices.find(v => v.lang === language);
        
        // If no exact match, try language code only (e.g., 'en' from 'en-US')
        if (!voice) {
            const langCode = language.split('-')[0];
            voice = this.voices.find(v => v.lang.startsWith(langCode));
        }

        // Prefer local voices over remote ones
        if (voice && !voice.localService) {
            const localVoice = this.voices.find(v => 
                (v.lang === language || v.lang.startsWith(language.split('-')[0])) && 
                v.localService
            );
            if (localVoice) {
                voice = localVoice;
            }
        }

        return voice;
    }

    setLanguage(languageCode) {
        console.log('Setting language to:', languageCode);
        this.currentLanguage = languageCode;
        
        // Update audio status in UI
        const audioStatus = document.getElementById('audioStatus');
        if (audioStatus) {
            audioStatus.textContent = `Ready (${languageCode})`;
        }
    }

    stop() {
        if (this.isSupported) {
            this.speechSynthesis.cancel();
            this.urgencyQueue = [];
            this.isSpeaking = false;
        }
    }

    getAvailableLanguages() {
        if (!this.voices) {
            return [];
        }
        
        return [...new Set(this.voices.map(voice => voice.lang))].sort();
    }
}

class VideoDisplayController {
    constructor(onStatusChange) {
        // Use dynamic URL from configuration
        this.streamUrl = window.AI_NAV_CONFIG.getVideoStreamUrl();
        this.videoElement = document.getElementById('videoStream');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.onStatusChange = onStatusChange;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        this.reconnectDelay = 2000;
        this.shouldReconnect = true;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.videoElement.onload = () => {
            console.log('Video stream connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.hideLoadingSpinner();
            this.onStatusChange('connected', 'Video stream active');
        };

        this.videoElement.onerror = (error) => {
            console.error('Video stream error:', error);
            this.handleStreamError();
        };

        this.videoElement.onabort = () => {
            console.log('Video stream aborted');
            this.handleStreamDisconnection();
        };
    }

    connectToStream() {
        if (this.isConnected) {
            console.log('Video stream already connected');
            return;
        }

        console.log('Connecting to video stream:', this.streamUrl);
        this.showLoadingSpinner();
        this.onStatusChange('connecting', 'Connecting to video stream...');
        
        // Switch to processed stream view
        this.switchToProcessedStream();
        
        // Add timestamp to prevent caching issues
        const timestampedUrl = `${this.streamUrl}?t=${Date.now()}`;
        this.videoElement.src = timestampedUrl;
    }

    switchToProcessedStream() {
        const videoModeIndicator = document.getElementById('videoModeIndicator');
        const videoModeText = document.getElementById('videoModeText');
        
        // Show processed stream
        this.videoElement.classList.remove('hidden');
        
        // Update mode indicator
        videoModeIndicator.classList.remove('hidden');
        videoModeText.textContent = 'Processed Stream';
    }

    disconnect() {
        this.shouldReconnect = false;
        this.isConnected = false;
        this.videoElement.src = '';
        this.hideLoadingSpinner();
        
        // Hide video elements and mode indicator
        this.videoElement.classList.add('hidden');
        const videoModeIndicator = document.getElementById('videoModeIndicator');
        videoModeIndicator.classList.add('hidden');
        
        this.onStatusChange('disconnected', 'Video stream disconnected');
    }

    handleStreamError() {
        this.isConnected = false;
        this.hideLoadingSpinner();
        this.onStatusChange('error', 'Video stream error');
        
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        } else {
            this.onStatusChange('error', 'Video stream failed - max retries reached');
        }
    }

    handleStreamDisconnection() {
        this.isConnected = false;
        this.onStatusChange('disconnected', 'Video stream disconnected');
        
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * this.reconnectAttempts;
        
        console.log(`Attempting to reconnect video stream in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.onStatusChange('warning', `Reconnecting video... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (this.shouldReconnect) {
                this.connectToStream();
            }
        }, delay);
    }

    showLoadingSpinner() {
        this.loadingSpinner.classList.remove('hidden');
    }

    hideLoadingSpinner() {
        this.loadingSpinner.classList.add('hidden');
    }

    isStreamConnected() {
        return this.isConnected;
    }

    resetReconnectAttempts() {
        this.reconnectAttempts = 0;
        this.shouldReconnect = true;
    }
}

// WebRTC functionality removed from frontend - this is handled by the client device
// Frontend is now purely for preview and control

class NavigationApp {
    constructor() {
        this.authManager = new AuthManager();
        this.websocketManager = null;
        this.audioSystem = null;
        this.videoController = null;
        this.isNavigating = false;
        this.currentState = 'idle';
        this.init();
    }

    init() {
        // Check if user is already authenticated
        if (this.authManager.checkStoredAuth()) {
            this.hideAuthModal();
            this.initializeApp();
        } else {
            this.showAuthModal();
        }
        this.setupAuthEventListeners();
    }

    showAuthModal() {
        const modal = document.getElementById('authModal');
        if (modal) {
            modal.style.display = 'flex';
            document.getElementById('authCodeInput').focus();
        }
    }

    hideAuthModal() {
        const modal = document.getElementById('authModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    setupAuthEventListeners() {
        const authSubmitBtn = document.getElementById('authSubmitBtn');
        const authCodeInput = document.getElementById('authCodeInput');
        const authError = document.getElementById('authError');

        const handleAuth = () => {
            const code = authCodeInput.value.trim();
            if (this.authManager.authenticate(code)) {
                authError.classList.add('hidden');
                this.hideAuthModal();
                this.initializeApp();
            } else {
                authError.classList.remove('hidden');
                authCodeInput.value = '';
                authCodeInput.focus();
                // Add shake animation
                authCodeInput.style.animation = 'shake 0.5s ease-in-out';
                setTimeout(() => {
                    authCodeInput.style.animation = '';
                }, 500);
            }
        };

        authSubmitBtn.addEventListener('click', handleAuth);
        authCodeInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleAuth();
            }
        });
    }

    initializeApp() {
        this.setupEventListeners();
        this.initializeAudioSystem();
        this.initializeVideoController();
        this.initializeWebSocket();
        this.updateUI();
    }

    initializeVideoController() {
        this.videoController = new VideoDisplayController(
            (status, message) => this.handleVideoStatusChange(status, message)
        );
    }

    initializeAudioSystem() {
        this.audioSystem = new AudioFeedbackSystem();
        
        // Update audio status based on support
        const audioStatus = document.getElementById('audioStatus');
        if (this.audioSystem.isSupported) {
            audioStatus.textContent = 'Ready';
            audioStatus.className = 'status-value ready';
        } else {
            audioStatus.textContent = 'Not Supported';
            audioStatus.className = 'status-value error';
        }
    }

    setupEventListeners() {
        const startStopBtn = document.getElementById('startStopBtn');
        startStopBtn.addEventListener('click', () => this.toggleNavigation());
    }

    initializeWebSocket() {
        // Use dynamic URL from configuration
        const wsUrl = window.AI_NAV_CONFIG.getWebSocketUrl();
        
        this.websocketManager = new WebSocketManager(
            wsUrl,
            (message) => this.handleWebSocketMessage(message),
            (state, message) => this.handleStateChange(state, message),
            (status) => this.handleConnectionChange(status)
        );

        this.websocketManager.connect();
    }

    handleWebSocketMessage(message) {
        console.log('Received message:', message);
        
        // Handle different message types
        if (message.speak) {
            this.handleAudioMessage(message.speak, message.state);
        }
        
        if (message.set_lang) {
            this.audioSystem.setLanguage(message.set_lang);
        }
    }

    handleAudioMessage(text, currentState) {
        // Always show visual feedback for monitoring
        this.showVisualFeedback(text, currentState);
        
        // Only play audio guidance when navigation is active
        if (!this.isNavigating) {
            console.log('Navigation not active - audio guidance muted:', text);
            return;
        }
        
        if (!this.audioSystem.isSupported) {
            console.warn('Audio feedback not available - displaying text instead:', text);
            this.showTextFallback(text);
            return;
        }

        // Determine urgency based on message content and state
        let urgency = 'normal';
        
        if (currentState === 'blocked' || text.toLowerCase().includes('stop') || text.toLowerCase().includes('danger')) {
            urgency = 'emergency';
        } else if (text.toLowerCase().includes('caution') || text.toLowerCase().includes('careful')) {
            urgency = 'urgent';
        }

        // Speak the message with appropriate urgency
        this.audioSystem.speak(text, urgency);
        
        // Also show visual feedback for important messages
        if (urgency === 'emergency' || urgency === 'urgent') {
            this.showVisualAlert(text, urgency);
        }
    }
    
    showVisualFeedback(text, currentState) {
        // Always show visual feedback in the status area for monitoring
        const systemStatus = document.getElementById('systemStatus');
        const originalText = systemStatus.textContent;
        const originalClass = systemStatus.className;
        
        // Determine visual style based on state and content
        let statusClass = 'status-value ready';
        if (currentState === 'blocked' || text.toLowerCase().includes('stop') || text.toLowerCase().includes('danger')) {
            statusClass = 'status-value error';
        } else if (text.toLowerCase().includes('caution') || text.toLowerCase().includes('careful')) {
            statusClass = 'status-value warning';
        }
        
        systemStatus.textContent = text;
        systemStatus.className = statusClass;
        
        // Restore original status after 5 seconds if navigation is not active
        if (!this.isNavigating) {
            setTimeout(() => {
                systemStatus.textContent = originalText;
                systemStatus.className = originalClass;
            }, 5000);
        }
    }

    showTextFallback(text) {
        // Show text feedback when audio is not available
        const systemStatus = document.getElementById('systemStatus');
        const originalText = systemStatus.textContent;
        const originalClass = systemStatus.className;
        
        systemStatus.textContent = text;
        systemStatus.className = 'status-value warning';
        
        // Restore original status after 3 seconds
        setTimeout(() => {
            systemStatus.textContent = originalText;
            systemStatus.className = originalClass;
        }, 3000);
    }

    showVisualAlert(text, urgency) {
        // Create temporary visual alert for urgent messages
        const alertDiv = document.createElement('div');
        alertDiv.className = `visual-alert ${urgency}`;
        alertDiv.textContent = text;
        
        // Style the alert
        alertDiv.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: ${urgency === 'emergency' ? '#f44336' : '#ff9800'};
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1rem;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            animation: slideDown 0.3s ease-out;
        `;
        
        document.body.appendChild(alertDiv);
        
        // Remove alert after 4 seconds
        setTimeout(() => {
            alertDiv.style.animation = 'slideUp 0.3s ease-in';
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.parentNode.removeChild(alertDiv);
                }
            }, 300);
        }, 4000);
    }

    handleStateChange(state, message) {
        console.log('State changed to:', state);
        this.currentState = state;
        this.updateFSMDisplay(state, message);
        
        // Video stream is always available for monitoring
        // Audio guidance only plays when navigation is active (not idle)
        // No need to start/stop video stream based on state changes
    }

    handleVideoStatusChange(status, message) {
        console.log('Video status changed:', status, message);
        this.updateVideoStatus(status, message);
    }

    handleConnectionChange(status) {
        console.log('Connection status changed to:', status);
        this.updateConnectionStatus(status);
        
        // Handle connection state changes
        switch (status) {
            case 'connected':
                // Connection restored - clear any error states
                this.clearConnectionErrors();
                
                // Update system status to ready
                this.updateSystemStatus('ready', 'System ready - monitoring active');
                
                // Start video stream for monitoring (always available)
                setTimeout(() => {
                    if (!this.videoController.isStreamConnected()) {
                        this.videoController.connectToStream();
                    }
                }, 2000); // Give backend time to be ready
                break;
                
            case 'failed':
                // Permanent connection failure
                this.handlePermanentConnectionFailure();
                break;
                
            case 'reconnecting':
                // Show reconnection status
                this.showReconnectionStatus();
                break;
                
            case 'disconnected':
                // Temporary disconnection
                this.handleTemporaryDisconnection();
                break;
        }
    }

    clearConnectionErrors() {
        // Remove any existing error notifications
        const errorNotifications = document.querySelectorAll('.connection-error-notification');
        errorNotifications.forEach(notification => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }

    handlePermanentConnectionFailure() {
        // Stop navigation if active
        if (this.isNavigating) {
            this.isNavigating = false;
            const startStopBtn = document.getElementById('startStopBtn');
            startStopBtn.textContent = 'Start Navigation';
            startStopBtn.className = 'control-btn start';
            startStopBtn.disabled = true; // Disable until connection restored
        }
        
        // Stop video stream
        if (this.videoController) {
            this.videoController.disconnect();
        }
        
        this.updateSystemStatus('error', 'Connection lost - navigation disabled');
    }

    handleTemporaryDisconnection() {
        this.updateSystemStatus('warning', 'Connection lost - attempting to reconnect...');
    }

    showReconnectionStatus() {
        this.updateSystemStatus('warning', 'Reconnecting to server...');
    }

    async toggleNavigation() {
        const startStopBtn = document.getElementById('startStopBtn');
        
        if (!this.websocketManager.isConnected()) {
            this.updateSystemStatus('error', 'Not connected to server');
            return;
        }

        if (this.isNavigating) {
            // Stop navigation (but keep video stream for monitoring)
            if (this.websocketManager.sendCommand('stop')) {
                this.isNavigating = false;
                startStopBtn.textContent = 'Start Navigation';
                startStopBtn.className = 'control-btn start';
                this.updateSystemStatus('ready', 'Navigation stopped - monitoring continues');
                
                // Update audio status to show guidance is muted
                const audioStatus = document.getElementById('audioStatus');
                audioStatus.textContent = 'Ready (guidance muted)';
                audioStatus.className = 'status-value ready';
                
                // Stop audio guidance but keep video stream
                this.audioSystem.stop();
                
                // Video stream continues for monitoring purposes
            }
        } else {
            // Start navigation - audio guidance begins
            startStopBtn.disabled = true;
            startStopBtn.textContent = 'Starting...';
            
            try {
                if (this.websocketManager.sendCommand('start')) {
                    this.isNavigating = true;
                    startStopBtn.textContent = 'Stop Navigation';
                    startStopBtn.className = 'control-btn stop';
                    this.updateSystemStatus('active', 'Navigation started - audio guidance active');
                    
                    // Update audio status to show guidance is active
                    const audioStatus = document.getElementById('audioStatus');
                    audioStatus.textContent = 'Guidance Active';
                    audioStatus.className = 'status-value active';
                    
                    // Video stream should already be running for monitoring
                    // If not connected, start it now
                    if (!this.videoController.isStreamConnected()) {
                        setTimeout(() => {
                            this.videoController.connectToStream();
                        }, 1000);
                        this.videoController.resetReconnectAttempts();
                    }
                } else {
                    this.updateSystemStatus('error', 'Failed to start navigation - server communication error');
                    startStopBtn.textContent = 'Start Navigation';
                    startStopBtn.className = 'control-btn start';
                }
            } catch (error) {
                console.error('Error starting navigation:', error);
                this.updateSystemStatus('error', 'Failed to start navigation: ' + error.message);
                startStopBtn.textContent = 'Start Navigation';
                startStopBtn.className = 'control-btn start';
            } finally {
                startStopBtn.disabled = false;
            }
        }
    }

    // WebRTC status handling removed - frontend doesn't handle camera access

    updateConnectionStatus(status) {
        const indicator = document.getElementById('connectionIndicator');
        const text = document.getElementById('connectionText');
        
        indicator.className = `status-indicator ${status}`;
        
        switch (status) {
            case 'connected':
                text.textContent = 'Connected';
                this.updateSystemStatus('ready', 'System ready');
                // Re-enable navigation button
                document.getElementById('startStopBtn').disabled = false;
                break;
                
            case 'connecting':
                text.textContent = 'Connecting...';
                this.updateSystemStatus('warning', 'Connecting to server');
                break;
                
            case 'reconnecting':
                text.textContent = 'Reconnecting...';
                this.updateSystemStatus('warning', 'Reconnecting to server');
                break;
                
            case 'disconnected':
                text.textContent = 'Disconnected';
                this.updateSystemStatus('error', 'Server disconnected');
                break;
                
            case 'failed':
                text.textContent = 'Connection Failed';
                this.updateSystemStatus('error', 'Connection failed - please refresh');
                // Disable navigation button
                document.getElementById('startStopBtn').disabled = true;
                break;
                
            default:
                text.textContent = status;
                this.updateSystemStatus('warning', `Connection status: ${status}`);
        }
    }

    updateFSMDisplay(state, message) {
        const stateDisplay = document.getElementById('fsmState');
        const stateDescription = document.getElementById('fsmDescription');
        
        // Remove all state classes
        stateDisplay.className = 'state-display';
        stateDisplay.classList.add(state.toLowerCase());
        stateDisplay.textContent = state.toUpperCase();
        
        // Update description based on state
        const descriptions = {
            'idle': 'System ready to start navigation',
            'scanning': 'Scanning environment for safe path',
            'guiding': 'Providing navigation guidance',
            'blocked': 'Obstacle detected - please stop'
        };
        
        stateDescription.textContent = descriptions[state.toLowerCase()] || 'Unknown state';
    }

    updateSystemStatus(type, message) {
        const systemStatus = document.getElementById('systemStatus');
        systemStatus.textContent = message;
        systemStatus.className = `status-value ${type}`;
    }

    updateVideoStatus(status, message) {
        const videoStatus = document.getElementById('videoStatus');
        videoStatus.textContent = message;
        
        switch (status) {
            case 'connected':
                videoStatus.className = 'status-value ready';
                break;
            case 'connecting':
                videoStatus.className = 'status-value warning';
                break;
            case 'error':
                videoStatus.className = 'status-value error';
                break;
            case 'disconnected':
            default:
                videoStatus.className = 'status-value inactive';
                break;
        }
    }

    updateUI() {
        // Initialize UI state
        this.updateSystemStatus('inactive', 'Connecting to monitoring feed...');
        
        const audioStatus = document.getElementById('audioStatus');
        audioStatus.textContent = 'Ready (guidance muted)';
        audioStatus.className = 'status-value ready';
        
        this.updateVideoStatus('disconnected', 'Connecting to video stream...');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new NavigationApp();
});