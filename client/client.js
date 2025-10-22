// AI Navigation Assistant - Client Device Component

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

class NavigationClient {
    constructor() {
        this.authManager = new AuthManager();
        this.localStream = null;
        this.videoStream = null;
        this.audioStream = null;
        this.peerConnection = null;
        this.websocket = null;
        this.isConnected = false;
        this.cameraEnabled = false;
        this.micEnabled = false;
        this.biometricEnabled = false;
        this.heartRateMonitor = null;
        this.clientId = this.generateClientId(); // Generate once and reuse
        
        // Biometric sensor properties (placeholder for optional task 6.3)
        // this.availableSensors = {};
        // this.bluetoothDevice = null;
        // this.heartRateCharacteristic = null;
        
        // Reconnection settings
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        this.reconnectDelay = 2000; // 2 seconds
        
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
        this.checkBrowserSupport();
        this.setupEventListeners();
        this.updateAllStatuses();
        this.checkBiometricAvailability();
        this.setupMixedContentHandling();
    }

    setupMixedContentHandling() {
        // Check if we're on HTTPS and need to handle Mixed Content
        if (window.AI_NAV_CONFIG && window.AI_NAV_CONFIG.isHttpsPage()) {
            console.warn('🔒 HTTPS page detected - Mixed Content may be blocked');
            this.setupMixedContentWarning();
        }
    }

    setupMixedContentWarning() {
        const dismissBtn = document.getElementById('dismissWarningBtn');
        const retryBtn = document.getElementById('retryConnectionBtn');
        
        if (dismissBtn) {
            dismissBtn.addEventListener('click', () => {
                this.hideMixedContentWarning();
            });
        }
        
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                this.hideMixedContentWarning();
                // Try to connect again
                this.connectToServer();
            });
        }
    }

    showMixedContentWarning() {
        const warning = document.getElementById('mixedContentWarning');
        if (warning) {
            warning.classList.remove('hidden');
        }
    }

    hideMixedContentWarning() {
        const warning = document.getElementById('mixedContentWarning');
        if (warning) {
            warning.classList.add('hidden');
        }
    }

    checkBrowserSupport() {
        const support = {
            webrtc: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            websocket: !!window.WebSocket,
            speechSynthesis: !!window.speechSynthesis
        };
        
        const supportText = Object.entries(support)
            .map(([key, value]) => `${key}: ${value ? '✓' : '✗'}`)
            .join(', ');
            
        document.getElementById('browserSupport').textContent = supportText;
        
        if (!support.webrtc) {
            this.showError('WebRTC not supported in this browser');
        }
    }

    async createPeerConnection() {
        // ICE servers configuration
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };
        
        this.peerConnection = new RTCPeerConnection(configuration);
        
        // Set up event handlers
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.websocket) {
                this.websocket.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate
                }));
            }
        };
        
        this.peerConnection.onconnectionstatechange = () => {
            const state = this.peerConnection.connectionState;
            this.updateConnectionState(state);
            document.getElementById('connectionState').textContent = state;
            
            console.log(`WebRTC connection state changed to: ${state}`);
            
            switch (state) {
                case 'connected':
                    document.getElementById('connectionIndicator').className = 'status-indicator active';
                    this.reconnectAttempts = 0; // Reset on successful connection
                    break;
                    
                case 'connecting':
                    document.getElementById('connectionIndicator').className = 'status-indicator warning';
                    break;
                    
                case 'disconnected':
                    document.getElementById('connectionIndicator').className = 'status-indicator error';
                    this.showError('WebRTC connection lost');
                    this.handleConnectionFailure();
                    break;
                    
                case 'failed':
                    document.getElementById('connectionIndicator').className = 'status-indicator error';
                    this.showError('WebRTC connection failed');
                    this.handleConnectionFailure();
                    break;
                    
                case 'closed':
                    document.getElementById('connectionIndicator').className = 'status-indicator inactive';
                    break;
                    
                default:
                    console.log(`Unhandled WebRTC connection state: ${state}`);
            }
        };
        
        this.peerConnection.oniceconnectionstatechange = () => {
            const iceState = this.peerConnection.iceConnectionState;
            document.getElementById('iceState').textContent = iceState;
            
            if (iceState === 'failed') {
                this.handleConnectionFailure();
            }
        };
    }

    // WebSocket signaling removed - client now uses HTTP for WebRTC offers
    // This simplifies the connection process and removes WebSocket dependency

    // Signaling message handling removed - using HTTP for WebRTC negotiation

    async createAndSendOffer() {
        if (!this.peerConnection) {
            throw new Error('Peer connection not available');
        }
        
        try {
            const offer = await this.peerConnection.createOffer({
                offerToReceiveAudio: false,
                offerToReceiveVideo: false
            });
            
            await this.peerConnection.setLocalDescription(offer);
            
            // Send offer to server via HTTP POST (not WebSocket)
            const serverUrl = document.getElementById('serverUrl').value.trim();
            const httpUrl = serverUrl.replace('ws://', 'http://').replace('wss://', 'https://');
            
            console.log('Sending WebRTC offer to:', `${httpUrl}/webrtc/offer`);
            console.log('Client ID:', this.clientId);
            console.log('Offer SDP:', offer.sdp.substring(0, 100) + '...');
            
            const response = await fetch(`${httpUrl}/webrtc/offer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    client_id: this.clientId,
                    type: offer.type,
                    sdp: offer.sdp
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const answerData = await response.json();
            console.log('Received answer from server:', answerData);
            
            if (answerData.status === 'success') {
                const answer = new RTCSessionDescription({
                    type: answerData.answer.type,
                    sdp: answerData.answer.sdp
                });
                
                await this.peerConnection.setRemoteDescription(answer);
                console.log('WebRTC connection established');
                this.updateStreamStatus('Connected');
            } else {
                throw new Error('Failed to get answer from server');
            }
            
        } catch (error) {
            // Check if this is a Mixed Content error
            if (error.message.includes('Mixed Content') || 
                error.message.includes('blocked') || 
                error.message.includes('HTTPS') ||
                (window.AI_NAV_CONFIG && window.AI_NAV_CONFIG.isHttpsPage() && error.name === 'TypeError')) {
                console.error('Mixed Content error detected:', error);
                this.showMixedContentWarning();
            }
            throw new Error('Failed to create WebRTC offer: ' + error.message);
        }
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }

    handleConnectionFailure() {
        console.log('Handling connection failure...');
        
        // Stop current streaming if active
        if (this.localStream) {
            this.stopStreaming();
        }
        
        // Close existing connections
        this.cleanupConnections();
        
        // Implement comprehensive error recovery with circuit breaker pattern
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateConnectionStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            // Exponential backoff with jitter for reconnection
            const baseDelay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            const jitter = Math.random() * 1000; // Add randomness to prevent thundering herd
            const delay = Math.min(baseDelay + jitter, 30000); // Cap at 30 seconds
            
            // Store timeout ID for potential cancellation
            this.reconnectTimeoutId = setTimeout(() => {
                this.attemptReconnection();
            }, delay);
        } else {
            this.handlePermanentConnectionFailure();
        }
    }

    handlePermanentConnectionFailure() {
        console.error('Permanent connection failure detected');
        this.updateConnectionStatus('Connection failed permanently');
        document.getElementById('connectionIndicator').className = 'status-indicator error';
        
        // Show comprehensive error message with recovery options
        this.showConnectionFailureDialog();
        
        // Reset for manual retry
        this.reconnectAttempts = 0;
        this.updateConnectButtonState();
        
        // Disable streaming controls until connection restored
        this.disableStreamingControls();
    }

    showConnectionFailureDialog() {
        const errorDialog = document.createElement('div');
        errorDialog.className = 'error-dialog';
        errorDialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 3px solid #f44336;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 2000;
            max-width: 400px;
            text-align: center;
        `;
        
        errorDialog.innerHTML = `
            <h3 style="color: #f44336; margin-top: 0;">Connection Failed</h3>
            <p>Unable to connect to the navigation server after multiple attempts.</p>
            <p><strong>Possible causes:</strong></p>
            <ul style="text-align: left; margin: 15px 0;">
                <li>Server is offline or unreachable</li>
                <li>Network connectivity issues</li>
                <li>Firewall blocking connection</li>
                <li>Incorrect server URL</li>
            </ul>
            <div style="margin-top: 20px;">
                <button id="retryConnection" style="background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-right: 10px; cursor: pointer;">
                    Retry Connection
                </button>
                <button id="checkNetwork" style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-right: 10px; cursor: pointer;">
                    Check Network
                </button>
                <button id="closeDialog" style="background: #757575; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Close
                </button>
            </div>
        `;
        
        document.body.appendChild(errorDialog);
        
        // Add event listeners
        document.getElementById('retryConnection').onclick = () => {
            document.body.removeChild(errorDialog);
            this.retryConnection();
        };
        
        document.getElementById('checkNetwork').onclick = () => {
            this.performNetworkDiagnostics();
        };
        
        document.getElementById('closeDialog').onclick = () => {
            document.body.removeChild(errorDialog);
        };
    }

    async performNetworkDiagnostics() {
        console.log('Performing network diagnostics...');
        this.updateConnectionStatus('Running diagnostics...');
        
        const diagnostics = {
            online: navigator.onLine,
            serverReachable: false,
            webrtcSupport: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            websocketSupport: !!window.WebSocket
        };
        
        // Test server reachability
        try {
            const serverUrl = document.getElementById('serverUrl').value.trim();
            if (serverUrl) {
                const testUrl = serverUrl.replace(/\/$/, '') + '/health';
                const response = await fetch(testUrl, { 
                    method: 'GET',
                    timeout: 5000 
                });
                diagnostics.serverReachable = response.ok;
            }
        } catch (error) {
            console.error('Server reachability test failed:', error);
            diagnostics.serverReachable = false;
        }
        
        // Display diagnostics results
        this.showDiagnosticsResults(diagnostics);
    }

    showDiagnosticsResults(diagnostics) {
        const resultsDialog = document.createElement('div');
        resultsDialog.className = 'diagnostics-dialog';
        resultsDialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 2px solid #2196F3;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 2000;
            max-width: 400px;
        `;
        
        const getStatusIcon = (status) => status ? '✅' : '❌';
        
        resultsDialog.innerHTML = `
            <h3 style="color: #2196F3; margin-top: 0;">Network Diagnostics</h3>
            <div style="text-align: left;">
                <p>${getStatusIcon(diagnostics.online)} Internet Connection: ${diagnostics.online ? 'Online' : 'Offline'}</p>
                <p>${getStatusIcon(diagnostics.serverReachable)} Server Reachable: ${diagnostics.serverReachable ? 'Yes' : 'No'}</p>
                <p>${getStatusIcon(diagnostics.webrtcSupport)} WebRTC Support: ${diagnostics.webrtcSupport ? 'Available' : 'Not Available'}</p>
                <p>${getStatusIcon(diagnostics.websocketSupport)} WebSocket Support: ${diagnostics.websocketSupport ? 'Available' : 'Not Available'}</p>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button id="closeDiagnostics" style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Close
                </button>
            </div>
        `;
        
        document.body.appendChild(resultsDialog);
        
        document.getElementById('closeDiagnostics').onclick = () => {
            document.body.removeChild(resultsDialog);
        };
    }

    disableStreamingControls() {
        document.getElementById('startStreamBtn').disabled = true;
        document.getElementById('stopStreamBtn').disabled = true;
        document.getElementById('requestCameraBtn').disabled = true;
        document.getElementById('requestMicBtn').disabled = true;
    }

    enableStreamingControls() {
        document.getElementById('requestCameraBtn').disabled = false;
        document.getElementById('requestMicBtn').disabled = false;
        this.updateStreamButton(); // Re-enable based on current state
    }

    retryConnection() {
        console.log('Manual connection retry initiated');
        this.reconnectAttempts = 0;
        
        // Cancel any pending reconnection attempts
        if (this.reconnectTimeoutId) {
            clearTimeout(this.reconnectTimeoutId);
            this.reconnectTimeoutId = null;
        }
        
        // Re-enable controls
        this.enableStreamingControls();
        
        // Attempt connection
        this.connectToServer();
    }

    cleanupConnections() {
        try {
            // Close WebSocket connection
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
            
            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }
            
            this.isConnected = false;
            
        } catch (error) {
            console.error('Error during connection cleanup:', error);
        }
    }

    async attemptReconnection() {
        try {
            console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            
            // Clean up any existing connections first
            this.cleanupConnections();
            
            // Wait a moment before attempting reconnection
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            const serverUrl = document.getElementById('serverUrl').value.trim();
            if (!serverUrl) {
                throw new Error('Server URL is required for reconnection');
            }
            
            // Attempt to reconnect
            await this.connectToServer();
            
            // If successful, reset reconnect attempts
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('Reconnected successfully');
            
            // Restart streaming if it was active before
            if (this.cameraEnabled && this.micEnabled) {
                setTimeout(() => {
                    this.startStreaming();
                }, 2000);
            }
            
        } catch (error) {
            console.error('Reconnection failed:', error);
            this.showError(`Reconnection attempt ${this.reconnectAttempts} failed: ${error.message}`);
        }
    }

    setupAudioLevelMonitoring(audioStream) {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const microphone = audioContext.createMediaStreamSource(audioStream);
            
            analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            microphone.connect(analyser);
            
            const updateAudioLevel = () => {
                analyser.getByteFrequencyData(dataArray);
                
                // Calculate average volume
                let sum = 0;
                for (let i = 0; i < bufferLength; i++) {
                    sum += dataArray[i];
                }
                const average = sum / bufferLength;
                const percentage = (average / 255) * 100;
                
                // Update audio level indicator
                const audioLevel = document.getElementById('audioLevel');
                audioLevel.style.setProperty('--level', `${percentage}%`);
                
                if (this.micEnabled) {
                    requestAnimationFrame(updateAudioLevel);
                }
            };
            
            updateAudioLevel();
            
        } catch (error) {
            console.warn('Audio level monitoring not available:', error);
        }
    }

    setupEventListeners() {
        // Media permission buttons
        document.getElementById('requestCameraBtn').addEventListener('click', () => this.requestCameraPermission());
        document.getElementById('requestMicBtn').addEventListener('click', () => this.requestMicPermission());
        
        // Stream control buttons
        document.getElementById('startStreamBtn').addEventListener('click', () => this.startStreaming());
        document.getElementById('stopStreamBtn').addEventListener('click', () => this.stopStreaming());
        
        // Connection buttons
        document.getElementById('connectBtn').addEventListener('click', () => this.connectToServer());
        document.getElementById('disconnectBtn').addEventListener('click', () => this.disconnectFromServer());
        
        // Biometric button
        document.getElementById('enableBiometricBtn').addEventListener('click', () => this.enableBiometrics());
        
        // Server URL input
        document.getElementById('serverUrl').addEventListener('input', (e) => {
            this.updateConnectButtonState();
        });
    }

    async requestCameraPermission() {
        try {
            this.updateCameraStatus('Requesting permission...');
            
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access not supported in this browser');
            }
            
            // Request camera access using getUserMedia
            const videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                }
            });
            
            // Store video stream
            this.videoStream = videoStream;
            this.cameraEnabled = true;
            
            // Display video in the video element
            const videoElement = document.getElementById('localVideo');
            videoElement.srcObject = videoStream;
            
            // Add error handling for video element
            videoElement.onerror = (error) => {
                console.error('Video element error:', error);
                this.showError('Video display error');
                this.handleMediaStreamError('video');
            };
            
            // Monitor stream for track ending
            videoStream.getVideoTracks().forEach(track => {
                track.onended = () => {
                    console.log('Video track ended');
                    this.handleMediaStreamError('video');
                };
            });
            
            this.updateCameraStatus('Ready');
            this.updateStreamButton();
            document.getElementById('cameraIndicator').className = 'status-indicator ready';
            document.getElementById('videoOverlay').classList.add('hidden');
            
        } catch (error) {
            console.error('Camera permission error:', error);
            
            let errorMessage = 'Camera access failed';
            
            // Provide specific error messages based on error type
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Camera permission denied by user';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No camera device found';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera is already in use by another application';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = 'Camera does not support requested settings';
            } else if (error.name === 'SecurityError') {
                errorMessage = 'Camera access blocked by security policy';
            } else {
                errorMessage = `Camera error: ${error.message}`;
            }
            
            this.showError(errorMessage);
            this.updateCameraStatus('Error');
            document.getElementById('cameraIndicator').className = 'status-indicator error';
        }
    }

    async requestMicPermission() {
        try {
            this.updateMicStatus('Requesting permission...');
            
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Microphone access not supported in this browser');
            }
            
            // Request microphone access using getUserMedia
            const audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000 // Optimal for speech recognition
                }
            });
            
            // Store audio stream
            this.audioStream = audioStream;
            this.micEnabled = true;
            
            // Monitor stream for track ending
            audioStream.getAudioTracks().forEach(track => {
                track.onended = () => {
                    console.log('Audio track ended');
                    this.handleMediaStreamError('audio');
                };
            });
            
            // Set up audio level monitoring with error handling
            try {
                this.setupAudioLevelMonitoring(audioStream);
            } catch (audioError) {
                console.warn('Audio level monitoring setup failed:', audioError);
                // Continue without audio level monitoring
            }
            
            this.updateMicStatus('Ready');
            this.updateStreamButton();
            document.getElementById('micIndicator').className = 'status-indicator ready';
            
        } catch (error) {
            console.error('Microphone permission error:', error);
            
            let errorMessage = 'Microphone access failed';
            
            // Provide specific error messages based on error type
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Microphone permission denied by user';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No microphone device found';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Microphone is already in use by another application';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = 'Microphone does not support requested settings';
            } else if (error.name === 'SecurityError') {
                errorMessage = 'Microphone access blocked by security policy';
            } else {
                errorMessage = `Microphone error: ${error.message}`;
            }
            
            this.showError(errorMessage);
            this.updateMicStatus('Error');
            document.getElementById('micIndicator').className = 'status-indicator error';
        }
    }

    handleMediaStreamError(mediaType) {
        console.error(`${mediaType} stream error detected`);
        
        if (mediaType === 'video') {
            this.cameraEnabled = false;
            this.videoStream = null;
            this.updateCameraStatus('Stream lost');
            document.getElementById('cameraIndicator').className = 'status-indicator error';
            document.getElementById('videoOverlay').classList.remove('hidden');
        } else if (mediaType === 'audio') {
            this.micEnabled = false;
            this.audioStream = null;
            this.updateMicStatus('Stream lost');
            document.getElementById('micIndicator').className = 'status-indicator error';
        }
        
        // Stop streaming if active
        if (this.localStream) {
            this.stopStreaming();
        }
        
        this.updateStreamButton();
        this.showError(`${mediaType} stream was lost. Please grant permissions again.`);
    }

    async startStreaming() {
        console.log('startStreaming called');
        console.log('Camera enabled:', this.cameraEnabled);
        console.log('Mic enabled:', this.micEnabled);
        console.log('Is connected:', this.isConnected);
        
        if (!this.cameraEnabled || !this.micEnabled) {
            this.showError('Both camera and microphone must be enabled before streaming');
            return;
        }

        try {
            this.updateStreamStatus('Starting local stream...');
            
            // First, combine video and audio streams locally
            const combinedStream = new MediaStream();
            
            if (this.videoStream) {
                this.videoStream.getVideoTracks().forEach(track => {
                    combinedStream.addTrack(track);
                });
            }
            
            if (this.audioStream) {
                this.audioStream.getAudioTracks().forEach(track => {
                    combinedStream.addTrack(track);
                });
            }
            
            this.localStream = combinedStream;
            console.log('Local stream created successfully with', this.localStream.getTracks().length, 'tracks');
            
            // Update UI to show local streaming is active
            this.updateStreamStatus('Local stream active');
            document.getElementById('streamIndicator').className = 'status-indicator active';
            document.getElementById('startStreamBtn').disabled = true;
            document.getElementById('stopStreamBtn').disabled = false;
            
            // Now automatically connect to backend if not already connected
            if (!this.isConnected) {
                console.log('Local stream ready - connecting to backend...');
                this.updateStreamStatus('Connecting to backend...');
                
                try {
                    await this.connectToServer();
                    console.log('Successfully connected to backend');
                } catch (error) {
                    console.warn('Failed to connect to backend:', error.message);
                    this.showError('Backend connection failed - streaming locally only: ' + error.message);
                    this.updateStreamStatus('Local only (backend unavailable)');
                    // Continue with local streaming even if backend connection fails
                    return;
                }
            }
            
            // Backend connection successful - set up WebRTC
            console.log('Setting up WebRTC connection...');
            this.updateStreamStatus('Setting up WebRTC...');
            
            // Create WebRTC peer connection if not exists
            if (!this.peerConnection) {
                console.log('Creating peer connection...');
                await this.createPeerConnection();
            }
            
            // Add stream to peer connection
            console.log('Adding tracks to peer connection...');
            this.localStream.getTracks().forEach(track => {
                console.log('Adding track:', track.kind);
                this.peerConnection.addTrack(track, this.localStream);
            });
            
            // Create and send offer
            console.log('Creating and sending WebRTC offer...');
            await this.createAndSendOffer();
            
            // WebRTC connection successful
            this.updateStreamStatus('Streaming to backend');
            console.log('✅ Full streaming pipeline active: Local → WebRTC → Backend');
            
        } catch (error) {
            this.showError('Failed to start streaming: ' + error.message);
            this.updateStreamStatus('Error');
            document.getElementById('streamIndicator').className = 'status-indicator error';
        }
    }

    stopStreaming() {
        try {
            // Stop all tracks in the local stream
            if (this.localStream) {
                this.localStream.getTracks().forEach(track => {
                    track.stop();
                });
                this.localStream = null;
            }
            
            // Close peer connection if exists
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }
            
            this.updateStreamStatus('Inactive');
            document.getElementById('streamIndicator').className = 'status-indicator';
            document.getElementById('startStreamBtn').disabled = false;
            document.getElementById('stopStreamBtn').disabled = true;
            
        } catch (error) {
            this.showError('Error stopping stream: ' + error.message);
        }
    }

    async connectToServer() {
        const serverUrl = document.getElementById('serverUrl').value.trim();
        if (!serverUrl) {
            this.showError('Please enter a server URL');
            return;
        }

        try {
            this.updateConnectionStatus('Connecting...');
            document.getElementById('connectBtn').disabled = true;
            
            // Test server connectivity with a simple HTTP request
            const testUrl = serverUrl.replace(/\/$/, '') + '/health';
            
            // Create a timeout promise
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Connection timeout')), 5000);
            });
            
            // Race between fetch and timeout
            const response = await Promise.race([
                fetch(testUrl, { method: 'GET' }),
                timeoutPromise
            ]);
            
            if (!response.ok) {
                throw new Error(`Server not responding (${response.status})`);
            }
            
            const healthData = await response.json();
            console.log('Server health check passed:', healthData);
            
            this.isConnected = true;
            this.updateConnectionStatus('Connected');
            document.getElementById('connectionIndicator').className = 'status-indicator active';
            document.getElementById('disconnectBtn').disabled = false;
            this.updateConnectionState('connected');
            this.updateConnectButtonState();
            
        } catch (error) {
            // Check if this is a Mixed Content error
            if (error.message.includes('Mixed Content') || 
                error.message.includes('blocked') || 
                error.message.includes('HTTPS') ||
                (window.AI_NAV_CONFIG && window.AI_NAV_CONFIG.isHttpsPage() && error.name === 'TypeError')) {
                console.error('Mixed Content error in server connection:', error);
                this.showMixedContentWarning();
            } else {
                this.showError('Failed to connect to server: ' + error.message);
            }
            this.updateConnectionStatus('Connection failed');
            document.getElementById('connectionIndicator').className = 'status-indicator error';
            this.updateConnectButtonState();
        }
    }

    disconnectFromServer() {
        try {
            // Close peer connection
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }
            
            // Stop streaming if active
            if (document.getElementById('stopStreamBtn').disabled === false) {
                this.stopStreaming();
            }
            
            this.isConnected = false;
            this.updateConnectionStatus('Disconnected');
            document.getElementById('connectionIndicator').className = 'status-indicator';
            document.getElementById('disconnectBtn').disabled = true;
            this.updateConnectionState('disconnected');
            this.updateConnectButtonState();
            
        } catch (error) {
            this.showError('Error during disconnection: ' + error.message);
        }
    }

    checkBiometricAvailability() {
        // Biometric sensor integration is optional (task 6.3)
        // This is a placeholder for future implementation
        console.log('Biometric availability check - placeholder for optional task 6.3');
        
        // Hide biometric section since it's not implemented
        document.getElementById('biometricSection').style.display = 'none';
        document.getElementById('biometricStatus').textContent = 'Not Implemented (Optional)';
    }

    enableBiometrics() {
        // Biometric sensor integration is optional (task 6.3)
        // This is a placeholder for future implementation
        console.log('Biometric enablement - placeholder for optional task 6.3');
        this.showError('Biometric sensors not implemented (optional feature)');
    }

    startHeartRateSimulation() {
        // Simulate heart rate data for UI testing
        this.heartRateMonitor = setInterval(() => {
            const heartRate = 60 + Math.floor(Math.random() * 40); // 60-100 BPM
            document.getElementById('heartRateValue').textContent = `${heartRate} BPM`;
        }, 2000);
    }

    updateStreamButton() {
        const startBtn = document.getElementById('startStreamBtn');
        startBtn.disabled = !(this.cameraEnabled && this.micEnabled);
    }

    updateCameraStatus(status) {
        document.getElementById('cameraStatus').textContent = status;
    }

    updateMicStatus(status) {
        document.getElementById('micStatus').textContent = status;
        document.getElementById('audioStatus').textContent = `Microphone: ${status}`;
    }

    updateConnectionStatus(status) {
        document.getElementById('connectionStatus').textContent = status;
    }

    updateStreamStatus(status) {
        document.getElementById('streamStatus').textContent = status;
    }

    updateConnectionState(state) {
        document.getElementById('connectionState').textContent = state;
    }

    updateAllStatuses() {
        this.updateCameraStatus('Not Ready');
        this.updateMicStatus('Not Ready');
        this.updateConnectionStatus('Disconnected');
        this.updateStreamStatus('Inactive');
        this.updateConnectionState('new');
        this.updateConnectButtonState();
    }
    
    updateConnectButtonState() {
        const connectBtn = document.getElementById('connectBtn');
        const serverUrl = document.getElementById('serverUrl').value.trim();
        
        // Enable connect button if there's a server URL and not connected
        connectBtn.disabled = !serverUrl || this.isConnected;
        
        console.log('Connect button state updated:', {
            serverUrl: serverUrl,
            isConnected: this.isConnected,
            disabled: connectBtn.disabled
        });
    }

    showError(message) {
        document.getElementById('lastError').textContent = message;
        console.error('NavigationClient Error:', message);
        
        // Show error in a more prominent way
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #f44336;
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
            z-index: 1000;
            max-width: 300px;
        `;
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        
        // Remove error message after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Initialize the client when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing NavigationClient...');
    const client = new NavigationClient();
    
    // Force update connect button state after a short delay to ensure DOM is ready
    setTimeout(() => {
        client.updateConnectButtonState();
    }, 100);
});