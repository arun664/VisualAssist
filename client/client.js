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
        this.dataChannel = null;
        this.websocket = null;
        this.isConnected = false;
        this.cameraEnabled = false;
        this.micEnabled = false;
        this.biometricEnabled = false;
        this.heartRateMonitor = null;
        this.clientId = this.generateClientId(); // Generate once and reuse
        this.currentCameraFacing = 'user'; // 'user' for front camera, 'environment' for back camera
        this.availableCameras = []; // Store available camera devices
        
        // Navigation guidance properties
        this.isSpeaking = false;
        this.lastGuidanceMessage = '';
        this.lastGuidanceTime = 0;
        
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
        // Skip authentication for now - initialize app directly
        this.hideAuthModal();
        this.initializeApp();
        this.setupAuthEventListeners();
        
        // Automatically detect and connect to backend
        this.autoDetectAndConnectBackend();
    }

    async autoDetectAndConnectBackend() {
        console.log('🔍 Auto-detecting backend server...');
        
        // Possible backend URLs to try
        const backendUrls = [
            'http://localhost:8000',
            'http://127.0.0.1:8000',
            window.AI_NAV_CONFIG?.getBackendUrl?.() || null,
            window.AI_NAV_CONFIG?.getDirectBackendUrl?.() || null
        ].filter(url => url && url.trim()); // Remove null/empty values
        
        for (const url of backendUrls) {
            try {
                console.log(`Testing backend at: ${url}`);
                
                const response = await fetch(`${url}/health`, {
                    method: 'GET',
                    headers: {
                        'ngrok-skip-browser-warning': 'true'
                    },
                    timeout: 3000
                });
                
                if (response.ok) {
                    const healthData = await response.json();
                    console.log(`✅ Backend found at: ${url}`, healthData);
                    
                    // Update the server URL input and connect
                    document.getElementById('serverUrl').value = url;
                    this.updateConnectButtonState();
                    
                    // Show backend connection status
                    this.showBackendDetectionResult(url, healthData);
                    
                    // Automatically connect
                    await this.connectToServer();
                    return; // Stop trying other URLs
                }
            } catch (error) {
                console.log(`❌ Backend not available at: ${url} - ${error.message}`);
            }
        }
        
        // No backend found
        console.log('⚠️ No backend server detected');
        this.showBackendDetectionResult(null, null);
    }

    showBackendDetectionResult(url, healthData) {
        const statusElement = document.getElementById('backendDetectionStatus');
        if (!statusElement) {
            // Create status element if it doesn't exist
            const connectionSection = document.querySelector('.connection-section');
            const statusDiv = document.createElement('div');
            statusDiv.id = 'backendDetectionStatus';
            statusDiv.style.cssText = `
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            `;
            connectionSection.appendChild(statusDiv);
        }
        
        const statusEl = document.getElementById('backendDetectionStatus');
        
        if (url && healthData) {
            statusEl.innerHTML = `
                <div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 5px;">
                    ✅ <strong>Backend Auto-Detected</strong><br>
                    URL: ${url}<br>
                    Status: ${healthData.status || 'Running'}<br>
                    Version: ${healthData.version || 'Unknown'}
                </div>
            `;
        } else {
            statusEl.innerHTML = `
                <div style="background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;">
                    ⚠️ <strong>No Backend Detected</strong><br>
                    You can still use local preview mode, or manually enter a backend URL above.
                </div>
            `;
        }
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
        // Enhanced ICE servers configuration for better connectivity
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
                { urls: 'stun:stun2.l.google.com:19302' },
                { urls: 'stun:stun3.l.google.com:19302' },
                { urls: 'stun:stun4.l.google.com:19302' },
                // Add more public STUN servers for redundancy
                { urls: 'stun:stun.stunprotocol.org:3478' },
                { urls: 'stun:stun.services.mozilla.com' }
            ],
            iceCandidatePoolSize: 10,
            // Optimize for better connectivity
            bundlePolicy: 'balanced',
            rtcpMuxPolicy: 'require',
            // Enable more aggressive ICE gathering
            iceTransportPolicy: 'all'
        };
        
        this.peerConnection = new RTCPeerConnection(configuration);
        
        // Create data channel for receiving navigation guidance
        this.dataChannel = this.peerConnection.createDataChannel('navigationGuidance', {
            ordered: true
        });
        
        // Set up data channel event handlers
        this.dataChannel.onopen = () => {
            console.log('✅ Navigation guidance data channel opened');
            this.updateGuidanceStatus('Connected to navigation service');
        };
        
        this.dataChannel.onclose = () => {
            console.log('❌ Navigation guidance data channel closed');
            this.updateGuidanceStatus('Navigation service disconnected');
        };
        
        this.dataChannel.onmessage = (event) => {
            console.log('📡 Received navigation guidance:', event.data);
            try {
                const guidanceData = JSON.parse(event.data);
                this.handleNavigationGuidance(guidanceData);
            } catch (error) {
                console.error('Error parsing navigation guidance:', error);
            }
        };
        
        // Handle data channel from the server
        this.peerConnection.ondatachannel = (event) => {
            const receivedChannel = event.channel;
            
            receivedChannel.onopen = () => {
                console.log('✅ Server-initiated data channel opened:', receivedChannel.label);
            };
            
            receivedChannel.onmessage = (messageEvent) => {
                console.log(`📡 Message received on ${receivedChannel.label}:`, messageEvent.data);
                
                try {
                    // Handle guidance data
                    if (receivedChannel.label === 'navigationGuidance' || receivedChannel.label === 'guidance') {
                        const guidanceData = JSON.parse(messageEvent.data);
                        this.handleNavigationGuidance(guidanceData);
                    }
                } catch (error) {
                    console.error('Error handling channel message:', error);
                }
            };
            
            receivedChannel.onclose = () => {
                console.log('❌ Server-initiated data channel closed:', receivedChannel.label);
            };
        };
        
        // Set up event handlers
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('ICE candidate generated:', {
                    type: event.candidate.type,
                    protocol: event.candidate.protocol,
                    candidate: event.candidate.candidate.substring(0, 50) + '...'
                });
                // Note: No WebSocket for ICE candidates in this implementation
                // ICE candidates are handled via HTTP polling or other mechanisms
            } else {
                console.log('ICE gathering completed');
            }
        };
        
        // Add ICE connection state monitoring
        this.peerConnection.oniceconnectionstatechange = () => {
            const iceState = this.peerConnection.iceConnectionState;
            console.log('ICE connection state changed to:', iceState);
            document.getElementById('iceState').textContent = iceState;
            
            switch (iceState) {
                case 'connected':
                case 'completed':
                    console.log('✅ ICE connection established successfully');
                    break;
                case 'failed':
                    console.log('❌ ICE connection failed - may need TURN servers');
                    this.handleIceConnectionFailure();
                    break;
                case 'disconnected':
                    console.log('⚠️ ICE connection disconnected');
                    break;
            }
        };
        
        // Add ICE gathering state monitoring
        this.peerConnection.onicegatheringstatechange = () => {
            console.log('ICE gathering state changed to:', this.peerConnection.iceGatheringState);
        };
        
        this.peerConnection.onconnectionstatechange = () => {
            const state = this.peerConnection.connectionState;
            this.updateConnectionState(state);
            document.getElementById('connectionState').textContent = state;
            
            console.log(`WebRTC connection state changed to: ${state}`);
            
            switch (state) {
                case 'connected':
                    document.getElementById('connectionIndicator').className = 'status-indicator active';
                    document.getElementById('streamIndicator').className = 'status-indicator active';
                    this.updateStreamStatus('Streaming to backend');
                    this.reconnectAttempts = 0; // Reset on successful connection
                    console.log('✅ WebRTC connection established - now streaming to backend');
                    
                    // Update navigation guidance UI
                    this.updateGuidanceStatus('Ready for navigation guidance');
                    
                    // Log media streaming statistics
                    setTimeout(() => {
                        this.peerConnection.getStats().then(stats => {
                            let videoTrackSent = false;
                            let audioTrackSent = false;
                            let videoStats = {};
                            let audioStats = {};
                            
                            stats.forEach(report => {
                                if (report.type === 'outbound-rtp') {
                                    if (report.mediaType === 'video') {
                                        videoTrackSent = true;
                                        videoStats = {
                                            packetsSent: report.packetsSent,
                                            bytesSent: report.bytesSent,
                                            framesSent: report.framesSent,
                                            framesPerSecond: report.framesPerSecond
                                        };
                                    } else if (report.mediaType === 'audio') {
                                        audioTrackSent = true;
                                        audioStats = {
                                            packetsSent: report.packetsSent,
                                            bytesSent: report.bytesSent
                                        };
                                    }
                                }
                            });
                            
                            console.log('📊 Media streaming status:', {
                                videoSending: videoTrackSent,
                                audioSending: audioTrackSent,
                                videoStats: videoStats,
                                audioStats: audioStats
                            });
                            
                            if (!videoTrackSent || !audioTrackSent) {
                                console.warn('⚠️ Media not being sent properly:', {
                                    video: videoTrackSent ? 'OK' : 'NOT SENDING',
                                    audio: audioTrackSent ? 'OK' : 'NOT SENDING'
                                });
                            }
                        });
                    }, 2000); // Check stats after 2 seconds
                    break;
                    
                case 'connecting':
                    document.getElementById('connectionIndicator').className = 'status-indicator warning';
                    this.updateStreamStatus('Establishing WebRTC connection...');
                    break;
                    
                case 'disconnected':
                    document.getElementById('connectionIndicator').className = 'status-indicator error';
                    document.getElementById('streamIndicator').className = 'status-indicator error';
                    this.updateStreamStatus('Connection lost');
                    this.showError('WebRTC connection lost');
                    this.updateGuidanceStatus('Navigation service disconnected');
                    this.handleConnectionFailure();
                    break;
                    
                case 'failed':
                    document.getElementById('connectionIndicator').className = 'status-indicator error';
                    document.getElementById('streamIndicator').className = 'status-indicator error';
                    this.updateStreamStatus('Connection failed');
                    console.error('WebRTC connection failed - possible causes: firewall, NAT, or network restrictions');
                    this.showError('WebRTC connection failed - check network/firewall settings');
                    this.updateGuidanceStatus('Navigation service unavailable');
                    this.handleConnectionFailure();
                    break;
                    
                case 'closed':
                    document.getElementById('connectionIndicator').className = 'status-indicator inactive';
                    document.getElementById('streamIndicator').className = 'status-indicator inactive';
                    this.updateStreamStatus('Inactive');
                    this.updateGuidanceStatus('Navigation service disconnected');
                    break;
                    
                default:
                    console.log(`Unhandled WebRTC connection state: ${state}`);
            }
        };
        
        this.peerConnection.oniceconnectionstatechange = () => {
            const iceState = this.peerConnection.iceConnectionState;
            document.getElementById('iceState').textContent = iceState;
            console.log(`ICE connection state changed to: ${iceState}`);
            
            switch (iceState) {
                case 'connected':
                case 'completed':
                    console.log('✅ ICE connection established successfully');
                    break;
                case 'disconnected':
                    console.warn('⚠️ ICE connection disconnected');
                    break;
                case 'failed':
                    console.error('❌ ICE connection failed - network connectivity issues');
                    this.showError('Network connection failed - check firewall/NAT settings');
                    this.handleConnectionFailure();
                    break;
                case 'checking':
                    console.log('🔍 ICE connectivity checks in progress...');
                    break;
                default:
                    console.log(`ICE connection state: ${iceState}`);
            }
        };
        
        this.peerConnection.onicegatheringstatechange = () => {
            const gatheringState = this.peerConnection.iceGatheringState;
            console.log(`ICE gathering state: ${gatheringState}`);
            
            if (gatheringState === 'complete') {
                console.log('✅ ICE candidate gathering completed');
            }
        };
    }

    async testWebRTCConnectivity() {
        console.log('🔍 Testing WebRTC connectivity...');
        
        try {
            // Enhanced ICE servers configuration for better connectivity
            const testConfig = {
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' },
                    { urls: 'stun:stun2.l.google.com:19302' },
                    { urls: 'stun:stun3.l.google.com:19302' },
                    { urls: 'stun:stun4.l.google.com:19302' },
                    // Add more public STUN servers for redundancy
                    { urls: 'stun:stun.stunprotocol.org:3478' },
                    { urls: 'stun:stun.services.mozilla.com' }
                ],
                iceCandidatePoolSize: 10,
                // Optimize for better connectivity
                bundlePolicy: 'balanced',
                rtcpMuxPolicy: 'require'
            };
            
            const testPC = new RTCPeerConnection(testConfig);
            let candidatesReceived = 0;
            let hasHostCandidate = false;
            let hasSrflxCandidate = false;
            
            // Track ICE candidates as they arrive
            testPC.onicecandidate = (event) => {
                if (event.candidate) {
                    candidatesReceived++;
                    const candidate = event.candidate.candidate;
                    
                    // Check for different types of candidates
                    if (candidate.includes('typ host')) {
                        hasHostCandidate = true;
                        console.log('✅ Host candidate found (local network)');
                    } else if (candidate.includes('typ srflx')) {
                        hasSrflxCandidate = true;
                        console.log('✅ Server reflexive candidate found (through STUN)');
                    }
                    
                    console.log(`ICE candidate #${candidatesReceived}:`, candidate.substring(0, 60) + '...');
                }
            };
            
            // Create a data channel to ensure ICE gathering
            const dataChannel = testPC.createDataChannel('connectivity-test', {
                ordered: true
            });
            
            // Create an offer to trigger ICE gathering
            const offer = await testPC.createOffer({
                offerToReceiveAudio: false,
                offerToReceiveVideo: false
            });
            await testPC.setLocalDescription(offer);
            
            // Enhanced ICE gathering with multiple conditions
            const iceGatheringPromise = new Promise((resolve, reject) => {
                // Reduced timeout for faster feedback but allow some flexibility
                const timeout = setTimeout(() => {
                    if (candidatesReceived > 0) {
                        console.log(`⚠️ ICE gathering timeout reached, but ${candidatesReceived} candidates were found`);
                        resolve(testPC.localDescription);
                    } else {
                        reject(new Error('ICE gathering timeout - no candidates found'));
                    }
                }, 8000); // Reduced from 10s to 8s
                
                // Track ICE gathering state changes
                testPC.onicegatheringstatechange = () => {
                    const state = testPC.iceGatheringState;
                    console.log(`ICE gathering state: ${state}`);
                    
                    if (state === 'complete') {
                        clearTimeout(timeout);
                        console.log('✅ ICE gathering completed normally');
                        resolve(testPC.localDescription);
                    }
                };
                
                // Alternative completion check - if we get good candidates early
                const candidateCheckInterval = setInterval(() => {
                    if (candidatesReceived >= 2 && hasHostCandidate) {
                        clearTimeout(timeout);
                        clearInterval(candidateCheckInterval);
                        console.log('✅ Sufficient ICE candidates found early');
                        resolve(testPC.localDescription);
                    }
                }, 1000);
                
                // Clean up interval if we timeout or complete normally
                setTimeout(() => clearInterval(candidateCheckInterval), 8000);
            });
            
            const localDesc = await iceGatheringPromise;
            const candidateCount = (localDesc.sdp.match(/a=candidate/g) || []).length;
            
            // Analyze connectivity quality
            let connectivityQuality = 'unknown';
            if (hasHostCandidate && hasSrflxCandidate) {
                connectivityQuality = 'excellent';
            } else if (hasHostCandidate || hasSrflxCandidate) {
                connectivityQuality = 'good';
            } else if (candidatesReceived > 0) {
                connectivityQuality = 'limited';
            } else {
                connectivityQuality = 'poor';
            }
            
            console.log(`✅ WebRTC connectivity test passed - ${candidateCount} ICE candidates found`);
            console.log(`📊 Connectivity quality: ${connectivityQuality}`);
            console.log(`📈 Candidates: ${candidatesReceived} received, Host: ${hasHostCandidate}, STUN: ${hasSrflxCandidate}`);
            
            testPC.close();
            
            return { 
                success: true, 
                candidateCount,
                connectivityQuality,
                hasHostCandidate,
                hasSrflxCandidate,
                candidatesReceived
            };
            
        } catch (error) {
            console.error('❌ WebRTC connectivity test failed:', error);
            
            // Provide more specific error guidance
            let errorGuidance = '';
            if (error.message.includes('timeout')) {
                errorGuidance = 'Network/firewall may be blocking STUN servers. Try: 1) Check firewall settings, 2) Try different network, 3) Contact network administrator.';
            } else if (error.message.includes('no candidates')) {
                errorGuidance = 'No ICE candidates found. This usually indicates severe network restrictions or lack of internet connectivity.';
            }
            
            return { 
                success: false, 
                error: error.message,
                guidance: errorGuidance
            };
        }
    }

    // WebSocket signaling removed - client now uses HTTP for WebRTC offers
    // This simplifies the connection process and removes WebSocket dependency

    // Navigation guidance handling
    handleNavigationGuidance(guidanceData) {
        if (!guidanceData) return;
        
        const guidanceMessage = document.getElementById('guidanceMessage');
        const guidancePulse = document.getElementById('guidancePulse');
        
        // Reset previous classes
        guidanceMessage.className = 'guidance-message';
        
        let message = '';
        let directionClass = '';
        let icon = '';
        
        // Determine guidance message and styling based on direction
        switch(guidanceData.direction) {
            case 'forward':
                message = 'Move Forward';
                directionClass = 'direction-forward';
                icon = '⬆️';
                break;
            case 'left':
                message = 'Turn Left';
                directionClass = 'direction-left';
                icon = '⬅️';
                break;
            case 'right':
                message = 'Turn Right';
                directionClass = 'direction-right';
                icon = '➡️';
                break;
            case 'turn_around':
                message = 'Turn Around';
                directionClass = 'direction-turn-around';
                icon = '🔄';
                break;
            case 'wait':
                message = 'Please Wait';
                directionClass = 'direction-wait';
                icon = '⏳';
                break;
            case 'no_path':
                message = 'No Path Detected';
                directionClass = 'direction-no-path';
                icon = '⚠️';
                break;
            default:
                message = guidanceData.message || 'Processing...';
        }
        
        // Set message with icon
        guidanceMessage.innerHTML = `${icon} ${message}`;
        if (guidanceData.distance) {
            guidanceMessage.innerHTML += ` (${guidanceData.distance}m)`;
        }
        
        // Apply direction-specific styling
        guidanceMessage.classList.add(directionClass);
        
        // Trigger pulse animation for new guidance
        guidancePulse.classList.remove('active');
        setTimeout(() => {
            guidancePulse.classList.add('active');
        }, 10);
        
        // Speak guidance using text-to-speech if available
        this.speakGuidanceMessage(`${message}${guidanceData.distance ? ', ' + guidanceData.distance + ' meters' : ''}`);
    }
    
    speakGuidanceMessage(message) {
        // Use speech synthesis if available
        if (window.speechSynthesis && !this.isSpeaking) {
            this.isSpeaking = true;
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.volume = 1; // 0 to 1
            utterance.rate = 1; // 0.1 to 10
            utterance.pitch = 1; //0 to 2
            utterance.lang = 'en-US';
            
            utterance.onend = () => {
                this.isSpeaking = false;
            };
            
            window.speechSynthesis.speak(utterance);
            
            // Safety fallback in case onend doesn't fire
            setTimeout(() => {
                this.isSpeaking = false;
            }, 5000);
        }
    }

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
            
            // Debug: Log SDP content to verify media tracks are included
            console.log('Offer SDP contains:');
            console.log('- Video tracks:', (offer.sdp.match(/m=video/g) || []).length);
            console.log('- Audio tracks:', (offer.sdp.match(/m=audio/g) || []).length);
            console.log('- Video codecs:', offer.sdp.includes('H264') ? 'H264' : 'Other');
            console.log('- Audio codecs:', offer.sdp.includes('opus') ? 'Opus' : 'Other');
            
            // Log first 200 chars of SDP for debugging
            console.log('SDP preview:', offer.sdp.substring(0, 200) + '...');
            
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
                    'ngrok-skip-browser-warning': 'true'
                },
                body: JSON.stringify({
                    client_id: this.clientId,
                    type: offer.type,
                    sdp: offer.sdp
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('WebRTC offer failed:', response.status, errorText);
                throw new Error(`HTTP error! status: ${response.status}, response: ${errorText.substring(0, 100)}...`);
            }
            
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const responseText = await response.text();
                console.error('WebRTC non-JSON response:', responseText.substring(0, 200));
                throw new Error(`WebRTC endpoint returned HTML instead of JSON. Backend may be busy or misconfigured.`);
            }
            
            const answerData = await response.json();
            console.log('Received answer from server:', answerData);
            
            if (answerData.status === 'success') {
                const answer = new RTCSessionDescription({
                    type: answerData.answer.type,
                    sdp: answerData.answer.sdp
                });
                
                await this.peerConnection.setRemoteDescription(answer);
                console.log('WebRTC answer processed - waiting for connection establishment');
                // Don't update stream status here - let the connection state handler do it
            } else {
                throw new Error('Failed to get answer from server');
            }
            
        } catch (error) {
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

    handleIceConnectionFailure() {
        console.log('🧊 ICE connection failed - this usually indicates NAT/firewall issues');
        this.showError('WebRTC connection failed: Network/firewall may be blocking connection. Consider using TURN servers.');
        
        // Attempt to restart ICE if possible
        if (this.peerConnection && this.peerConnection.iceConnectionState === 'failed') {
            console.log('Attempting ICE restart...');
            try {
                this.peerConnection.restartIce();
            } catch (error) {
                console.error('ICE restart failed:', error);
            }
        }
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
        
        // Check if switch camera button exists before adding listener
        const switchCameraBtn = document.getElementById('switchCameraBtn');
        if (switchCameraBtn) {
            switchCameraBtn.addEventListener('click', () => this.switchCamera());
        }
        
        document.getElementById('requestMicBtn').addEventListener('click', () => this.requestMicPermission());
        
        // Stream control buttons
        document.getElementById('startStreamBtn').addEventListener('click', () => this.startStreaming());
        document.getElementById('stopStreamBtn').addEventListener('click', () => this.stopStreaming());
        
        // Connection buttons
        document.getElementById('connectBtn').addEventListener('click', () => this.connectToServer());
        document.getElementById('disconnectBtn').addEventListener('click', () => this.disconnectFromServer());
        
        // Biometric button
        document.getElementById('enableBiometricBtn').addEventListener('click', () => this.enableBiometrics());
        
        // WebRTC connectivity test button
        const testWebRTCBtn = document.getElementById('testWebRTCBtn');
        if (testWebRTCBtn) {
            testWebRTCBtn.addEventListener('click', async () => {
                // Disable button during test
                testWebRTCBtn.disabled = true;
                testWebRTCBtn.textContent = 'Testing Connectivity...';
                
                try {
                    const result = await this.testWebRTCConnectivity();
                    
                    if (result.success) {
                        // Show success message with details
                        const qualityEmoji = {
                            'excellent': '🟢',
                            'good': '🟡', 
                            'limited': '🟠',
                            'poor': '🔴',
                            'unknown': '⚪'
                        };
                        
                        const emoji = qualityEmoji[result.connectivityQuality] || '⚪';
                        const message = `${emoji} WebRTC connectivity test passed!\n\n` +
                                      `Quality: ${result.connectivityQuality}\n` +
                                      `ICE candidates: ${result.candidateCount}\n` +
                                      `Local network: ${result.hasHostCandidate ? 'Yes' : 'No'}\n` +
                                      `STUN server: ${result.hasSrflxCandidate ? 'Yes' : 'No'}`;
                        
                        this.showConnectivityTestResult(message, 'success');
                    } else {
                        // Show failure message with guidance
                        const message = `❌ WebRTC connectivity test failed!\n\n` +
                                      `Error: ${result.error}\n\n` +
                                      `${result.guidance || 'Check your network connection and firewall settings.'}`;
                        
                        this.showConnectivityTestResult(message, 'error');
                    }
                } catch (error) {
                    this.showError('Connectivity test failed: ' + error.message);
                } finally {
                    // Re-enable button
                    testWebRTCBtn.disabled = false;
                    testWebRTCBtn.textContent = 'Test WebRTC Connectivity';
                }
            });
        }
        
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
                    frameRate: { ideal: 30 },
                    facingMode: { ideal: this.currentCameraFacing }
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
            
            // Enumerate cameras and enable switch button if multiple cameras available
            await this.enumerateCameras();
            
            // Update switch button text
            const switchBtn = document.getElementById('switchCameraBtn');
            const cameraType = this.currentCameraFacing === 'user' ? 'Front' : 'Back';
            switchBtn.innerHTML = `<span class="btn-icon">🔄</span> ${cameraType} Camera`;
            
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

    async enumerateCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.availableCameras = devices.filter(device => device.kind === 'videoinput');
            
            // Enable camera switch button if we have multiple cameras
            const switchBtn = document.getElementById('switchCameraBtn');
            if (this.availableCameras.length > 1) {
                switchBtn.disabled = false;
                console.log(`Found ${this.availableCameras.length} cameras:`, this.availableCameras);
            } else {
                switchBtn.disabled = true;
                console.log('Only one camera available, switch disabled');
            }
            
            return this.availableCameras;
        } catch (error) {
            console.error('Error enumerating cameras:', error);
            return [];
        }
    }

    async switchCamera() {
        try {
            // Toggle camera facing mode
            this.currentCameraFacing = this.currentCameraFacing === 'user' ? 'environment' : 'user';
            
            // Stop current video stream
            if (this.videoStream) {
                this.videoStream.getTracks().forEach(track => track.stop());
            }
            
            // Request new camera with switched facing mode
            await this.requestCameraWithFacing(this.currentCameraFacing);
            
            // Update button text to show current camera
            const switchBtn = document.getElementById('switchCameraBtn');
            const cameraType = this.currentCameraFacing === 'user' ? 'Front' : 'Back';
            switchBtn.innerHTML = `<span class="btn-icon">🔄</span> ${cameraType} Camera`;
            
            console.log(`Switched to ${cameraType.toLowerCase()} camera`);
            
        } catch (error) {
            console.error('Error switching camera:', error);
            this.showError('Failed to switch camera: ' + error.message);
            
            // Revert facing mode on error
            this.currentCameraFacing = this.currentCameraFacing === 'user' ? 'environment' : 'user';
        }
    }

    async requestCameraWithFacing(facingMode) {
        // Update status
        this.updateCameraStatus('Switching camera...');
        
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera access not supported in this browser');
        }
        
        // Request camera access with specific facing mode
        const videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
                facingMode: { ideal: facingMode }
            }
        });
        
        // Store video stream
        this.videoStream = videoStream;
        
        // Update local video preview
        const localVideo = document.getElementById('localVideo');
        localVideo.srcObject = videoStream;
        
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
        console.log('Video stream tracks:', this.videoStream ? this.videoStream.getTracks().length : 0);
        console.log('Audio stream tracks:', this.audioStream ? this.audioStream.getTracks().length : 0);
        
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
            
            // Update UI to show local streaming is ready (but not backend streaming yet)
            this.updateStreamStatus('Local stream ready');
            document.getElementById('streamIndicator').className = 'status-indicator warning';
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
            
            // Test WebRTC connectivity first (but be more lenient)
            console.log('Testing WebRTC connectivity before establishing connection...');
            this.updateStreamStatus('Testing WebRTC connectivity...');
            
            const connectivityTest = await this.testWebRTCConnectivity();
            
            if (!connectivityTest.success) {
                const errorMsg = `WebRTC connectivity test failed: ${connectivityTest.error}`;
                const guidance = connectivityTest.guidance || 'Check network/firewall settings.';
                
                console.warn(`⚠️ ${errorMsg}`);
                
                // Show warning but try to continue anyway (more lenient approach)
                this.showError(`WebRTC connectivity issue detected: ${connectivityTest.error}. Attempting connection anyway...`);
                this.updateStreamStatus('WebRTC connectivity limited - attempting connection...');
                
                // Continue with WebRTC setup despite connectivity issues
                console.log('Continuing with WebRTC setup despite connectivity issues...');
            } else {
                // Log connectivity quality for debugging
                console.log(`✅ WebRTC connectivity OK (${connectivityTest.candidateCount} ICE candidates, quality: ${connectivityTest.connectivityQuality})`);
                
                // Show warning if connectivity quality is poor but continue
                if (connectivityTest.connectivityQuality === 'limited' || connectivityTest.connectivityQuality === 'poor') {
                    console.warn('⚠️ Limited WebRTC connectivity detected - connection may be unstable');
                    this.showError('Limited network connectivity detected - WebRTC connection may be unstable but will attempt connection.');
                }
            }
            
            // Add a small delay to ensure backend is ready for WebRTC
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Create WebRTC peer connection if not exists
            if (!this.peerConnection) {
                console.log('Creating peer connection...');
                await this.createPeerConnection();
            }
            
            // Add stream to peer connection
            console.log('Adding tracks to peer connection...');
            console.log('Local stream has', this.localStream.getTracks().length, 'tracks:');
            this.localStream.getTracks().forEach((track, index) => {
                console.log(`Track ${index}: ${track.kind} - enabled: ${track.enabled} - readyState: ${track.readyState}`);
                this.peerConnection.addTrack(track, this.localStream);
            });
            
            // Verify tracks were added to peer connection
            const senders = this.peerConnection.getSenders();
            console.log('Peer connection now has', senders.length, 'senders');
            senders.forEach((sender, index) => {
                if (sender.track) {
                    console.log(`Sender ${index}: ${sender.track.kind} track - enabled: ${sender.track.enabled}`);
                } else {
                    console.log(`Sender ${index}: No track`);
                }
            });
            
            // Create and send offer
            console.log('Creating and sending WebRTC offer...');
            this.updateStreamStatus('Establishing WebRTC connection...');
            await this.createAndSendOffer();
            
            // WebRTC offer sent successfully - actual connection status will be updated by connection state handler
            console.log('✅ WebRTC offer sent successfully - waiting for connection establishment');
            
        } catch (error) {
            this.showError('Failed to start streaming: ' + error.message);
            this.updateStreamStatus('Streaming failed');
            document.getElementById('streamIndicator').className = 'status-indicator error';
            
            // Reset button states on error
            document.getElementById('startStreamBtn').disabled = false;
            document.getElementById('stopStreamBtn').disabled = true;
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
            
            // Close data channel if exists
            if (this.dataChannel) {
                this.dataChannel.close();
                this.dataChannel = null;
            }
            
            // Close peer connection if exists
            if (this.peerConnection) {
                this.peerConnection.close();
                this.peerConnection = null;
            }
            
            // Update UI
            this.updateStreamStatus('Inactive');
            this.updateGuidanceStatus('Start streaming to receive navigation assistance');
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
            this.showError('Failed to connect to server: ' + error.message);
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
        this.updateGuidanceStatus('Start streaming to receive navigation assistance');
        this.updateConnectButtonState();
    }
    
    updateGuidanceStatus(message) {
        const guidanceMessage = document.getElementById('guidanceMessage');
        if (guidanceMessage) {
            // Reset any previous styling
            guidanceMessage.className = 'guidance-message';
            guidanceMessage.textContent = message;
        }
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

    showConnectivityTestResult(message, type) {
        // Create a modal dialog for test results
        const resultDialog = document.createElement('div');
        resultDialog.className = 'connectivity-test-dialog';
        resultDialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 3px solid ${type === 'success' ? '#4CAF50' : '#f44336'};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 2000;
            max-width: 400px;
            font-family: monospace;
            white-space: pre-line;
        `;
        
        resultDialog.innerHTML = `
            <h3 style="color: ${type === 'success' ? '#4CAF50' : '#f44336'}; margin-top: 0;">
                WebRTC Connectivity Test
            </h3>
            <div style="text-align: left; margin: 15px 0; line-height: 1.6;">
                ${message}
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button id="closeTestResult" style="background: ${type === 'success' ? '#4CAF50' : '#f44336'}; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Close
                </button>
                ${type === 'error' ? `
                <button id="showTroubleshooting" style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin-left: 10px; cursor: pointer;">
                    Troubleshooting Tips
                </button>
                ` : ''}
            </div>
        `;
        
        document.body.appendChild(resultDialog);
        
        // Add event listeners
        document.getElementById('closeTestResult').onclick = () => {
            document.body.removeChild(resultDialog);
        };
        
        if (type === 'error') {
            const troubleshootingBtn = document.getElementById('showTroubleshooting');
            if (troubleshootingBtn) {
                troubleshootingBtn.onclick = () => {
                    this.showTroubleshootingTips();
                    document.body.removeChild(resultDialog);
                };
            }
        }
    }

    showWebRTCFailureOptions(errorMsg, guidance) {
        const optionsDialog = document.createElement('div');
        optionsDialog.className = 'webrtc-failure-dialog';
        optionsDialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 3px solid #ff9800;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 2000;
            max-width: 450px;
        `;
        
        optionsDialog.innerHTML = `
            <h3 style="color: #ff9800; margin-top: 0;">⚠️ WebRTC Connection Failed</h3>
            <p style="margin: 15px 0;">
                <strong>Issue:</strong> ${errorMsg}<br>
                <strong>Suggestion:</strong> ${guidance}
            </p>
            <p style="margin: 15px 0;">
                <strong>Available Options:</strong>
            </p>
            <div style="margin: 15px 0;">
                <button id="continueLocalOnly" style="background: #4CAF50; color: white; border: none; padding: 10px 15px; border-radius: 5px; margin: 5px; cursor: pointer; width: 100%;">
                    📹 Continue with Local Preview Only
                </button>
                <button id="retryWebRTC" style="background: #2196F3; color: white; border: none; padding: 10px 15px; border-radius: 5px; margin: 5px; cursor: pointer; width: 100%;">
                    🔄 Retry WebRTC Connection
                </button>
                <button id="showNetworkTips" style="background: #ff9800; color: white; border: none; padding: 10px 15px; border-radius: 5px; margin: 5px; cursor: pointer; width: 100%;">
                    🛠️ Show Network Troubleshooting
                </button>
                <button id="cancelStreaming" style="background: #757575; color: white; border: none; padding: 10px 15px; border-radius: 5px; margin: 5px; cursor: pointer; width: 100%;">
                    ❌ Cancel Streaming
                </button>
            </div>
        `;
        
        document.body.appendChild(optionsDialog);
        
        // Add event listeners
        document.getElementById('continueLocalOnly').onclick = () => {
            document.body.removeChild(optionsDialog);
            console.log('User chose to continue with local preview only');
            // Don't start WebRTC, just keep local streams active
        };
        
        document.getElementById('retryWebRTC').onclick = async () => {
            document.body.removeChild(optionsDialog);
            console.log('User chose to retry WebRTC connection');
            
            // Update status and retry
            this.updateStreamStatus('Retrying WebRTC connection...');
            
            try {
                // Retry the WebRTC setup process
                await this.continueWithWebRTCSetup();
            } catch (error) {
                this.showError('WebRTC retry failed: ' + error.message);
                this.updateStreamStatus('WebRTC retry failed - local preview only');
            }
        };
        
        document.getElementById('showNetworkTips').onclick = () => {
            document.body.removeChild(optionsDialog);
            this.showTroubleshootingTips();
        };
        
        document.getElementById('cancelStreaming').onclick = () => {
            document.body.removeChild(optionsDialog);
            console.log('User cancelled streaming');
            this.stopStreaming();
        };
    }

    async continueWithWebRTCSetup() {
        // This method continues with the WebRTC setup process
        // after the initial local stream is established
        
        console.log('Continuing with WebRTC setup...');
        
        // Create WebRTC peer connection if not exists
        if (!this.peerConnection) {
            console.log('Creating peer connection...');
            await this.createPeerConnection();
        }
        
        // Add stream to peer connection
        console.log('Adding tracks to peer connection...');
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => {
                console.log('Adding track:', track.kind);
                this.peerConnection.addTrack(track, this.localStream);
            });
        }
        
        // Create and send offer
        console.log('Creating and sending WebRTC offer...');
        this.updateStreamStatus('Establishing WebRTC connection...');
        await this.createAndSendOffer();
        
        console.log('✅ WebRTC setup continued successfully');
    }

    showTroubleshootingTips() {
        const tipsDialog = document.createElement('div');
        tipsDialog.className = 'troubleshooting-dialog';
        tipsDialog.style.cssText = `
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
            max-width: 500px;
            max-height: 70vh;
            overflow-y: auto;
        `;
        
        tipsDialog.innerHTML = `
            <h3 style="color: #2196F3; margin-top: 0;">🔧 WebRTC Troubleshooting Tips</h3>
            <div style="text-align: left;">
                <h4>Common Issues & Solutions:</h4>
                <ul>
                    <li><strong>ICE Gathering Timeout:</strong>
                        <ul>
                            <li>Check if your firewall is blocking UDP traffic on ports 3478, 19302</li>
                            <li>Try disabling Windows Firewall temporarily</li>
                            <li>Switch to a different network (mobile hotspot, different WiFi)</li>
                        </ul>
                    </li>
                    <li><strong>Corporate/Restricted Networks:</strong>
                        <ul>
                            <li>Contact your network administrator about WebRTC/STUN server access</li>
                            <li>Request firewall exceptions for STUN servers</li>
                            <li>Consider using a VPN</li>
                        </ul>
                    </li>
                    <li><strong>Browser Issues:</strong>
                        <ul>
                            <li>Try a different browser (Chrome, Firefox, Edge)</li>
                            <li>Clear browser cache and cookies</li>
                            <li>Disable browser extensions temporarily</li>
                        </ul>
                    </li>
                    <li><strong>Network Debugging:</strong>
                        <ul>
                            <li>Test internet connectivity</li>
                            <li>Check if STUN servers are reachable: ping stun.l.google.com</li>
                            <li>Try running from localhost vs remote server</li>
                        </ul>
                    </li>
                </ul>
                
                <h4>Advanced Solutions:</h4>
                <ul>
                    <li>Configure TURN servers for NAT traversal</li>
                    <li>Use fallback to WebSocket streaming</li>
                    <li>Consider using a different WebRTC signaling approach</li>
                </ul>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button id="closeTroubleshooting" style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Close
                </button>
            </div>
        `;
        
        document.body.appendChild(tipsDialog);
        
        document.getElementById('closeTroubleshooting').onclick = () => {
            document.body.removeChild(tipsDialog);
        };
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
    
    // Store client instance for testing
    window.navigatorClientInstances = window.navigatorClientInstances || [];
    window.navigatorClientInstances.push(client);
    
    // Force update connect button state after a short delay to ensure DOM is ready
    setTimeout(() => {
        client.updateConnectButtonState();
    }, 100);
    
    // Load test script if 'test=true' is in URL
    const urlParams = new URLSearchParams(window.location.search);
    const testMode = urlParams.get('test') === 'true';
    
    if (testMode && !document.getElementById('navTestScript')) {
        const testScript = document.createElement('script');
        testScript.id = 'navTestScript';
        testScript.src = 'test-navigation.js';
        document.body.appendChild(testScript);
        console.log('🧪 Navigation test mode enabled - loaded test script');
    }
});