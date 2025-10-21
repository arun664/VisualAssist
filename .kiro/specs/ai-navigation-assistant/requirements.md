# Requirements Document

## Introduction

The AI Navigation Assistant is a real-time navigation assistance system that provides audio-guided walking directions using computer vision and AI processing. The system uses a three-component architecture consisting of a mobile client for sensor data capture, a backend server for AI processing and finite state machine logic, and a frontend for user interaction and audio feedback. The system employs closed-loop feedback mechanisms via optical flow analysis to confirm user actions and enhance safety during navigation.

## Glossary

- **Client Device**: Mobile phone or device with camera and microphone capabilities that captures and streams sensor data
- **Backend Server**: Central processing unit running AI models and finite state machine logic for navigation decisions
- **Frontend Interface**: Web-based user interface that displays processed video and provides audio feedback
- **FSM**: Finite State Machine that manages the navigation guidance process through defined states
- **Optical Flow Analysis**: Computer vision technique used to detect motion between video frames
- **YOLOv11**: Object detection AI model used to identify obstacles and calculate safe paths
- **WebRTC**: Real-time communication protocol for streaming video and audio data
- **Vosk STT**: Speech-to-text library for processing voice commands
- **MJPEG Stream**: Motion JPEG video streaming format for processed video delivery

## Requirements

### Requirement 1

**User Story:** As a visually impaired user, I want to receive real-time audio navigation guidance, so that I can safely navigate through environments with obstacle detection.

#### Acceptance Criteria

1. WHEN the user activates the navigation system, THE Backend Server SHALL enter STATE_SCANNING and provide audio instructions to stand still
2. WHILE the user is moving during scanning, THE Backend Server SHALL continuously monitor optical flow and request the user to remain stationary
3. WHEN the user becomes stationary and a clear path is detected, THE Backend Server SHALL transition to STATE_GUIDING and provide "path clear" audio feedback
4. IF an obstacle is detected one step before during navigation, THEN THE Backend Server SHALL immediately transition to STATE_BLOCKED and issue "Stop!" audio commands
5. WHERE the user continues moving after a stop command, THE Backend Server SHALL escalate to urgent "DANGER! STOP!" warnings

### Requirement 2

**User Story:** As a user, I want to capture and stream real-time video and audio data from my mobile device, so that the AI system can process my environment for navigation assistance.

#### Acceptance Criteria

1. THE Client Device SHALL capture video using navigator.mediaDevices.getUserMedia API
2. THE Client Device SHALL capture audio using navigator.mediaDevices.getUserMedia API
3. THE Client Device SHALL establish WebRTC connection to stream raw video and audio feeds to the backend server
4. THE Client Device SHALL maintain continuous real-time streaming during active navigation sessions
5. WHEN network connectivity is lost, THE Client Device SHALL attempt to re-establish the WebRTC connection

### Requirement 3

**User Story:** As a user, I want to see processed video with visual overlays indicating safe paths and obstacles, so that I can understand the navigation guidance being provided.

#### Acceptance Criteria

1. THE Backend Server SHALL process incoming video frames using YOLOv11 object detection during SCANNING and GUIDING states
2. THE Backend Server SHALL draw green tiles indicating safe paths using OpenCV drawing functions
3. THE Backend Server SHALL draw red boxes around detected obstacles using cv2.rectangle
4. THE Backend Server SHALL stream processed video with overlays via MJPEG endpoint
5. THE Frontend Interface SHALL display the processed video stream with overlays in real-time

### Requirement 4

**User Story:** As a user, I want to control the navigation system through simple voice commands and button interactions, so that I can operate the system hands-free when needed.

#### Acceptance Criteria

1. THE Frontend Interface SHALL provide a start/stop button for system activation and deactivation
2. WHEN the user says "scan" during STATE_BLOCKED, THE Backend Server SHALL recognize the command using Vosk STT
3. WHEN a "scan" command is recognized, THE Backend Server SHALL transition from STATE_BLOCKED to STATE_SCANNING
4. THE Frontend Interface SHALL send start/stop commands to the backend via WebSocket connection
5. THE Backend Server SHALL process voice commands only during appropriate FSM states

### Requirement 5

**User Story:** As a user, I want to receive clear audio feedback about my navigation status and required actions, so that I can respond appropriately to changing conditions.

#### Acceptance Criteria

1. THE Frontend Interface SHALL use browser SpeechSynthesis API to provide audio feedback
2. WHEN the backend sends speak commands via WebSocket, THE Frontend Interface SHALL immediately vocalize the instructions
3. THE Backend Server SHALL provide state-appropriate audio messages for each FSM transition
4. WHEN language change commands are received, THE Frontend Interface SHALL update speech synthesis language settings
5. THE Frontend Interface SHALL prioritize urgent safety messages over routine navigation feedback

### Requirement 6

**User Story:** As a system administrator, I want the backend to operate as a reliable finite state machine, so that navigation guidance follows predictable and safe behavioral patterns.

#### Acceptance Criteria

1. THE Backend Server SHALL implement four distinct states: STATE_IDLE, STATE_SCANNING, STATE_GUIDING, and STATE_BLOCKED
2. WHEN in STATE_IDLE, THE Backend Server SHALL wait for start commands via WebSocket
3. WHILE in STATE_SCANNING, THE Backend Server SHALL use optical flow analysis to confirm user stillness before path detection
4. WHEN in STATE_GUIDING, THE Backend Server SHALL continuously run YOLOv11 for obstacle detection
5. WHILE in STATE_BLOCKED, THE Backend Server SHALL use optical flow analysis to confirm user has stopped moving

### Requirement 7

**User Story:** As a system operator, I want continuous visual monitoring with optional audio guidance, so that I can monitor navigation activity and provide guidance only when needed.

#### Acceptance Criteria

1. THE Frontend Interface SHALL automatically display processed video stream with AI overlays when connected to backend
2. THE Frontend Interface SHALL show real-time FSM state changes and object detection results for continuous monitoring
3. WHEN "Start Navigation" is activated, THE Frontend Interface SHALL enable audio guidance with appropriate urgency levels
4. WHEN "Stop Navigation" is activated, THE Frontend Interface SHALL mute audio guidance while maintaining visual monitoring
5. THE Frontend Interface SHALL provide visual feedback for all navigation events regardless of audio guidance state

### Requirement 7.1

**User Story:** As a developer, I want the system to maintain real-time communication between all components, so that navigation guidance remains responsive and synchronized.

#### Acceptance Criteria

1. THE Backend Server SHALL maintain WebSocket connections for bidirectional communication with the frontend
2. THE Backend Server SHALL provide processed video via dedicated MJPEG streaming endpoint accessible at all times
3. WHEN state changes occur, THE Backend Server SHALL immediately send updates via WebSocket
4. THE Frontend Interface SHALL maintain persistent WebSocket connection during active sessions
5. WHEN WebSocket connection is lost, THE Frontend Interface SHALL attempt automatic reconnection

### Requirement 8

**User Story:** As a user with health monitoring needs, I want the system to optionally integrate vital sensor data to provide personalized navigation commands, so that I can receive guidance appropriate to my physical condition.

#### Acceptance Criteria

1. WHERE vital sensor data is available, THE Client Device SHALL capture and stream biometric information alongside video and audio
2. WHERE heart rate data indicates elevated stress, THE Backend Server SHALL provide "walk slowly" audio commands during STATE_GUIDING
3. WHERE sensor data indicates fatigue or distress, THE Backend Server SHALL suggest rest breaks through audio feedback
4. WHERE vital sensor integration is enabled, THE Backend Server SHALL adjust navigation pace recommendations based on real-time biometric data
5. THE Backend Server SHALL operate normally without vital sensor data when this optional feature is not available