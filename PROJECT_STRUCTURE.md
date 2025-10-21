# AI Navigation Assistant - Project Structure

```
ai-navigation-assistant/
├── backend/                    # Python FastAPI backend server
│   ├── core/                  # Core system modules (FSM, etc.)
│   ├── models/                # Data models and structures
│   ├── services/              # Service layer (CV, STT, etc.)
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── requirements.txt      # Python dependencies
│   └── .env.example         # Environment configuration template
│
├── frontend/                  # Web frontend interface
│   ├── index.html           # Main HTML page
│   ├── styles.css           # CSS styling
│   ├── app.js              # Frontend JavaScript logic
│   └── package.json        # Frontend configuration
│
├── client/                   # Client device capture component
│   ├── index.html          # Client HTML page
│   ├── client-styles.css   # Client CSS styling
│   ├── client.js           # WebRTC streaming logic
│   └── package.json        # Client configuration
│
├── .kiro/specs/ai-navigation-assistant/  # Specification documents
│   ├── requirements.md     # System requirements
│   ├── design.md          # System design document
│   └── tasks.md           # Implementation task list
│
├── config.json            # Global configuration
├── setup.py              # Automated setup script
├── README.md             # Project documentation
└── PROJECT_STRUCTURE.md  # This file
```

## Component Overview

### Backend Server (Port 8000)
- **FastAPI** web server with WebSocket support
- **YOLOv11** object detection for obstacle identification
- **OpenCV** for optical flow analysis and video processing
- **Vosk** speech-to-text for voice command recognition
- **Finite State Machine** for navigation logic

### Frontend Interface (Port 3000)
- **HTML/CSS/JS** web interface for user interaction
- **WebSocket** communication with backend
- **Web Speech API** for audio feedback
- **MJPEG** video stream display

### Client Device (Port 3001)
- **WebRTC** video/audio capture and streaming
- **Camera/Microphone** access via getUserMedia
- **Optional biometric** sensor integration

## Development Workflow

1. **Setup**: Run `python setup.py` for automated environment setup
2. **Backend**: Start with `python backend/main.py`
3. **Frontend**: Serve with `python -m http.server 3000` in frontend/
4. **Client**: Serve with `python -m http.server 3001` in client/
5. **Access**: Open frontend and client URLs in browser

## Next Implementation Steps

Refer to `.kiro/specs/ai-navigation-assistant/tasks.md` for detailed implementation tasks.
Current status: ✅ Task 1 Complete - Project structure and dependencies set up.