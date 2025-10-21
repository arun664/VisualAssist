# AI Navigation Assistant - Complete System Test Guide

## Overview
This guide tests the complete navigation flow from client device to processed video output with audio guidance.

## System Architecture
```
Client (3001) → WebRTC → Backend (8000) → Processed Stream → Frontend (3000)
     ↓                        ↓                              ↓
Camera/Mic              AI Processing                Audio Guidance
                       Object Detection              Visual Overlays
```

## Prerequisites
1. All servers running: `scripts\start.bat`
2. Backend accessible: http://localhost:8000/health
3. Frontend accessible: http://localhost:3000
4. Client accessible: http://localhost:3001

## Complete Test Flow

### Phase 1: Client Setup (Port 3001)
1. **Open Client**: http://localhost:3001
2. **Open Browser Console** (F12 → Console tab)
3. **Enable Camera**:
   - Click "Enable Camera" button
   - Grant camera permission when prompted
   - Verify: Camera preview appears
4. **Enable Microphone**:
   - Click "Enable Microphone" button  
   - Grant microphone permission when prompted
   - Verify: Audio level indicator shows activity
5. **Connect to Backend**:
   - Ensure server URL is: `http://localhost:8000`
   - Click "Connect to Backend" button
   - Verify: Status shows "Connected"

### Phase 2: Start Streaming
6. **Start Streaming**:
   - Click "Start Streaming" button
   - **Watch Console** for debug messages:
     ```
     startStreaming called
     Camera enabled: true
     Mic enabled: true
     Is connected: true
     Setting up WebRTC connection...
     Creating peer connection...
     Adding tracks to peer connection...
     Sending WebRTC offer to: http://localhost:8000/webrtc/offer
     Client ID: client_xxxxxxxxx
     Received answer from server: {status: "success", ...}
     WebRTC connection established
     ```
   - Verify: Stream status shows "Connected"

### Phase 3: Verify Backend Connection
7. **Check WebRTC Connections**:
   - Open: http://localhost:8000/webrtc/connections
   - **Expected**: `{"active_connections": 1, "connections": {...}}`
   - Verify: Shows 1 active connection with client ID

### Phase 4: Frontend Navigation (Port 3000)
8. **Open Frontend**: http://localhost:3000
9. **Start Navigation**:
   - Click "Start Navigation" button
   - **Expected Behavior**:
     - Button changes to "Stop Navigation"
     - System status: "Navigation started"
     - FSM state changes from "IDLE" to "SCANNING"
     - Processed video stream starts automatically
     - Audio guidance begins (if obstacles detected)

### Phase 5: Verify Complete Flow
10. **Check Processed Video Stream**:
    - Frontend should show processed video with:
      - Object detection overlays
      - Grid overlay for navigation
      - Real-time AI analysis
    - Video should update in real-time from client camera

11. **Test Audio Guidance**:
    - Move objects in front of client camera
    - **Expected**: Frontend plays audio guidance:
      - "Obstacle detected ahead"
      - "Path clear, continue forward"
      - Emergency alerts for immediate dangers

12. **Test State Transitions**:
    - Watch FSM state display on frontend
    - States should change based on camera input:
      - `IDLE` → `SCANNING` → `GUIDING` → `BLOCKED` (if obstacle)

## Verification Checklist

### ✅ Client (3001)
- [ ] Camera preview working
- [ ] Microphone level indicator active
- [ ] Connected to backend server
- [ ] Streaming status: "Connected"
- [ ] Console shows successful WebRTC negotiation

### ✅ Backend (8000)
- [ ] Health endpoint responding: `/health`
- [ ] WebRTC connections: `/webrtc/connections` shows 1 connection
- [ ] Processed video stream available: `/processed_video_stream`
- [ ] FSM status accessible: `/fsm/status`

### ✅ Frontend (3000)
- [ ] WebSocket connected to backend
- [ ] Navigation controls working
- [ ] Processed video stream displaying
- [ ] Audio guidance playing
- [ ] FSM state updates in real-time
- [ ] Visual alerts for urgent messages

## Expected User Experience

### Normal Navigation Flow:
1. **User starts client** → Enables camera/mic → Connects to backend → Starts streaming
2. **User opens frontend** → Starts navigation → Sees processed video + hears guidance
3. **System provides**:
   - Real-time object detection overlays
   - Audio navigation instructions
   - Visual alerts for obstacles
   - Continuous guidance based on camera input

### Audio Guidance Examples:
- **Normal**: "Path clear, continue forward"
- **Caution**: "Obstacle detected on the right, move left"
- **Emergency**: "STOP! Large obstacle directly ahead"

## Troubleshooting

### No WebRTC Connection (0 connections):
- Check client console for WebRTC errors
- Verify backend is running on port 8000
- Ensure camera/microphone permissions granted
- Check if "Start Streaming" was clicked after connecting

### No Video Stream on Frontend:
- Verify WebRTC connection established first
- Check if navigation was started on frontend
- Ensure processed video stream endpoint is accessible
- Check browser console for video loading errors

### No Audio Guidance:
- Check browser audio permissions
- Verify WebSocket connection between frontend and backend
- Ensure navigation state is not "IDLE"
- Check if browser has audio autoplay restrictions

### Performance Issues:
- Monitor CPU usage during AI processing
- Check network latency between components
- Verify camera resolution settings (640x480 recommended)
- Consider reducing frame rate if needed

## Success Criteria
✅ **Complete Success**: Client streams → Backend processes → Frontend displays processed video with audio guidance

The system should provide a seamless navigation experience with real-time AI analysis and audio feedback.