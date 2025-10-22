# WebSocket Connection Manager
# Handles bidirectional communication with frontend clients

import json
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        # Track active connections with client IDs
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_counter = 0
        
        # Custom endpoint handlers
        self.custom_endpoints = {}
        
        # Navigation feed handler (will be set during initialization)
        self.navigation_feed = None
        
        # Set up FSM state change callback
        self._setup_fsm_callback()
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept new WebSocket connection and assign client ID"""
        await websocket.accept()
        
        # Generate unique client ID
        client_id = f"client_{self.connection_counter}"
        self.connection_counter += 1
        
        # Store connection
        self.active_connections[client_id] = websocket
        
        logger.info(f"WebSocket client {client_id} connected. Total connections: {len(self.active_connections)}")
        return client_id
    
    def disconnect(self, client_id: str):
        """Remove client connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
            
    def register_endpoint(self, endpoint_path: str, connect_handler, message_handler):
        """Register custom WebSocket endpoint handlers"""
        self.custom_endpoints[endpoint_path] = {
            "connect_handler": connect_handler,
            "message_handler": message_handler
        }
        logger.info(f"Registered WebSocket endpoint: {endpoint_path}")
    
    async def handle_endpoint_connection(self, websocket: WebSocket, path: str):
        """Handle connection for a specific endpoint"""
        if path in self.custom_endpoints:
            # Use the registered handler for this endpoint
            await self.custom_endpoints[path]["connect_handler"](websocket, path)
        else:
            # Default WebSocket connection handling
            await self.connect(websocket)
    
    async def handle_endpoint_message(self, websocket: WebSocket, message: str, path: str):
        """Handle message for a specific endpoint"""
        if path in self.custom_endpoints:
            # Use the registered handler for this endpoint
            await self.custom_endpoints[path]["message_handler"](websocket, message)
        else:
            # Default message handling
            try:
                parsed_message = self.parse_message(message)
                # Process the message using default handler
            except ValueError as e:
                logger.error(f"Error parsing message on {path}: {e}")
    
    async def broadcast_navigation(self, navigation_data: dict):
        """Broadcast navigation guidance to connected clients"""
        if not self.active_connections:
            return
            
        # Format message for WebSocket transmission
        message = {
            "type": "navigation_guidance",
            "guidance": navigation_data.get("navigation", {}),
            "timestamp": navigation_data.get("timestamp", ""),
            "client_id": navigation_data.get("client_id", "unknown"),
            "frame_skipped": navigation_data.get("frame_skipped", False)
        }
        
        # Always include voice guidance for all navigation messages
        if "navigation_message" in navigation_data.get("navigation", {}):
            navigation_message = navigation_data["navigation"]["navigation_message"]
            message["speak"] = navigation_message
            
            # Log that we're sending audio message
            logger.debug(f"Sending audio guidance: {navigation_message}")
        elif "navigation" in navigation_data:
            # Ensure every navigation message has voice guidance
            navigation_guidance = navigation_data.get("navigation", {})
            if navigation_guidance.get("path_found"):
                direction = navigation_guidance.get("direction", "forward")
                message["speak"] = f"Move {direction} along the path."
            else:
                message["speak"] = "Looking for a safe path."
            
            logger.debug(f"Generated default audio guidance: {message['speak']}")
        
        # Broadcast to main WebSocket clients
        await self.broadcast(message)
        
        # Also update navigation feed if it exists
        if self.navigation_feed:
            await self.navigation_feed.update_navigation(message)
    
    def parse_message(self, raw_message: str) -> dict:
        """Parse incoming WebSocket message"""
        try:
            message = json.loads(raw_message)
            
            # Validate message structure
            if not isinstance(message, dict):
                raise ValueError("Message must be a JSON object")
            
            # Ensure required fields
            if "type" not in message:
                raise ValueError("Message must contain 'type' field")
            
            return message
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in WebSocket message: {e}")
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {e}")
            raise
    
    async def handle_command_message(self, message: dict, client_id: str) -> dict:
        """Process command messages (start/stop) and integrate with FSM"""
        command_type = message.get("type")
        
        # Import here to avoid circular imports
        from navigation_fsm import navigation_fsm
        
        if command_type == "start":
            logger.info(f"Start command received from client {client_id}")
            
            # Process start command through FSM
            success = await navigation_fsm.handle_start_command()
            
            if success:
                return {
                    "type": "command_response",
                    "command": "start",
                    "status": "success",
                    "message": "Navigation system started",
                    "current_state": navigation_fsm.get_current_state().value
                }
            else:
                return {
                    "type": "command_response",
                    "command": "start", 
                    "status": "failed",
                    "message": "Cannot start navigation from current state",
                    "current_state": navigation_fsm.get_current_state().value
                }
        
        elif command_type == "stop":
            logger.info(f"Stop command received from client {client_id}")
            
            # Process stop command through FSM
            success = await navigation_fsm.handle_stop_command()
            
            if success:
                return {
                    "type": "command_response",
                    "command": "stop",
                    "status": "success", 
                    "message": "Navigation system stopped",
                    "current_state": navigation_fsm.get_current_state().value
                }
            else:
                return {
                    "type": "command_response",
                    "command": "stop",
                    "status": "failed",
                    "message": "Error stopping navigation system",
                    "current_state": navigation_fsm.get_current_state().value
                }
        
        elif command_type == "scan":
            logger.info(f"Scan command received from client {client_id}")
            
            # Process scan command through FSM
            success = await navigation_fsm.handle_scan_command()
            
            if success:
                return {
                    "type": "command_response",
                    "command": "scan",
                    "status": "success",
                    "message": "Scanning environment",
                    "current_state": navigation_fsm.get_current_state().value
                }
            else:
                return {
                    "type": "command_response",
                    "command": "scan",
                    "status": "failed", 
                    "message": "Cannot scan from current state",
                    "current_state": navigation_fsm.get_current_state().value
                }
        
        elif command_type == "emergency_stop":
            logger.critical(f"Emergency stop command received from client {client_id}")
            
            # Process emergency stop through FSM
            success = await navigation_fsm.handle_emergency_stop()
            
            return {
                "type": "command_response",
                "command": "emergency_stop",
                "status": "success" if success else "failed",
                "message": "Emergency stop activated" if success else "Emergency stop failed",
                "current_state": navigation_fsm.get_current_state().value
            }
        
        else:
            logger.warning(f"Unknown command type: {command_type}")
            return {
                "type": "error",
                "message": f"Unknown command type: {command_type}",
                "current_state": navigation_fsm.get_current_state().value
            }

    def _setup_fsm_callback(self):
        """Set up FSM state change callback for WebSocket notifications"""
        try:
            from navigation_fsm import navigation_fsm
            navigation_fsm.set_state_change_callback(self._handle_fsm_state_change)
            logger.info("FSM state change callback registered with WebSocket manager")
            
            # Set up speech recognition integration
            self._setup_speech_recognition_integration()
        except ImportError:
            logger.warning("Could not import navigation_fsm - callback not registered")
    
    def _setup_speech_recognition_integration(self):
        """Set up speech recognition integration with FSM"""
        try:
            from speech_recognition import speech_processor
            from navigation_fsm import navigation_fsm
            
            # Set up speech recognition callback to handle voice commands
            speech_processor.set_command_callback(self._handle_voice_command)
            
            # Set up FSM voice command handler
            navigation_fsm.set_voice_command_callback(self._handle_fsm_voice_command)
            
            logger.info("Speech recognition integration set up successfully")
        except ImportError as e:
            logger.warning(f"Could not set up speech recognition integration: {e}")
    
    async def _handle_voice_command(self, command: str, command_data: Dict[str, Any]):
        """
        Handle voice commands from speech recognition
        Requirement 4.4: Create voice command response system
        """
        try:
            logger.info(f"Voice command received: {command}")
            
            # Import FSM
            from navigation_fsm import navigation_fsm
            
            # Process voice command through FSM
            success = await navigation_fsm.handle_voice_command(command, command_data)
            
            # Create response message
            response_message = {
                "type": "voice_command_processed",
                "command": command,
                "success": success,
                "command_data": command_data,
                "current_state": navigation_fsm.get_current_state().value,
                "timestamp": command_data.get("timestamp")
            }
            
            # Broadcast voice command result to all clients
            await self.broadcast(response_message)
            
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
    
    async def _handle_fsm_voice_command(self, command: str, command_data: Dict[str, Any]):
        """Handle voice commands processed by FSM"""
        logger.info(f"FSM processed voice command: {command}")
        # Additional processing if needed
    
    async def _handle_fsm_state_change(self, message):
        """Handle FSM state change notifications and broadcast to clients"""
        try:
            # Convert NavigationMessage to dict for WebSocket transmission
            state_message = message.to_dict()
            
            logger.info(f"Broadcasting FSM state change: {message.state.value}")
            
            # Broadcast state change to all connected clients
            await self.broadcast(state_message)
            
            # Notify WebRTC handler about state change for processing coordination
            try:
                from webrtc_handler import webrtc_manager
                await webrtc_manager.handle_fsm_state_change(message)
            except Exception as webrtc_error:
                logger.error(f"Error notifying WebRTC handler of state change: {webrtc_error}")
            
        except Exception as e:
            logger.error(f"Error handling FSM state change: {e}")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()