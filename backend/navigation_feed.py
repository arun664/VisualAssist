"""
Navigation Feed Module for Visual Assist
This module creates a dedicated WebSocket endpoint for real-time navigation instructions.
It responds to ping requests from the frontend and delivers navigation guidance with minimal latency.
"""

import asyncio
import json
import logging
from datetime import datetime

from websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class NavigationFeed:
    """Dedicated navigation feed handler that provides real-time navigation instructions"""
    
    def __init__(self, websocket_manager):
        """Initialize the navigation feed handler
        
        Args:
            websocket_manager: The main WebSocketManager instance
        """
        self.websocket_manager = websocket_manager
        self.last_navigation = None
        self.navigation_clients = set()
        self.register_handlers()
        logger.info("Navigation feed handler initialized")
        
    def register_handlers(self):
        """Register WebSocket handlers for the navigation feed"""
        # Register a handler for the /navigation endpoint
        self.websocket_manager.register_endpoint("/navigation", 
                                                self.handle_navigation_client_connected,
                                                self.handle_navigation_client_message)
    
    async def handle_navigation_client_connected(self, websocket, path):
        """Handle a new client connection to the navigation feed
        
        Args:
            websocket: The WebSocket connection
            path: The connection path
        """
        logger.info(f"Navigation client connected: {websocket.remote_address}")
        self.navigation_clients.add(websocket)
        
        # Send the latest navigation data if available
        if self.last_navigation:
            try:
                await websocket.send(json.dumps(self.last_navigation))
                logger.debug("Sent initial navigation data to new client")
            except Exception as e:
                logger.error(f"Error sending initial navigation data: {e}")
    
    async def handle_navigation_client_message(self, websocket, message):
        """Handle a message from a navigation client
        
        Args:
            websocket: The WebSocket connection
            message: The message received
        """
        try:
            data = json.loads(message)
            
            # Handle navigation_request ping messages
            if data.get("type") == "navigation_request":
                # Respond with the latest navigation data
                if self.last_navigation:
                    response = {**self.last_navigation, 
                               "timestamp": datetime.now().timestamp() * 1000,  # ms timestamp
                               "is_response": True}
                    await websocket.send(json.dumps(response))
                    logger.debug("Responded to navigation ping")
                else:
                    # No navigation data available yet
                    await websocket.send(json.dumps({
                        "type": "navigation_guidance",
                        "guidance": {
                            "path_found": False,
                            "direction": "unknown",
                            "navigation_message": "No navigation data available yet"
                        },
                        "timestamp": datetime.now().timestamp() * 1000,
                        "is_response": True
                    }))
                    logger.debug("Responded to navigation ping with no data")
        except Exception as e:
            logger.error(f"Error handling navigation client message: {e}")
    
    async def update_navigation(self, navigation_data):
        """Update the current navigation data and broadcast to all connected clients
        
        Args:
            navigation_data: The navigation data to broadcast
        """
        # Store the latest navigation data
        self.last_navigation = {
            **navigation_data,
            "timestamp": datetime.now().timestamp() * 1000  # ms timestamp
        }
        
        # Broadcast to all connected navigation clients
        if self.navigation_clients:
            disconnected_clients = set()
            
            for client in self.navigation_clients:
                try:
                    await client.send(json.dumps(self.last_navigation))
                except Exception as e:
                    logger.error(f"Error sending navigation update: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.navigation_clients.discard(client)
            
            logger.debug(f"Broadcast navigation update to {len(self.navigation_clients)} clients")
    
    def cleanup(self):
        """Clean up resources"""
        self.navigation_clients.clear()
        logger.info("Navigation feed handler cleaned up")