# AI Navigation Assistant - Service Manager
# Controls activation and deactivation of AI processing services

import asyncio
import logging
import time
from typing import Dict, Optional, Set
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ServiceStatus:
    state: ServiceState
    started_at: Optional[float] = None
    last_activity: Optional[float] = None
    error_message: Optional[str] = None
    initiated_by: Optional[str] = None

class AIServiceManager:
    """Manages AI processing services - only starts when requested by frontend"""
    
    def __init__(self):
        self.services_active = False
        self.service_status: Dict[str, ServiceStatus] = {
            "computer_vision": ServiceStatus(ServiceState.INACTIVE),
            "speech_recognition": ServiceStatus(ServiceState.INACTIVE),
            "navigation_fsm": ServiceStatus(ServiceState.INACTIVE),
            "webrtc_processing": ServiceStatus(ServiceState.INACTIVE)
        }
        
        # Track which frontends have requested services
        self.active_requesters: Set[str] = set()
        
        # Auto-shutdown timer
        self.auto_shutdown_delay = 300  # 5 minutes of inactivity
        self.shutdown_timer: Optional[asyncio.Task] = None
        
    async def start_ai_services(self, requester_id: str = "frontend") -> bool:
        """Start AI processing services when requested by frontend"""
        try:
            if self.services_active:
                logger.info(f"AI services already active, adding requester {requester_id}")
                self.active_requesters.add(requester_id)
                self._update_activity()
                return True
            
            logger.info(f"Starting AI services initiated by {requester_id}")
            
            # Update all service states to starting
            for service_name in self.service_status:
                self.service_status[service_name].state = ServiceState.STARTING
                self.service_status[service_name].initiated_by = requester_id
            
            # Start computer vision service
            await self._start_computer_vision()
            
            # Start speech recognition service
            await self._start_speech_recognition()
            
            # Start navigation FSM
            await self._start_navigation_fsm()
            
            # Start WebRTC processing
            await self._start_webrtc_processing()
            
            # Mark services as active
            self.services_active = True
            self.active_requesters.add(requester_id)
            
            # Update all service states to active
            current_time = time.time()
            for service_name in self.service_status:
                self.service_status[service_name].state = ServiceState.ACTIVE
                self.service_status[service_name].started_at = current_time
                self.service_status[service_name].last_activity = current_time
            
            # Cancel any pending shutdown
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
                self.shutdown_timer = None
            
            logger.info(f"✅ All AI services started successfully by {requester_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start AI services: {e}")
            
            # Update service states to error
            for service_name in self.service_status:
                self.service_status[service_name].state = ServiceState.ERROR
                self.service_status[service_name].error_message = str(e)
            
            return False
    
    async def stop_ai_services(self, requester_id: str = "frontend") -> bool:
        """Stop AI processing services when no longer needed"""
        try:
            # Remove this requester
            self.active_requesters.discard(requester_id)
            
            # If other requesters are still active, don't stop services
            if self.active_requesters:
                logger.info(f"AI services still needed by {len(self.active_requesters)} requesters")
                return True
            
            logger.info(f"Stopping AI services, last requester was {requester_id}")
            
            # Update service states to stopping
            for service_name in self.service_status:
                self.service_status[service_name].state = ServiceState.STOPPING
            
            # Stop services in reverse order
            await self._stop_webrtc_processing()
            await self._stop_navigation_fsm()
            await self._stop_speech_recognition()
            await self._stop_computer_vision()
            
            # Mark services as inactive
            self.services_active = False
            
            # Update service states to inactive
            for service_name in self.service_status:
                self.service_status[service_name].state = ServiceState.INACTIVE
                self.service_status[service_name].started_at = None
                self.service_status[service_name].last_activity = None
                self.service_status[service_name].initiated_by = None
                self.service_status[service_name].error_message = None
            
            logger.info("✅ All AI services stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop AI services: {e}")
            return False
    
    def _update_activity(self):
        """Update last activity timestamp and reset shutdown timer"""
        current_time = time.time()
        for service_status in self.service_status.values():
            if service_status.state == ServiceState.ACTIVE:
                service_status.last_activity = current_time
        
        # Reset auto-shutdown timer
        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        
        self.shutdown_timer = asyncio.create_task(self._auto_shutdown_timer())
    
    async def _auto_shutdown_timer(self):
        """Auto-shutdown services after inactivity period"""
        try:
            await asyncio.sleep(self.auto_shutdown_delay)
            
            if self.services_active and not self.active_requesters:
                logger.info(f"Auto-shutting down AI services after {self.auto_shutdown_delay}s of inactivity")
                await self.stop_ai_services("auto_shutdown")
                
        except asyncio.CancelledError:
            logger.debug("Auto-shutdown timer cancelled")
    
    async def _start_computer_vision(self):
        """Start computer vision service"""
        try:
            from computer_vision import get_vision_processor
            vision_processor = get_vision_processor()
            logger.info("Computer vision service started")
        except Exception as e:
            logger.error(f"Failed to start computer vision: {e}")
            raise
    
    async def _start_speech_recognition(self):
        """Start speech recognition service"""
        try:
            from speech_recognition import speech_processor
            success = await speech_processor.initialize()
            if success:
                logger.info("Speech recognition service started")
            else:
                logger.warning("Speech recognition service failed to initialize")
        except Exception as e:
            logger.error(f"Failed to start speech recognition: {e}")
            # Don't raise - speech recognition is optional
    
    async def _start_navigation_fsm(self):
        """Start navigation FSM service"""
        try:
            from navigation_fsm import navigation_fsm
            # FSM is always available, just log activation
            logger.info("Navigation FSM service activated")
        except Exception as e:
            logger.error(f"Failed to start navigation FSM: {e}")
            raise
    
    async def _start_webrtc_processing(self):
        """Start WebRTC processing service"""
        try:
            from webrtc_handler import webrtc_manager
            # WebRTC manager is always available, just log activation
            logger.info("WebRTC processing service activated")
        except Exception as e:
            logger.error(f"Failed to start WebRTC processing: {e}")
            raise
    
    async def _stop_computer_vision(self):
        """Stop computer vision service"""
        try:
            # Computer vision cleanup if needed
            logger.info("Computer vision service stopped")
        except Exception as e:
            logger.error(f"Error stopping computer vision: {e}")
    
    async def _stop_speech_recognition(self):
        """Stop speech recognition service"""
        try:
            from speech_recognition import speech_processor
            await speech_processor.shutdown()
            logger.info("Speech recognition service stopped")
        except Exception as e:
            logger.error(f"Error stopping speech recognition: {e}")
    
    async def _stop_navigation_fsm(self):
        """Stop navigation FSM service"""
        try:
            # FSM cleanup if needed
            logger.info("Navigation FSM service deactivated")
        except Exception as e:
            logger.error(f"Error stopping navigation FSM: {e}")
    
    async def _stop_webrtc_processing(self):
        """Stop WebRTC processing service"""
        try:
            # WebRTC cleanup if needed
            logger.info("WebRTC processing service deactivated")
        except Exception as e:
            logger.error(f"Error stopping WebRTC processing: {e}")
    
    def get_service_status(self) -> Dict[str, any]:
        """Get current status of all AI services"""
        return {
            "services_active": self.services_active,
            "active_requesters": list(self.active_requesters),
            "requester_count": len(self.active_requesters),
            "services": {
                name: {
                    "state": status.state.value,
                    "started_at": status.started_at,
                    "last_activity": status.last_activity,
                    "uptime": time.time() - status.started_at if status.started_at else 0,
                    "initiated_by": status.initiated_by,
                    "error_message": status.error_message
                }
                for name, status in self.service_status.items()
            },
            "auto_shutdown": {
                "enabled": True,
                "delay_seconds": self.auto_shutdown_delay,
                "timer_active": self.shutdown_timer is not None and not self.shutdown_timer.done()
            }
        }
    
    def is_service_active(self, service_name: str) -> bool:
        """Check if a specific service is active"""
        return (self.services_active and 
                service_name in self.service_status and 
                self.service_status[service_name].state == ServiceState.ACTIVE)
    
    def update_service_activity(self, service_name: str):
        """Update activity timestamp for a specific service"""
        if service_name in self.service_status:
            self.service_status[service_name].last_activity = time.time()
        self._update_activity()

# Global service manager instance
service_manager = AIServiceManager()