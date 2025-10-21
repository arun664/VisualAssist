# Navigation Finite State Machine
# Manages the navigation guidance process through defined states

import logging
from enum import Enum
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

class NavigationState(Enum):
    """Four distinct navigation states as per requirement 6.1"""
    STATE_IDLE = "idle"
    STATE_SCANNING = "scanning"
    STATE_GUIDING = "guiding"
    STATE_BLOCKED = "blocked"

class NavigationMessage:
    """Message structure for state change notifications"""
    def __init__(self, state: NavigationState, speak: Optional[str] = None, 
                 set_lang: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self.state = state
        self.speak = speak
        self.set_lang = set_lang
        self.data = data or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for WebSocket transmission"""
        result = {
            "type": "state_change",
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.speak:
            result["speak"] = self.speak
        
        if self.set_lang:
            result["set_lang"] = self.set_lang
        
        if self.data:
            result["data"] = self.data
            
        return result

class NavigationFSM:
    """
    Finite State Machine for navigation guidance
    Implements requirements 6.1-6.5 for predictable and safe behavioral patterns
    """
    
    def __init__(self):
        # Current state tracking
        self.current_state = NavigationState.STATE_IDLE
        self.previous_state: Optional[NavigationState] = None
        
        # State transition history for debugging
        self.state_history = []
        
        # State-specific behavior handlers
        self.state_handlers: Dict[NavigationState, Callable] = {
            NavigationState.STATE_IDLE: self._handle_idle_state,
            NavigationState.STATE_SCANNING: self._handle_scanning_state,
            NavigationState.STATE_GUIDING: self._handle_guiding_state,
            NavigationState.STATE_BLOCKED: self._handle_blocked_state
        }
        
        # Valid state transitions for validation
        self.valid_transitions: Dict[NavigationState, set] = {
            NavigationState.STATE_IDLE: {
                NavigationState.STATE_SCANNING  # On "start" command
            },
            NavigationState.STATE_SCANNING: {
                NavigationState.STATE_GUIDING,  # When user stationary + clear path
                NavigationState.STATE_IDLE      # On "stop" command
            },
            NavigationState.STATE_GUIDING: {
                NavigationState.STATE_BLOCKED,  # When obstacle detected
                NavigationState.STATE_IDLE      # On "stop" command
            },
            NavigationState.STATE_BLOCKED: {
                NavigationState.STATE_SCANNING, # When user stopped + "scan" command
                NavigationState.STATE_IDLE      # On "stop" command
            }
        }
        
        # Callback for state change notifications
        self.state_change_callback: Optional[Callable[[NavigationMessage], None]] = None
        
        # Voice command processing callback
        self.voice_command_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        
        logger.info(f"NavigationFSM initialized in state: {self.current_state.value}")
    
    def set_state_change_callback(self, callback: Callable[[NavigationMessage], None]):
        """Set callback function for state change notifications"""
        self.state_change_callback = callback
        logger.info("State change callback registered")
    
    def set_voice_command_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set callback function for voice command processing"""
        self.voice_command_callback = callback
        logger.info("Voice command callback registered")
    
    async def handle_voice_command(self, command: str, command_data: Dict[str, Any]) -> bool:
        """
        Handle voice commands with state validation
        Requirement 4.2, 4.4: Voice command response system
        """
        logger.info(f"Processing voice command: {command} in state {self.current_state.value}")
        
        # Process scan command
        if command == "scan":
            return await self.handle_scan_command()
        else:
            logger.warning(f"Unknown voice command: {command}")
            return False
    
    def get_current_state(self) -> NavigationState:
        """Get the current navigation state"""
        return self.current_state
    
    def get_state_history(self) -> list:
        """Get the state transition history"""
        return self.state_history.copy()
    
    def is_valid_transition(self, from_state: NavigationState, to_state: NavigationState) -> bool:
        """
        Validate if a state transition is allowed
        Implements state transition validation as per requirement 6.1
        """
        return to_state in self.valid_transitions.get(from_state, set())
    
    async def transition_to(self, new_state: NavigationState, 
                           speak_message: Optional[str] = None,
                           additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transition to a new state with validation and logging
        Returns True if transition was successful, False if invalid
        """
        # Validate transition
        if not self.is_valid_transition(self.current_state, new_state):
            logger.error(f"Invalid state transition: {self.current_state.value} -> {new_state.value}")
            return False
        
        # Store previous state
        self.previous_state = self.current_state
        
        # Update current state
        self.current_state = new_state
        
        # Record transition in history
        transition_record = {
            "from_state": self.previous_state.value,
            "to_state": new_state.value,
            "timestamp": datetime.now(),
            "speak_message": speak_message,
            "data": additional_data
        }
        self.state_history.append(transition_record)
        
        logger.info(f"State transition: {self.previous_state.value} -> {new_state.value}")
        
        # Create state change message
        message = NavigationMessage(
            state=new_state,
            speak=speak_message,
            data=additional_data
        )
        
        # Notify via callback if registered
        if self.state_change_callback:
            try:
                await self._notify_state_change(message)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
        
        return True
    
    async def _notify_state_change(self, message: NavigationMessage):
        """Send state change notification via callback"""
        if self.state_change_callback:
            if asyncio.iscoroutinefunction(self.state_change_callback):
                await self.state_change_callback(message)
            else:
                self.state_change_callback(message)
    
    async def handle_start_command(self) -> bool:
        """
        Handle start command from WebSocket
        Requirement 6.2: WHEN in STATE_IDLE, wait for start commands
        """
        if self.current_state == NavigationState.STATE_IDLE:
            return await self.transition_to(
                NavigationState.STATE_SCANNING,
                speak_message="Navigation system starting. Please stand still while I scan your environment."
            )
        else:
            logger.warning(f"Start command received in invalid state: {self.current_state.value}")
            return False
    
    async def handle_stop_command(self) -> bool:
        """
        Handle stop command from WebSocket
        Any State → IDLE: On "stop" command
        """
        if self.current_state != NavigationState.STATE_IDLE:
            return await self.transition_to(
                NavigationState.STATE_IDLE,
                speak_message="Navigation system stopped."
            )
        else:
            logger.info("Stop command received while already in IDLE state")
            return True
    
    async def handle_scan_command(self) -> bool:
        """
        Handle scan command from voice recognition
        BLOCKED → SCANNING: When user stopped + "scan" command recognized
        """
        if self.current_state == NavigationState.STATE_BLOCKED:
            return await self.transition_to(
                NavigationState.STATE_SCANNING,
                speak_message="Scanning environment. Please remain still."
            )
        else:
            logger.warning(f"Scan command received in invalid state: {self.current_state.value}")
            return False
    
    async def handle_user_stationary_and_path_clear(self) -> bool:
        """
        Handle transition when user is stationary and path is clear
        SCANNING → GUIDING: When user stationary + clear path detected
        """
        if self.current_state == NavigationState.STATE_SCANNING:
            return await self.transition_to(
                NavigationState.STATE_GUIDING,
                speak_message="Path clear. You may proceed forward."
            )
        else:
            logger.warning(f"Path clear detected in invalid state: {self.current_state.value}")
            return False
    
    async def handle_obstacle_detected(self) -> bool:
        """
        Handle obstacle detection during navigation
        GUIDING → BLOCKED: When obstacle detected by YOLOv11
        """
        if self.current_state == NavigationState.STATE_GUIDING:
            return await self.transition_to(
                NavigationState.STATE_BLOCKED,
                speak_message="Stop! Obstacle detected ahead."
            )
        else:
            logger.warning(f"Obstacle detected in invalid state: {self.current_state.value}")
            return False
    
    async def handle_emergency_stop(self, reason: str = "Emergency stop activated") -> bool:
        """
        Enhanced emergency stop protocol with comprehensive safety measures
        Implements emergency stop protocol for safety with error recovery
        """
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        try:
            # Emergency stop bypasses normal transition validation
            # Store previous state
            self.previous_state = self.current_state
            
            # Force transition to BLOCKED state
            self.current_state = NavigationState.STATE_BLOCKED
            
            # Record transition in history with detailed emergency information
            transition_record = {
                "from_state": self.previous_state.value,
                "to_state": NavigationState.STATE_BLOCKED.value,
                "timestamp": datetime.now(),
                "speak_message": "DANGER! STOP IMMEDIATELY!",
                "data": {
                    "emergency": True,
                    "reason": reason,
                    "emergency_id": f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "system_state": self._get_emergency_system_state()
                }
            }
            self.state_history.append(transition_record)
            
            logger.critical(f"EMERGENCY: Force transition {self.previous_state.value} -> {NavigationState.STATE_BLOCKED.value}")
            
            # Create emergency state change message with multiple urgency levels
            emergency_messages = [
                "DANGER! STOP IMMEDIATELY!",
                "EMERGENCY! DO NOT MOVE!",
                "CRITICAL ALERT! STOP NOW!"
            ]
            
            # Send multiple emergency notifications
            for i, msg in enumerate(emergency_messages):
                message = NavigationMessage(
                    state=NavigationState.STATE_BLOCKED,
                    speak=msg,
                    data={
                        "emergency": True,
                        "urgency": "critical",
                        "sequence": i + 1,
                        "total_messages": len(emergency_messages),
                        "reason": reason
                    }
                )
                
                # Notify via callback if registered
                if self.state_change_callback:
                    try:
                        await self._notify_state_change(message)
                        # Small delay between emergency messages
                        if i < len(emergency_messages) - 1:
                            await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Error in emergency state change callback: {e}")
                        # Continue with other emergency notifications even if one fails
            
            # Activate additional safety protocols
            await self._activate_emergency_protocols()
            
            return True
            
        except Exception as e:
            logger.critical(f"CRITICAL ERROR in emergency stop protocol: {e}")
            
            # Last resort emergency handling
            try:
                await self._last_resort_emergency_handling(reason, str(e))
                return True
            except Exception as last_resort_error:
                logger.critical(f"LAST RESORT EMERGENCY HANDLING FAILED: {last_resort_error}")
                return False
    
    def _get_emergency_system_state(self) -> Dict[str, Any]:
        """Capture system state during emergency for debugging"""
        try:
            return {
                "current_state": self.current_state.value,
                "previous_state": self.previous_state.value if self.previous_state else None,
                "state_history_count": len(self.state_history),
                "callback_registered": self.state_change_callback is not None,
                "voice_callback_registered": self.voice_command_callback is not None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error capturing emergency system state: {e}")
            return {"error": str(e)}
    
    async def _activate_emergency_protocols(self):
        """Activate additional safety protocols during emergency"""
        try:
            logger.critical("Activating emergency safety protocols")
            
            # Protocol 1: Disable all non-critical processing
            await self._disable_non_critical_processing()
            
            # Protocol 2: Activate emergency audio loop
            await self._activate_emergency_audio_loop()
            
            # Protocol 3: Log emergency state for recovery
            await self._log_emergency_state()
            
            logger.critical("Emergency safety protocols activated successfully")
            
        except Exception as e:
            logger.error(f"Error activating emergency protocols: {e}")
    
    async def _disable_non_critical_processing(self):
        """Disable non-critical system processing during emergency"""
        try:
            # This would integrate with other system components
            # For now, log the action
            logger.critical("Non-critical processing disabled for emergency")
            
            # Set emergency flag for other components to check
            self.emergency_mode = True
            
        except Exception as e:
            logger.error(f"Error disabling non-critical processing: {e}")
    
    async def _activate_emergency_audio_loop(self):
        """Activate continuous emergency audio warnings"""
        try:
            # Send repeated emergency audio messages
            for i in range(3):  # Send 3 additional warnings
                await asyncio.sleep(2)  # 2-second intervals
                
                emergency_message = NavigationMessage(
                    state=NavigationState.STATE_BLOCKED,
                    speak=f"Emergency alert {i + 1}. Please stop and remain stationary.",
                    data={
                        "emergency": True,
                        "urgency": "critical",
                        "loop_sequence": i + 1
                    }
                )
                
                if self.state_change_callback:
                    await self._notify_state_change(emergency_message)
                    
        except Exception as e:
            logger.error(f"Error in emergency audio loop: {e}")
    
    async def _log_emergency_state(self):
        """Log detailed emergency state for recovery and analysis"""
        try:
            emergency_log = {
                "timestamp": datetime.now().isoformat(),
                "emergency_id": f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "system_state": self._get_emergency_system_state(),
                "state_history": self.state_history[-10:],  # Last 10 state changes
                "emergency_protocols_active": True
            }
            
            logger.critical(f"EMERGENCY STATE LOG: {emergency_log}")
            
        except Exception as e:
            logger.error(f"Error logging emergency state: {e}")
    
    async def _last_resort_emergency_handling(self, original_reason: str, error_details: str):
        """Last resort emergency handling when primary emergency protocols fail"""
        logger.critical("ACTIVATING LAST RESORT EMERGENCY PROTOCOLS")
        
        try:
            # Force state to BLOCKED without any complex processing
            self.current_state = NavigationState.STATE_BLOCKED
            
            # Create minimal emergency message
            if self.state_change_callback:
                try:
                    simple_message = NavigationMessage(
                        state=NavigationState.STATE_BLOCKED,
                        speak="EMERGENCY STOP",
                        data={"last_resort": True, "original_reason": original_reason, "error": error_details}
                    )
                    await self._notify_state_change(simple_message)
                except:
                    # If even this fails, log it but don't raise
                    logger.critical("Last resort emergency notification failed")
            
            logger.critical("Last resort emergency protocols completed")
            
        except Exception as e:
            logger.critical(f"LAST RESORT EMERGENCY PROTOCOLS FAILED: {e}")
            raise
    
    def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode"""
        return getattr(self, 'emergency_mode', False)
    
    async def clear_emergency_mode(self) -> bool:
        """Clear emergency mode after manual verification"""
        try:
            if hasattr(self, 'emergency_mode'):
                self.emergency_mode = False
                logger.info("Emergency mode cleared")
                return True
            return True
        except Exception as e:
            logger.error(f"Error clearing emergency mode: {e}")
            return False
    
    # State-specific behavior handlers
    
    async def _handle_idle_state(self, frame: Optional[np.ndarray] = None, 
                                audio_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Handle IDLE state behavior
        Requirement 6.2: WHEN in STATE_IDLE, wait for start commands via WebSocket
        """
        return {
            "state": "idle",
            "action": "waiting_for_start_command",
            "message": "System ready. Send start command to begin navigation."
        }
    
    async def _handle_scanning_state(self, frame: Optional[np.ndarray] = None,
                                   audio_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Handle SCANNING state behavior
        Requirement 6.3: WHILE in STATE_SCANNING, use optical flow analysis to confirm user stillness
        """
        return {
            "state": "scanning",
            "action": "optical_flow_analysis",
            "message": "Scanning environment. Checking user stillness and path detection.",
            "requires_optical_flow": True,
            "requires_path_detection": True
        }
    
    async def _handle_guiding_state(self, frame: Optional[np.ndarray] = None,
                                  audio_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Handle GUIDING state behavior  
        Requirement 6.4: WHEN in STATE_GUIDING, continuously run YOLOv11 for obstacle detection
        """
        return {
            "state": "guiding",
            "action": "obstacle_detection",
            "message": "Guiding user. Continuously monitoring for obstacles.",
            "requires_yolo_detection": True,
            "requires_path_monitoring": True
        }
    
    async def _handle_blocked_state(self, frame: Optional[np.ndarray] = None,
                                  audio_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Handle BLOCKED state behavior
        Requirement 6.5: WHILE in STATE_BLOCKED, use optical flow analysis to confirm user has stopped
        """
        return {
            "state": "blocked",
            "action": "confirm_user_stopped",
            "message": "User blocked. Confirming user has stopped moving.",
            "requires_optical_flow": True,
            "requires_voice_recognition": True
        }
    
    async def process_frame(self, frame: np.ndarray, processing_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process video frame based on current state and computer vision results
        Delegates to appropriate state handler
        """
        if self.current_state in self.state_handlers:
            handler = self.state_handlers[self.current_state]
            return await handler(frame=frame)
        else:
            logger.error(f"No handler found for state: {self.current_state.value}")
            return {"error": f"Invalid state: {self.current_state.value}"}
    
    async def handle_audio_input(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio input based on current state
        Delegates to appropriate state handler
        """
        if self.current_state in self.state_handlers:
            handler = self.state_handlers[self.current_state]
            return await handler(audio_data=audio_data)
        else:
            logger.error(f"No handler found for state: {self.current_state.value}")
            return {"error": f"Invalid state: {self.current_state.value}"}
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current FSM state"""
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "valid_transitions": [state.value for state in self.valid_transitions[self.current_state]],
            "state_history_count": len(self.state_history),
            "last_transition": self.state_history[-1] if self.state_history else None
        }

# Global FSM instance
navigation_fsm = NavigationFSM()