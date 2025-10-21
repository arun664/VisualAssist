#!/usr/bin/env python3
"""
Navigation Workflow Coordinator
Orchestrates the complete navigation workflow by coordinating all components

This module implements task 10.1 requirements:
- Integrate client video/audio streaming with backend processing
- Connect backend FSM state changes with frontend audio feedback
- Wire processed video streaming from backend to frontend display
- Test complete user journey from start to obstacle detection and recovery
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """Current state of the navigation workflow"""
    fsm_state: str
    client_connected: bool
    video_streaming: bool
    audio_processing: bool
    safety_monitoring: bool
    last_frame_time: Optional[float]
    last_audio_time: Optional[float]

class NavigationWorkflowCoordinator:
    """
    Coordinates the complete navigation workflow across all components
    Implements requirements 1.1, 1.3, 1.4, 1.5 for complete user journey
    """
    
    def __init__(self):
        self.workflow_state = WorkflowState(
            fsm_state="idle",
            client_connected=False,
            video_streaming=False,
            audio_processing=False,
            safety_monitoring=True,
            last_frame_time=None,
            last_audio_time=None
        )
        
        self.component_references = {}
        self.workflow_metrics = {
            "session_start_time": None,
            "total_frames_processed": 0,
            "total_state_transitions": 0,
            "total_audio_commands": 0,
            "safety_violations": 0,
            "emergency_activations": 0
        }
        
        self.active_sessions = {}
        self.workflow_callbacks = []
        
        logger.info("Navigation Workflow Coordinator initialized")
    
    async def initialize_components(self):
        """Initialize and connect all system components"""
        logger.info("Initializing navigation workflow components...")
        
        try:
            # Initialize FSM
            from navigation_fsm import navigation_fsm
            self.component_references["fsm"] = navigation_fsm
            
            # Set up FSM callback for workflow coordination
            navigation_fsm.set_state_change_callback(self._handle_fsm_state_change)
            
            # Initialize computer vision
            from computer_vision import get_vision_processor
            self.component_references["vision"] = get_vision_processor()
            
            # Initialize speech recognition
            from speech_recognition import speech_processor
            self.component_references["speech"] = speech_processor
            
            # Set up speech recognition callback
            speech_processor.set_command_callback(self._handle_voice_command)
            
            # Initialize WebSocket manager
            from websocket_manager import websocket_manager
            self.component_references["websocket"] = websocket_manager
            
            # Initialize WebRTC manager
            from webrtc_handler import webrtc_manager
            self.component_references["webrtc"] = webrtc_manager
            
            # Initialize safety monitor
            from safety_monitor import safety_monitor
            self.component_references["safety"] = safety_monitor
            
            # Set up safety monitoring callback
            safety_monitor.add_alert_callback(self._handle_safety_alert)
            
            logger.info("All workflow components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing workflow components: {e}")
            return False
    
    async def start_navigation_session(self, client_id: str) -> bool:
        """
        Start a complete navigation session for a client
        Coordinates all components for the complete user journey
        """
        logger.info(f"Starting navigation session for client {client_id}")
        
        try:
            # Record session start
            session_data = {
                "client_id": client_id,
                "start_time": time.time(),
                "state": "initializing",
                "components_ready": False
            }
            self.active_sessions[client_id] = session_data
            self.workflow_metrics["session_start_time"] = time.time()
            
            # Verify all components are ready
            components_ready = await self._verify_components_ready()
            if not components_ready:
                raise Exception("Not all components are ready for navigation session")
            
            session_data["components_ready"] = True
            session_data["state"] = "ready"
            
            # Update workflow state
            self.workflow_state.client_connected = True
            
            # Notify components about session start
            await self._notify_session_start(client_id)
            
            logger.info(f"Navigation session started successfully for client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting navigation session for {client_id}: {e}")
            if client_id in self.active_sessions:
                self.active_sessions[client_id]["state"] = "error"
                self.active_sessions[client_id]["error"] = str(e)
            return False
    
    async def handle_client_video_stream(self, client_id: str, video_frame) -> Dict[str, Any]:
        """
        Handle incoming video stream from client and coordinate processing
        Integrates client video/audio streaming with backend processing
        """
        try:
            frame_start_time = time.time()
            
            # Update workflow state
            self.workflow_state.video_streaming = True
            self.workflow_state.last_frame_time = frame_start_time
            self.workflow_metrics["total_frames_processed"] += 1
            
            # Process frame with computer vision
            vision_processor = self.component_references["vision"]
            processing_results = await vision_processor.process_frame_complete(video_frame)
            
            # Coordinate with FSM based on current state
            fsm_response = await self._coordinate_fsm_processing(processing_results)
            
            # Monitor processing latency
            frame_end_time = time.time()
            await self._monitor_frame_processing_latency(frame_start_time, frame_end_time)
            
            # Prepare response for frontend
            response = {
                "client_id": client_id,
                "processing_results": processing_results,
                "fsm_response": fsm_response,
                "workflow_state": self.workflow_state.fsm_state,
                "processing_time": frame_end_time - frame_start_time,
                "timestamp": frame_start_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling video stream from {client_id}: {e}")
            return {"error": str(e), "client_id": client_id}
    
    async def handle_client_audio_stream(self, client_id: str, audio_data: bytes) -> Dict[str, Any]:
        """
        Handle incoming audio stream from client and coordinate speech processing
        Integrates audio processing with FSM state management
        """
        try:
            audio_start_time = time.time()
            
            # Update workflow state
            self.workflow_state.audio_processing = True
            self.workflow_state.last_audio_time = audio_start_time
            
            # Process audio with speech recognition
            speech_processor = self.component_references["speech"]
            recognition_result = await speech_processor.process_audio_stream(audio_data)
            
            # Coordinate with FSM if commands detected
            fsm_response = None
            if recognition_result and recognition_result.get("status") == "success":
                fsm_response = await self._coordinate_voice_command_processing(recognition_result)
                self.workflow_metrics["total_audio_commands"] += 1
            
            # Monitor audio processing latency
            audio_end_time = time.time()
            await self._monitor_audio_processing_latency(audio_start_time, audio_end_time)
            
            # Prepare response
            response = {
                "client_id": client_id,
                "recognition_result": recognition_result,
                "fsm_response": fsm_response,
                "processing_time": audio_end_time - audio_start_time,
                "timestamp": audio_start_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling audio stream from {client_id}: {e}")
            return {"error": str(e), "client_id": client_id}
    
    async def _coordinate_fsm_processing(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate FSM processing based on computer vision results
        Implements state-specific processing logic
        """
        try:
            fsm = self.component_references["fsm"]
            current_state = fsm.get_current_state()
            
            fsm_response = {
                "current_state": current_state.value,
                "action_taken": None,
                "state_changed": False,
                "message": None
            }
            
            # State-specific coordination logic
            if current_state.value == "scanning":
                # Check for user stillness and clear path
                motion_analysis = processing_results.get("motion_analysis", {})
                path_analysis = processing_results.get("path_analysis", {})
                
                if motion_analysis.get("is_stationary", False) and path_analysis.get("path_clear", False):
                    success = await fsm.handle_user_stationary_and_path_clear()
                    if success:
                        fsm_response["action_taken"] = "transition_to_guiding"
                        fsm_response["state_changed"] = True
                        fsm_response["message"] = "User stationary, path clear - transitioning to guiding"
                        self.workflow_metrics["total_state_transitions"] += 1
                
            elif current_state.value == "guiding":
                # Check for obstacles
                detections = processing_results.get("detections", [])
                close_obstacles = self._identify_close_obstacles(detections)
                
                if close_obstacles:
                    success = await fsm.handle_obstacle_detected()
                    if success:
                        fsm_response["action_taken"] = "transition_to_blocked"
                        fsm_response["state_changed"] = True
                        fsm_response["message"] = f"Obstacles detected - transitioning to blocked ({len(close_obstacles)} obstacles)"
                        self.workflow_metrics["total_state_transitions"] += 1
                
            elif current_state.value == "blocked":
                # Monitor user compliance with stop command
                motion_analysis = processing_results.get("motion_analysis", {})
                
                if not motion_analysis.get("is_stationary", True):
                    # User still moving - escalate warnings
                    fsm_response["action_taken"] = "escalate_warnings"
                    fsm_response["message"] = "User still moving in blocked state - escalating warnings"
                    
                    # Could trigger additional safety protocols here
                    await self._handle_user_non_compliance()
            
            return fsm_response
            
        except Exception as e:
            logger.error(f"Error coordinating FSM processing: {e}")
            return {"error": str(e)}
    
    async def _coordinate_voice_command_processing(self, recognition_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate voice command processing with FSM
        Handles voice commands in appropriate states
        """
        try:
            fsm = self.component_references["fsm"]
            
            # Extract commands from recognition results
            commands_detected = []
            for result in recognition_result.get("results", []):
                if result.get("command_intent"):
                    commands_detected.append(result["command_intent"])
            
            fsm_response = {
                "commands_processed": [],
                "state_changes": [],
                "messages": []
            }
            
            # Process each detected command
            for command in commands_detected:
                if command == "scan":
                    success = await fsm.handle_scan_command()
                    if success:
                        fsm_response["commands_processed"].append("scan")
                        fsm_response["state_changes"].append("scanning")
                        fsm_response["messages"].append("Scan command processed - transitioning to scanning")
                        self.workflow_metrics["total_state_transitions"] += 1
            
            return fsm_response
            
        except Exception as e:
            logger.error(f"Error coordinating voice command processing: {e}")
            return {"error": str(e)}
    
    def _identify_close_obstacles(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify obstacles that are close enough to warrant stopping
        Uses detection confidence and bounding box analysis
        """
        close_obstacles = []
        
        for detection in detections:
            try:
                bbox = detection.get("bbox", [0, 0, 0, 0])
                confidence = detection.get("confidence", 0.0)
                
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # Calculate obstacle characteristics
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Determine if obstacle is close/dangerous
                    is_large = area > 10000  # Large objects are likely close
                    is_in_path = 200 < center_x < 440  # Center third of 640px frame
                    is_low = center_y > 300  # Lower half of 480px frame
                    is_confident = confidence > 0.5
                    
                    if is_large and is_in_path and is_low and is_confident:
                        close_obstacles.append(detection)
                        
            except Exception as e:
                logger.error(f"Error analyzing obstacle detection: {e}")
                continue
        
        return close_obstacles
    
    async def _handle_user_non_compliance(self):
        """
        Handle cases where user doesn't comply with stop commands
        Escalates safety measures
        """
        try:
            logger.warning("User non-compliance detected - escalating safety measures")
            
            # Trigger additional safety protocols
            safety_monitor = self.component_references["safety"]
            await safety_monitor.activate_emergency_protocols("User non-compliance with stop command")
            
            # Send urgent audio messages
            websocket_manager = self.component_references["websocket"]
            urgent_message = {
                "type": "urgent_safety_alert",
                "speak": "DANGER! You must stop immediately!",
                "urgency": "emergency",
                "reason": "user_non_compliance"
            }
            await websocket_manager.broadcast(urgent_message)
            
        except Exception as e:
            logger.error(f"Error handling user non-compliance: {e}")
    
    async def _handle_fsm_state_change(self, message):
        """
        Handle FSM state changes and coordinate workflow
        Connects backend FSM state changes with frontend audio feedback
        """
        try:
            logger.info(f"Workflow coordinator handling FSM state change: {message.state.value}")
            
            # Update workflow state
            self.workflow_state.fsm_state = message.state.value
            
            # Coordinate with other components based on new state
            if message.state.value == "scanning":
                await self._coordinate_scanning_state()
            elif message.state.value == "guiding":
                await self._coordinate_guiding_state()
            elif message.state.value == "blocked":
                await self._coordinate_blocked_state()
            elif message.state.value == "idle":
                await self._coordinate_idle_state()
            
            # Notify workflow callbacks
            for callback in self.workflow_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as callback_error:
                    logger.error(f"Error in workflow callback: {callback_error}")
            
        except Exception as e:
            logger.error(f"Error handling FSM state change in workflow coordinator: {e}")
    
    async def _coordinate_scanning_state(self):
        """Coordinate components for scanning state"""
        logger.info("Coordinating components for SCANNING state")
        
        # Configure computer vision for optical flow priority
        # Configure speech recognition for reduced processing
        # Configure safety monitoring for scanning-specific thresholds
    
    async def _coordinate_guiding_state(self):
        """Coordinate components for guiding state"""
        logger.info("Coordinating components for GUIDING state")
        
        # Configure computer vision for obstacle detection priority
        # Configure safety monitoring for guiding-specific thresholds
    
    async def _coordinate_blocked_state(self):
        """Coordinate components for blocked state"""
        logger.info("Coordinating components for BLOCKED state")
        
        # Configure speech recognition for active voice command processing
        # Configure safety monitoring for critical thresholds
    
    async def _coordinate_idle_state(self):
        """Coordinate components for idle state"""
        logger.info("Coordinating components for IDLE state")
        
        # Configure all components for minimal processing
        # Reset any emergency protocols
    
    async def _handle_voice_command(self, command: str, command_data: Dict[str, Any]):
        """
        Handle voice commands from speech recognition
        Coordinates voice command processing across components
        """
        try:
            logger.info(f"Workflow coordinator handling voice command: {command}")
            
            # Update metrics
            self.workflow_metrics["total_audio_commands"] += 1
            
            # Coordinate command processing with FSM
            fsm_response = await self._coordinate_voice_command_processing({
                "results": [{"command_intent": command, "data": command_data}]
            })
            
            logger.info(f"Voice command coordination result: {fsm_response}")
            
        except Exception as e:
            logger.error(f"Error handling voice command in workflow coordinator: {e}")
    
    async def _handle_safety_alert(self, metric):
        """
        Handle safety alerts and coordinate emergency responses
        Implements comprehensive safety coordination
        """
        try:
            logger.warning(f"Workflow coordinator handling safety alert: {metric.message}")
            
            # Update metrics
            self.workflow_metrics["safety_violations"] += 1
            if metric.level.value == "emergency":
                self.workflow_metrics["emergency_activations"] += 1
            
            # Coordinate emergency response across all components
            if metric.level.value in ["critical", "emergency"]:
                await self._coordinate_emergency_response(metric)
            
        except Exception as e:
            logger.error(f"Error handling safety alert in workflow coordinator: {e}")
    
    async def _coordinate_emergency_response(self, metric):
        """
        Coordinate emergency response across all components
        Implements comprehensive emergency coordination
        """
        try:
            logger.critical(f"Coordinating emergency response for: {metric.message}")
            
            # 1. Trigger FSM emergency stop
            fsm = self.component_references["fsm"]
            await fsm.handle_emergency_stop(f"Safety alert: {metric.message}")
            
            # 2. Send emergency audio messages
            websocket_manager = self.component_references["websocket"]
            emergency_message = {
                "type": "emergency_alert",
                "speak": "EMERGENCY! Navigation system error. Please stop immediately!",
                "urgency": "emergency",
                "safety_metric": {
                    "name": metric.name,
                    "level": metric.level.value,
                    "message": metric.message
                }
            }
            await websocket_manager.broadcast(emergency_message)
            
            # 3. Update workflow state
            self.workflow_state.fsm_state = "emergency"
            
            logger.critical("Emergency response coordination completed")
            
        except Exception as e:
            logger.critical(f"Error coordinating emergency response: {e}")
    
    async def _monitor_frame_processing_latency(self, start_time: float, end_time: float):
        """Monitor frame processing latency"""
        safety_monitor = self.component_references["safety"]
        await safety_monitor.monitor_processing_latency("workflow_frame_processing", start_time, end_time)
    
    async def _monitor_audio_processing_latency(self, start_time: float, end_time: float):
        """Monitor audio processing latency"""
        safety_monitor = self.component_references["safety"]
        await safety_monitor.monitor_processing_latency("workflow_audio_processing", start_time, end_time)
    
    async def _verify_components_ready(self) -> bool:
        """Verify all components are ready for navigation session"""
        try:
            # Check FSM
            fsm = self.component_references.get("fsm")
            if not fsm or fsm.get_current_state().value != "idle":
                logger.error("FSM not ready - not in idle state")
                return False
            
            # Check computer vision
            vision = self.component_references.get("vision")
            if not vision:
                logger.error("Computer vision processor not available")
                return False
            
            # Check safety monitoring
            safety = self.component_references.get("safety")
            if not safety or not safety.monitoring_active:
                logger.error("Safety monitoring not active")
                return False
            
            logger.info("All components verified ready")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying components ready: {e}")
            return False
    
    async def _notify_session_start(self, client_id: str):
        """Notify all components about session start"""
        try:
            # Notify WebSocket manager
            websocket_manager = self.component_references["websocket"]
            session_message = {
                "type": "session_started",
                "client_id": client_id,
                "timestamp": time.time()
            }
            await websocket_manager.broadcast(session_message)
            
            logger.info(f"Session start notifications sent for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error notifying session start: {e}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "workflow_state": {
                "fsm_state": self.workflow_state.fsm_state,
                "client_connected": self.workflow_state.client_connected,
                "video_streaming": self.workflow_state.video_streaming,
                "audio_processing": self.workflow_state.audio_processing,
                "safety_monitoring": self.workflow_state.safety_monitoring
            },
            "metrics": self.workflow_metrics.copy(),
            "active_sessions": len(self.active_sessions),
            "components_initialized": len(self.component_references)
        }
    
    def add_workflow_callback(self, callback):
        """Add callback for workflow events"""
        self.workflow_callbacks.append(callback)

# Global workflow coordinator instance
workflow_coordinator = NavigationWorkflowCoordinator()