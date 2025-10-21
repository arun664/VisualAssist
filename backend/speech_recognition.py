# Speech Recognition Module
# Implements Vosk speech-to-text integration for voice command processing

import json
import logging
import asyncio
import wave
import io
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np

try:
    import vosk
except ImportError:
    vosk = None
    logging.warning("Vosk not available. Speech recognition will be disabled.")

logger = logging.getLogger(__name__)

class VoiceCommandProcessor:
    """
    Vosk-based speech recognition for voice command processing
    Implements requirements 4.2, 4.3 for "scan" command intent recognition
    """
    
    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.vosk_model = None
        self.recognizer = None
        self.is_initialized = False
        
        # Command intent patterns
        self.scan_command_patterns = [
            "scan", "start scan", "begin scan", "scan environment",
            "look around", "check area", "analyze", "search"
        ]
        
        # Audio processing buffer
        self.audio_buffer = bytearray()
        self.buffer_size = 4096  # Process audio in chunks
        
        # Callback for recognized commands
        self.command_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        
        # State-aware processing - only process commands in appropriate states
        self.state_aware_processing = True
        self.current_fsm_state = None
        
        logger.info(f"VoiceCommandProcessor initialized with sample_rate={sample_rate}")
    
    async def initialize(self, model_path: Optional[str] = None) -> bool:
        """
        Initialize Vosk model and recognizer
        Set up Vosk model loading and audio stream processing
        """
        if vosk is None:
            logger.error("Vosk library not available. Cannot initialize speech recognition.")
            return False
        
        try:
            # Use provided model path or try to find default model
            if model_path:
                self.model_path = model_path
            elif not self.model_path:
                self.model_path = self._find_default_model()
            
            if not self.model_path or not Path(self.model_path).exists():
                logger.error(f"Vosk model not found at path: {self.model_path}")
                return False
            
            # Initialize Vosk model
            logger.info(f"Loading Vosk model from: {self.model_path}")
            self.vosk_model = vosk.Model(self.model_path)
            
            # Create recognizer
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
            
            # Configure recognizer for better command recognition
            self.recognizer.SetWords(True)
            
            self.is_initialized = True
            logger.info("Vosk speech recognition initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vosk speech recognition: {e}")
            self.is_initialized = False
            return False
    
    def _find_default_model(self) -> Optional[str]:
        """Try to find default Vosk model in common locations"""
        possible_paths = [
            "models/vosk-model-en-us-0.22",
            "models/vosk-model-small-en-us-0.15",
            "../models/vosk-model-en-us-0.22",
            "/opt/vosk-models/vosk-model-en-us-0.22"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"Found default Vosk model at: {path}")
                return path
        
        logger.warning("No default Vosk model found. Please specify model_path.")
        return None
    
    def set_command_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set callback function for recognized commands"""
        self.command_callback = callback
        logger.info("Command callback registered")
    
    def set_current_fsm_state(self, state):
        """
        Set current FSM state for state-appropriate processing
        Requirement 4.2, 4.3: Add audio processing only during BLOCKED state
        """
        self.current_fsm_state = state
        logger.debug(f"Speech recognition FSM state updated to: {state}")
    
    def should_process_audio(self) -> bool:
        """
        Determine if audio should be processed based on current FSM state
        Requirement 4.2: Add audio processing only during BLOCKED state
        """
        if not self.state_aware_processing:
            return True
        
        # Only process voice commands during BLOCKED state
        if self.current_fsm_state and hasattr(self.current_fsm_state, 'value'):
            return self.current_fsm_state.value == "blocked"
        
        return False
    
    async def process_audio_stream(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process audio data stream for speech recognition with comprehensive error handling
        Create audio data ingestion from WebRTC streams
        Implements command validation and state-appropriate processing
        """
        if not self.is_initialized:
            logger.warning("Speech recognition not initialized. Attempting fallback processing.")
            return await self._process_audio_fallback(audio_data)
        
        # Check if we should process audio based on FSM state
        if not self.should_process_audio():
            logger.debug(f"Skipping audio processing - current state: {self.current_fsm_state}")
            return {
                "status": "skipped",
                "reason": "audio_processing_disabled_for_current_state",
                "current_state": self.current_fsm_state.value if self.current_fsm_state else None
            }
        
        try:
            # Validate audio data
            if not audio_data or len(audio_data) == 0:
                logger.warning("Empty audio data received")
                return {
                    "status": "error",
                    "error": "empty_audio_data",
                    "timestamp": asyncio.get_event_loop().time()
                }
            
            # Add audio data to buffer with overflow protection
            if len(self.audio_buffer) > 100000:  # 100KB buffer limit
                logger.warning("Audio buffer overflow - clearing buffer")
                self.audio_buffer.clear()
            
            self.audio_buffer.extend(audio_data)
            
            # Process audio in chunks with error recovery
            results = []
            processing_errors = 0
            max_processing_errors = 3
            
            while len(self.audio_buffer) >= self.buffer_size:
                # Extract chunk from buffer
                chunk = bytes(self.audio_buffer[:self.buffer_size])
                self.audio_buffer = self.audio_buffer[self.buffer_size:]
                
                try:
                    # Process chunk with Vosk
                    result = await self._process_audio_chunk(chunk)
                    if result:
                        results.append(result)
                        processing_errors = 0  # Reset error counter on success
                        
                except Exception as chunk_error:
                    processing_errors += 1
                    logger.error(f"Error processing audio chunk: {chunk_error}")
                    
                    if processing_errors >= max_processing_errors:
                        logger.error("Too many audio processing errors - switching to fallback")
                        return await self._process_audio_fallback(audio_data)
            
            # Return combined results
            if results:
                return {
                    "status": "success",
                    "results": results,
                    "timestamp": asyncio.get_event_loop().time(),
                    "processed_in_state": self.current_fsm_state.value if self.current_fsm_state else None,
                    "processing_errors": processing_errors
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Critical error in audio stream processing: {e}")
            
            # Attempt to recover by reinitializing
            try:
                logger.info("Attempting to recover speech recognition system")
                recovery_success = await self._attempt_recovery()
                
                if recovery_success:
                    logger.info("Speech recognition recovery successful")
                    return {
                        "status": "recovered",
                        "message": "Speech recognition system recovered",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                else:
                    logger.error("Speech recognition recovery failed")
                    return await self._process_audio_fallback(audio_data)
                    
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {recovery_error}")
                return await self._process_audio_fallback(audio_data)
    
    async def _process_audio_fallback(self, audio_data: bytes) -> Dict[str, Any]:
        """Fallback audio processing when Vosk is unavailable"""
        logger.info("Using fallback audio processing")
        
        try:
            # Simple volume-based activity detection
            if len(audio_data) > 0:
                # Convert bytes to numpy array for analysis
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_array**2))
                
                # If volume is above threshold, assume potential voice command
                if volume > 1000:  # Adjust threshold as needed
                    logger.info("Audio activity detected - prompting for manual command")
                    return {
                        "status": "fallback_activity_detected",
                        "message": "Voice activity detected but speech recognition unavailable",
                        "volume": float(volume),
                        "suggestion": "Please use manual controls or check speech recognition system",
                        "timestamp": asyncio.get_event_loop().time()
                    }
            
            return {
                "status": "fallback_no_activity",
                "message": "No significant audio activity detected",
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Fallback audio processing failed: {e}")
            return {
                "status": "fallback_error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def _attempt_recovery(self) -> bool:
        """Attempt to recover speech recognition system"""
        try:
            # Clear buffers
            self.reset_audio_buffer()
            
            # Reinitialize recognizer
            if self.vosk_model:
                self.recognizer = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
                self.recognizer.SetWords(True)
                logger.info("Speech recognition recognizer reinitialized")
                return True
            else:
                # Try to reload model
                return await self.initialize(self.model_path)
                
        except Exception as e:
            logger.error(f"Speech recognition recovery failed: {e}")
            return False
    
    async def _process_audio_chunk(self, audio_chunk: bytes) -> Optional[Dict[str, Any]]:
        """Process individual audio chunk with Vosk recognizer"""
        try:
            # Feed audio data to recognizer
            if self.recognizer.AcceptWaveform(audio_chunk):
                # Final result available
                result_json = self.recognizer.Result()
                result_data = json.loads(result_json)
                
                if result_data.get("text"):
                    transcription = result_data["text"].strip().lower()
                    logger.info(f"Speech recognized: '{transcription}'")
                    
                    # Check for command intents
                    command_intent = self.detect_scan_intent(transcription)
                    
                    result = {
                        "type": "final",
                        "transcription": transcription,
                        "confidence": result_data.get("confidence", 0.0),
                        "command_intent": command_intent
                    }
                    
                    # Process command with validation if detected
                    if command_intent:
                        validation_result = self.validate_command_for_state(command_intent, self.current_fsm_state)
                        command_response = await self.create_voice_command_response(command_intent, validation_result)
                        
                        result["validation"] = validation_result
                        result["response"] = command_response
                        
                        # Trigger callback with processed command
                        if self.command_callback:
                            await self._trigger_command_callback(command_intent, result)
                    
                    return result
            else:
                # Partial result
                partial_json = self.recognizer.PartialResult()
                partial_data = json.loads(partial_json)
                
                if partial_data.get("partial"):
                    return {
                        "type": "partial",
                        "transcription": partial_data["partial"].strip().lower(),
                        "confidence": 0.0
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def detect_scan_intent(self, transcription: str) -> Optional[str]:
        """
        Implement "scan" command intent recognition
        Requirement 4.3: Implement "scan" command intent recognition
        """
        if not transcription:
            return None
        
        transcription_lower = transcription.lower().strip()
        
        # Check for scan command patterns
        for pattern in self.scan_command_patterns:
            if pattern in transcription_lower:
                logger.info(f"Scan command detected: '{transcription}' matches pattern '{pattern}'")
                return "scan"
        
        # Check for exact word matches
        words = transcription_lower.split()
        if "scan" in words:
            logger.info(f"Scan command detected: word 'scan' found in '{transcription}'")
            return "scan"
        
        return None
    
    def validate_command_for_state(self, command: str, current_state) -> Dict[str, Any]:
        """
        Implement command validation and state-appropriate processing
        Requirement 4.2, 4.4: Command validation and state-appropriate processing
        """
        validation_result = {
            "command": command,
            "current_state": current_state.value if current_state else None,
            "is_valid": False,
            "reason": "",
            "action": None
        }
        
        if not current_state:
            validation_result["reason"] = "No current state available"
            return validation_result
        
        # Validate scan command
        if command == "scan":
            if current_state.value == "blocked":
                validation_result["is_valid"] = True
                validation_result["action"] = "transition_to_scanning"
                validation_result["reason"] = "Scan command valid in BLOCKED state"
            else:
                validation_result["reason"] = f"Scan command not valid in {current_state.value} state"
        else:
            validation_result["reason"] = f"Unknown command: {command}"
        
        return validation_result
    
    async def create_voice_command_response(self, command: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create voice command response system
        Requirement 4.4: Create voice command response system
        """
        response = {
            "type": "voice_command_response",
            "command": command,
            "validation": validation_result,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        if validation_result["is_valid"]:
            if command == "scan" and validation_result["action"] == "transition_to_scanning":
                response["speak_message"] = "Scan command received. Starting environment scan."
                response["fsm_action"] = "handle_scan_command"
                response["success"] = True
            else:
                response["speak_message"] = f"Command '{command}' processed successfully."
                response["success"] = True
        else:
            # Provide helpful feedback for invalid commands
            if validation_result["current_state"] != "blocked":
                response["speak_message"] = "Voice commands are only available when navigation is blocked."
            else:
                response["speak_message"] = f"Command '{command}' not recognized. Try saying 'scan'."
            response["success"] = False
        
        return response
    
    async def _trigger_command_callback(self, command: str, result_data: Dict[str, Any]):
        """Trigger command callback with recognized command"""
        try:
            if self.command_callback:
                if asyncio.iscoroutinefunction(self.command_callback):
                    await self.command_callback(command, result_data)
                else:
                    self.command_callback(command, result_data)
        except Exception as e:
            logger.error(f"Error in command callback: {e}")
    
    async def process_webrtc_audio_frame(self, audio_frame) -> Optional[Dict[str, Any]]:
        """
        Process audio frame from WebRTC stream
        Convert WebRTC audio frame to format suitable for Vosk processing
        """
        try:
            # Convert audio frame to bytes
            # Note: This is a placeholder - actual implementation depends on WebRTC frame format
            if hasattr(audio_frame, 'to_ndarray'):
                # Convert to numpy array
                audio_array = audio_frame.to_ndarray()
                
                # Ensure correct format for Vosk (16-bit PCM, mono)
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)
                
                # Convert to mono if stereo
                if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1).astype(np.int16)
                
                # Convert to bytes
                audio_bytes = audio_array.tobytes()
                
                # Process with speech recognition
                return await self.process_audio_stream(audio_bytes)
            
            else:
                logger.warning("Unsupported audio frame format for speech recognition")
                return None
                
        except Exception as e:
            logger.error(f"Error processing WebRTC audio frame: {e}")
            return None
    
    def reset_audio_buffer(self):
        """Reset audio processing buffer"""
        self.audio_buffer.clear()
        logger.debug("Audio buffer reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of speech recognition system"""
        return {
            "initialized": self.is_initialized,
            "model_path": self.model_path,
            "sample_rate": self.sample_rate,
            "vosk_available": vosk is not None,
            "buffer_size": len(self.audio_buffer),
            "supported_commands": self.scan_command_patterns
        }
    
    async def shutdown(self):
        """Clean up speech recognition resources"""
        try:
            self.reset_audio_buffer()
            self.recognizer = None
            self.vosk_model = None
            self.is_initialized = False
            logger.info("Speech recognition system shut down")
        except Exception as e:
            logger.error(f"Error during speech recognition shutdown: {e}")

# Global speech recognition processor instance
speech_processor = VoiceCommandProcessor()