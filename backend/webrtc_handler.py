# WebRTC Handler
# Manages WebRTC connections for video/audio streaming from client devices

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder

logger = logging.getLogger(__name__)

class WebRTCConnectionManager:
    """Manages WebRTC peer connections and media streams"""
    
    def __init__(self):
        # Track active peer connections by client ID
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.media_recorders: Dict[str, MediaRecorder] = {}
        self.connection_states: Dict[str, str] = {}
        
        # Error tracking for circuit breaker pattern
        self.audio_error_counts: Dict[str, int] = {}
        self.video_error_counts: Dict[str, int] = {}
        self.max_consecutive_errors = 10  # Circuit breaker threshold
        self.error_reset_time = 60  # Reset error count after 60 seconds
        self.last_error_time: Dict[str, float] = {}
        
        # Frame rate limiting for performance optimization
        self.target_fps = 0.5  # Process 1 frame every 2 seconds
        self.last_processed_frame_time: Dict[str, float] = {}
        self.latest_processed_frame: Dict[str, Any] = {}  # Store latest processed frame for MJPEG streaming
        
        # Parallel processing setup
        import concurrent.futures
        import threading
        from queue import Queue
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,  # Adjust based on CPU cores
            thread_name_prefix="ai_processing"
        )
        
        # Frame processing queue for each client
        self.frame_queues: Dict[str, Queue] = {}
        self.processing_locks: Dict[str, threading.Lock] = {}
        
        # Background processing status
        self.background_processors: Dict[str, bool] = {}
    
    async def create_peer_connection(self, client_id: str) -> RTCPeerConnection:
        """Create new RTCPeerConnection for client"""
        
        # Create peer connection
        pc = RTCPeerConnection()
        
        # Store connection
        self.peer_connections[client_id] = pc
        self.connection_states[client_id] = "new"
        
        # Set up event handlers
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            self.connection_states[client_id] = state
            logger.info(f"WebRTC connection state for {client_id}: {state}")
            
            if state == "closed" or state == "failed":
                await self.cleanup_connection(client_id)
        
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received {track.kind} track from {client_id}")
            
            if track.kind == "video":
                self._handle_video_track(track, client_id)
            elif track.kind == "audio":
                self._handle_audio_track(track, client_id)
        
        logger.info(f"Created WebRTC peer connection for client {client_id}")
        return pc
    
    def _should_process_frames(self, client_id: str, frame_type: str) -> bool:
        """Check if we should continue processing frames based on error count (circuit breaker)"""
        current_time = time.time()
        error_key = f"{client_id}_{frame_type}"
        
        # Reset error count if enough time has passed
        if error_key in self.last_error_time:
            if current_time - self.last_error_time[error_key] > self.error_reset_time:
                if frame_type == "audio":
                    self.audio_error_counts[client_id] = 0
                else:
                    self.video_error_counts[client_id] = 0
                logger.info(f"Reset {frame_type} error count for {client_id}")
        
        # Check if we've exceeded error threshold
        error_count = (self.audio_error_counts.get(client_id, 0) if frame_type == "audio" 
                      else self.video_error_counts.get(client_id, 0))
        
        if error_count >= self.max_consecutive_errors:
            logger.warning(f"Circuit breaker active for {client_id} {frame_type} processing "
                          f"({error_count} consecutive errors)")
            return False
        
        return True
    
    def _record_frame_error(self, client_id: str, frame_type: str, error: Exception):
        """Record a frame processing error and update circuit breaker state"""
        current_time = time.time()
        
        if frame_type == "audio":
            self.audio_error_counts[client_id] = self.audio_error_counts.get(client_id, 0) + 1
            error_count = self.audio_error_counts[client_id]
        else:
            self.video_error_counts[client_id] = self.video_error_counts.get(client_id, 0) + 1
            error_count = self.video_error_counts[client_id]
        
        self.last_error_time[f"{client_id}_{frame_type}"] = current_time
        
        # Log error with rate limiting
        if error_count <= 3:  # Only log first 3 errors
            logger.error(f"Error processing {frame_type} frame from {client_id} "
                        f"(error #{error_count}): {error}")
        elif error_count == self.max_consecutive_errors:
            logger.error(f"Circuit breaker activated for {client_id} {frame_type} processing "
                        f"after {error_count} consecutive errors. Last error: {error}")
        
        return error_count
    
    def _setup_parallel_processing(self, client_id: str):
        """Setup parallel processing infrastructure for a client"""
        from queue import Queue
        import threading
        
        # Create frame queue for this client
        self.frame_queues[client_id] = Queue(maxsize=3)  # Small buffer to prevent memory buildup
        self.processing_locks[client_id] = threading.Lock()
        self.background_processors[client_id] = True
        
        # Start background processing thread
        processing_thread = threading.Thread(
            target=self._background_frame_processor,
            args=(client_id,),
            name=f"frame_processor_{client_id}",
            daemon=True
        )
        processing_thread.start()
        
        logger.info(f"Started parallel processing for client {client_id}")
    
    def _background_frame_processor(self, client_id: str):
        """Background thread for processing frames in parallel"""
        import asyncio
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self.background_processors.get(client_id, False):
                try:
                    # Get frame from queue (blocking with timeout)
                    frame_data = self.frame_queues[client_id].get(timeout=1.0)
                    
                    if frame_data is None:  # Shutdown signal
                        break
                    
                    # Process frame asynchronously
                    loop.run_until_complete(self._process_frame_async(client_id, frame_data))
                    
                except Exception as e:
                    logger.error(f"Error in background frame processor for {client_id}: {e}")
                    
        finally:
            loop.close()
            logger.info(f"Background frame processor stopped for {client_id}")
    
    async def _process_frame_async(self, client_id: str, frame_data: dict):
        """Asynchronously process a single frame with parallel AI processing"""
        try:
            img = frame_data['image']
            timestamp = frame_data['timestamp']
            
            # Use thread pool for CPU-intensive AI processing
            loop = asyncio.get_event_loop()
            
            # Parallel AI processing tasks
            tasks = []
            
            # Task 1: Computer Vision Processing (CPU intensive)
            cv_task = loop.run_in_executor(
                self.thread_pool,
                self._process_computer_vision_sync,
                img
            )
            tasks.append(cv_task)
            
            # Task 2: Optical Flow (if previous frame exists) - can run in parallel
            if hasattr(self, '_previous_frames') and client_id in self._previous_frames:
                flow_task = loop.run_in_executor(
                    self.thread_pool,
                    self._process_optical_flow_sync,
                    self._previous_frames[client_id],
                    img
                )
                tasks.append(flow_task)
            
            # Wait for all parallel tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            cv_results = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else {}
            
            if cv_results and isinstance(cv_results, dict):
                # Update latest processed frame
                with self.processing_locks[client_id]:
                    self.latest_processed_frame[client_id] = {
                        "frame": cv_results.get("processed_frame", img),
                        "timestamp": timestamp,
                        "processing_time": cv_results.get("processing_time", 0),
                        "results": cv_results
                    }
                
                # Store for MJPEG streaming
                self._store_processed_frame(client_id, cv_results.get("processed_frame", img))
                
                # Handle FSM processing asynchronously
                asyncio.create_task(self._handle_fsm_async(client_id, img, cv_results))
            
            # Store current frame for next optical flow calculation
            if not hasattr(self, '_previous_frames'):
                self._previous_frames = {}
            self._previous_frames[client_id] = img.copy()
            
        except Exception as e:
            logger.error(f"Error in async frame processing for {client_id}: {e}")
    
    def _process_computer_vision_sync(self, img):
        """Synchronous computer vision processing for thread pool"""
        try:
            from computer_vision import get_vision_processor
            import asyncio
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                vision_processor = get_vision_processor()
                # Run async method in sync context
                result = loop.run_until_complete(vision_processor.process_frame_complete(img))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in sync computer vision processing: {e}")
            return {"processed_frame": img, "processing_time": 0}
    
    def _process_optical_flow_sync(self, prev_frame, curr_frame):
        """Synchronous optical flow processing for thread pool"""
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Simple frame difference for motion detection
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Calculate motion magnitude
            motion_magnitude = np.mean(diff)
            
            return {"optical_flow": motion_magnitude}
            
        except Exception as e:
            logger.error(f"Error in optical flow processing: {e}")
            return {"optical_flow": None}
    
    async def _handle_fsm_async(self, client_id: str, img, processing_results):
        """Handle FSM processing asynchronously"""
        try:
            from navigation_fsm import navigation_fsm
            from safety_monitor import safety_monitor
            
            fsm_start_time = time.time()
            await self._handle_fsm_processing(navigation_fsm, img, processing_results)
            fsm_end_time = time.time()
            
            # Monitor FSM processing latency
            await safety_monitor.monitor_processing_latency(
                "state_transition", 
                fsm_start_time, 
                fsm_end_time
            )
            
        except Exception as e:
            logger.error(f"Error in async FSM processing for {client_id}: {e}")
    
    def _handle_video_track(self, track: MediaStreamTrack, client_id: str):
        """Handle incoming video track"""
        logger.info(f"[VIDEO] Setting up video track handler for {client_id}")
        logger.info(f"Video track details - Kind: {track.kind}, ReadyState: {track.readyState}")
        
        # Create async task to process video frames
        asyncio.create_task(self._process_video_frames(track, client_id))

    def _handle_audio_track(self, track: MediaStreamTrack, client_id: str):
        """Handle incoming audio track"""
        logger.info(f"[AUDIO] Setting up audio track handler for {client_id}")
        logger.info(f"Audio track details - Kind: {track.kind}, ReadyState: {track.readyState}")
        
        # Create async task to process audio frames
        asyncio.create_task(self._process_audio_frames(track, client_id))

    async def _process_video_frames(self, track: MediaStreamTrack, client_id: str):
        """Process incoming video frames with parallel processing and safety monitoring"""
        try:
            from safety_monitor import safety_monitor
            
            # Initialize error tracking and parallel processing for this client
            self.video_error_counts[client_id] = 0
            self._setup_parallel_processing(client_id)
            
            logger.info(f"Started parallel video processing for {client_id}")
            
            while True:
                # Check circuit breaker before processing
                if not self._should_process_frames(client_id, "video"):
                    logger.info(f"Video processing paused for {client_id} due to circuit breaker")
                    await asyncio.sleep(5)  # Wait 5 seconds before checking again
                    continue
                
                frame_start_time = time.time()
                
                try:
                    frame = await track.recv()
                    logger.info(f"[VIDEO] Received video frame from {client_id} at {time.time():.2f}")
                    
                    # Reset error count on successful frame reception
                    if self.video_error_counts.get(client_id, 0) > 0:
                        logger.info(f"Video frame reception recovered for {client_id}")
                        self.video_error_counts[client_id] = 0
                    
                    # Frame rate limiting: Only process frames at target FPS (1 FPS)
                    current_time = time.time()
                    last_processed = self.last_processed_frame_time.get(client_id, 0)
                    time_since_last = current_time - last_processed
                    min_interval = 1.0 / self.target_fps  # 1 second for 1 FPS
                    
                    if time_since_last < min_interval:
                        # Skip this frame - too soon since last processed frame
                        logger.debug(f"Skipping frame from {client_id} - rate limiting "
                                   f"(last processed {time_since_last:.2f}s ago, min interval {min_interval:.2f}s)")
                        
                        # Still convert and store frame for MJPEG streaming (lightweight operation)
                        # Simple placeholder frame for rate limiting case
                        import numpy as np
                        img = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame
                        self._store_processed_frame(client_id, img)
                        continue
                    
                    # Process this frame - enough time has passed
                    self.last_processed_frame_time[client_id] = current_time
                    logger.debug(f"Queuing frame from {client_id} for parallel processing")
                    
                    # Convert WebRTC frame to OpenCV format
                    # TODO: Implement proper aiortc frame conversion
                    # For now, use a placeholder frame to avoid compilation errors
                    import numpy as np
                    logger.info(f"Processing frame from {client_id} (using placeholder for now)")
                    img = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame
                    
                    # Queue frame for parallel processing (non-blocking)
                    frame_data = {
                        "image": img,
                        "timestamp": current_time,
                        "frame_start_time": frame_start_time
                    }
                    
                    try:
                        # Try to add to queue (non-blocking)
                        if client_id in self.frame_queues:
                            self.frame_queues[client_id].put_nowait(frame_data)
                            logger.debug(f"Frame queued for parallel processing: {client_id}")
                        else:
                            logger.warning(f"No processing queue found for {client_id}")
                            
                    except Exception as queue_error:
                        # Queue is full - skip this frame to prevent blocking
                        logger.debug(f"Processing queue full for {client_id}, skipping frame")
                        
                        # Still store raw frame for MJPEG streaming
                        self._store_processed_frame(client_id, img)
                    
                    # Monitor frame reception latency (lightweight)
                    frame_end_time = time.time()
                    await safety_monitor.monitor_processing_latency(
                        "frame_reception", 
                        frame_start_time, 
                        frame_end_time
                    )
                    
                except Exception as frame_error:
                    # Use circuit breaker error recording
                    error_count = self._record_frame_error(client_id, "video", frame_error)
                    
                    # Check if this is a critical processing failure
                    frame_end_time = time.time()
                    if frame_end_time - frame_start_time > 2.0:  # 2 second timeout
                        await safety_monitor.activate_emergency_protocols(
                            f"Critical frame processing failure: {frame_error}"
                        )
                    
                    # If circuit breaker threshold reached, the next iteration will pause processing
                    continue
                
        except Exception as e:
            logger.error(f"Critical error in video frame processing for {client_id}: {e}")
            
            # Trigger emergency protocols for critical video processing failure
            try:
                from safety_monitor import safety_monitor
                await safety_monitor.activate_emergency_protocols(
                    f"Video processing pipeline failure: {e}"
                )
            except:
                pass  # Don't let safety monitoring errors crash the handler
    
    async def _handle_fsm_processing(self, navigation_fsm, img, processing_results):
        """
        Handle FSM processing with state-specific logic
        Implements complete navigation workflow integration
        """
        try:
            current_state = navigation_fsm.get_current_state()
            
            # State-specific processing logic
            if current_state.value == "scanning":
                # During scanning, check if user is stationary and path is clear
                motion_analysis = processing_results.get("motion_analysis", {})
                path_analysis = processing_results.get("path_analysis", {})
                
                if motion_analysis.get("is_stationary", False) and path_analysis.get("path_clear", False):
                    # User is stationary and path is clear - transition to guiding
                    await navigation_fsm.handle_user_stationary_and_path_clear()
                    logger.info("FSM transition: SCANNING -> GUIDING (user stationary, path clear)")
                
            elif current_state.value == "guiding":
                # During guiding, continuously monitor for obstacles
                detections = processing_results.get("detections", [])
                
                # Check if any obstacles are detected close to the user
                close_obstacles = [d for d in detections if self._is_obstacle_close(d)]
                
                if close_obstacles:
                    # Obstacle detected - transition to blocked
                    await navigation_fsm.handle_obstacle_detected()
                    logger.info(f"FSM transition: GUIDING -> BLOCKED (obstacles detected: {len(close_obstacles)})")
                
            elif current_state.value == "blocked":
                # During blocked state, verify user has stopped moving
                motion_analysis = processing_results.get("motion_analysis", {})
                
                if not motion_analysis.get("is_stationary", True):
                    # User is still moving - send additional stop warnings
                    logger.warning("User still moving in BLOCKED state - escalating warnings")
                    # This could trigger additional safety protocols
            
            # Always process frame for general FSM handling
            await navigation_fsm.process_frame(img, processing_results)
            
        except Exception as e:
            logger.error(f"Error in FSM processing: {e}")
            raise
    
    def _is_obstacle_close(self, detection):
        """
        Determine if an obstacle is close enough to warrant stopping
        Uses bounding box size and position to estimate proximity
        """
        try:
            x1, y1, x2, y2 = detection["bbox"]
            
            # Calculate obstacle size and position
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Calculate center position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Consider obstacle close if:
            # 1. It's large (close to camera)
            # 2. It's in the lower portion of the frame (close to user)
            # 3. It's in the center path
            
            is_large = area > 10000  # Adjust threshold as needed
            is_in_lower_frame = center_y > 300  # Lower half of typical 640x480 frame
            is_in_center_path = 200 < center_x < 440  # Center third of frame
            
            return is_large and is_in_lower_frame and is_in_center_path
            
        except Exception as e:
            logger.error(f"Error checking obstacle proximity: {e}")
            return False  # Default to not close if error occurs
    
    def _store_processed_frame(self, client_id: str, processed_frame):
        """Store processed frame for MJPEG streaming"""
        if not hasattr(self, 'processed_frames'):
            self.processed_frames = {}
        
        self.processed_frames[client_id] = processed_frame
    
    async def handle_fsm_state_change(self, message):
        """
        Handle FSM state changes and coordinate with video processing
        Integrates backend FSM state changes with frontend audio feedback
        """
        try:
            logger.info(f"WebRTC handler received FSM state change: {message.state.value}")
            
            # Adjust processing based on FSM state
            if message.state.value == "scanning":
                # During scanning, focus on optical flow analysis
                self.processing_mode = "optical_flow_priority"
                logger.info("WebRTC processing mode: optical flow priority for scanning")
                
            elif message.state.value == "guiding":
                # During guiding, focus on obstacle detection
                self.processing_mode = "obstacle_detection_priority"
                logger.info("WebRTC processing mode: obstacle detection priority for guiding")
                
            elif message.state.value == "blocked":
                # During blocked state, maintain obstacle detection but prepare for voice commands
                self.processing_mode = "obstacle_detection_with_audio"
                logger.info("WebRTC processing mode: obstacle detection with audio processing")
                
            elif message.state.value == "idle":
                # During idle, minimal processing
                self.processing_mode = "minimal"
                logger.info("WebRTC processing mode: minimal processing for idle state")
            
            # Store current FSM state for processing decisions
            self.current_fsm_state = message.state
            
        except Exception as e:
            logger.error(f"Error handling FSM state change in WebRTC handler: {e}")
    
    async def _process_audio_frames(self, track: MediaStreamTrack, client_id: str):
        """Process incoming audio frames with speech recognition and safety monitoring"""
        try:
            # Import speech processor and safety monitor
            from speech_recognition import speech_processor
            from safety_monitor import safety_monitor
            
            # Initialize error tracking for this client
            self.audio_error_counts[client_id] = 0
            
            while True:
                # Check circuit breaker before processing
                if not self._should_process_frames(client_id, "audio"):
                    logger.info(f"Audio processing paused for {client_id} due to circuit breaker")
                    await asyncio.sleep(5)  # Wait 5 seconds before checking again
                    continue
                
                audio_start_time = time.time()
                
                try:
                    frame = await track.recv()
                    logger.debug(f"Received audio frame from {client_id}")
                    
                    # Reset error count on successful frame reception
                    if self.audio_error_counts.get(client_id, 0) > 0:
                        logger.info(f"Audio frame reception recovered for {client_id}")
                        self.audio_error_counts[client_id] = 0
                    
                    # Process audio frame with speech recognition (with timing)
                    if speech_processor.is_initialized:
                        processing_start_time = time.time()
                        recognition_result = await speech_processor.process_webrtc_audio_frame(frame)
                        processing_end_time = time.time()
                        
                        # Monitor audio processing latency
                        await safety_monitor.monitor_processing_latency(
                            "audio_processing", 
                            processing_start_time, 
                            processing_end_time
                        )
                        
                        if recognition_result:
                            if recognition_result.get("status") == "success":
                                logger.info(f"Speech recognition result from {client_id}: {recognition_result}")
                                
                                # Check for command intents in results
                                for result in recognition_result.get("results", []):
                                    if result.get("command_intent"):
                                        logger.info(f"Command detected from {client_id}: {result['command_intent']}")
                                        
                                        # Voice commands are automatically processed via callback
                                        # No additional action needed here
                            
                            elif recognition_result.get("status") == "skipped":
                                logger.debug(f"Audio processing skipped for {client_id}: {recognition_result.get('reason')}")
                            
                            elif recognition_result.get("status") == "error":
                                logger.error(f"Speech recognition error for {client_id}: {recognition_result.get('error')}")
                                
                                # Check if this is a critical audio processing failure
                                if "critical" in recognition_result.get("error", "").lower():
                                    await safety_monitor.activate_emergency_protocols(
                                        f"Critical audio processing error: {recognition_result.get('error')}"
                                    )
                    
                    # Monitor overall audio frame processing time
                    audio_end_time = time.time()
                    if audio_end_time - audio_start_time > 1.0:  # 1 second warning threshold
                        logger.warning(f"Slow audio processing detected: {audio_end_time - audio_start_time:.3f}s")
                    
                except Exception as frame_error:
                    # Use circuit breaker error recording
                    error_count = self._record_frame_error(client_id, "audio", frame_error)
                    
                    # Check if this is a critical processing failure
                    audio_end_time = time.time()
                    if audio_end_time - audio_start_time > 2.0:  # 2 second timeout
                        await safety_monitor.activate_emergency_protocols(
                            f"Critical audio processing failure: {frame_error}"
                        )
                    
                    # If circuit breaker threshold reached, the next iteration will pause processing
                    continue
                
        except Exception as e:
            logger.error(f"Critical error in audio frame processing for {client_id}: {e}")
            
            # Trigger emergency protocols for critical audio processing failure
            try:
                from safety_monitor import safety_monitor
                await safety_monitor.activate_emergency_protocols(
                    f"Audio processing pipeline failure: {e}"
                )
            except:
                pass  # Don't let safety monitoring errors crash the handler
    
    async def handle_offer(self, client_id: str, offer_data: dict) -> dict:
        """Handle WebRTC offer from client"""
        try:
            # Create peer connection if it doesn't exist
            if client_id not in self.peer_connections:
                await self.create_peer_connection(client_id)
            
            pc = self.peer_connections[client_id]
            
            # Create RTCSessionDescription from offer
            offer = RTCSessionDescription(
                sdp=offer_data["sdp"],
                type=offer_data["type"]
            )
            
            # Set remote description
            await pc.setRemoteDescription(offer)
            logger.info(f"Set remote description for {client_id}")
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            logger.info(f"Created answer for {client_id}")
            
            return {
                "type": "answer",
                "sdp": pc.localDescription.sdp,
                "client_id": client_id
            }
            
        except Exception as e:
            logger.error(f"Error handling offer from {client_id}: {e}")
            raise
    
    async def handle_answer(self, client_id: str, answer_data: dict) -> dict:
        """Handle WebRTC answer from client"""
        try:
            if client_id not in self.peer_connections:
                raise ValueError(f"No peer connection found for client {client_id}")
            
            pc = self.peer_connections[client_id]
            
            # Create RTCSessionDescription from answer
            answer = RTCSessionDescription(
                sdp=answer_data["sdp"],
                type=answer_data["type"]
            )
            
            # Set remote description
            await pc.setRemoteDescription(answer)
            logger.info(f"Set remote description (answer) for {client_id}")
            
            return {
                "status": "success",
                "message": "Answer processed successfully",
                "client_id": client_id
            }
            
        except Exception as e:
            logger.error(f"Error handling answer from {client_id}: {e}")
            raise
    
    async def cleanup_connection(self, client_id: str):
        """Clean up WebRTC connection resources"""
        try:
            # Close peer connection
            if client_id in self.peer_connections:
                pc = self.peer_connections[client_id]
                await pc.close()
                del self.peer_connections[client_id]
            
            # Clean up media recorder
            if client_id in self.media_recorders:
                recorder = self.media_recorders[client_id]
                await recorder.stop()
                del self.media_recorders[client_id]
            
            # Remove connection state
            if client_id in self.connection_states:
                del self.connection_states[client_id]
            
            # Clean up error tracking
            if client_id in self.audio_error_counts:
                del self.audio_error_counts[client_id]
            if client_id in self.video_error_counts:
                del self.video_error_counts[client_id]
            
            # Clean up error timing
            audio_key = f"{client_id}_audio"
            video_key = f"{client_id}_video"
            if audio_key in self.last_error_time:
                del self.last_error_time[audio_key]
            if video_key in self.last_error_time:
                del self.last_error_time[video_key]
            
            # Clean up frame rate limiting data
            if client_id in self.last_processed_frame_time:
                del self.last_processed_frame_time[client_id]
            if client_id in self.latest_processed_frame:
                del self.latest_processed_frame[client_id]
            
            # Clean up parallel processing resources
            if client_id in self.background_processors:
                self.background_processors[client_id] = False  # Signal shutdown
                
            if client_id in self.frame_queues:
                # Send shutdown signal to background processor
                try:
                    self.frame_queues[client_id].put_nowait(None)
                except:
                    pass
                del self.frame_queues[client_id]
                
            if client_id in self.processing_locks:
                del self.processing_locks[client_id]
            
            # Clean up previous frames cache
            if hasattr(self, '_previous_frames') and client_id in self._previous_frames:
                del self._previous_frames[client_id]
            
            logger.info(f"Cleaned up WebRTC resources, error tracking, frame rate limiting, and parallel processing for {client_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up WebRTC connection for {client_id}: {e}")
    
    def get_connection_state(self, client_id: str) -> Optional[str]:
        """Get current connection state for client"""
        return self.connection_states.get(client_id)
    
    def get_active_connections(self) -> Dict[str, str]:
        """Get all active connections and their states"""
        return self.connection_states.copy()
    
    def get_latest_processed_frame(self, client_id: Optional[str] = None):
        """Get the latest processed frame for MJPEG streaming"""
        if client_id:
            frame_data = self.latest_processed_frame.get(client_id)
            return frame_data["frame"] if frame_data else None
        else:
            # Return the most recent frame from any client
            if not self.latest_processed_frame:
                return None
            
            # Find the most recently processed frame
            latest_client = max(self.latest_processed_frame.keys(), 
                              key=lambda k: self.latest_processed_frame[k]["timestamp"])
            return self.latest_processed_frame[latest_client]["frame"]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get performance statistics for frame processing including parallel processing metrics"""
        stats = {
            "target_fps": self.target_fps,
            "active_clients": len(self.latest_processed_frame),
            "parallel_processing": {
                "thread_pool_workers": self.thread_pool._max_workers,
                "active_background_processors": sum(1 for active in self.background_processors.values() if active),
                "queue_sizes": {client_id: queue.qsize() for client_id, queue in self.frame_queues.items()}
            },
            "clients": {}
        }
        
        current_time = time.time()
        for client_id, frame_data in self.latest_processed_frame.items():
            time_since_last = current_time - frame_data["timestamp"]
            
            # Get queue status
            queue_size = self.frame_queues[client_id].qsize() if client_id in self.frame_queues else 0
            is_processing = self.background_processors.get(client_id, False)
            
            stats["clients"][client_id] = {
                "last_processed": time_since_last,
                "processing_time": frame_data["processing_time"],
                "effective_fps": 1.0 / time_since_last if time_since_last > 0 else 0,
                "queue_size": queue_size,
                "parallel_processing_active": is_processing,
                "processing_mode": "parallel" if is_processing else "sequential"
            }
        
        return stats

# Global WebRTC connection manager instance
webrtc_manager = WebRTCConnectionManager()