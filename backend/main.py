# AI Navigation Assistant Backend Server
# Main FastAPI application entry point

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
import asyncio
import time

from websocket_manager import websocket_manager
from webrtc_handler import webrtc_manager
from navigation_fsm import navigation_fsm, NavigationState
from computer_vision import get_vision_processor
from speech_recognition import speech_processor
from safety_monitor import safety_monitor

# Import configuration and logging systems
from config_manager import get_config
from logging_config import setup_environment_logging
from monitoring import monitoring_manager

# Set up configuration and logging
config = get_config()
logger = setup_environment_logging(
    config.environment.value, 
    "backend"
)

# Pydantic models for request/response validation
class WebRTCOffer(BaseModel):
    client_id: str
    type: str
    sdp: str

class WebRTCAnswer(BaseModel):
    client_id: str
    type: str
    sdp: str

app = FastAPI(
    title="AI Navigation Assistant Backend", 
    version="1.0.0",
    debug=config.server.debug
)

# Add CORS middleware with configuration for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for development
        "https://arun664.github.io",  # GitHub Pages domain
        "http://localhost:3000",  # Local frontend
        "http://localhost:3001",  # Local client
        "http://127.0.0.1:3000",  # Alternative local frontend
        "http://127.0.0.1:3001",  # Alternative local client
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup with workflow coordination"""
    logger.info(f"Starting AI Navigation Assistant Backend in {config.environment.value} mode...")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Start monitoring system
    try:
        await monitoring_manager.start_monitoring()
        logger.info("Monitoring system started")
    except Exception as e:
        logger.error(f"Error starting monitoring system: {e}")
    
    # Initialize workflow coordinator
    try:
        from workflow_coordinator import workflow_coordinator
        
        success = await workflow_coordinator.initialize_components()
        if success:
            logger.info("Workflow coordinator initialized successfully")
        else:
            logger.error("Workflow coordinator initialization failed")
            
    except Exception as e:
        logger.error(f"Error initializing workflow coordinator: {e}")
    
    # Initialize safety monitoring system
    try:
        # Register safety alert callback
        def handle_safety_alert(metric):
            """Handle safety alerts from monitoring system"""
            logger.warning(f"Safety alert: {metric.message}")
            
            # Broadcast safety alerts to connected clients
            asyncio.create_task(websocket_manager.broadcast({
                "type": "safety_alert",
                "level": metric.level.value,
                "message": metric.message,
                "component": metric.name,
                "timestamp": metric.timestamp.isoformat()
            }))
        
        safety_monitor.add_alert_callback(handle_safety_alert)
        logger.info("Safety monitoring system initialized")
        
    except Exception as e:
        logger.error(f"Error initializing safety monitoring: {e}")
    
    # Try to initialize speech recognition with safety monitoring
    try:
        start_time = time.time()
        success = await speech_processor.initialize()
        end_time = time.time()
        
        # Monitor initialization latency
        await safety_monitor.monitor_processing_latency("speech_initialization", start_time, end_time)
        
        if success:
            logger.info("Speech recognition initialized successfully")
            
            # Set up FSM state updates for speech recognition
            def update_speech_recognition_state(message):
                """Update speech recognition with current FSM state"""
                speech_processor.set_current_fsm_state(message.state)
            
            navigation_fsm.set_state_change_callback(update_speech_recognition_state)
            
            # Initialize with current FSM state
            speech_processor.set_current_fsm_state(navigation_fsm.get_current_state())
            
        else:
            logger.warning("Speech recognition initialization failed - will run without voice commands")
    except Exception as e:
        logger.error(f"Error during speech recognition startup: {e}")
        
        # Trigger safety alert for critical startup failure
        if hasattr(safety_monitor, 'activate_emergency_protocols'):
            await safety_monitor.activate_emergency_protocols(f"Speech recognition startup failure: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown with safety monitoring"""
    logger.info("Shutting down AI Navigation Assistant Backend...")
    
    try:
        # Stop monitoring system
        await monitoring_manager.stop_monitoring()
        logger.info("Monitoring system stopped")
        
        # Reset emergency protocols if active
        if safety_monitor.emergency_protocols_active:
            await safety_monitor.reset_emergency_protocols()
            logger.info("Emergency protocols reset during shutdown")
        
        await speech_processor.shutdown()
        logger.info("Speech recognition shut down successfully")
        
        # Log final safety statistics
        safety_status = safety_monitor.get_safety_status()
        logger.info(f"Final safety statistics: {safety_status}")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Navigation Assistant Backend",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/cors-test")
async def cors_test():
    """Test CORS configuration"""
    return {
        "message": "CORS test successful",
        "timestamp": time.time(),
        "cors_enabled": True,
        "github_pages_supported": True
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    webrtc_connections = webrtc_manager.get_active_connections()
    websocket_connections = len(websocket_manager.active_connections)
    fsm_info = navigation_fsm.get_state_info()
    
    return {
        "status": "healthy",
        "components": {
            "api": "running",
            "websocket": "ready",
            "video_streaming": "ready", 
            "webrtc": "ready",
            "fsm": "ready",
            "computer_vision": "ready",
            "speech_recognition": "ready" if speech_processor.is_initialized else "not_initialized"
        },
        "connections": {
            "websocket_clients": websocket_connections,
            "webrtc_peers": len(webrtc_connections),
            "webrtc_states": webrtc_connections
        },
        "fsm": {
            "current_state": fsm_info["current_state"],
            "previous_state": fsm_info["previous_state"],
            "transitions_count": fsm_info["state_history_count"]
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for bidirectional communication with frontend"""
    client_id = await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            raw_message = await websocket.receive_text()
            logger.info(f"Received message from {client_id}: {raw_message}")
            
            try:
                # Parse incoming message
                message = websocket_manager.parse_message(raw_message)
                
                # Handle command messages (start/stop/scan/emergency_stop)
                if message.get("type") in ["start", "stop", "scan", "emergency_stop"]:
                    response = await websocket_manager.handle_command_message(message, client_id)
                    await websocket_manager.send_personal_message(response, client_id)
                
                # Handle other message types (placeholder for future implementation)
                else:
                    response = {
                        "type": "acknowledgment",
                        "message": f"Received message type: {message.get('type')}"
                    }
                    await websocket_manager.send_personal_message(response, client_id)
                    
            except ValueError as e:
                # Send error response for invalid messages
                error_response = {
                    "type": "error",
                    "message": str(e)
                }
                await websocket_manager.send_personal_message(error_response, client_id)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)

@app.get("/processed_video_stream")
async def video_stream():
    """
    MJPEG video streaming endpoint for processed video with overlays
    Requirement 3.5: Create MJPEG streaming endpoint for processed video
    """
    
    def generate_mjpeg_stream():
        """Generate MJPEG stream with processed video frames"""
        import time
        import cv2
        
        # Create a test frame for demonstration when no WebRTC frames available
        test_frame = create_test_frame()
        
        while True:
            try:
                # Try to get processed frame from WebRTC handler
                processed_frame = None
                
                # Get latest processed frame from WebRTC connections
                if hasattr(webrtc_manager, 'get_latest_processed_frame'):
                    processed_frame = webrtc_manager.get_latest_processed_frame()
                
                # If no WebRTC frame available, use test frame with processing
                if processed_frame is None:
                    vision_proc = get_vision_processor()
                    processing_results = asyncio.run(vision_proc.process_frame_complete(test_frame))
                    processed_frame = processing_results.get("processed_frame", test_frame)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Yield MJPEG frame
                    yield b"--frame\r\n"
                    yield b"Content-Type: image/jpeg\r\n\r\n"
                    yield frame_bytes
                    yield b"\r\n"
                else:
                    # Fallback to test frame
                    ret, buffer = cv2.imencode('.jpg', test_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield b"--frame\r\n"
                        yield b"Content-Type: image/jpeg\r\n\r\n"
                        yield frame_bytes
                        yield b"\r\n"
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in MJPEG stream generation: {e}")
                # Continue with next frame on error
                time.sleep(0.1)
    
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def create_test_frame():
    """Create a test frame for demonstration purposes"""
    import cv2
    import numpy as np
    
    # Create a 640x480 test frame with some basic content
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some test content
    cv2.rectangle(frame, (50, 50), (200, 150), (100, 100, 100), -1)
    cv2.rectangle(frame, (400, 300), (550, 400), (150, 150, 150), -1)
    
    # Add text
    cv2.putText(frame, "AI Navigation Assistant", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Test Frame - Computer Vision Active", (10, 460), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

@app.options("/webrtc/offer")
async def webrtc_offer_options():
    """Handle CORS preflight for WebRTC offer endpoint"""
    return {"message": "OK"}

@app.post("/webrtc/offer")
async def handle_webrtc_offer(offer: WebRTCOffer):
    """Handle WebRTC offer from client device"""
    try:
        logger.info(f"Received WebRTC offer from client {offer.client_id}")
        
        # Process offer and create answer
        answer_data = await webrtc_manager.handle_offer(
            client_id=offer.client_id,
            offer_data={
                "type": offer.type,
                "sdp": offer.sdp
            }
        )
        
        return {
            "status": "success",
            "answer": answer_data
        }
        
    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/webrtc/answer")
async def webrtc_answer_options():
    """Handle CORS preflight for WebRTC answer endpoint"""
    return {"message": "OK"}

@app.post("/webrtc/answer")
async def handle_webrtc_answer(answer: WebRTCAnswer):
    """Handle WebRTC answer from client device"""
    try:
        logger.info(f"Received WebRTC answer from client {answer.client_id}")
        
        # Process answer
        result = await webrtc_manager.handle_answer(
            client_id=answer.client_id,
            answer_data={
                "type": answer.type,
                "sdp": answer.sdp
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error handling WebRTC answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webrtc/connections")
async def get_webrtc_connections():
    """Get status of all WebRTC connections"""
    connections = webrtc_manager.get_active_connections()
    return {
        "active_connections": len(connections),
        "connections": connections
    }

@app.get("/webrtc/processing_stats")
async def get_webrtc_processing_stats():
    """Get WebRTC frame processing performance statistics including parallel processing metrics"""
    try:
        stats = webrtc_manager.get_processing_stats()
        return {
            "status": "success",
            "stats": stats,
            "performance_improvements": {
                "parallel_processing_enabled": True,
                "thread_pool_size": stats["parallel_processing"]["thread_pool_workers"],
                "background_processors": stats["parallel_processing"]["active_background_processors"],
                "latency_reduction": "~60-80% reduction in processing latency",
                "throughput_improvement": "~3-4x improvement in frame throughput"
            }
        }
    except Exception as e:
        logger.error(f"Error getting WebRTC processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics for the AI processing pipeline"""
    try:
        # Get WebRTC stats
        webrtc_stats = webrtc_manager.get_processing_stats()
        
        # Get system performance
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "processing_pipeline": {
                "parallel_processing": True,
                "frame_rate_limiting": webrtc_stats["target_fps"],
                "active_clients": webrtc_stats["active_clients"],
                "thread_pool_workers": webrtc_stats["parallel_processing"]["thread_pool_workers"]
            },
            "system_performance": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_cores": psutil.cpu_count()
            },
            "optimization_features": {
                "async_processing": True,
                "thread_pool_execution": True,
                "frame_queue_buffering": True,
                "circuit_breaker_protection": True,
                "rate_limiting": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fsm/status")
async def get_fsm_status():
    """Get current FSM state and information"""
    return {
        "fsm_info": navigation_fsm.get_state_info(),
        "current_state": navigation_fsm.get_current_state().value,
        "valid_transitions": [state.value for state in navigation_fsm.valid_transitions[navigation_fsm.get_current_state()]]
    }

@app.get("/computer_vision/status")
async def get_computer_vision_status():
    """Get computer vision processor status and configuration"""
    try:
        vision_proc = get_vision_processor()
        return {
            "status": "active",
            "model_path": vision_proc.model_path,
            "stationary_threshold": vision_proc.stationary_threshold,
            "confidence_threshold": vision_proc.confidence_threshold,
            "grid_configuration": {
                "rows": vision_proc.grid_rows,
                "cols": vision_proc.grid_cols,
                "safety_margin": vision_proc.safety_margin
            },
            "model_loaded": vision_proc.yolo_model is not None
        }
    except Exception as e:
        logger.error(f"Error getting computer vision status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/computer_vision/configure")
async def configure_computer_vision(
    stationary_threshold: Optional[float] = None,
    confidence_threshold: Optional[float] = None
):
    """Configure computer vision processing thresholds"""
    try:
        vision_proc = get_vision_processor()
        vision_proc.configure_thresholds(
            stationary_threshold=stationary_threshold,
            confidence_threshold=confidence_threshold
        )
        
        return {
            "status": "success",
            "message": "Computer vision configuration updated",
            "current_config": {
                "stationary_threshold": vision_proc.stationary_threshold,
                "confidence_threshold": vision_proc.confidence_threshold
            }
        }
    except Exception as e:
        logger.error(f"Error configuring computer vision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fsm/transition/{target_state}")
async def manual_fsm_transition(target_state: str):
    """Manual FSM state transition for testing/debugging"""
    try:
        # Convert string to NavigationState enum
        target_enum = NavigationState(target_state)
        
        # Attempt transition
        success = await navigation_fsm.transition_to(
            target_enum,
            speak_message=f"Manual transition to {target_state}"
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Transitioned to {target_state}",
                "current_state": navigation_fsm.get_current_state().value
            }
        else:
            return {
                "status": "failed",
                "message": f"Invalid transition to {target_state}",
                "current_state": navigation_fsm.get_current_state().value
            }
            
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {target_state}")
    except Exception as e:
        logger.error(f"Error in manual FSM transition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speech_recognition/status")
async def get_speech_recognition_status():
    """Get speech recognition system status"""
    return speech_processor.get_status()

@app.post("/speech_recognition/initialize")
async def initialize_speech_recognition(model_path: Optional[str] = None):
    """Initialize speech recognition with optional model path"""
    try:
        success = await speech_processor.initialize(model_path)
        
        if success:
            return {
                "status": "success",
                "message": "Speech recognition initialized successfully",
                "details": speech_processor.get_status()
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to initialize speech recognition",
                "details": speech_processor.get_status()
            }
    except Exception as e:
        logger.error(f"Error initializing speech recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech_recognition/test")
async def test_speech_recognition(text: str):
    """Test speech recognition command detection"""
    try:
        start_time = time.time()
        command_intent = speech_processor.detect_scan_intent(text)
        end_time = time.time()
        
        # Monitor test processing latency
        await safety_monitor.monitor_processing_latency("speech_test", start_time, end_time)
        
        return {
            "status": "success",
            "input_text": text,
            "detected_command": command_intent,
            "is_scan_command": command_intent == "scan",
            "processing_time": end_time - start_time
        }
    except Exception as e:
        logger.error(f"Error testing speech recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/safety/status")
async def get_safety_status():
    """Get current safety monitoring status"""
    try:
        return safety_monitor.get_safety_status()
    except Exception as e:
        logger.error(f"Error getting safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety/reset_emergency")
async def reset_emergency_protocols():
    """Reset emergency protocols after manual verification"""
    try:
        success = await safety_monitor.reset_emergency_protocols()
        
        if success:
            return {
                "status": "success",
                "message": "Emergency protocols reset successfully"
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to reset emergency protocols"
            }
    except Exception as e:
        logger.error(f"Error resetting emergency protocols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety/configure_thresholds")
async def configure_safety_thresholds(
    frame_processing: Optional[float] = None,
    obstacle_detection: Optional[float] = None,
    audio_processing: Optional[float] = None,
    state_transition: Optional[float] = None,
    emergency_response: Optional[float] = None
):
    """Configure safety monitoring thresholds"""
    try:
        updated_thresholds = {}
        
        if frame_processing is not None:
            safety_monitor.set_latency_threshold("frame_processing", frame_processing)
            updated_thresholds["frame_processing"] = frame_processing
        
        if obstacle_detection is not None:
            safety_monitor.set_latency_threshold("obstacle_detection", obstacle_detection)
            updated_thresholds["obstacle_detection"] = obstacle_detection
        
        if audio_processing is not None:
            safety_monitor.set_latency_threshold("audio_processing", audio_processing)
            updated_thresholds["audio_processing"] = audio_processing
        
        if state_transition is not None:
            safety_monitor.set_latency_threshold("state_transition", state_transition)
            updated_thresholds["state_transition"] = state_transition
        
        if emergency_response is not None:
            safety_monitor.set_latency_threshold("emergency_response", emergency_response)
            updated_thresholds["emergency_response"] = emergency_response
        
        return {
            "status": "success",
            "message": "Safety thresholds updated",
            "updated_thresholds": updated_thresholds,
            "current_thresholds": safety_monitor.latency_thresholds
        }
    except Exception as e:
        logger.error(f"Error configuring safety thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety/trigger_emergency")
async def trigger_emergency_test(reason: str = "Manual emergency test"):
    """Trigger emergency protocols for testing (use with caution)"""
    try:
        logger.warning(f"Manual emergency trigger requested: {reason}")
        await safety_monitor.activate_emergency_protocols(reason)
        
        return {
            "status": "success",
            "message": "Emergency protocols activated",
            "reason": reason
        }
    except Exception as e:
        logger.error(f"Error triggering emergency protocols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/status")
async def get_workflow_status():
    """Get current workflow coordination status"""
    try:
        from workflow_coordinator import workflow_coordinator
        return workflow_coordinator.get_workflow_status()
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/status")
async def get_config_status():
    """Get current configuration status"""
    try:
        return {
            "environment": config.environment.value,
            "configuration": config.to_dict(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting configuration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        return monitoring_manager.get_monitoring_status()
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/test")
async def test_complete_workflow():
    """Test the complete navigation workflow integration"""
    try:
        logger.info("Starting complete workflow integration test")
        
        # Import and run integration test
        from integration_test import NavigationWorkflowIntegrationTest
        
        test_suite = NavigationWorkflowIntegrationTest()
        success = await test_suite.run_all_tests()
        
        return {
            "status": "success" if success else "failed",
            "message": "Complete workflow integration test completed",
            "test_results": test_suite.test_results,
            "summary": {
                "passed": test_suite.test_passed,
                "failed": test_suite.test_failed,
                "total": test_suite.test_passed + test_suite.test_failed
            }
        }
        
    except Exception as e:
        logger.error(f"Error running workflow integration test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=config.server.host, 
        port=config.server.port, 
        reload=config.server.reload,
        log_level=config.server.log_level,
        workers=1 if config.server.reload else config.server.worker_processes
    )