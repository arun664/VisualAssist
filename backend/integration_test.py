#!/usr/bin/env python3
"""
Integration Test for Complete Navigation Workflow
Tests the end-to-end integration of all components for task 10.1

This test verifies:
- Client video/audio streaming with backend processing
- Backend FSM state changes with frontend audio feedback
- Processed video streaming from backend to frontend display
- Complete user journey from start to obstacle detection and recovery
"""

import asyncio
import logging
import time
import numpy as np
import cv2
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NavigationWorkflowIntegrationTest:
    """
    Comprehensive integration test for the complete navigation workflow
    Requirement 1.1, 1.3, 1.4, 1.5: Test complete user journey
    """
    
    def __init__(self):
        self.test_results = []
        self.test_passed = 0
        self.test_failed = 0
        
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting Navigation Workflow Integration Tests")
        
        # Test 1: Component initialization
        await self.test_component_initialization()
        
        # Test 2: WebSocket communication
        await self.test_websocket_communication()
        
        # Test 3: WebRTC video streaming integration
        await self.test_webrtc_video_integration()
        
        # Test 4: FSM state transitions with audio feedback
        await self.test_fsm_audio_integration()
        
        # Test 5: Computer vision processing pipeline
        await self.test_computer_vision_pipeline()
        
        # Test 6: Speech recognition integration
        await self.test_speech_recognition_integration()
        
        # Test 7: Safety monitoring integration
        await self.test_safety_monitoring_integration()
        
        # Test 8: Complete navigation workflow
        await self.test_complete_navigation_workflow()
        
        # Test 9: Error recovery scenarios
        await self.test_error_recovery_scenarios()
        
        # Test 10: Performance and latency
        await self.test_performance_latency()
        
        # Print results
        self.print_test_results()
        
        return self.test_failed == 0
    
    async def test_component_initialization(self):
        """Test that all components initialize correctly"""
        test_name = "Component Initialization"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Test FSM initialization
            from navigation_fsm import navigation_fsm
            assert navigation_fsm.get_current_state().value == "idle"
            
            # Test computer vision initialization
            from computer_vision import get_vision_processor
            vision_processor = get_vision_processor()
            assert vision_processor is not None
            
            # Test speech recognition initialization
            from speech_recognition import speech_processor
            assert speech_processor is not None
            
            # Test WebSocket manager initialization
            from websocket_manager import websocket_manager
            assert websocket_manager is not None
            
            # Test WebRTC manager initialization
            from webrtc_handler import webrtc_manager
            assert webrtc_manager is not None
            
            # Test safety monitor initialization
            from safety_monitor import safety_monitor
            assert safety_monitor is not None
            
            self.record_test_result(test_name, True, "All components initialized successfully")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Component initialization failed: {e}")
    
    async def test_websocket_communication(self):
        """Test WebSocket communication between components"""
        test_name = "WebSocket Communication"
        logger.info(f"Running test: {test_name}")
        
        try:
            from websocket_manager import websocket_manager
            from navigation_fsm import navigation_fsm
            
            # Test message parsing
            test_message = '{"type": "start", "timestamp": "2024-01-01T00:00:00"}'
            parsed = websocket_manager.parse_message(test_message)
            assert parsed["type"] == "start"
            
            # Test FSM state change callback
            state_changes_received = []
            
            def test_callback(message):
                state_changes_received.append(message.state.value)
            
            navigation_fsm.set_state_change_callback(test_callback)
            
            # Trigger state change
            from navigation_fsm import NavigationState
            await navigation_fsm.transition_to(
                NavigationState.STATE_SCANNING,
                speak_message="Test transition"
            )
            
            # Verify callback was called
            assert len(state_changes_received) > 0
            assert state_changes_received[-1] == "scanning"
            
            # Reset to idle
            await navigation_fsm.transition_to(NavigationState.STATE_IDLE)
            
            self.record_test_result(test_name, True, "WebSocket communication working correctly")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"WebSocket communication failed: {e}")
    
    async def test_webrtc_video_integration(self):
        """Test WebRTC video streaming integration with computer vision"""
        test_name = "WebRTC Video Integration"
        logger.info(f"Running test: {test_name}")
        
        try:
            from webrtc_handler import webrtc_manager
            from computer_vision import get_vision_processor
            
            # Create test frame
            test_frame = self.create_test_frame_with_obstacles()
            
            # Test computer vision processing
            vision_processor = get_vision_processor()
            processing_results = await vision_processor.process_frame_complete(test_frame)
            
            # Verify processing results structure
            assert "detections" in processing_results
            assert "motion_analysis" in processing_results
            assert "path_analysis" in processing_results
            assert "processed_frame" in processing_results
            
            # Test processed frame storage
            webrtc_manager._store_processed_frame("test_client", processing_results["processed_frame"])
            stored_frame = webrtc_manager.get_latest_processed_frame("test_client")
            assert stored_frame is not None
            
            self.record_test_result(test_name, True, "WebRTC video integration working correctly")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"WebRTC video integration failed: {e}")
    
    async def test_fsm_audio_integration(self):
        """Test FSM state transitions with audio feedback integration"""
        test_name = "FSM Audio Integration"
        logger.info(f"Running test: {test_name}")
        
        try:
            from navigation_fsm import navigation_fsm
            from websocket_manager import websocket_manager
            
            # Track audio messages
            audio_messages = []
            
            async def test_audio_callback(message):
                if hasattr(message, 'speak') and message.speak:
                    audio_messages.append(message.speak)
            
            navigation_fsm.set_state_change_callback(test_audio_callback)
            
            # Test complete state transition cycle with audio
            await navigation_fsm.handle_start_command()
            await asyncio.sleep(0.1)  # Allow callback processing
            
            await navigation_fsm.handle_user_stationary_and_path_clear()
            await asyncio.sleep(0.1)
            
            await navigation_fsm.handle_obstacle_detected()
            await asyncio.sleep(0.1)
            
            await navigation_fsm.handle_scan_command()
            await asyncio.sleep(0.1)
            
            await navigation_fsm.handle_stop_command()
            await asyncio.sleep(0.1)
            
            # Verify audio messages were generated
            assert len(audio_messages) > 0
            logger.info(f"Audio messages generated: {audio_messages}")
            
            self.record_test_result(test_name, True, f"FSM audio integration working - {len(audio_messages)} messages")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"FSM audio integration failed: {e}")
    
    async def test_computer_vision_pipeline(self):
        """Test computer vision processing pipeline"""
        test_name = "Computer Vision Pipeline"
        logger.info(f"Running test: {test_name}")
        
        try:
            from computer_vision import get_vision_processor
            
            vision_processor = get_vision_processor()
            
            # Test with frame containing obstacles
            test_frame = self.create_test_frame_with_obstacles()
            processing_results = await vision_processor.process_frame_complete(test_frame)
            
            # Verify obstacle detection
            detections = processing_results["detections"]
            assert isinstance(detections, list)
            
            # Test with frame for motion analysis
            test_frame2 = self.create_test_frame_with_motion()
            motion_results = await vision_processor.process_frame_complete(test_frame2)
            
            # Verify motion analysis
            motion_analysis = motion_results["motion_analysis"]
            assert "flow_magnitude" in motion_analysis
            assert "is_stationary" in motion_analysis
            
            # Test path calculation
            path_analysis = processing_results["path_analysis"]
            assert "safe_path_grid" in path_analysis
            assert "path_clear" in path_analysis
            
            self.record_test_result(test_name, True, "Computer vision pipeline working correctly")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Computer vision pipeline failed: {e}")
    
    async def test_speech_recognition_integration(self):
        """Test speech recognition integration with FSM"""
        test_name = "Speech Recognition Integration"
        logger.info(f"Running test: {test_name}")
        
        try:
            from speech_recognition import speech_processor
            from navigation_fsm import navigation_fsm
            
            # Test command detection
            test_transcriptions = [
                "scan the area",
                "start scan",
                "scan environment",
                "look around"
            ]
            
            scan_commands_detected = 0
            for transcription in test_transcriptions:
                command = speech_processor.detect_scan_intent(transcription)
                if command == "scan":
                    scan_commands_detected += 1
            
            assert scan_commands_detected > 0
            
            # Test state-aware processing
            from navigation_fsm import NavigationState
            speech_processor.set_current_fsm_state(NavigationState.STATE_BLOCKED)
            should_process = speech_processor.should_process_audio()
            assert should_process == True
            
            speech_processor.set_current_fsm_state(NavigationState.STATE_IDLE)
            should_process = speech_processor.should_process_audio()
            assert should_process == False
            
            self.record_test_result(test_name, True, f"Speech recognition integration working - {scan_commands_detected} commands detected")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Speech recognition integration failed: {e}")
    
    async def test_safety_monitoring_integration(self):
        """Test safety monitoring integration"""
        test_name = "Safety Monitoring Integration"
        logger.info(f"Running test: {test_name}")
        
        try:
            from safety_monitor import safety_monitor
            
            # Test latency monitoring
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing
            end_time = time.time()
            
            metric = await safety_monitor.monitor_processing_latency(
                "test_component", start_time, end_time
            )
            
            assert metric.component == "test_component"
            assert metric.duration > 0
            
            # Test safety status
            status = safety_monitor.get_safety_status()
            assert "monitoring_active" in status
            assert "performance_stats" in status
            
            # Test threshold configuration
            safety_monitor.set_latency_threshold("test_component", 0.05)
            assert safety_monitor.latency_thresholds["test_component"] == 0.05
            
            self.record_test_result(test_name, True, "Safety monitoring integration working correctly")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Safety monitoring integration failed: {e}")
    
    async def test_complete_navigation_workflow(self):
        """Test complete navigation workflow from start to finish"""
        test_name = "Complete Navigation Workflow"
        logger.info(f"Running test: {test_name}")
        
        try:
            from navigation_fsm import navigation_fsm
            from computer_vision import get_vision_processor
            from webrtc_handler import webrtc_manager
            
            # Reset to initial state
            from navigation_fsm import NavigationState
            await navigation_fsm.transition_to(NavigationState.STATE_IDLE)
            
            # Track workflow states
            workflow_states = []
            
            def track_states(message):
                workflow_states.append(message.state.value)
            
            navigation_fsm.set_state_change_callback(track_states)
            
            # Simulate complete workflow
            logger.info("Testing complete navigation workflow...")
            
            # 1. Start navigation
            success = await navigation_fsm.handle_start_command()
            assert success == True
            await asyncio.sleep(0.1)
            
            # 2. Simulate user becoming stationary with clear path
            test_frame = self.create_test_frame_clear_path()
            vision_processor = get_vision_processor()
            processing_results = await vision_processor.process_frame_complete(test_frame)
            
            # Simulate FSM processing the clear path
            await webrtc_manager._handle_fsm_processing(navigation_fsm, test_frame, processing_results)
            await asyncio.sleep(0.1)
            
            # 3. Simulate obstacle detection
            obstacle_frame = self.create_test_frame_with_obstacles()
            obstacle_results = await vision_processor.process_frame_complete(obstacle_frame)
            await webrtc_manager._handle_fsm_processing(navigation_fsm, obstacle_frame, obstacle_results)
            await asyncio.sleep(0.1)
            
            # 4. Simulate scan command
            success = await navigation_fsm.handle_scan_command()
            await asyncio.sleep(0.1)
            
            # 5. Stop navigation
            success = await navigation_fsm.handle_stop_command()
            await asyncio.sleep(0.1)
            
            # Verify workflow progression
            expected_states = ["scanning", "idle"]  # Minimum expected states
            for expected_state in expected_states:
                assert expected_state in workflow_states, f"Expected state {expected_state} not found in {workflow_states}"
            
            logger.info(f"Workflow states traversed: {workflow_states}")
            
            self.record_test_result(test_name, True, f"Complete workflow tested - states: {workflow_states}")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Complete navigation workflow failed: {e}")
    
    async def test_error_recovery_scenarios(self):
        """Test error recovery and emergency scenarios"""
        test_name = "Error Recovery Scenarios"
        logger.info(f"Running test: {test_name}")
        
        try:
            from navigation_fsm import navigation_fsm, NavigationState
            from safety_monitor import safety_monitor
            
            # Test emergency stop
            initial_state = navigation_fsm.get_current_state()
            success = await navigation_fsm.handle_emergency_stop("Integration test emergency")
            assert success == True
            assert navigation_fsm.get_current_state().value == "blocked"
            
            # Test safety monitoring emergency protocols
            emergency_triggered = False
            
            def emergency_callback(metric):
                nonlocal emergency_triggered
                if metric.level.value == "emergency":
                    emergency_triggered = True
            
            safety_monitor.add_alert_callback(emergency_callback)
            
            # Simulate critical latency violation
            start_time = time.time()
            await asyncio.sleep(0.6)  # Exceed frame processing threshold
            end_time = time.time()
            
            await safety_monitor.monitor_processing_latency("frame_processing", start_time, end_time)
            await asyncio.sleep(0.1)  # Allow callback processing
            
            # Reset emergency protocols
            await safety_monitor.reset_emergency_protocols()
            
            # Reset FSM to idle
            await navigation_fsm.transition_to(NavigationState.STATE_IDLE)
            
            self.record_test_result(test_name, True, "Error recovery scenarios tested successfully")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Error recovery scenarios failed: {e}")
    
    async def test_performance_latency(self):
        """Test system performance and latency requirements"""
        test_name = "Performance and Latency"
        logger.info(f"Running test: {test_name}")
        
        try:
            from computer_vision import get_vision_processor
            from safety_monitor import safety_monitor
            
            vision_processor = get_vision_processor()
            
            # Test frame processing performance
            test_frame = self.create_test_frame_with_obstacles()
            
            processing_times = []
            for i in range(5):  # Test 5 frames
                start_time = time.time()
                await vision_processor.process_frame_complete(test_frame)
                end_time = time.time()
                processing_times.append(end_time - start_time)
            
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            
            # Verify performance meets requirements
            assert avg_processing_time < 0.5, f"Average processing time {avg_processing_time}s exceeds 0.5s threshold"
            assert max_processing_time < 1.0, f"Max processing time {max_processing_time}s exceeds 1.0s threshold"
            
            logger.info(f"Performance metrics - Avg: {avg_processing_time:.3f}s, Max: {max_processing_time:.3f}s")
            
            self.record_test_result(test_name, True, f"Performance requirements met - Avg: {avg_processing_time:.3f}s")
            
        except Exception as e:
            self.record_test_result(test_name, False, f"Performance testing failed: {e}")
    
    def create_test_frame_with_obstacles(self):
        """Create test frame with simulated obstacles"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some obstacles (rectangles)
        cv2.rectangle(frame, (100, 200), (200, 350), (100, 100, 100), -1)  # Left obstacle
        cv2.rectangle(frame, (450, 250), (550, 400), (150, 150, 150), -1)  # Right obstacle
        
        # Add some background texture
        cv2.rectangle(frame, (0, 400), (640, 480), (50, 50, 50), -1)  # Ground
        
        return frame
    
    def create_test_frame_clear_path(self):
        """Create test frame with clear path"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background without obstacles in center path
        cv2.rectangle(frame, (0, 400), (640, 480), (50, 50, 50), -1)  # Ground
        cv2.rectangle(frame, (0, 0), (100, 400), (30, 30, 30), -1)    # Left wall
        cv2.rectangle(frame, (540, 0), (640, 400), (30, 30, 30), -1)  # Right wall
        
        return frame
    
    def create_test_frame_with_motion(self):
        """Create test frame with motion patterns"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some moving elements (different from previous frame)
        cv2.circle(frame, (320, 240), 50, (200, 200, 200), -1)  # Moving object
        
        return frame
    
    def record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result"""
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": time.time()
        }
        
        self.test_results.append(result)
        
        if passed:
            self.test_passed += 1
            logger.info(f"✅ {test_name}: {message}")
        else:
            self.test_failed += 1
            logger.error(f"❌ {test_name}: {message}")
    
    def print_test_results(self):
        """Print comprehensive test results"""
        logger.info("\n" + "="*80)
        logger.info("NAVIGATION WORKFLOW INTEGRATION TEST RESULTS")
        logger.info("="*80)
        
        for result in self.test_results:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            logger.info(f"{status} | {result['test']}: {result['message']}")
        
        logger.info("-"*80)
        logger.info(f"SUMMARY: {self.test_passed} passed, {self.test_failed} failed")
        
        if self.test_failed == 0:
            logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
            logger.info("Complete navigation workflow is working correctly.")
        else:
            logger.error(f"⚠️  {self.test_failed} TESTS FAILED")
            logger.error("Integration issues detected - review failed tests.")
        
        logger.info("="*80)

async def main():
    """Run integration tests"""
    test_suite = NavigationWorkflowIntegrationTest()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("Integration test completed successfully!")
        return 0
    else:
        logger.error("Integration test failed!")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)